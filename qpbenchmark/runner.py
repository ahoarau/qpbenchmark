"""Benchmark runner and execution logic."""

import multiprocessing
import signal
from time import perf_counter, sleep
from typing import Any, Dict, List, Tuple

import qpsolvers
from tqdm import tqdm

from .results import Results
from .spdlog import logging
from .worker import _solve_task, _worker_init


def _format_runtime(seconds: float) -> str:
    """Format a runtime duration with an appropriate unit."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    else:
        return f"{seconds:.2f}s"


class BenchmarkRunner:
    """Orchestrates the process pool and task queue."""

    def __init__(self, max_workers: int, verbose: bool):
        """Initialize the runner."""
        self.max_workers = max_workers
        self.verbose = verbose
        self.status_interval = 4.0
        self.write_interval = 30.0

        self.stop_event = multiprocessing.Event()
        self.pool_cell: List = []

        self.original_sigint = signal.signal(
            signal.SIGINT, self._signal_handler
        )

    def _signal_handler(self, signum, frame):
        logging.warning("Received interrupt signal (CTRL+C), stopping...")
        self.stop_event.set()
        if self.pool_cell:
            self.pool_cell[0].terminate()

    def execute(
        self,
        task_generator,
        nb_total: int,
        results: Results,
        stats: Dict[str, int],
    ):
        """Execute tasks from the generator using a process pool."""
        pool = multiprocessing.Pool(
            processes=self.max_workers, initializer=_worker_init
        )
        self.pool_cell.append(pool)

        progress_bar = None
        if not self.verbose:
            progress_bar = tqdm(
                total=nb_total,
                initial=0,
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.1,
                maxinterval=1.0,
                smoothing=0.1,
            )

        start_counter = perf_counter()
        last_status_time = perf_counter()
        last_write_time = perf_counter()
        unsaved_results = 0
        nb_calls = 0

        max_queued = self.max_workers * 2
        pending: List[Tuple[Any, str, str, str]] = []
        generator_exhausted = False

        logging.info(
            f"Launching process pool with {self.max_workers} workers "
            f"(streaming mode)..."
        )

        try:
            while (
                pending or not generator_exhausted
            ) and not self.stop_event.is_set():
                # Refill pending queue
                while (
                    len(pending) < max_queued
                    and not generator_exhausted
                    and not self.stop_event.is_set()
                ):
                    try:
                        action, problem, solver, settings, kwargs = next(
                            task_generator
                        )

                        if action == "skip":
                            stats["skip"] += 1
                            logging.debug(
                                f"{problem.name} already solved by {solver} "
                                f"with {settings} settings, skipping."
                            )
                            if progress_bar is not None:
                                progress_bar.initial += 1
                                progress_bar.n += 1

                        elif action == "skip_timeout":
                            stats["skip_timeout"] += 1
                            logging.info(
                                f"Skipping {problem.name} / {solver} / "
                                f"{settings} (previous timeout)."
                            )
                            if progress_bar is not None:
                                progress_bar.initial += 1
                                progress_bar.n += 1

                        elif action == "record_failure":
                            stats["record_failure"] += 1
                            solution = qpsolvers.Solution(problem)
                            results.update(
                                problem, solver, settings, solution, 0.0
                            )
                            unsaved_results += 1
                            logging.debug(
                                f"Recording failure for "
                                f"{problem.name} / {solver} / {settings}."
                            )
                            if progress_bar is not None:
                                progress_bar.update(1)

                        elif action == "solve":
                            stats["solve"] += 1
                            ar = pool.apply_async(
                                _solve_task,
                                args=(
                                    problem,
                                    solver,
                                    settings,
                                    kwargs,
                                    self.verbose,
                                ),
                            )
                            # Let problem get garbage collected by only
                            # saving its name in pending queue.
                            pending.append(
                                (ar, problem.name, solver, settings)
                            )

                    except StopIteration:
                        generator_exhausted = True
                        break

                # Process pending results
                still_pending = []
                newly_done = []
                for ar, problem_name, solver, settings in pending:
                    if ar.ready():
                        newly_done.append((ar, problem_name, solver, settings))
                    else:
                        still_pending.append(
                            (ar, problem_name, solver, settings)
                        )

                for ar, problem_name, solver, settings in newly_done:
                    try:
                        (
                            found,
                            primal_res,
                            dual_res,
                            duality_gap,
                            runtime,
                            success,
                        ) = ar.get()
                    except Exception as exc:
                        logging.error(
                            f"Unexpected error collecting result for "
                            f"{problem_name}/{solver}/{settings}: {exc}"
                        )
                        found = None
                        primal_res = float("inf")
                        dual_res = float("inf")
                        duality_gap = float("inf")
                        runtime = 0.0
                        success = False

                    if success:
                        nb_calls += 1
                    logging.info(
                        f"{'Solved' if success else 'Failed'} "
                        f"{problem_name} / {solver} / {settings} "
                        f"in {_format_runtime(runtime)} "
                        f"(found={found})"
                    )

                    results.update(
                        problem_name,
                        solver,
                        settings,
                        runtime=runtime,
                        found=found,
                        primal_residual=primal_res,
                        dual_residual=dual_res,
                        duality_gap=duality_gap,
                    )
                    unsaved_results += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

                pending = still_pending

                # Periodic saves
                if (
                    perf_counter() - last_write_time >= self.write_interval
                    and unsaved_results > 0
                ):
                    results.write()
                    last_write_time = perf_counter()
                    unsaved_results = 0

                # Periodic status log
                now = perf_counter()
                if now - last_status_time >= self.status_interval:
                    n_done = stats["solve"] - len(pending)
                    n_in_flight = len(pending)
                    if n_in_flight > 0:
                        logging.info(
                            f"Progress: {n_done}/{stats['solve']} solves "
                            f"done, {n_in_flight} in flight"
                        )
                    last_status_time = now

                if progress_bar is not None:
                    progress_bar.refresh()

                # Avoid busy-waiting when tasks are still running
                if not newly_done and not generator_exhausted:
                    if len(pending) >= max_queued:
                        pending[0][0].wait(timeout=0.05)
                elif not newly_done and generator_exhausted:
                    if pending:
                        pending[0][0].wait(timeout=0.05)
                    else:
                        sleep(0.05)

            if self.stop_event.is_set():
                pool.terminate()
            else:
                pool.close()
            pool.join()

        except KeyboardInterrupt:
            self.stop_event.set()
            pool.terminate()
            pool.join()
            logging.warning("Interrupted, stopping gracefully...")

        finally:
            signal.signal(signal.SIGINT, self.original_sigint)
            if progress_bar is not None:
                progress_bar.close()

        # Final safety-net write
        if unsaved_results > 0 or self.stop_event.is_set():
            logging.info("Writing final results to disk...")
            results.write()

        duration = perf_counter() - start_counter

        logging.info(
            f"Task breakdown: {stats['solve']} solved, "
            f"{stats['skip']} skipped (existing), "
            f"{stats['skip_timeout']} skipped (timeout), "
            f"{stats['record_failure']} known failures"
        )

        if not self.stop_event.is_set():
            logging.info(f"Ran the test set in {duration:.0f} seconds")
            logging.info(f"Made {nb_calls} QP solver calls")
        else:
            logging.info(f"Partial run completed in {duration:.0f} seconds")
            logging.info(
                f"Made {nb_calls} QP solver calls before interruption"
            )
