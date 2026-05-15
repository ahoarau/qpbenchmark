#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parallel task execution helpers for benchmarks."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import List, Optional, Tuple

from tqdm import tqdm

from .spdlog import logging
from .utils import time_solve_problem

_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class _SolveResultProxy:
    """Minimal picklable stand-in for qpsolvers.Solution."""

    def __init__(
        self,
        found,
        primal_res: float,
        dual_res: float,
        duality_gap_val: float,
    ):
        self.found = found
        self._primal_res = primal_res
        self._dual_res = dual_res
        self._duality_gap = duality_gap_val

    def primal_residual(self) -> float:
        return self._primal_res

    def dual_residual(self) -> float:
        return self._dual_res

    def duality_gap(self) -> float:
        return self._duality_gap


def _format_runtime(seconds: float) -> str:
    """Format a runtime duration with an appropriate unit."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}\u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.2f}s"


def _find_blas_lib(lib_prefix: str) -> Optional[str]:
    """Return the filesystem path of a loaded BLAS library, or None."""
    import ctypes.util

    path = ctypes.util.find_library(lib_prefix)
    if path is not None:
        return path
    try:
        with open("/proc/self/maps") as maps_file:
            for line in maps_file:
                if lib_prefix in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


def _worker_init():
    """Initializer for every pool worker process."""
    import ctypes

    for var, val in _THREAD_ENV_VARS.items():
        os.environ[var] = val

    blas_setters = [
        ("libopenblas", "openblas_set_num_threads"),
        ("libmkl_rt", "MKL_Set_Num_Threads"),
        ("libblis", "bli_thread_set_num_threads"),
    ]
    for lib_prefix, func_name in blas_setters:
        lib_path = _find_blas_lib(lib_prefix)
        if lib_path is None:
            continue
        try:
            lib = ctypes.CDLL(lib_path)
            getattr(lib, func_name)(1)
        except Exception:
            pass


def _solve_task(
    problem,
    solver: str,
    settings: str,
    solver_kwargs: dict,
    verbose: bool,
):
    """Execute a single QP solve from a process-pool worker."""
    logging.debug(
        f"Worker {os.getpid()} started: "
        f"{problem.name}/{solver}/{settings}"
    )

    if verbose:
        logging.info(
            f"Solving {problem.name} by {solver} "
            f"with {settings} settings..."
        )

    try:
        solution, runtime = time_solve_problem(
            problem, solver, **solver_kwargs
        )
        proxy = _SolveResultProxy(
            found=solution.found,
            primal_res=solution.primal_residual(),
            dual_res=solution.dual_residual(),
            duality_gap_val=solution.duality_gap(),
        )
        success = True
    except Exception as exc:
        logging.error(
            f"Unhandled exception solving {problem.name} "
            f"with {solver}/{settings}: {exc}"
        )
        proxy = _SolveResultProxy(
            found=None,
            primal_res=float("inf"),
            dual_res=float("inf"),
            duality_gap_val=float("inf"),
        )
        runtime = 0.0
        success = False

    return proxy, runtime, success


def _get_future_result(future, future_to_task):
    """Unpack a completed future into (problem, solver, settings, solution, runtime, success)."""
    problem, solver, settings = future_to_task[future]
    try:
        solution, runtime, success = future.result()
    except Exception as exc:
        logging.error(
            f"Unexpected error collecting result for "
            f"{problem.name}/{solver}/{settings}: {exc}"
        )
        solution = _SolveResultProxy(
            found=None,
            primal_res=float("inf"),
            dual_res=float("inf"),
            duality_gap_val=float("inf"),
        )
        runtime = 0.0
        success = False
    return problem, solver, settings, solution, runtime, success


def _get_executor_context():
    """Return the multiprocessing context for the process pool.

    Uses ``fork`` on Linux (fast, safe for single-threaded startup).
    Uses the platform default (``spawn``) on macOS and Windows to avoid
    fork-after-threads deadlocks caused by solver libraries that start
    background threads at import time (Gurobi, JAX, OpenBLAS, etc.).
    """
    import sys

    if sys.platform.startswith("linux"):
        try:
            return multiprocessing.get_context("fork")
        except ValueError:
            pass
    return None  # spawn on macOS / Windows (system default)


def _execute_tasks(
    solve_tasks: List[Tuple],
    test_set,
    results,
    max_workers: int,
    verbose: bool,
    stop_event,
    pool_cell: List,
    progress_bar: Optional[tqdm],
) -> int:
    """Execute all solve tasks in a process pool using as_completed."""
    status_interval = 4.0
    write_interval = 30.0
    last_status_time = perf_counter()
    last_write_time = perf_counter()
    unsaved_results = 0
    nb_calls = 0

    logging.info(
        f"Launching process pool with {max_workers} workers "
        f"for {len(solve_tasks)} solve tasks..."
    )

    future_to_task = {}
    futures = set()

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            mp_context=_get_executor_context(),
        ) as executor:
            pool_cell.append(executor)

            for problem, solver, settings in solve_tasks:
                if stop_event.is_set():
                    break
                future = executor.submit(
                    _solve_task,
                    problem,
                    solver,
                    settings,
                    test_set.solver_settings[settings][solver],
                    verbose,
                )
                future_to_task[future] = (problem, solver, settings)
                futures.add(future)

            completed_count = 0
            for future in as_completed(futures):
                if stop_event.is_set():
                    break

                problem, solver, settings, solution, runtime, success = (
                    _get_future_result(future, future_to_task)
                )

                if success:
                    nb_calls += 1
                logging.info(
                    f"{'Solved' if success else 'Failed'} "
                    f"{problem.name} / {solver} / {settings} "
                    f"in {_format_runtime(runtime)} "
                    f"(found={solution.found})"
                )
                results.update(
                    problem, solver, settings, solution, runtime
                )
                unsaved_results += 1
                completed_count += 1

                if perf_counter() - last_write_time >= write_interval:
                    results.write()
                    last_write_time = perf_counter()
                    unsaved_results = 0

                if progress_bar is not None:
                    progress_bar.update(1)

                now = perf_counter()
                if now - last_status_time >= status_interval:
                    n_in_flight = len(futures) - completed_count
                    if n_in_flight > 0:
                        logging.info(
                            f"Progress: {completed_count}/{len(solve_tasks)} "
                            f"solves done, {n_in_flight} in flight"
                        )
                    last_status_time = now

                if progress_bar is not None:
                    progress_bar.refresh()

            if stop_event.is_set():
                for future in futures:
                    future.cancel()
                logging.warning(
                    "Interrupted, cancelled remaining futures..."
                )

    except KeyboardInterrupt:
        stop_event.set()
        logging.warning("Interrupted, stopping gracefully...")

    if unsaved_results > 0 or stop_event.is_set():
        logging.info("Writing final results to disk...")
        results.write()

    return nb_calls
