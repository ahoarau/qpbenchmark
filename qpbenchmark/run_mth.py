#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Multithreaded variant of the benchmark runner."""

import multiprocessing
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas
import qpsolvers
from qpsolvers.exceptions import SolverNotFound
from tqdm import tqdm

from .results import Results
from .spdlog import logging
from .test_set import TestSet
from .utils import time_solve_problem

# Environment variables that limit BLAS/OpenMP thread counts to 1 per worker.
# _init_worker() applies them in each subprocess so the setting holds across
# process boundaries regardless of what the main process has set.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class _SolveResult:
    """Picklable result returned from a worker process.

    Stands in for ``qpsolvers.Solution`` across process boundaries, exposing
    only the four fields that :meth:`Results.update` reads.
    """

    __slots__ = ("found", "_primal", "_dual", "_gap")

    def __init__(self, found, primal: float, dual: float, gap: float):
        self.found = found
        self._primal = primal
        self._dual = dual
        self._gap = gap

    def primal_residual(self) -> float:
        return self._primal

    def dual_residual(self) -> float:
        return self._dual

    def duality_gap(self) -> float:
        return self._gap


@dataclass
class _PoolStats:
    """Timing and throughput counters for the pool execution path."""

    task_fetch_time: float = 0.0
    submit_time: float = 0.0
    collect_time: float = 0.0
    update_time: float = 0.0
    write_time: float = 0.0
    tasks_submitted: int = 0
    tasks_completed: int = 0


@dataclass
class _BufferedResult:
    """Result row buffered in main process before batch DataFrame update."""

    problem: str
    solver: str
    settings: str
    runtime: float
    found: bool
    primal_residual: float
    dual_residual: float
    duality_gap: float


def _log_pool_stats(
    stats: _PoolStats,
    first_submit: Optional[float],
    first_completion: Optional[float],
) -> None:
    """Emit debug-only runtime counters for bottleneck inspection."""
    startup_to_first_completion = (
        None
        if first_submit is None or first_completion is None
        else max(0.0, first_completion - first_submit)
    )
    logging.debug(
        "run_mth pool stats: submitted=%d completed=%d "
        "fetch=%.3fs submit=%.3fs collect=%.3fs update=%.3fs write=%.3fs "
        "submit_to_first_completion=%s",
        stats.tasks_submitted,
        stats.tasks_completed,
        stats.task_fetch_time,
        stats.submit_time,
        stats.collect_time,
        stats.update_time,
        stats.write_time,
        "n/a"
        if startup_to_first_completion is None
        else f"{startup_to_first_completion:.3f}s",
    )


def _init_worker() -> None:
    """Initializer called once per worker process: enforce single-threaded BLAS.

    The parent process also sets these vars before creating the pool so
    that workers inherit them before any BLAS library initialises its thread
    pool. Repeating them here also covers contexts where the library may
    already be loaded.
    """
    for var, val in _SINGLE_THREAD_ENV.items():
        os.environ[var] = val
    try:
        import threadpoolctl  # limits already-running pools if preloaded

        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _solve_one(
    problem,
    solver: str,
    settings: str,
    solver_kwargs: dict,
    verbose: bool,
) -> Tuple[_SolveResult, float, bool]:
    """Solve one (problem, solver, settings) triple inside a worker process.

    Returns:
        Tuple of ``(result, runtime, success)``.
    """
    if verbose:
        logging.info(f"Solving {problem.name} with {solver}/{settings}...")
    try:
        solution, runtime = time_solve_problem(
            problem, solver, **solver_kwargs
        )
        return (
            _SolveResult(
                found=solution.found,
                primal=solution.primal_residual(),
                dual=solution.dual_residual(),
                gap=solution.duality_gap(),
            ),
            runtime,
            True,
        )
    except Exception as exc:
        logging.error(
            f"Unhandled error solving {problem.name} with {solver}/{settings}: {exc}"
        )
        return (
            _SolveResult(None, float("inf"), float("inf"), float("inf")),
            0.0,
            False,
        )


def _solve_chunk(
    chunk: List[Tuple],
    verbose: bool,
) -> List[Tuple[_SolveResult, float, bool]]:
    """Solve a chunk of tasks in one worker call to reduce IPC overhead."""
    outputs: List[Tuple[_SolveResult, float, bool]] = []
    for problem, solver, settings, solver_kwargs in chunk:
        outputs.append(
            _solve_one(problem, solver, settings, solver_kwargs, verbose)
        )
    return outputs


def _make_buffered_result(
    problem,
    solver: str,
    settings: str,
    solution,
    runtime: float,
) -> _BufferedResult:
    """Convert a solve output into the flat row format used by Results.df."""
    return _BufferedResult(
        problem=problem.name,
        solver=solver,
        settings=settings,
        runtime=runtime,
        found=True if solution.found else False,
        primal_residual=solution.primal_residual(),
        dual_residual=solution.dual_residual(),
        duality_gap=solution.duality_gap(),
    )


def _flush_buffered_results(
    results: Results,
    buffered_results: List[_BufferedResult],
) -> None:
    """Append buffered result rows to results.df in one DataFrame operation.

    No deduplication is needed here: the planning phase guarantees each
    (problem, solver, settings) triple appears at most once in the task list.
    """
    if not buffered_results:
        return

    rows_df = pandas.DataFrame(
        {
            "problem": [r.problem for r in buffered_results],
            "solver": [r.solver for r in buffered_results],
            "settings": [r.settings for r in buffered_results],
            "runtime": [r.runtime for r in buffered_results],
            "found": [r.found for r in buffered_results],
            "primal_residual": [r.primal_residual for r in buffered_results],
            "dual_residual": [r.dual_residual for r in buffered_results],
            "duality_gap": [r.duality_gap for r in buffered_results],
        }
    ).astype(
        {
            "problem": str,
            "solver": str,
            "settings": str,
            "runtime": float,
            "found": bool,
            "primal_residual": float,
            "dual_residual": float,
            "duality_gap": float,
        }
    )

    results.df = pandas.concat([results.df, rows_df], ignore_index=True)


def _failed_chunk_outputs(
    chunk_len: int,
) -> List[Tuple[_SolveResult, float, bool]]:
    """Build failure outputs for every task in a chunk."""
    return [
        (
            _SolveResult(None, float("inf"), float("inf"), float("inf")),
            0.0,
            False,
        )
        for _ in range(chunk_len)
    ]


def _run_chunk_job(
    job: Tuple[int, List[Tuple], bool, Callable],
) -> Tuple[int, List[Tuple[_SolveResult, float, bool]], Optional[str]]:
    """Execute one chunk job in a worker process.

    Returns:
        ``(chunk_id, outputs, error_message)`` where ``error_message`` is
        ``None`` when execution succeeds.
    """
    chunk_id, chunk, verbose, solve_chunk_fn = job
    try:
        outputs = solve_chunk_fn(chunk, verbose)
        return chunk_id, outputs, None
    except Exception as exc:
        return chunk_id, [], str(exc)


def _worker_ping(worker_id: int) -> int:
    """No-op callable used to force worker process startup."""
    return worker_id


def _run_pool(
    tasks: Iterable[Tuple],
    results: Results,
    n_workers: int,
    verbose: bool,
    progress_bar,
    mp_context=None,
    chunk_size: Optional[int] = None,
    result_flush_size: int = 256,
    solve_chunk_fn=None,
) -> int:
    """Distribute solve tasks across a worker-process pool.

    The execution path is map-style: task chunks are generated lazily and fed
    to workers with ``imap_unordered``. Results are consumed as chunks finish,
    then buffered into ``results.df`` in batches.

    Args:
        tasks: Iterable of ``(problem, solver, settings, solver_kwargs)``
            tuples to solve.
        results: Updated in the main process as results arrive.
        n_workers: Number of worker processes.
        verbose: Passed through to each worker.
        progress_bar: Advanced by one for each completed task.
        mp_context: Multiprocessing context used to create the pool.
            Defaults to ``spawn``.
        chunk_size: Number of solve tasks bundled into one chunk.
            Defaults to ``1`` on small worker counts and ``32`` otherwise.
        result_flush_size: Number of collected rows to buffer before one batch
            append to ``Results.df``.
        solve_chunk_fn: Optional callable used to solve one chunk. Defaults
            to :func:`_solve_chunk`.

    Returns:
        Number of successful solver calls.
    """
    nb_calls = 0
    last_write = perf_counter()
    write_interval = 30.0
    stats = _PoolStats()
    first_submit: Optional[float] = None
    first_completion: Optional[float] = None

    if mp_context is None:
        mp_context = multiprocessing.get_context("spawn")
    if chunk_size is None:
        chunk_size = 1 if n_workers < 8 else 32
    else:
        chunk_size = max(1, chunk_size)
    result_flush_size = max(1, result_flush_size)
    if solve_chunk_fn is None:
        solve_chunk_fn = _solve_chunk

    for var, val in _SINGLE_THREAD_ENV.items():
        os.environ[var] = val

    logging.info(
        "Starting pool on %d workers (chunk_size=%d)",
        n_workers,
        chunk_size,
    )
    if verbose:
        logging.info("run_mth scheduler tracing is enabled")

    buffered_results: List[_BufferedResult] = []
    pending_chunks: Dict[int, List[Tuple]] = {}
    next_chunk_id = 0

    def flush_buffer() -> None:
        if not buffered_results:
            return
        if verbose:
            logging.info(
                "Flushing %d buffered rows to results DataFrame",
                len(buffered_results),
            )
        update_start = perf_counter()
        _flush_buffered_results(results, buffered_results)
        buffered_results.clear()
        stats.update_time += perf_counter() - update_start

    def write_results() -> None:
        if results.file_path is None:
            return
        if verbose:
            logging.info("Writing results to '%s'", results.file_path)
        write_start = perf_counter()
        results.write()
        stats.write_time += perf_counter() - write_start

    def iter_chunk_jobs():
        nonlocal first_submit
        nonlocal next_chunk_id

        tasks_iter = iter(tasks)
        while True:
            target_chunk = 1 if next_chunk_id < n_workers else chunk_size
            chunk: List[Tuple] = []
            fetch_start = perf_counter()
            while len(chunk) < target_chunk:
                try:
                    problem, solver, settings, solver_kwargs = next(tasks_iter)
                except StopIteration:
                    break
                chunk.append((problem, solver, settings, solver_kwargs))
            stats.task_fetch_time += perf_counter() - fetch_start

            if not chunk:
                break

            chunk_id = next_chunk_id
            next_chunk_id += 1
            stats.tasks_submitted += len(chunk)
            pending_chunks[chunk_id] = chunk
            if first_submit is None:
                first_submit = perf_counter()
            if verbose:
                logging.info(
                    "Submitting chunk %d with %d task(s)",
                    chunk_id,
                    len(chunk),
                )
            yield (chunk_id, chunk, verbose, solve_chunk_fn)

    pool = mp_context.Pool(
        processes=n_workers,
        initializer=_init_worker,
    )
    try:
        # For non-sized iterables (typically generators), warm workers first
        # so task generation can overlap with already-ready worker execution.
        if not hasattr(tasks, "__len__"):
            if verbose:
                logging.info("Pre-warming %d worker process(es)", n_workers)
            pool.map(_worker_ping, range(n_workers), chunksize=1)

        for chunk_id, chunk_outputs, error_msg in pool.imap_unordered(
            _run_chunk_job,
            iter_chunk_jobs(),
            chunksize=1,
        ):
            collect_start = perf_counter()
            chunk = pending_chunks.pop(chunk_id, None)
            if chunk is None:
                logging.error("Unknown chunk id %d returned from worker", chunk_id)
                continue
            if verbose:
                logging.info(
                    "Collected chunk %d (expected_tasks=%d, outputs=%d)",
                    chunk_id,
                    len(chunk),
                    len(chunk_outputs),
                )

            if error_msg is not None:
                logging.error("Collecting chunk result failed: %s", error_msg)
                chunk_outputs = _failed_chunk_outputs(len(chunk))

            if len(chunk_outputs) != len(chunk):
                logging.error(
                    "Worker returned %d outputs for %d inputs; "
                    "marking missing outputs as failures",
                    len(chunk_outputs),
                    len(chunk),
                )
                if len(chunk_outputs) < len(chunk):
                    chunk_outputs = list(chunk_outputs) + _failed_chunk_outputs(
                        len(chunk) - len(chunk_outputs)
                    )
                else:
                    chunk_outputs = list(chunk_outputs[: len(chunk)])
            stats.collect_time += perf_counter() - collect_start

            for task_item, output in zip(chunk, chunk_outputs):
                problem, solver, settings, _ = task_item
                result, runtime, success = output
                buffered_results.append(
                    _make_buffered_result(
                        problem, solver, settings, result, runtime
                    )
                )
                stats.tasks_completed += 1
                if first_completion is None:
                    first_completion = perf_counter()
                if success:
                    nb_calls += 1
                if progress_bar is not None:
                    progress_bar.update(1)

            if len(buffered_results) >= result_flush_size:
                flush_buffer()

            if perf_counter() - last_write >= write_interval:
                flush_buffer()
                write_results()
                last_write = perf_counter()

    except KeyboardInterrupt:
        logging.warning("Interrupted — cancelling pending tasks...")
        pool.terminate()
        flush_buffer()
        _log_pool_stats(stats, first_submit, first_completion)
        raise
    else:
        pool.close()
    finally:
        pool.join()

    flush_buffer()
    write_results()
    _log_pool_stats(stats, first_submit, first_completion)
    return nb_calls


def _get_cpu_count() -> int:
    """Return number of physical CPU cores to use as worker processes.

    Returns:
        Physical core count, falling back to 1 on detection failure.
    """
    try:
        import psutil

        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        return max(1, (os.cpu_count() or 1) // 2)


def _limit_solver_threads(solver_settings) -> None:
    """Force each solver to single-threaded mode in its settings object.

    Prevents thread explosion when N worker processes each launch a
    multi-threaded solver.

    Args:
        solver_settings: Mapping of settings name → :class:`SolverSettings`.
    """
    thread_params = {
        "gurobi": ("Threads", 1),
        "mosek": ("MSK_IPAR_NUM_THREADS", 1),
        "highs": ("threads", 1),
        "clarabel": ("max_threads", 1),
    }
    for settings_obj in solver_settings.values():
        for solver, (param, value) in thread_params.items():
            if solver in settings_obj.solvers:
                settings_obj.set_param(solver, param, value)


def _build_existing_result_keys(
    results: Results,
    filtered_solvers: List[str],
    filtered_settings: List[str],
) -> Set[Tuple[str, str, str]]:
    """Build O(1) set of already-solved (problem, solver, settings) keys.

    Args:
        results: Existing results to index.
        filtered_solvers: Only include rows for these solvers.
        filtered_settings: Only include rows for these settings.

    Returns:
        Set of ``(problem_name, solver, settings)`` keys.
    """
    if results.df.empty:
        return set()

    filtered_df = results.df[
        results.df["solver"].isin(filtered_solvers)
        & results.df["settings"].isin(filtered_settings)
    ]
    return {
        (row.problem, row.solver, row.settings)
        for row in filtered_df.itertuples(index=False)
    }


def _build_task_plan(
    test_set,
    results: Results,
    only_problem: Optional[str],
    filtered_solvers: List[str],
    filtered_settings: List[str],
) -> Tuple[List[Tuple], List[Tuple]]:
    """Iterate the test set once and decide exactly what needs to run.

    Args:
        test_set: Provides problems, tolerances and solver settings.
        results: Used to look up already-solved triples.
        only_problem: If set, restrict to this one problem name.
        filtered_solvers: Solvers to consider.
        filtered_settings: Settings names to consider.

    Returns:
        tasks: ``(problem, solver, settings, solver_kwargs)`` 4-tuples to
            pass to the pool — one entry per solve call needed.
        skip_records: ``(problem, solver, settings)`` triples that should be
            immediately written as null results (solver issue / known timeout).
    """
    existing_keys = _build_existing_result_keys(
        results, filtered_solvers, filtered_settings
    )

    tasks: List[Tuple] = []
    skip_records: List[Tuple] = []

    for problem in test_set:
        if only_problem and problem.name != only_problem:
            continue
        for solver in filtered_solvers:
            for settings in filtered_settings:
                time_limit = test_set.tolerances[settings].runtime
                key = (problem.name, solver, settings)
                if key in existing_keys:
                    continue

                if test_set.skip_solver_issue(problem, solver):
                    skip_records.append((problem, solver, settings))
                    continue
                if test_set.skip_solver_timeout(
                    time_limit, problem, solver, settings
                ):
                    skip_records.append((problem, solver, settings))
                    continue

                solver_kwargs = test_set.solver_settings[settings][solver]
                tasks.append((problem, solver, settings, solver_kwargs))

    return tasks, skip_records


def run_mth(
    test_set: TestSet,
    results: Results,
    only_problem: Optional[str] = None,
    only_settings: Optional[str] = None,
    only_solver: Optional[str] = None,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    """Run a test set using multiple worker processes and store results.

    Drop-in companion to :func:`run` with one extra parameter. Problems are
    distributed across a pool of worker processes so that multiple solves
    execute simultaneously. Each worker is restricted to one BLAS/OpenMP
    thread to prevent thread explosion.

    Execution is split into two phases:

    1. **Planning** — the test set is iterated once in the main process to
         decide what to run and what to skip. Existing rows in ``results`` are
         never re-run in this multithreaded path.
    2. **Pool** — the pre-computed task list is distributed across worker
       processes.  The progress bar shows the exact remaining count from the
       start.

    On CTRL+C, tasks that have not started are cancelled, already-collected
    results are flushed, and all results are written to disk.

    Args:
        test_set: Test set to run.
        results: Results instance to write to.
        only_problem: If set, only run that specific problem in the set.
        only_settings: If set, only run with these solver settings.
        only_solver: If set, only run that specific solver.
        verbose: If set, emit solver and scheduler trace logs.
        max_workers: Number of worker processes.
            ``None`` or ``1`` → sequential (same behaviour as :func:`run`).
            ``0`` → auto-detect physical CPU cores.
            ``N`` → use exactly N workers.
    """
    if only_settings and only_settings not in test_set.solver_settings:
        raise ValueError(
            f"settings '{only_settings}' not in the list of settings "
            f"for this test set: {list(test_set.solver_settings.keys())}"
        )
    if only_solver and only_solver not in test_set.solvers:
        raise SolverNotFound(
            f"solver '{only_solver}' not in the list of "
            f"available solvers for this test set: {test_set.solvers}"
        )

    if max_workers is None or max_workers == 1:
        n_workers = 1
    elif max_workers == 0:
        n_workers = _get_cpu_count()
    else:
        n_workers = max(1, max_workers)

    logging.info(f"run_mth: {n_workers} worker process(es)")

    filtered_solvers = [
        solver
        for solver in test_set.solvers
        if only_solver is None or solver == only_solver
    ]
    filtered_settings = [
        settings
        for settings in test_set.solver_settings
        if only_settings is None or settings == only_settings
    ]

    _limit_solver_threads(test_set.solver_settings)

    start_counter = perf_counter()

    logging.info("Planning tasks...")
    tasks, skip_records = _build_task_plan(
        test_set,
        results,
        only_problem,
        filtered_solvers,
        filtered_settings,
    )
    logging.info(
        "Plan: %d to run, %d to skip",
        len(tasks),
        len(skip_records),
    )
    if verbose:
        for problem, solver, settings in skip_records:
            logging.info(
                "Planned skip: problem=%s solver=%s settings=%s",
                problem.name,
                solver,
                settings,
            )
        for problem, solver, settings, _ in tasks:
            logging.info(
                "Planned task: problem=%s solver=%s settings=%s",
                problem.name,
                solver,
                settings,
            )

    for problem, solver, settings in skip_records:
        results.update(
            problem,
            solver,
            settings,
            qpsolvers.Solution(problem),
            0.0,
        )

    progress_bar = tqdm(
        total=len(tasks), mininterval=0.1, miniters=1, dynamic_ncols=True
    )

    nb_calls = 0
    try:
        nb_calls = _run_pool(
            tasks,
            results,
            n_workers,
            verbose,
            progress_bar,
        )
    except KeyboardInterrupt:
        logging.warning(
            "Benchmark interrupted — saving partial results..."
        )
        if results.file_path is not None:
            results.write()
        return
    finally:
        if progress_bar is not None:
            progress_bar.close()

    duration = perf_counter() - start_counter
    logging.info(f"Ran the test set in {duration:.0f} seconds")
    logging.info(f"Made {nb_calls} QP solver calls")
