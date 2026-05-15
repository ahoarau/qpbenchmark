#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Main function of the benchmark."""

import multiprocessing
import os
import signal
from time import perf_counter
from typing import List, Optional, Tuple

import qpsolvers
from qpsolvers.exceptions import SolverNotFound
from tqdm import tqdm

from .parallel_execution import _execute_tasks
from .results import Results
from .spdlog import logging
from .test_set import TestSet


def _get_cpu_count(enable_hyperthreading: bool = False) -> int:
    """Get number of CPU cores.

    Note:
        On modern laptops, CPUs might be divided into performance and 
        efficiency cores, which can result in different benchmarking 
        results compared to regular cores and Hyper-Threading.

    Args:
        enable_hyperthreading: Set to True to count logical cores instead of physical cores.

    Returns:
        Number of cores, defaulting to 1 if detection fails.
    """
    try:
        import psutil
        count = psutil.cpu_count(logical=enable_hyperthreading) or 1
        core_type = "logical" if enable_hyperthreading else "physical"
        logging.info(f"CPU detection (psutil): {count} {core_type} cores")
        return count
    except ImportError:
        logical = os.cpu_count() or 1
        core_type = "logical" if enable_hyperthreading else "physical"
        if enable_hyperthreading:
            count = logical
        else:
            # Fallback: estimate as half of logical cores (common for hyperthreading)
            count = max(1, logical // 2)
        logging.info(
            f"CPU detection (fallback, no psutil): {logical} logical cores "
            f"-> using {count} {core_type} cores as estimate"
        )
        return count


def _display_system_info(max_workers: int) -> None:
    """Display user-friendly system information before benchmark.

    Args:
        max_workers: Number of worker threads that will be used.
    """
    try:
        import psutil
        from cpuinfo import get_cpu_info
        
        # CPU Information
        cpu_info = get_cpu_info()
        cpu_brand = cpu_info.get('brand_raw', 'Unknown CPU')
        
        # Core counts
        physical_cores = psutil.cpu_count(logical=False) or 0
        logical_cores = psutil.cpu_count(logical=True) or 0
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # Display formatted information
        logging.info("="*60)
        logging.info("Benchmark System Information")
        logging.info("="*60)
        logging.info(f"CPU: {cpu_brand}")
        logging.info(f"Physical cores: {physical_cores} | Logical cores: {logical_cores}")
        logging.info(f"Memory: {available_memory_gb:.1f} GB available / {total_memory_gb:.1f} GB total")
        logging.info(f"Worker processes: {max_workers} (each pinned to 1 solver thread)")
        logging.info(f"Parallel mode: {'Yes' if max_workers > 1 else 'No (sequential)'}")
        logging.info("="*60)
        
    except ImportError:
        # Minimal fallback if psutil or cpuinfo not available
        logical_cores = os.cpu_count() or "unknown"
        logging.info("="*60)
        logging.info("Benchmark System Information")
        logging.info("="*60)
        logging.info(f"Logical cores: {logical_cores}")
        logging.info(f"Worker processes: {max_workers}")
        logging.info("="*60)
    except Exception as e:
        # Silently skip if there's any issue getting system info
        logging.debug(f"Could not display full system info: {e}")


def _limit_solver_threads(solver_settings):
    """Apply thread limits to individual solver parameters.

    Forces all QP solvers to use single-threaded mode to prevent thread
    explosion when running multiple problems in parallel.

    The underlying threading library environment variables (OMP_NUM_THREADS,
    OPENBLAS_NUM_THREADS, RAYON_NUM_THREADS, etc.) are set early in
    benchmark.py, before solver libraries are imported, so they take effect
    at library initialization time.

    Args:
        solver_settings: Dictionary of settings name -> SolverSettings object.
    """
    for settings_name, settings_obj in solver_settings.items():
        for solver in settings_obj.solvers:
            if solver == "gurobi":
                settings_obj.set_param(solver, "Threads", 1)
            elif solver == "mosek":
                settings_obj.set_param(solver, "MSK_IPAR_NUM_THREADS", 1)
            elif solver == "highs":
                settings_obj.set_param(solver, "threads", 1)
            elif solver == "clarabel":
                settings_obj.set_param(solver, "max_threads", 1)


# ---------------------------------------------------------------------------
# Core benchmarking routines
# ---------------------------------------------------------------------------

def _load_problems(
    test_set: TestSet,
    only_problem: Optional[str],
    filtered_solvers: List[str],
    filtered_settings: List[str],
    stop_event: multiprocessing.Event,
) -> Tuple[List, int]:
    """Load problems from the test set, filtering if requested."""
    logging.info("Loading problems from test set...")
    problems = []
    for problem in test_set:
        if stop_event.is_set():
            logging.warning("Interrupted during problem loading.")
            break
        if only_problem is None or problem.name == only_problem:
            problems.append(problem)
    
    nb_total = len(problems) * len(filtered_solvers) * len(filtered_settings)
    logging.info(
        f"Loaded {len(problems)} problems, "
        f"{nb_total} total tasks "
        f"({len(problems)} problems x {len(filtered_solvers)} solvers "
        f"x {len(filtered_settings)} settings)"
    )
    return problems, nb_total


def _resolve_action(
    key: Tuple,
    existing_results: dict,
    time_limit: float,
    problem,
    solver: str,
    settings: str,
    test_set: TestSet,
    rerun: bool,
    rerun_timeouts: bool,
) -> str:
    """Return the action string for a single (problem, solver, settings) tuple."""
    if key not in existing_results:
        if test_set.skip_solver_issue(problem, solver):
            return "record_failure"
        if test_set.skip_solver_timeout(time_limit, problem, solver, settings):
            return "record_failure"
        return "solve"

    if not rerun:
        return "skip"
    if rerun_timeouts:
        return "solve"
    runtime = existing_results[key]
    return "skip_timeout" if runtime > 0.99 * time_limit else "solve"


def _triage_tasks(
    problems: List,
    filtered_solvers: List[str],
    filtered_settings: List[str],
    test_set: TestSet,
    results: Results,
    rerun: bool,
    rerun_timeouts: bool,
    stop_event: multiprocessing.Event,
    progress_bar: Optional[tqdm],
) -> List[Tuple]:
    """Decide which tasks to skip, report as failed, or actually solve."""
    logging.info("Preparing tasks (checking existing results)...")
    solve_tasks = []
    nb_skip = 0
    nb_skip_timeout = 0
    nb_record_failure = 0

    # Optimization: Pre-load existing results for O(1) lookup
    existing_results = {
        (row.problem, row.solver, row.settings): row.runtime
        for row in results.df.itertuples(index=False)
    }

    for problem in problems:
        if stop_event.is_set():
            logging.warning("Interrupted during task triage.")
            break
        for solver in filtered_solvers:
            for settings in filtered_settings:
                time_limit = test_set.tolerances[settings].runtime
                key = (problem.name, solver, settings)
                action = _resolve_action(
                    key, existing_results, time_limit,
                    problem, solver, settings,
                    test_set, rerun, rerun_timeouts,
                )

                if action == "skip":
                    nb_skip += 1
                    logging.debug(
                        f"{problem.name} already solved by {solver} "
                        f"with {settings} settings, skipping."
                    )
                elif action == "skip_timeout":
                    nb_skip_timeout += 1
                    logging.info(
                        f"Skipping {problem.name} / {solver} / "
                        f"{settings} (previous timeout)."
                    )
                elif action == "record_failure":
                    nb_record_failure += 1
                    solution = qpsolvers.Solution(problem)
                    results.update(
                        problem, solver, settings, solution, 0.0
                    )
                    logging.debug(
                        f"Recording failure for "
                        f"{problem.name} / {solver} / {settings}."
                    )
                elif action == "solve":
                    solve_tasks.append((problem, solver, settings))
                    continue

                if progress_bar is not None:
                    progress_bar.update(1)

    # Persist any recorded failures in one batch
    if nb_record_failure > 0:
        results.write()

    logging.info(
        f"Task breakdown: {len(solve_tasks)} to solve, "
        f"{nb_skip} skipped (existing), "
        f"{nb_skip_timeout} skipped (timeout), "
        f"{nb_record_failure} known failures"
    )

    return solve_tasks


def run(
    test_set: TestSet,
    results: Results,
    only_problem: Optional[str] = None,
    only_settings: Optional[str] = None,
    only_solver: Optional[str] = None,
    rerun: bool = False,
    rerun_timeouts: bool = False,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    enable_hyperthreading: bool = False,
) -> None:
    """Run a given test set and store results.

    Args:
        test_set: Test set to run.
        results: Results instance to write to.
        only_problem: If set, only run that specific problem in the set.
        only_settings: If set, only run with these solver settings.
        only_solver: If set, only run that specific solver.
        rerun: If set, rerun instances that already have a result.
        rerun_timeouts: If set, also rerun known timeouts.
        verbose: If set, log info messages for each QP solver call.
        max_workers: Maximum number of worker threads for parallel execution.
            If None, uses 1 for sequential execution. If 0, uses physical CPU
            core count (or logical if enable_hyperthreading is True) for optimal 
            performance in CPU-bound benchmarks. Problems are processed in 
            parallel while solvers for each problem run sequentially.
        enable_hyperthreading: Set to True to count logical cores instead of physical cores.
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

    # Determine number of worker threads
    if max_workers is None:
        max_workers = 1
    elif max_workers == 0:
        max_workers = _get_cpu_count(enable_hyperthreading)
    else:
        max_workers = max(1, max_workers)

    # Display system information
    _display_system_info(max_workers)

    # Install signal handler early so CTRL+C is clean.
    stop_event = multiprocessing.Event()
    pool_cell: List = []

    def signal_handler(signum, frame):
        logging.warning("Received interrupt signal (CTRL+C), stopping...")
        stop_event.set()
        if pool_cell:
            try:
                pool_cell[0].shutdown(wait=False)
            except Exception:
                pass

    original_sigint = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Filter solvers and settings based on user preferences
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

        logging.info(
            f"Solvers ({len(filtered_solvers)}): "
            + ", ".join(sorted(filtered_solvers))
        )
        logging.info(
            f"Settings ({len(filtered_settings)}): "
            + ", ".join(filtered_settings)
        )

        # Apply thread limits to solver settings to prevent thread explosion
        _limit_solver_threads(test_set.solver_settings)

        start_counter = perf_counter()

        # Phase 1 – Load problems
        problems, nb_total = _load_problems(
            test_set, only_problem, filtered_solvers, filtered_settings, stop_event
        )

        if not problems or stop_event.is_set():
            return

        # Initialize progress bar
        progress_bar = None
        if not verbose:
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

        try:
            # Phase 2 – Triage tasks
            solve_tasks = _triage_tasks(
                problems,
                filtered_solvers,
                filtered_settings,
                test_set,
                results,
                rerun,
                rerun_timeouts,
                stop_event,
                progress_bar,
            )

            # Phase 3 – Solve
            if not solve_tasks:
                logging.info("Nothing to solve – all tasks were skipped or failed.")
            elif not stop_event.is_set():
                nb_calls = _execute_tasks(
                    solve_tasks,
                    test_set,
                    results,
                    max_workers,
                    verbose,
                    stop_event,
                    pool_cell,
                    progress_bar,
                )
            else:
                nb_calls = 0

            duration = perf_counter() - start_counter
            if not stop_event.is_set():
                logging.info(f"Ran the test set in {duration:.0f} seconds")
                logging.info(f"Made {nb_calls} QP solver calls")
            else:
                logging.info(f"Partial run completed in {duration:.0f} seconds")
                logging.info(f"Made {nb_calls} QP solver calls before interruption")
                
        finally:
            if progress_bar is not None:
                progress_bar.close()

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)

