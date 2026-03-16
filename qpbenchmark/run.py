"""Main function of the benchmark."""

from typing import Optional

from qpsolvers.exceptions import SolverNotFound

from .results import Results
from .runner import BenchmarkRunner
from .spdlog import logging
from .system import (
    display_system_info,
    get_logical_cpu_count,
    get_physical_cpu_count,
)
from .test_set import TestSet


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


def _triage_tasks(
    test_set,
    results,
    filtered_solvers,
    filtered_settings,
    only_problem,
    rerun,
    rerun_timeouts,
    stop_event,
):
    """Generate tasks for the process pool, applying skips and failures."""
    # Optimization: Pre-load existing results for O(1) lookup
    existing_results = {
        (row.problem, row.solver, row.settings): row.runtime
        for row in results.df.itertuples(index=False)
    }

    for problem in test_set:
        if stop_event.is_set():
            break
        if only_problem is not None and problem.name != only_problem:
            continue

        for solver in filtered_solvers:
            for settings in filtered_settings:
                time_limit = test_set.tolerances[settings].runtime
                key = (problem.name, solver, settings)

                if key in existing_results:
                    if not rerun:
                        action = "skip"
                    elif not rerun_timeouts:
                        runtime = existing_results[key]
                        if runtime > 0.99 * time_limit:
                            action = "skip_timeout"
                        else:
                            action = "solve"
                    else:
                        action = "solve"
                elif test_set.skip_solver_issue(problem, solver):
                    action = "record_failure"
                elif test_set.skip_solver_timeout(
                    time_limit, problem, solver, settings
                ):
                    action = "record_failure"
                else:
                    action = "solve"

                kwargs = test_set.solver_settings[settings][solver]
                yield action, problem, solver, settings, kwargs


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
            If None (default), uses physical CPU core count for optimal
            performance in CPU-bound benchmarks. Set to 1 for sequential
            execution. Problems are processed in parallel while solvers for
            each problem run sequentially to ensure fair timing.
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

    # Determine number of worker threads (use physical cores by default)
    if max_workers is None:
        max_workers = get_physical_cpu_count()
    elif max_workers == 0:
        max_workers = get_logical_cpu_count()

    # Display system information
    display_system_info(max_workers)

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
        f"Settings ({len(filtered_settings)}): " + ", ".join(filtered_settings)
    )

    # Apply thread limits to solver settings to prevent thread explosion
    _limit_solver_threads(test_set.solver_settings)

    # Phase 1: initialization
    if only_problem is not None:
        nb_problems = 1
    elif test_set.limit:
        nb_problems = min(test_set.limit, test_set.count_problems())
    else:
        nb_problems = test_set.count_problems()

    nb_total = nb_problems * len(filtered_solvers) * len(filtered_settings)

    logging.info(
        f"Test set has {nb_problems} problems, "
        f"{nb_total} total tasks "
        f"({nb_problems} problems x {len(filtered_solvers)} solvers "
        f"x {len(filtered_settings)} settings)"
    )

    runner = BenchmarkRunner(max_workers, verbose)

    stats = {"solve": 0, "skip": 0, "skip_timeout": 0, "record_failure": 0}

    task_generator = _triage_tasks(
        test_set,
        results,
        filtered_solvers,
        filtered_settings,
        only_problem,
        rerun,
        rerun_timeouts,
        runner.stop_event,
    )

    runner.execute(task_generator, nb_total, results, stats)
