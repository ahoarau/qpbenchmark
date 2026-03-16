"""Multiprocessing worker definitions for qpbenchmark."""

import ctypes
import ctypes.util
import os

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


def _worker_init():
    """Initializer for every pool worker process.

    After ``fork`` the BLAS libraries (OpenBLAS, MKL, …) reset their
    internal thread-pool sizes to the machine's core count because the
    parent's helper threads are dead.  We re-apply the limit here via
    both environment variables (for any late-initialising libs) and
    direct ctypes calls (for already-loaded shared objects).
    """
    import signal

    # Ignore SIGINT in pool workers so the main process can handle Ctrl+C
    # gracefully and terminate the pool via pool.terminate() rather than
    # workers crashing with KeyboardInterrupt.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Re-set env vars so any library initialised later in this worker
    # picks up the right value.
    for var, val in _THREAD_ENV_VARS.items():
        os.environ[var] = val

    # Force-limit already-loaded BLAS shared objects via their C API.
    _BLAS_SETTERS = [
        # (library glob pattern, function name)
        ("libopenblas", "openblas_set_num_threads"),
        ("libmkl_rt", "MKL_Set_Num_Threads"),
        ("libblis", "bli_thread_set_num_threads"),
    ]
    for lib_prefix, func_name in _BLAS_SETTERS:
        lib_path = ctypes.util.find_library(lib_prefix)
        if lib_path is None:
            # Try common suffixes directly
            # Walk /proc/self/maps to find the loaded library
            try:
                with open("/proc/self/maps") as f:
                    for line in f:
                        if lib_prefix in line:
                            lib_path = line.split()[-1]
                            break
            except OSError:
                pass
        if lib_path:
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
    """Execute a single QP solve.  Called from a process-pool worker.

    Args:
        problem: The QP problem to solve.
        solver: Name of the QP solver.
        settings: Name of the solver settings.
        solver_kwargs: Keyword arguments for the solver.
        verbose: If True, log before each solve.

    Returns:
        Tuple ``(found, primal_res, dual_res, duality_gap, runtime, success)``.
    """
    logging.debug(
        f"Worker {os.getpid()} started: {problem.name}/{solver}/{settings}"
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
        # Compute residuals here while the full Solution is still available,
        # then return plain scalars to the process boundary (avoids pickling
        # failures on C-extension extras).
        found = solution.found
        primal_res = solution.primal_residual()
        dual_res = solution.dual_residual()
        duality_gap = solution.duality_gap()
        success = True
    except Exception as exc:
        logging.error(
            f"Unhandled exception solving {problem.name} "
            f"with {solver}/{settings}: {exc}"
        )
        found = None
        primal_res = float("inf")
        dual_res = float("inf")
        duality_gap = float("inf")
        runtime = 0.0
        success = False

    return found, primal_res, dual_res, duality_gap, runtime, success
