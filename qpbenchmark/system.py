"""System information and environment utilities."""

import os

from .spdlog import logging


def get_logical_cpu_count() -> int:
    """Get number of logical CPU cores (including hyperthreaded).

    Returns:
        Number of logical cores, defaulting to 1 if detection fails.
    """
    try:
        import psutil

        logical = psutil.cpu_count(logical=True) or 1
        return logical
    except ImportError:
        logical = os.cpu_count() or 1
        return logical


def get_physical_cpu_count() -> int:
    """Get number of physical CPU cores (not hyperthreaded).

    Returns:
        Number of physical cores, defaulting to 1 if detection fails.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or 1
        logging.info(
            f"CPU detection (psutil): {physical} physical cores, "
            f"{logical} logical cores"
        )
        return physical
    except ImportError:
        # Fallback: estimate as half of logical cores
        # (common for hyperthreading)
        logical = os.cpu_count() or 1
        physical = max(1, logical // 2)
        logging.info(
            f"CPU detection (fallback, no psutil): "
            f"{logical} logical cores -> using {physical} as estimate"
        )
        return physical


def display_system_info(max_workers: int) -> None:
    """Display user-friendly system information before benchmark.

    Args:
        max_workers: Number of worker threads that will be used.
    """
    try:
        import psutil
        from cpuinfo import get_cpu_info

        # CPU Information
        cpu_info = get_cpu_info()
        cpu_brand = cpu_info.get("brand_raw", "Unknown CPU")

        # Core counts
        physical_cores = psutil.cpu_count(logical=False) or 0
        logical_cores = psutil.cpu_count(logical=True) or 0

        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)

        # Display formatted information
        logging.info("=" * 60)
        logging.info("Benchmark System Information")
        logging.info("=" * 60)
        logging.info(f"CPU: {cpu_brand}")
        logging.info(
            f"Physical cores: {physical_cores} | "
            f"Logical cores: {logical_cores}"
        )
        logging.info(
            f"Memory: {available_memory_gb:.1f} GB available / "
            f"{total_memory_gb:.1f} GB total"
        )
        logging.info(
            f"Worker processes: {max_workers} (each pinned to 1 solver thread)"
        )
        logging.info(
            f"Parallel mode: {'Yes' if max_workers > 1 else 'No (sequential)'}"
        )
        logging.info("=" * 60)

    except ImportError:
        # Minimal fallback if psutil or cpuinfo not available
        logical_cores = os.cpu_count() or "unknown"
        logging.info("=" * 60)
        logging.info("Benchmark System Information")
        logging.info("=" * 60)
        logging.info(f"Logical cores: {logical_cores}")
        logging.info(f"Worker processes: {max_workers}")
        logging.info("=" * 60)
    except Exception as e:
        # Silently skip if there's any issue getting system info
        logging.debug(f"Could not display full system info: {e}")
