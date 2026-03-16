# Throughput Bottleneck Fix & Max-Workers Refactor

This document explains the changes made to the `qpbenchmark` repository to improve runtime throughput and add support for the `max-workers=0` argument.

## 1. Uncapping Asynchronous Task Throughput
**File Modified**: `qpbenchmark/runner.py`

### **Problem:**
The core main loop inside `BenchmarkRunner.execute()` was unintentionally limiting the speed of tasks due to an arbitrary continuous `time.sleep(0.05)` block. This executed whenever the asynchronous `apply_async` multiprocessing queue filled up, effectively limiting tasks to `worker queue / 0.05` tasks per second. When running lightweight tasks with a single worker `max-workers=1`, this bottleneck became glaringly apparent (capping around ~20 tasks/s).

### **Solution:**
Replaced the `sleep(0.05)` code blocks with a block that waits directly on the multiprocessor's worker tasks.

```python
pending[0][0].wait(timeout=0.05)
```
Instead of putting the main controller explicitly to sleep, this change binds the wait cycle on the first `ApplyResult` returned by the queue. While still bounding the total latency at `0.05` seconds, the `.wait()` call immediately releases execution to the thread natively as soon as a worker returns its payload—entirely bypassing the artificial throughput threshold.


## 2. Using All Logical Cores explicitly (`max-workers=0`)
**Files Modified**: `qpbenchmark/system.py` and `qpbenchmark/run.py`

### **Problem:**
The user wanted the ability to input `--max-workers 0` into the `pixi run qpbenchmark` client config, translating directly to using all available *logical* processor threads (like `hyper-threading` support) instead of default `None` returning all available *physical* cores.

### **Solution:**

**A.** Created a logical CPU utility inside `system.py`:
A logical cpu fallback utility `get_logical_cpu_count()` was added to `qpbenchmark/system.py`. Similar to the existing physical hardware utility, it checks with `psutil` or falls back to standard python library commands.

```python
def get_logical_cpu_count() -> int:
    try:
        import psutil
        logical = psutil.cpu_count(logical=True) or 1
        return logical
    except ImportError:
        logical = os.cpu_count() or 1
        return logical
```

**B.** Handled `0` input inside `run.py` options parsing block:
The `qpbenchmark/run.py` config evaluation block dynamically evaluates `max_workers=None` to use the physical cpu limit wrapper. We implemented intercepting `0` to immediately pull counts using the newly written `get_logical_cpu_count()` utility function instead.

```python
    # Determine number of worker threads (use physical cores by default)
    if max_workers is None:
        max_workers = get_physical_cpu_count()
    elif max_workers == 0:
        max_workers = get_logical_cpu_count()
```
This guarantees that standard input correctly falls back while specific configurations natively support maximum throughput processing!
