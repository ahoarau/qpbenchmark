#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test the run function."""

import multiprocessing
import os
import sys
import tempfile
import time
import unittest

from qpsolvers import SolverNotFound, available_solvers

import qpbenchmark
from qpbenchmark import Results
from qpbenchmark.run import _solve_task

from .custom_test_set import CustomTestSet

# ---------------------------------------------------------------------------
# Module-level helpers for TestProcessPoolParallelism
# (must be at module level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __getitem__(self, solver):
        return {}


class _FakeTestSet:
    solver_settings = {"default": _FakeSettings()}


class TestRun(unittest.TestCase):
    def setUp(self):
        csv_path = tempfile.mktemp(".csv")
        self.results = Results(file_path=csv_path, test_set=CustomTestSet())
        self.test_set = CustomTestSet()

    def test_run_available_solvers(self):
        self.assertEqual(len(self.results.df), 0)
        qpbenchmark.run(
            self.test_set,
            self.results,
            only_problem="custom",
            only_settings="default",
            rerun=False,
            rerun_timeouts=False,
        )
        self.assertEqual(len(self.results.df), len(available_solvers))

    def test_only_solver(self):
        self.assertEqual(len(self.results.df), 0)
        qpbenchmark.run(
            self.test_set,
            self.results,
            only_problem="custom",
            only_settings="default",
            only_solver="daqp",  # listed in tox.ini
            rerun=False,
            rerun_timeouts=False,
        )
        self.assertEqual(len(self.results.df), 1)

    def test_settings_not_found(self):
        with self.assertRaises(ValueError):
            qpbenchmark.run(
                self.test_set,
                self.results,
                only_settings="unknown",
                rerun=False,
                rerun_timeouts=False,
            )

    def test_solver_not_found(self):
        with self.assertRaises(SolverNotFound):
            qpbenchmark.run(
                self.test_set,
                self.results,
                only_solver="unknown",
                rerun=False,
                rerun_timeouts=False,
            )


class TestProcessPoolParallelism(unittest.TestCase):
    """Verify that the process pool executes solve tasks in parallel.

    The test injects a slow 'solve' function and checks that N tasks taking
    T seconds each complete in roughly T seconds with N workers (not N*T).
    """

    def setUp(self):
        self.task_delay = 0.3  # seconds per simulated solve
        self.num_tasks = 8
        self.num_workers = 4

    def _fake_problem(self, name: str):
        import numpy as np

        return qpbenchmark.Problem(
            P=np.eye(1),
            q=np.ones(1),
            G=None,
            h=None,
            A=None,
            b=None,
            lb=None,
            ub=None,
            name=name,
        )

    def test_workers_run_in_parallel(self):
        """Tasks must finish in ~task_delay seconds, not num_tasks*task_delay."""
        from unittest.mock import patch

        import qpsolvers

        fake_test_set = _FakeTestSet()
        fake_problems = [
            self._fake_problem(f"problem_{i}") for i in range(self.num_tasks)
        ]

        manager = multiprocessing.Manager()
        active_tasks = manager.dict()

        # Use a manager list to collect (problem_name, pid) pairs
        # across processes (inherited via fork).
        solve_records = manager.list()

        def slow_solve(problem, solver, **kwargs):
            """Simulate a CPU-bound solve that lives in a worker process."""
            solve_records.append((problem.name, os.getpid()))
            time.sleep(self.task_delay)
            return qpsolvers.Solution(problem), self.task_delay

        run_module = sys.modules["qpbenchmark.run"]

        with patch.object(run_module, "time_solve_problem", slow_solve):
            start = time.perf_counter()

            pool = multiprocessing.Pool(processes=self.num_workers)
            async_results = [
                pool.apply_async(
                    _solve_task,
                    args=(
                        prob,
                        f"solver_{i}",
                        "default",
                        {},
                        False,
                        active_tasks,
                    ),
                )
                for i, prob in enumerate(fake_problems)
            ]
            pool.close()
            for ar in async_results:
                ar.get()  # propagate exceptions
            pool.join()

            elapsed = time.perf_counter() - start

        records = list(solve_records)
        manager.shutdown()
        sequential_time = self.num_tasks * self.task_delay
        # With num_workers parallel workers: ceil(num_tasks / num_workers) rounds
        expected_parallel = (
            self.num_tasks / self.num_workers
        ) * self.task_delay
        # Allow 50% overhead for process startup and scheduling
        tolerance = 1.5

        self.assertLess(
            elapsed,
            expected_parallel * tolerance,
            f"Process pool took {elapsed:.2f}s; expected < "
            f"{expected_parallel * tolerance:.2f}s "
            f"(sequential would have taken {sequential_time:.2f}s). "
            "Workers are likely NOT running in parallel.",
        )

        self.assertEqual(
            len(records),
            self.num_tasks,
            f"Expected {self.num_tasks} solver calls, got {len(records)}.",
        )

        worker_pids = {pid for _, pid in records}
        self.assertGreater(
            len(worker_pids),
            1,
            f"All tasks ran in a single process (PIDs: {worker_pids}). "
            "Workers are likely NOT running in parallel.",
        )


class PerformanceTestSet(qpbenchmark.TestSet):
    def __init__(self, num_problems: int):
        self.num_problems = num_problems
        super().__init__()

    @property
    def description(self) -> str:
        return "Performance test set"

    @property
    def title(self) -> str:
        return "Performance test set"

    @property
    def sparse_only(self) -> bool:
        return False

    def define_tolerances(self, runtime: float = 100.0) -> None:
        from qpbenchmark import Tolerance

        self.tolerances = {
            "default": Tolerance(
                primal=1.0,
                dual=1.0,
                gap=1.0,
                runtime=runtime,
            )
        }

    def define_solver_settings(self) -> None:
        from qpbenchmark.solver_settings import SolverSettings

        default = SolverSettings()
        default.set_time_limit(self.tolerances["default"].runtime)
        self.solver_settings = {"default": default}
        # Use a real solver name so we don't trip up validation,
        # but we will mock the execution.
        self.solvers = {"daqp"}

    def __iter__(self):
        import numpy as np
        from qpbenchmark import Problem

        for i in range(self.num_problems):
            yield Problem(
                P=np.eye(1),
                q=np.ones(1),
                G=None,
                h=None,
                A=None,
                b=None,
                lb=None,
                ub=None,
                name=f"problem_{i}",
            )


class TestPerformance(unittest.TestCase):
    def test_parallel_speedup(self):
        """
        Verify that parallel execution provides expected speedup.
        Simulate a slow computation (sleep 1s) with 10 records.
        """
        from unittest.mock import patch

        import qpsolvers

        num_records = 10
        sleep_time = 1.0  # seconds

        test_set = PerformanceTestSet(num_records)
        csv_path = tempfile.mktemp(".csv")
        results = Results(file_path=csv_path, test_set=test_set)

        def slow_solve(problem, solver, **kwargs):
            time.sleep(sleep_time)
            # Return a valid solution object and runtime
            return qpsolvers.Solution(problem), sleep_time

        # Patch time_solve_problem where it is imported in run.py
        with patch(
            "qpbenchmark.run.time_solve_problem", side_effect=slow_solve
        ):

            # 1. Test sequential (max_workers=1)
            # This should take ~10 seconds
            start_seq = time.perf_counter()
            qpbenchmark.run(
                test_set,
                results,
                only_settings="default",
                only_solver="daqp",
                max_workers=1,
                verbose=False,
            )
            duration_seq = time.perf_counter() - start_seq

            # 2. Test parallel (max_workers=16)
            # This should take ~1 second
            start_par = time.perf_counter()
            qpbenchmark.run(
                test_set,
                results,
                only_settings="default",
                only_solver="daqp",
                max_workers=16,
                rerun=True,
                verbose=False,
            )
            duration_par = time.perf_counter() - start_par

        # Assertions
        # Sequential: ~10s
        self.assertGreaterEqual(duration_seq, num_records * sleep_time * 0.95)

        # Parallel: ~1s
        # Allow up to 3.5s to account for process startup/shutdown overhead
        self.assertLess(duration_par, 3.5 * sleep_time)
