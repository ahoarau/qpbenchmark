#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Inria

import multiprocessing
import tempfile
import time
import unittest

from qpsolvers import SolverNotFound

import qpbenchmark
from qpbenchmark import Results
from qpbenchmark.run_mth import _run_pool

from .custom_test_set import CustomTestSet


class _FakeProblem:
    def __init__(self, name: str):
        self.name = name


class _FakeSolution:
    found = True

    def primal_residual(self) -> float:
        return 0.0

    def dual_residual(self) -> float:
        return 0.0

    def duality_gap(self) -> float:
        return 0.0


def _solve_chunk_with_sleep(chunk, verbose):
    _ = verbose
    outputs = []
    for _, _, _, solver_kwargs in chunk:
        delay_s = solver_kwargs["sleep_s"]
        time.sleep(delay_s)
        outputs.append((_FakeSolution(), delay_s, True))
    return outputs


class TestRunMthValidation(unittest.TestCase):
    def setUp(self):
        csv_path = tempfile.mktemp(".csv")
        self.test_set = CustomTestSet()
        self.results = Results(file_path=csv_path, test_set=self.test_set)

    def test_settings_not_found(self):
        with self.assertRaises(ValueError):
            qpbenchmark.run_mth(
                self.test_set,
                self.results,
                only_settings="unknown",
            )

    def test_solver_not_found(self):
        with self.assertRaises(SolverNotFound):
            qpbenchmark.run_mth(
                self.test_set,
                self.results,
                only_solver="unknown",
            )


class TestProcessPoolParallelism(unittest.TestCase):
    def test_sleep_tasks_parallelize_and_record_all_results(self):
        num_tasks = 50
        sleep_s = 0.2
        n_workers = 2
        tasks = [
            (
                _FakeProblem(f"problem_{i}"),
                "daqp",
                "default",
                {"sleep_s": sleep_s},
            )
            for i in range(num_tasks)
        ]
        results = Results(file_path=None, test_set=CustomTestSet())

        start = time.perf_counter()
        nb_calls = _run_pool(
            tasks,
            results,
            n_workers=n_workers,
            verbose=False,
            progress_bar=None,
            mp_context=multiprocessing.get_context("spawn"),
            chunk_size=1,
            solve_chunk_fn=_solve_chunk_with_sleep,
        )
        elapsed = time.perf_counter() - start

        self.assertEqual(nb_calls, num_tasks)
        self.assertEqual(len(results.df), num_tasks)
        self.assertLess(elapsed, num_tasks * sleep_s * 0.85)
