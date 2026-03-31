#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test the run function."""

import tempfile
import unittest

from qpsolvers import SolverNotFound, available_solvers

import qpbenchmark
from qpbenchmark import Results

from .custom_test_set import CustomTestSet


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

    def test_check_problems(self):
        from qpbenchmark.run import check_problems
        
        # Initially, all solvers should be scheduled
        tasks_generator = check_problems(
            self.test_set,
            self.results,
            list(self.test_set.solvers),
            list(self.test_set.solver_settings.keys()),
            only_problem="custom",
            rerun=False,
            rerun_timeouts=False,
            progress_bar=None,
        )
        problem_tasks = list(tasks_generator)
        self.assertEqual(len(problem_tasks), 1)
        problem, tasks = problem_tasks[0]
        self.assertEqual(
            len(tasks),
            len(self.test_set.solvers) * len(self.test_set.solver_settings)
        )
        
        # Run tests to populate results
        qpbenchmark.run(
            self.test_set,
            self.results,
            only_problem="custom",
            only_settings="default",
            rerun=False,
            rerun_timeouts=False,
        )
        
        # Running check_problems again should yield no tasks (since results exist and rerun=False)
        tasks_generator2 = check_problems(
            self.test_set,
            self.results,
            list(self.test_set.solvers),
            ["default"],
            only_problem="custom",
            rerun=False,
            rerun_timeouts=False,
            progress_bar=None,
        )
        problem_tasks2 = list(tasks_generator2)
        self.assertEqual(len(problem_tasks2), 0)

