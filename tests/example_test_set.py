#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

from custom_problem import custom_problem

import qpbenchmark


class ExampleTestSet(qpbenchmark.TestSet):
    @property
    def description(self) -> str:
        return "Example test set"

    @property
    def title(self) -> str:
        return "Example test set"

    @property
    def sparse_only(self) -> bool:
        return False

    def __iter__(self):
        yield custom_problem(name="custom")
        yield custom_problem(name="custom_again")
