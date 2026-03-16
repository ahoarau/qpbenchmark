#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test case results."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas
import qpsolvers

from .exceptions import BenchmarkError, ResultsError
from .problem import Problem
from .shgeom import shgeom
from .spdlog import logging
from .test_set import TestSet


class Results:
    """Test set results.

    Attributes:
        df: Data frame storing the results.
        file_path: Path to the results CSV file.
        test_set: Test set from which results were produced.
    """

    df: pandas.DataFrame
    file_path: Optional[Path]
    test_set: TestSet
    __new_results: list

    @staticmethod
    def check_df(df) -> None:
        """Check consistency of a full results dataframe.

        Raises:
            ResultsError: if the dataframe is inconsitent.
        """
        if not isinstance(df["found"].dtype, np.dtypes.BoolDType):
            raise ResultsError('"found" column has some non-boolean values')

    @staticmethod
    def read_from_file(path: Union[str, Path]) -> Optional[pandas.DataFrame]:
        """Load a pandas dataframe from a CSV or Parquet file.

        Args:
            path: Path to the file to load.

        Returns:
            Loaded dataframe, or None if the file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            return None
        elif file_path.stat().st_size == 0:
            logging.warning(
                "Results file '%s' is empty, treating as if it "
                "does not exist.",
                file_path,
            )
            return None
        elif file_path.suffix not in (".csv", ".parquet"):
            raise BenchmarkError(
                f"unknown file extension to read results from in '{file_path}'"
            )
        logging.info("Loading existing results from '%s'...", file_path)
        read_func = (
            pandas.read_csv
            if file_path.suffix == ".csv"
            else pandas.read_parquet
        )
        df = read_func(file_path)
        logging.info("Loaded %d rows from '%s'", df.shape[0], file_path)
        return df

    def __init__(
        self, file_path: Optional[Union[str, Path]], test_set: TestSet
    ):
        """Initialize results.

        Args:
            file_path: Path to the results file (format: CSV or Parquet), or
                `None` if there is no file associated with these results.
            test_set: Test set from which results were produced.
        """
        df = pandas.DataFrame(
            [],
            columns=[
                "problem",
                "solver",
                "settings",
                "runtime",
                "found",
                "primal_residual",
                "dual_residual",
                "duality_gap",
            ],
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
        if file_path is not None:
            df_from_file = Results.read_from_file(file_path)
            if df_from_file is not None:
                df = pandas.concat([df, df_from_file])
        Results.check_df(df)

        # Filter out problems from the CSV that are in the test set
        problems = set(problem.name for problem in test_set)
        test_set_df = df[df["problem"].isin(problems)]
        complementary_df = df[~df["problem"].isin(problems)]

        self.__complementary_df = complementary_df
        self.__new_results: list = []
        self.df = test_set_df
        self.file_path = Path(file_path) if file_path is not None else None
        self.test_set = test_set

    @property
    def nb_rows(self) -> int:
        """Number of rows in the dataframe."""
        return self.df.shape[0] + len(self.__new_results)

    def write(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write results to their CSV file for persistence.

        Args:
            path: Optional path to a separate file to write to.
        """
        path_check = path or self.file_path
        if path_check is None:
            raise BenchmarkError("no path to save results to")
        self.__merge_new_results()
        save_path = Path(path_check)
        save_df = pandas.concat([self.df, self.__complementary_df])
        save_df = save_df.sort_values(by=["problem", "solver", "settings"])
        if save_path.suffix == ".csv":
            save_df.to_csv(save_path, index=False)
        elif save_path.suffix == ".parquet":
            save_df.to_parquet(save_path, index=False)
        else:  # unknown file extension
            raise BenchmarkError(
                f"unknown results file extension in '{save_path}'"
            )
        logging.debug(
            "Test set results written to '%s' (%d rows saved)",
            save_path,
            save_df.shape[0],
        )

    def has(self, problem: Problem, solver: str, settings: str) -> bool:
        """Check if results contain a given run of a solver on a problem.

        Args:
            problem: Test set problem.
            solver: Name of the QP solver.
            settings: Name of the corresponding solver settings.

        Returns:
            True if a result for this instance is present.
        """
        self.__merge_new_results()
        return (
            (self.df["problem"] == problem.name)
            & (self.df["solver"] == solver)
            & (self.df["settings"] == settings)
        ).any()

    def is_timeout(
        self, problem: Problem, solver: str, settings: str, time_limit: float
    ) -> bool:
        """Check whether a particular result was a timeout."""
        self.__merge_new_results()
        runtime = self.df[
            (self.df["problem"] == problem.name)
            & (self.df["solver"] == solver)
            & (self.df["settings"] == settings)
        ]["runtime"].iat[0]
        return runtime > 0.99 * time_limit

    def __merge_new_results(self) -> None:
        if not self.__new_results:
            return
        new_df = pandas.DataFrame(self.__new_results)
        self.df = pandas.concat([self.df, new_df], ignore_index=True)
        self.df = self.df.drop_duplicates(
            subset=["problem", "solver", "settings"], keep="last"
        )
        self.__new_results = []

    def update(
        self,
        problem: Union[Problem, str],
        solver: str,
        settings: str,
        solution: Optional[qpsolvers.Solution] = None,
        runtime: float = 0.0,
        *,
        found: Optional[bool] = None,
        primal_residual: Optional[float] = None,
        dual_residual: Optional[float] = None,
        duality_gap: Optional[float] = None,
    ) -> None:
        """Update entry for a given (problem, solver) pair.

        Args:
            problem: Problem solved or problem name.
            solver: Solver name.
            settings: Solver settings.
            solution: Solution found by the solver (optional).
            runtime: Duration the solver took, in seconds.
            found: Whether the solver found a solution.
            primal_residual: Primal residual of the solution.
            dual_residual: Dual residual of the solution.
            duality_gap: Duality gap of the solution.
        """
        problem_name = problem if isinstance(problem, str) else problem.name

        if solution is not None:
            found_val: Optional[bool] = solution.found
            primal_res_val: Optional[float] = solution.primal_residual()
            dual_res_val: Optional[float] = solution.dual_residual()
            gap_val: Optional[float] = solution.duality_gap()
        else:
            found_val = found
            primal_res_val = primal_residual
            dual_res_val = dual_residual
            gap_val = duality_gap

        self.__new_results.append(
            {
                "problem": problem_name,
                "solver": solver,
                "settings": settings,
                "runtime": runtime,
                "found": True if found_val else False,
                "primal_residual": primal_res_val,
                "dual_residual": dual_res_val,
                "duality_gap": gap_val,
            }
        )

    def build_success_rate_df(
        self,
        primal_tolerances: Dict[str, float],
        dual_tolerances: Dict[str, float],
        gap_tolerances: Dict[str, float],
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """Build the success-rate data frame.

        Args:
            primal_tolerances: Primal-residual tolerance for each settings.
            dual_tolerances: Dual-residual tolerance for each settings.
            gap_tolerances: Duality-gap tolerance for each settings.

        Returns:
            Success-rate data frames.
        """
        self.__merge_new_results()
        solvers = set(self.df["solver"].to_list())
        all_settings = set(self.df["settings"].to_list())
        df = self.df.fillna(value=np.nan)  # replace None by NaN for abs()
        found_and_valid = {
            settings: df["found"]
            & (df["primal_residual"] < primal_tolerances[settings])
            & (df["dual_residual"] < dual_tolerances[settings])
            & (df["duality_gap"] < gap_tolerances[settings])
            for settings in all_settings
        }
        success_rate_df = (
            pandas.DataFrame(
                {
                    settings: {
                        solver: 100.0
                        * found_and_valid[settings][
                            (df["settings"] == settings)
                            & (df["solver"] == solver)
                        ]
                        .astype(float)
                        .mean()
                        for solver in solvers
                    }
                    for settings in all_settings
                }
            )
            .reindex(columns=sorted(all_settings))
            .sort_index()
        )
        return success_rate_df

    def build_correct_rate_df(
        self,
        primal_tolerances: Dict[str, float],
        dual_tolerances: Dict[str, float],
        gap_tolerances: Dict[str, float],
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """Build the correctness-rate data frame.

        Args:
            primal_tolerances: Primal-residual tolerance for each settings.
            dual_tolerances: Dual-residual tolerance for each settings.
            gap_tolerances: Duality-gap tolerance for each settings.

        Returns:
            Correctness-rate data frames.
        """
        self.__merge_new_results()
        solvers = set(self.df["solver"].to_list())
        all_settings = set(self.df["settings"].to_list())
        df = self.df.fillna(value=np.nan)  # replace None by NaN for abs()
        found_and_valid = {
            settings: df["found"]
            & (df["primal_residual"] < primal_tolerances[settings])
            & (df["dual_residual"] < dual_tolerances[settings])
            & (df["duality_gap"] < gap_tolerances[settings])
            for settings in all_settings
        }
        correctness_rate_df = (
            pandas.DataFrame(
                {
                    settings: {
                        solver: 100.0
                        * (
                            df[
                                (df["settings"] == settings)
                                & (df["solver"] == solver)
                            ]["found"]
                            == found_and_valid[settings][
                                (df["settings"] == settings)
                                & (df["solver"] == solver)
                            ]
                        )
                        .astype(float)
                        .mean()
                        for solver in solvers
                    }
                    for settings in all_settings
                }
            )
            .reindex(columns=sorted(all_settings))
            .sort_index()
        )
        return correctness_rate_df

    def get_shgeom_for_metric_and_settings(
        self,
        metric: str,
        settings: str,
        shift: float,
        not_found_value: float,
    ) -> Dict[str, float]:
        """Get shifted geometric means for a given metric with given settings.

        Args:
            metric: Name of the metric column to average.
            settings: Name of the settings column to filter on.
            shift: Shift of the shifted geometric mean.
            not_found_value: Value to apply when a solver has not found a
                solution.

        Returns:
            Dictionary with the shifted geometric mean of each solver.
        """
        self.__merge_new_results()
        solvers = set(self.df["solver"].to_list())
        means = {}
        for solver in solvers:
            solver_df = self.df[
                (self.df["solver"] == solver)
                & (self.df["settings"] == settings)
            ]
            column_values = np.array(
                [
                    (
                        solver_df.at[i, metric]
                        if solver_df.at[i, "found"]
                        else not_found_value
                    )
                    for i in solver_df.index
                ]
            )
            try:
                means[solver] = shgeom(column_values, shift)
            except BenchmarkError as exn:
                raise BenchmarkError(
                    f"Cannot evaluate mean for {settings=} of {solver=}"
                ) from exn
        best_mean = np.min(list(means.values()))
        return {solver: means[solver] / best_mean for solver in solvers}

    def build_shgeom_df(
        self, metric: str, shift: float, not_found_values: Dict[str, float]
    ) -> pandas.DataFrame:
        """Compute the shifted geometric mean for a given metric.

        Args:
            metric: Name of the metric column to average.
            shift: Shift of the shifted geometric mean.
            not_found_values: Values to apply when a solver has not found a
                solution (one per settings). For instance, time limits are used
                for the runtime of a solver that fails to solve a problem.

        Returns:
            Shifted geometric mean of the prescribed column.
        """
        all_settings = set(self.df["settings"].to_list())
        return (
            pandas.DataFrame(
                {
                    settings: self.get_shgeom_for_metric_and_settings(
                        metric,
                        settings,
                        shift=shift,
                        not_found_value=not_found_values[settings],
                    )
                    for settings in all_settings
                }
            )
            .reindex(columns=sorted(all_settings))
            .sort_index()
        )
