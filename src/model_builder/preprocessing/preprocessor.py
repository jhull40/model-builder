from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from model_builder.config.schema import PipelineConfig
from model_builder.eda.analyzer import _describe_distribution

_STANDARD_SCALE_DISTRIBUTIONS = {"gaussian", "right-skewed", "left-skewed"}


class Preprocessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._scalers: Dict[str, Any] = {}
        self._impute_values: Dict[str, float] = {}
        self._distributions: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Split *df* into (train, test, val) before any fitting/scaling.

        Strategy priority:
        1. **Date-based** – when any start/stop date is configured, rows are
           filtered by the date ranges defined in the split config.
        2. **Column-based** – when ``split_column`` is set, unique values of
           that column are partitioned so entire groups stay together (useful
           for player IDs, subject IDs, etc. to prevent leakage).
        3. **Random** – plain ``train_test_split`` with optional stratification.

        ``val_df`` is ``None`` when ``val_size == 0``.
        """
        sc = self.config.split
        if self._is_date_split():
            return self._date_split(df)
        elif sc.split_column is not None:
            return self._column_split(df)
        else:
            return self._random_split(df)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        target = self.config.data.target_column
        feature_cols = self._numeric_feature_cols(df, target)

        for col in feature_cols:
            dist = _describe_distribution(df[col])
            self._distributions[col] = dist

            if dist == "binary":
                continue

            scaler = (
                StandardScaler()
                if dist in _STANDARD_SCALE_DISTRIBUTIONS
                else MinMaxScaler()
            )
            scaler.fit(df[[col]].dropna())
            self._scalers[col] = scaler

        strategy = self.config.data.impute_strategy
        if strategy in ("mean", "median"):
            for col in feature_cols:
                self._impute_values[col] = (
                    float(df[col].mean())
                    if strategy == "mean"
                    else float(df[col].median())
                )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        strategy = self.config.data.impute_strategy

        if strategy == "drop":
            df = df.dropna()
        else:
            for col, value in self._impute_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)

        for col, scaler in self._scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]]).ravel()

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def _is_date_split(self) -> bool:
        sc = self.config.split
        return any(
            [
                sc.start_train_date,
                sc.stop_train_date,
                sc.start_test_date,
                sc.stop_test_date,
            ]
        )

    def _date_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        date_col = self.config.data.date_column
        if date_col is None:
            raise ValueError(
                "data.date_column must be set when using date-based splitting"
            )

        sc = self.config.split
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        def _filter(start: Optional[str], stop: Optional[str]) -> pd.DataFrame:
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df[date_col] >= pd.Timestamp(start)
            if stop:
                mask &= df[date_col] <= pd.Timestamp(stop)
            return df[mask]

        train_df = _filter(sc.start_train_date, sc.stop_train_date)
        test_df = _filter(sc.start_test_date, sc.stop_test_date)
        val_df = (
            _filter(sc.start_val_date, sc.stop_val_date)
            if (sc.start_val_date or sc.stop_val_date)
            else None
        )
        return train_df, test_df, val_df

    def _column_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Partition on unique values of *split_column* to prevent row-level leakage."""
        sc = self.config.split
        rng = np.random.default_rng(self.config.base.seed)

        unique_vals = df[sc.split_column].unique()
        rng.shuffle(unique_vals)

        n_total = len(unique_vals)
        n_train = max(1, int(np.ceil(n_total * sc.train_size)))
        train_vals = unique_vals[:n_train]
        remaining_vals = unique_vals[n_train:]

        train_df = df[df[sc.split_column].isin(train_vals)]

        if sc.val_size > 0 and len(remaining_vals) > 1:
            total_remaining = sc.test_size + sc.val_size
            n_test = max(
                1,
                int(np.ceil(len(remaining_vals) * (sc.test_size / total_remaining))),
            )
            test_df = df[df[sc.split_column].isin(remaining_vals[:n_test])]
            val_df = df[df[sc.split_column].isin(remaining_vals[n_test:])]
        else:
            test_df = df[df[sc.split_column].isin(remaining_vals)]
            val_df = None

        return train_df, test_df, val_df

    def _random_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        sc = self.config.split
        target = self.config.data.target_column
        stratify = (
            df[target]
            if (sc.stratified_split and target and target in df.columns)
            else None
        )

        train_df, temp_df = _sklearn_split(
            df,
            train_size=sc.train_size,
            shuffle=sc.shuffle,
            stratify=stratify,
            random_state=self.config.base.seed,
        )

        if sc.val_size > 0 and len(temp_df) > 1:
            total_remaining = sc.test_size + sc.val_size
            val_relative = sc.val_size / total_remaining
            stratify_temp = (
                temp_df[target]
                if (sc.stratified_split and target and target in temp_df.columns)
                else None
            )
            test_df, val_df = _sklearn_split(
                temp_df,
                test_size=val_relative,
                shuffle=sc.shuffle,
                stratify=stratify_temp,
                random_state=self.config.base.seed,
            )
        else:
            test_df = temp_df
            val_df = None

        return train_df, test_df, val_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_feature_cols(df: pd.DataFrame, target: Optional[str]) -> list[str]:
        numeric = df.select_dtypes(include="number").columns.tolist()
        return [c for c in numeric if c != target]
