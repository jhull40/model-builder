"""Tests for Preprocessor: fit/transform, split strategies."""

import numpy as np
import pandas as pd
import pytest

from model_builder.config.schema import (
    BaseConfig,
    DataConfig,
    PipelineConfig,
    TrainTestSplitConfig,
)
from model_builder.preprocessing.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path, *, data=None, split=None):
    return PipelineConfig(
        base=BaseConfig(name="test", output_dir=str(tmp_path), seed=42),
        data=data or DataConfig(),
        split=split or TrainTestSplitConfig(),
    )


def _numeric_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.exponential(2, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _df_with_nulls(n=100, seed=1):
    rng = np.random.default_rng(seed)
    df = _numeric_df(n=n, seed=seed)
    null_idx = rng.choice(n, size=10, replace=False)
    df.loc[null_idx, "a"] = np.nan
    return df


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------


class TestFitTransform:
    def test_fit_returns_self(self, tmp_path):
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        assert pre.fit(_numeric_df()) is pre

    def test_transform_no_nulls_preserves_shape(self, tmp_path):
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        df = _numeric_df()
        result = pre.fit_transform(df)
        assert result.shape == df.shape

    def test_scalers_fitted_for_non_binary_cols(self, tmp_path):
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        pre.fit(_numeric_df())
        # target (binary) should not get a scaler; a and b should
        assert "target" not in pre._scalers
        assert "a" in pre._scalers or "b" in pre._scalers

    def test_target_excluded_from_scaling(self, tmp_path):
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        df = _numeric_df()
        result = pre.fit_transform(df)
        # target column values must be unchanged
        pd.testing.assert_series_equal(result["target"], df["target"])

    def test_transform_only_uses_train_stats(self, tmp_path):
        """Scaler fitted on train must not re-fit on test."""
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        train = _numeric_df(n=100, seed=0)
        test = _numeric_df(n=50, seed=7)
        pre.fit(train)
        result = pre.transform(test)
        assert result.shape == test.shape

    def test_distributions_recorded(self, tmp_path):
        cfg = _make_config(tmp_path, data=DataConfig(target_column="target"))
        pre = Preprocessor(cfg)
        pre.fit(_numeric_df())
        assert "a" in pre._distributions
        assert "b" in pre._distributions

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    def test_median_imputation_fills_nulls(self, tmp_path):
        cfg = _make_config(
            tmp_path, data=DataConfig(target_column="target", impute_strategy="median")
        )
        pre = Preprocessor(cfg)
        df = _df_with_nulls()
        result = pre.fit_transform(df)
        assert result["a"].isna().sum() == 0

    def test_mean_imputation_fills_nulls(self, tmp_path):
        cfg = _make_config(
            tmp_path, data=DataConfig(target_column="target", impute_strategy="mean")
        )
        pre = Preprocessor(cfg)
        df = _df_with_nulls()
        result = pre.fit_transform(df)
        assert result["a"].isna().sum() == 0

    def test_drop_imputation_removes_rows(self, tmp_path):
        cfg = _make_config(
            tmp_path, data=DataConfig(target_column="target", impute_strategy="drop")
        )
        pre = Preprocessor(cfg)
        df = _df_with_nulls(n=100)
        result = pre.fit_transform(df)
        assert len(result) < len(df)
        assert result["a"].isna().sum() == 0

    def test_impute_values_use_train_stats(self, tmp_path):
        """Mean/median stored during fit must not change when transforming different data."""
        cfg = _make_config(
            tmp_path, data=DataConfig(target_column="target", impute_strategy="mean")
        )
        pre = Preprocessor(cfg)
        train = _df_with_nulls(n=100, seed=1)
        pre.fit(train)
        stored_mean = pre._impute_values["a"]

        test = _df_with_nulls(n=50, seed=99)
        pre.transform(test)
        assert pre._impute_values["a"] == stored_mean


# ---------------------------------------------------------------------------
# split – random
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_basic_sizes(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
        )
        df = _numeric_df(n=200)
        train, test, val = Preprocessor(cfg).split(df)
        assert len(train) + len(test) == len(df)
        assert val is None

    def test_approximate_train_ratio(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
        )
        df = _numeric_df(n=500)
        train, test, _ = Preprocessor(cfg).split(df)
        ratio = len(train) / len(df)
        assert abs(ratio - 0.8) < 0.05

    def test_val_split_sizes(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.7, test_size=0.2, val_size=0.1),
        )
        df = _numeric_df(n=300)
        train, test, val = Preprocessor(cfg).split(df)
        assert val is not None
        assert len(train) + len(test) + len(val) == len(df)

    def test_no_overlap_between_splits(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.7, test_size=0.2, val_size=0.1),
        )
        df = _numeric_df(n=300)
        train, test, val = Preprocessor(cfg).split(df)
        train_idx = set(train.index)
        test_idx = set(test.index)
        val_idx = set(val.index)  # type: ignore[union-attr]
        assert train_idx.isdisjoint(test_idx)
        assert train_idx.isdisjoint(val_idx)
        assert test_idx.isdisjoint(val_idx)

    def test_deterministic_with_seed(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
        )
        df = _numeric_df(n=200)
        train1, test1, _ = Preprocessor(cfg).split(df)
        train2, test2, _ = Preprocessor(cfg).split(df)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)


# ---------------------------------------------------------------------------
# split – column-based
# ---------------------------------------------------------------------------


class TestColumnSplit:
    def _df_with_groups(self, n_groups=20, rows_per_group=10):
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "player_id": groups,
                "a": rng.normal(0, 1, n_groups * rows_per_group),
                "target": rng.integers(0, 2, n_groups * rows_per_group),
            }
        )

    def test_no_group_leakage(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(
                train_size=0.8, test_size=0.2, split_column="player_id"
            ),
        )
        df = self._df_with_groups()
        train, test, val = Preprocessor(cfg).split(df)
        train_groups = set(train["player_id"].unique())
        test_groups = set(test["player_id"].unique())
        assert train_groups.isdisjoint(test_groups)
        assert val is None

    def test_all_rows_accounted_for(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(
                train_size=0.8, test_size=0.2, split_column="player_id"
            ),
        )
        df = self._df_with_groups()
        train, test, _ = Preprocessor(cfg).split(df)
        assert len(train) + len(test) == len(df)

    def test_val_split_no_leakage(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(
                train_size=0.7,
                test_size=0.2,
                val_size=0.1,
                split_column="player_id",
            ),
        )
        df = self._df_with_groups(n_groups=30)
        train, test, val = Preprocessor(cfg).split(df)
        assert val is not None
        all_groups = (
            set(train["player_id"]) | set(test["player_id"]) | set(val["player_id"])
        )
        assert len(all_groups) == 30
        assert set(train["player_id"]).isdisjoint(set(test["player_id"]))
        assert set(train["player_id"]).isdisjoint(set(val["player_id"]))
        assert set(test["player_id"]).isdisjoint(set(val["player_id"]))

    def test_approximate_train_group_ratio(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(
                train_size=0.8, test_size=0.2, split_column="player_id"
            ),
        )
        df = self._df_with_groups(n_groups=100)
        train, test, _ = Preprocessor(cfg).split(df)
        ratio = train["player_id"].nunique() / df["player_id"].nunique()
        assert abs(ratio - 0.8) < 0.1


# ---------------------------------------------------------------------------
# split – date-based
# ---------------------------------------------------------------------------


class TestDateSplit:
    def _df_with_dates(self, n=365):
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.default_rng(5)
        return pd.DataFrame(
            {
                "date": dates.astype(str),
                "a": rng.normal(0, 1, n),
                "target": rng.integers(0, 2, n),
            }
        )

    def test_date_split_basic(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target", date_column="date"),
            split=TrainTestSplitConfig(
                start_train_date="2023-01-01",
                stop_train_date="2023-09-30",
                start_test_date="2023-10-01",
                stop_test_date="2023-12-31",
            ),
        )
        df = self._df_with_dates()
        train, test, val = Preprocessor(cfg).split(df)
        assert len(train) > 0 and len(test) > 0
        assert val is None

    def test_train_dates_within_range(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target", date_column="date"),
            split=TrainTestSplitConfig(
                start_train_date="2023-01-01",
                stop_train_date="2023-06-30",
                start_test_date="2023-07-01",
                stop_test_date="2023-12-31",
            ),
        )
        df = self._df_with_dates()
        train, test, _ = Preprocessor(cfg).split(df)
        train_dates = pd.to_datetime(train["date"])
        assert train_dates.min() >= pd.Timestamp("2023-01-01")
        assert train_dates.max() <= pd.Timestamp("2023-06-30")

    def test_test_dates_within_range(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target", date_column="date"),
            split=TrainTestSplitConfig(
                start_train_date="2023-01-01",
                stop_train_date="2023-06-30",
                start_test_date="2023-07-01",
                stop_test_date="2023-12-31",
            ),
        )
        df = self._df_with_dates()
        _, test, _ = Preprocessor(cfg).split(df)
        test_dates = pd.to_datetime(test["date"])
        assert test_dates.min() >= pd.Timestamp("2023-07-01")
        assert test_dates.max() <= pd.Timestamp("2023-12-31")

    def test_val_date_range(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target", date_column="date"),
            split=TrainTestSplitConfig(
                start_train_date="2023-01-01",
                stop_train_date="2023-06-30",
                start_test_date="2023-07-01",
                stop_test_date="2023-09-30",
                start_val_date="2023-10-01",
                stop_val_date="2023-12-31",
            ),
        )
        df = self._df_with_dates()
        _, _, val = Preprocessor(cfg).split(df)
        assert val is not None
        val_dates = pd.to_datetime(val["date"])
        assert val_dates.min() >= pd.Timestamp("2023-10-01")
        assert val_dates.max() <= pd.Timestamp("2023-12-31")

    def test_missing_date_column_raises(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target", date_column=None),
            split=TrainTestSplitConfig(
                start_train_date="2023-01-01",
                stop_train_date="2023-06-30",
                start_test_date="2023-07-01",
                stop_test_date="2023-12-31",
            ),
        )
        df = self._df_with_dates()
        with pytest.raises(ValueError, match="date_column"):
            Preprocessor(cfg).split(df)


# ---------------------------------------------------------------------------
# End-to-end: split then fit/transform
# ---------------------------------------------------------------------------


class TestSplitThenFitTransform:
    def test_scaler_fitted_on_train_applied_to_test(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
        )
        df = _numeric_df(n=200)
        pre = Preprocessor(cfg)
        train, test, _ = pre.split(df)
        train_out = pre.fit_transform(train)
        test_out = pre.transform(test)

        # scaled columns should have no NaNs
        assert train_out["a"].isna().sum() == 0
        assert test_out["a"].isna().sum() == 0

    def test_column_split_then_scale_no_leakage(self, tmp_path):
        """Groups in test must not appear in train after column split + fit/transform."""
        rng = np.random.default_rng(3)
        n_groups, rows = 20, 15
        df = pd.DataFrame(
            {
                "player_id": np.repeat(np.arange(n_groups), rows),
                "a": rng.normal(0, 1, n_groups * rows),
                "target": rng.integers(0, 2, n_groups * rows),
            }
        )
        cfg = _make_config(
            tmp_path,
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(
                train_size=0.8, test_size=0.2, split_column="player_id"
            ),
        )
        pre = Preprocessor(cfg)
        train, test, _ = pre.split(df)
        pre.fit(train)
        test_out = pre.transform(test)

        assert set(train["player_id"]).isdisjoint(set(test["player_id"]))
        assert test_out.shape == test.shape
