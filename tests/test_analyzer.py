"""Tests for DataAnalyzer and _describe_distribution."""

import os

import numpy as np
import pandas as pd

from model_builder.config.schema import BaseConfig, DataConfig, PipelineConfig
from model_builder.eda.analyzer import DataAnalyzer, _describe_distribution


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, target_column="target"):
    return PipelineConfig(
        base=BaseConfig(name="test_eda", output_dir=str(tmp_path)),
        data=DataConfig(target_column=target_column),
    )


def _iris_df():
    """Simple multi-column numeric dataframe similar to Iris."""
    rng = np.random.default_rng(0)
    n = 150
    return pd.DataFrame(
        {
            "a": rng.normal(5.0, 1.0, n),
            "b": rng.exponential(2.0, n),
            "c": rng.uniform(0.0, 1.0, n),
            "target": rng.integers(0, 3, n),
        }
    )


# ---------------------------------------------------------------------------
# _describe_distribution
# ---------------------------------------------------------------------------


class TestDescribeDistribution:
    def test_binary(self):
        s = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])
        assert _describe_distribution(s) == "binary"

    def test_ordinal_categorical(self):
        s = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 5)
        assert _describe_distribution(s) == "ordinal/categorical"

    def test_gaussian(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 1000))
        result = _describe_distribution(s)
        assert result == "gaussian"

    def test_right_skewed(self):
        rng = np.random.default_rng(0)
        # Exponential distribution is strongly right-skewed
        s = pd.Series(rng.exponential(1.0, 2000))
        result = _describe_distribution(s)
        assert result == "right-skewed"

    def test_left_skewed(self):
        rng = np.random.default_rng(0)
        # Reflect an exponential to get left skew
        s = pd.Series(-rng.exponential(1.0, 2000))
        result = _describe_distribution(s)
        assert result == "left-skewed"

    def test_ignores_nulls(self):
        rng = np.random.default_rng(7)
        values = list(rng.normal(0, 1, 200)) + [np.nan] * 20
        s = pd.Series(values)
        # Should not raise; nulls are dropped internally
        result = _describe_distribution(s)
        assert isinstance(result, str)

    def test_small_series_no_crash(self):
        # < 8 observations — should still return a string
        s = pd.Series([1.0, 2.5, 3.0, 4.0])
        result = _describe_distribution(s)
        assert isinstance(result, str)

    def test_returns_string(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(0, 1, 100))
        assert isinstance(_describe_distribution(s), str)


# ---------------------------------------------------------------------------
# DataAnalyzer.__init__ / directory creation
# ---------------------------------------------------------------------------


class TestDataAnalyzerInit:
    def test_eda_output_dir_created(self, tmp_path):
        cfg = _make_config(tmp_path)
        analyzer = DataAnalyzer(cfg)
        assert os.path.isdir(analyzer.output_path)

    def test_output_path_contains_run_name(self, tmp_path):
        cfg = _make_config(tmp_path)
        analyzer = DataAnalyzer(cfg)
        assert "test_eda" in analyzer.output_path

    def test_output_path_contains_eda_subdir(self, tmp_path):
        cfg = _make_config(tmp_path)
        analyzer = DataAnalyzer(cfg)
        assert analyzer.output_path.endswith("eda")


# ---------------------------------------------------------------------------
# DataAnalyzer.run — file outputs
# ---------------------------------------------------------------------------


class TestDataAnalyzerRunOutputs:
    def test_run_creates_pdf(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        assert os.path.isfile(os.path.join(str(tmp_path), "test_eda", "eda", "eda.pdf"))

    def test_run_creates_describe_csv(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        path = os.path.join(str(tmp_path), "test_eda", "eda", "describe.csv")
        assert os.path.isfile(path)

    def test_run_creates_correlations_csv(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        path = os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv")
        assert os.path.isfile(path)

    def test_run_creates_outliers_csv(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        path = os.path.join(str(tmp_path), "test_eda", "eda", "outliers.csv")
        assert os.path.isfile(path)

    def test_run_creates_histogram_pngs(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        plots_dir = os.path.join(str(tmp_path), "test_eda", "eda", "plots")
        pngs = [f for f in os.listdir(plots_dir) if f.endswith("_histogram.png")]
        assert len(pngs) > 0

    def test_histogram_png_per_numeric_column(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _iris_df()
        DataAnalyzer(cfg).run(df)
        plots_dir = os.path.join(str(tmp_path), "test_eda", "eda", "plots")
        pngs = {f for f in os.listdir(plots_dir) if f.endswith("_histogram.png")}
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            assert f"{col}_histogram.png" in pngs


# ---------------------------------------------------------------------------
# DataAnalyzer._add_describe
# ---------------------------------------------------------------------------


class TestDescribeCSV:
    def _describe_df(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        return pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "describe.csv"), index_col=0
        )

    def test_all_columns_present(self, tmp_path):
        desc = self._describe_df(tmp_path)
        for col in _iris_df().columns:
            assert col in desc.index

    def test_nulls_column_present(self, tmp_path):
        desc = self._describe_df(tmp_path)
        assert "nulls" in desc.columns

    def test_distribution_column_present(self, tmp_path):
        desc = self._describe_df(tmp_path)
        assert "distribution" in desc.columns

    def test_binary_target_labelled(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"x": rng.normal(0, 1, 100), "target": rng.integers(0, 2, 100)}
        )
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(df)
        desc = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "describe.csv"), index_col=0
        )
        assert desc.loc["target", "distribution"] == "binary"

    def test_nulls_counted_correctly(self, tmp_path):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "x": [np.nan] * 5 + list(rng.normal(0, 1, 95)),
                "target": rng.integers(0, 2, 100),
            }
        )
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(df)
        desc = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "describe.csv"), index_col=0
        )
        assert int(desc.loc["x", "nulls"]) == 5


# ---------------------------------------------------------------------------
# DataAnalyzer._add_correlations
# ---------------------------------------------------------------------------


class TestCorrelationsCSV:
    def test_correlations_is_square(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        corr = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv"),
            index_col=0,
        )
        assert corr.shape[0] == corr.shape[1]

    def test_diagonal_is_one(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        corr = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv"),
            index_col=0,
        )
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_values_in_range(self, tmp_path):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(_iris_df())
        corr = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv"),
            index_col=0,
        )
        assert corr.values.min() >= -1.0 - 1e-6
        assert corr.values.max() <= 1.0 + 1e-6

    def test_only_numeric_columns(self, tmp_path):
        """Correlations CSV should only contain numeric columns."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "num": rng.normal(0, 1, 50),
                "cat": ["a", "b"] * 25,
                "target": rng.integers(0, 2, 50),
            }
        )
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(df)
        corr = pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv"),
            index_col=0,
        )
        assert "cat" not in corr.columns

    def test_single_numeric_column_skips_correlations(self, tmp_path):
        """With only one numeric column the correlations CSV must not be written."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"cat": ["a", "b"] * 25, "target": rng.integers(0, 2, 50)})
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(df)
        path = os.path.join(str(tmp_path), "test_eda", "eda", "correlations.csv")
        # single numeric column → correlation skipped, file should not exist
        assert not os.path.isfile(path)


# ---------------------------------------------------------------------------
# DataAnalyzer._add_outliers
# ---------------------------------------------------------------------------


class TestOutliersCSV:
    def _outliers_df(self, tmp_path, df=None):
        cfg = _make_config(tmp_path)
        DataAnalyzer(cfg).run(df if df is not None else _iris_df())
        return pd.read_csv(
            os.path.join(str(tmp_path), "test_eda", "eda", "outliers.csv")
        )

    def test_columns_present(self, tmp_path):
        out = self._outliers_df(tmp_path)
        for col in (
            "column",
            "method",
            "n_outliers",
            "pct_outliers",
            "lower_fence",
            "upper_fence",
        ):
            assert col in out.columns

    def test_one_row_per_numeric_column(self, tmp_path):
        df = _iris_df()
        out = self._outliers_df(tmp_path, df)
        assert len(out) == len(df.select_dtypes(include="number").columns)

    def test_known_outlier_detected(self, tmp_path):
        rng = np.random.default_rng(0)
        values = list(rng.normal(0, 1, 98)) + [100.0, -100.0]
        df = pd.DataFrame({"x": values, "target": rng.integers(0, 2, 100)})
        out = self._outliers_df(tmp_path, df)
        row = out[out["column"] == "x"].iloc[0]
        assert row["n_outliers"] >= 2

    def test_pct_outliers_consistent_with_count(self, tmp_path):
        df = _iris_df()
        out = self._outliers_df(tmp_path, df)
        for _, row in out.iterrows():
            col_len = df[row["column"]].dropna().shape[0]
            expected_pct = round(100.0 * row["n_outliers"] / col_len, 2)
            assert abs(row["pct_outliers"] - expected_pct) < 0.01

    def test_no_numeric_cols_no_crash(self, tmp_path):
        df = pd.DataFrame({"a": ["x", "y", "z"] * 10, "b": ["p", "q", "r"] * 10})
        cfg = _make_config(tmp_path, target_column=None)
        # Should complete without error even with no numeric columns
        DataAnalyzer(cfg).run(df)

    def test_fences_ordered(self, tmp_path):
        out = self._outliers_df(tmp_path)
        assert (out["lower_fence"] <= out["upper_fence"]).all()
