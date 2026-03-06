"""Tests for BinaryClassificationEvaluator and its metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_builder.config.schema import (
    BaseConfig,
    DataConfig,
    PipelineConfig,
    TrainTestSplitConfig,
)
from model_builder.evaluation.evaluator import (
    BinaryClassificationEvaluator,
    _compute_metrics,
    _optimal_threshold_f1,
    _random_baseline_metrics,
)
from model_builder.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, *, name: str = "eval_run") -> PipelineConfig:
    return PipelineConfig(
        base=BaseConfig(name=name, output_dir=str(tmp_path), seed=42),
        data=DataConfig(target_column="target"),
        split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
    )


def _binary_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.exponential(2, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _multiclass_df(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.uniform(0, 1, n),
            "target": rng.integers(0, 3, n),
        }
    )


def _make_split(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# _optimal_threshold_f1
# ---------------------------------------------------------------------------


class TestOptimalThresholdF1:
    def test_returns_float_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(0, 1, 100)
        t = _optimal_threshold_f1(y, p)
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0

    def test_perfect_model_separates_classes(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        t = _optimal_threshold_f1(y, p)
        # Threshold is valid; applying it should yield perfect F1
        y_pred = (p >= t).astype(int)
        from sklearn.metrics import f1_score

        assert f1_score(y, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _sample(self, n: int = 60, seed: int = 1):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, n), rng.uniform(0, 1, n)

    def test_returns_all_expected_keys(self):
        y, p = self._sample()
        m = _compute_metrics(y, p)
        expected = {
            "n_rows",
            "actual_positive_rate",
            "sum_predicted_probs",
            "log_loss",
            "brier_score",
            "auc_pr",
            "auc_roc",
            "optimal_threshold_f1",
            "accuracy",
            "precision",
            "recall",
            "f1",
        }
        assert set(m.keys()) == expected

    def test_n_rows(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        assert _compute_metrics(y, p)["n_rows"] == 4

    def test_auc_roc_in_range(self):
        y, p = self._sample()
        assert 0.0 <= float(_compute_metrics(y, p)["auc_roc"]) <= 1.0

    def test_explicit_threshold_recorded(self):
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        m = _compute_metrics(y, p, threshold=0.6)
        assert float(m["optimal_threshold_f1"]) == pytest.approx(0.6)

    def test_perfect_predictor_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.05, 0.1, 0.9, 0.95])
        m = _compute_metrics(y, p)
        assert float(m["f1"]) == pytest.approx(1.0)
        assert float(m["accuracy"]) == pytest.approx(1.0)

    def test_sum_predicted_probs(self):
        y = np.array([0, 1, 1])
        p = np.array([0.2, 0.7, 0.8])
        m = _compute_metrics(y, p)
        assert float(m["sum_predicted_probs"]) == pytest.approx(1.7, abs=1e-4)

    def test_actual_positive_rate(self):
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0.1, 0.8, 0.9, 0.2, 0.7])
        m = _compute_metrics(y, p)
        assert float(m["actual_positive_rate"]) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _random_baseline_metrics
# ---------------------------------------------------------------------------


class TestRandomBaselineMetrics:
    def test_threshold_fixed_at_half(self):
        y = np.array([0, 1, 1, 0])
        m = _random_baseline_metrics(y, base_rate=0.4)
        assert float(m["optimal_threshold_f1"]) == pytest.approx(0.5)

    def test_sum_predicted_probs(self):
        y = np.array([0, 1, 1, 0, 1])
        base_rate = 0.6
        m = _random_baseline_metrics(y, base_rate)
        assert float(m["sum_predicted_probs"]) == pytest.approx(
            base_rate * len(y), abs=1e-4
        )

    def test_returns_all_keys(self):
        y = np.array([0, 1, 0, 1, 1])
        m = _random_baseline_metrics(y, base_rate=0.4)
        assert "log_loss" in m and "auc_roc" in m and "auc_pr" in m


# ---------------------------------------------------------------------------
# BinaryClassificationEvaluator — integration tests via Pipeline
# ---------------------------------------------------------------------------


def _eval_dir(tmp_path, cfg: PipelineConfig, model_id: int):
    return tmp_path / cfg.base.name / "evaluation" / str(model_id)


class TestBinaryClassificationEvaluator:
    def _run(self, tmp_path, name: str = "eval_run"):
        cfg = _make_config(tmp_path, name=name)
        pipeline = Pipeline(cfg).run(_binary_df())
        return cfg, pipeline

    # --- artifact existence ---

    def test_metrics_csv_created(self, tmp_path):
        cfg, p = self._run(tmp_path)
        assert (_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv").exists()

    def test_report_pdf_created(self, tmp_path):
        cfg, p = self._run(tmp_path)
        assert (_eval_dir(tmp_path, cfg, p.model_id) / "report.pdf").exists()

    # --- CSV content ---

    def test_csv_has_expected_splits(self, tmp_path):
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        for split in ("train", "test", "overall", "random"):
            assert split in df["split"].values, f"missing split: {split}"

    def test_all_splits_share_train_threshold(self, tmp_path):
        """Train threshold must be reused for test/overall — no leakage."""
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        non_random = df[df["split"] != "random"]["optimal_threshold_f1"]
        assert non_random.nunique() == 1, "thresholds differ across splits"

    def test_csv_row_order(self, tmp_path):
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        assert list(df["split"]) == ["train", "test", "overall", "random"]

    def test_csv_no_per_split_random_rows(self, tmp_path):
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        assert not any("_random" in s for s in df["split"].values)

    def test_csv_has_expected_columns(self, tmp_path):
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        for col in (
            "log_loss",
            "auc_roc",
            "auc_pr",
            "f1",
            "accuracy",
            "n_rows",
            "brier_score",
        ):
            assert col in df.columns, f"missing column: {col}"

    def test_csv_model_better_than_random_auc_roc(self, tmp_path):
        """A trained logistic regression should out-score random on AUC ROC."""
        cfg, p = self._run(tmp_path)
        df = pd.read_csv(_eval_dir(tmp_path, cfg, p.model_id) / "metrics.csv")
        model_roc = df.loc[df["split"] == "overall", "auc_roc"].iloc[0]
        random_roc = df.loc[df["split"] == "random", "auc_roc"].iloc[0]
        assert model_roc >= random_roc

    # --- val split ---

    def test_val_split_present_when_configured(self, tmp_path):
        cfg = PipelineConfig(
            base=BaseConfig(name="val_run", output_dir=str(tmp_path), seed=42),
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.7, test_size=0.15, val_size=0.15),
        )
        pipeline = Pipeline(cfg).run(_binary_df())
        df = pd.read_csv(_eval_dir(tmp_path, cfg, pipeline.model_id) / "metrics.csv")
        assert list(df["split"]) == ["train", "test", "val", "overall", "random"]

    # --- multiclass guard ---

    def test_no_evaluation_for_multiclass_target(self, tmp_path):
        """Evaluation should be silently skipped for > 2 target classes."""
        cfg = _make_config(tmp_path, name="mc_run")
        Pipeline(cfg).run(_multiclass_df())
        eval_dir = tmp_path / "mc_run" / "evaluation"
        assert not eval_dir.exists()

    # --- standalone evaluator (no pipeline) ---

    def test_standalone_evaluate(self, tmp_path):
        from model_builder.training.models.logr import LogisticRegressionModel

        X_train, y_train = _make_split(80, seed=0)
        X_test, y_test = _make_split(40, seed=1)
        model = LogisticRegressionModel(cv=2, max_iter=200)
        model.fit(X_train, y_train)

        cfg = _make_config(tmp_path, name="standalone")
        evaluator = BinaryClassificationEvaluator(cfg, model_id=1)
        evaluator.evaluate(model, (X_train, y_train), (X_test, y_test))

        out = tmp_path / "standalone" / "evaluation" / "1"
        assert (out / "metrics.csv").exists()
        assert (out / "report.pdf").exists()
