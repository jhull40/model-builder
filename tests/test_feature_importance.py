"""Tests for FeatureImportanceAnalyzer."""

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
from model_builder.feature_importance.analyzer import (
    FeatureImportanceAnalyzer,
    _extract_importances,
)
from model_builder.training.models.logr import LogisticRegressionModel
from model_builder.training.models.xgbc import XGBClassifierModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, *, name: str = "fi_run") -> PipelineConfig:
    return PipelineConfig(
        base=BaseConfig(name=name, output_dir=str(tmp_path), seed=42),
        data=DataConfig(target_column="target"),
        split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
    )


def _fit_xgb(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.uniform(size=n),
            "f3": rng.exponential(size=n),
        }
    )
    y = pd.Series((X["f1"] + X["f2"] > 1).astype(int))
    model = XGBClassifierModel(n_estimators=10, seed=seed)
    model.fit(X, y)
    return model, list(X.columns)


def _fit_logr(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.uniform(size=n),
            "f3": rng.exponential(size=n),
        }
    )
    y = pd.Series((X["f1"] + X["f2"] > 1).astype(int))
    model = LogisticRegressionModel(cv=3, max_iter=200, seed=seed)
    model.fit(X, y)
    return model, list(X.columns)


# ---------------------------------------------------------------------------
# _extract_importances
# ---------------------------------------------------------------------------


class TestExtractImportances:
    def test_xgb_returns_all_features(self):
        model, features = _fit_xgb()
        df = _extract_importances(model, features)
        assert list(df.columns) == ["feature", "importance"]
        assert set(df["feature"]) == set(features)

    def test_xgb_sorted_descending(self):
        model, features = _fit_xgb()
        df = _extract_importances(model, features)
        assert df["importance"].is_monotonic_decreasing

    def test_logr_returns_all_features(self):
        model, features = _fit_logr()
        df = _extract_importances(model, features)
        assert set(df["feature"]) == set(features)

    def test_logr_importances_non_negative(self):
        model, features = _fit_logr()
        df = _extract_importances(model, features)
        assert (df["importance"] >= 0).all()

    def test_logr_sorted_descending(self):
        model, features = _fit_logr()
        df = _extract_importances(model, features)
        assert df["importance"].is_monotonic_decreasing

    def test_unsupported_model_raises(self):
        from model_builder.training.base import Model

        class DummyModel(Model):
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.zeros(len(X))

            def score(self, X, y):
                return 0.0

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                return cls()

        with pytest.raises(TypeError, match="not supported"):
            _extract_importances(DummyModel(), ["a", "b"])


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer
# ---------------------------------------------------------------------------


class TestFeatureImportanceAnalyzer:
    def test_xgb_writes_csv_and_plot(self, tmp_path):
        config = _make_config(tmp_path)
        model, features = _fit_xgb()
        FeatureImportanceAnalyzer(config, model_id=1).analyze(model, features)

        out = tmp_path / "fi_run" / "feature_importance" / "1"
        assert (out / "feature_importance.csv").exists()
        assert (out / "feature_importance.png").exists()

    def test_logr_writes_csv_and_plot(self, tmp_path):
        config = _make_config(tmp_path)
        model, features = _fit_logr()
        FeatureImportanceAnalyzer(config, model_id=2).analyze(model, features)

        out = tmp_path / "fi_run" / "feature_importance" / "2"
        assert (out / "feature_importance.csv").exists()
        assert (out / "feature_importance.png").exists()

    def test_csv_contains_expected_columns(self, tmp_path):
        config = _make_config(tmp_path)
        model, features = _fit_xgb()
        FeatureImportanceAnalyzer(config, model_id=1).analyze(model, features)

        df = pd.read_csv(
            tmp_path / "fi_run" / "feature_importance" / "1" / "feature_importance.csv"
        )
        assert list(df.columns) == ["feature", "importance"]
        assert len(df) == len(features)

    def test_output_dir_uses_model_id(self, tmp_path):
        config = _make_config(tmp_path)
        model, features = _fit_xgb()
        FeatureImportanceAnalyzer(config, model_id=99).analyze(model, features)

        assert (tmp_path / "fi_run" / "feature_importance" / "99").is_dir()
