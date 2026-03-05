"""Tests for LogisticRegressionModel stages and its integration in the Pipeline."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_builder.config.schema import (
    BaseConfig,
    DataConfig,
    ModelConfig,
    LogisticRegressionConfig,
    PipelineConfig,
    TrainTestSplitConfig,
)
from model_builder.pipeline import Pipeline
from model_builder.training.models.logr import LogisticRegressionModel


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_Xy(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.exponential(2, n)})
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


def _make_df(n=150, seed=0):
    X, y = _make_Xy(n=n, seed=seed)
    return pd.concat([X, y], axis=1)


def _make_config(tmp_path, *, name="logr_run"):
    return PipelineConfig(
        base=BaseConfig(name=name, output_dir=str(tmp_path), seed=42),
        data=DataConfig(target_column="target"),
        split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
        model=ModelConfig(type="logr"),
    )


def _model_path(tmp_path, model_id, name="logr_run"):
    return Path(tmp_path) / name / "models" / str(model_id) / f"model_{model_id}"


# ---------------------------------------------------------------------------
# Unit tests: fit stage
# ---------------------------------------------------------------------------


class TestFitStage:
    def test_fit_returns_self(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200)
        result = model.fit(X, y)
        assert result is model

    def test_fit_enables_predict(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_fit_enables_score(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200)
        model.fit(X, y)
        score = model.score(X, y)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Unit tests: predict stage
# ---------------------------------------------------------------------------


class TestPredictStage:
    def test_predict_returns_ndarray(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_shape_matches_input(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_labels_are_valid_classes(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_multiclass_labels(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"a": rng.normal(0, 1, 150), "b": rng.normal(0, 1, 150)})
        y = pd.Series(rng.integers(0, 3, 150), name="target")
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Unit tests: score stage
# ---------------------------------------------------------------------------


class TestScoreStage:
    def test_score_returns_float(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        assert isinstance(model.score(X, y), float)

    def test_score_in_unit_interval(self):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_perfect_score_on_trivially_separable_data(self):
        X = pd.DataFrame({"a": [0.0] * 50 + [10.0] * 50})
        y = pd.Series([0] * 50 + [1] * 50)
        model = LogisticRegressionModel(cv=2, max_iter=1000).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unit tests: save / load stage
# ---------------------------------------------------------------------------


class TestSaveLoadStage:
    def test_save_creates_file(self, tmp_path):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        save_path = tmp_path / "model_1"
        model.save(save_path)
        assert save_path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        nested = tmp_path / "deep" / "nested" / "model_1"
        model.save(nested)
        assert nested.exists()

    def test_load_returns_logr_instance(self, tmp_path):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        save_path = tmp_path / "model_1"
        model.save(save_path)
        loaded = LogisticRegressionModel.load(save_path)
        assert isinstance(loaded, LogisticRegressionModel)

    def test_loaded_model_predict_matches_original(self, tmp_path):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        save_path = tmp_path / "model_1"
        model.save(save_path)
        loaded = LogisticRegressionModel.load(save_path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_loaded_model_score_matches_original(self, tmp_path):
        X, y = _make_Xy()
        model = LogisticRegressionModel(cv=2, max_iter=200).fit(X, y)
        save_path = tmp_path / "model_1"
        model.save(save_path)
        loaded = LogisticRegressionModel.load(save_path)
        assert model.score(X, y) == loaded.score(X, y)


# ---------------------------------------------------------------------------
# Pipeline integration: model file persistence
# ---------------------------------------------------------------------------


class TestPipelineModelPersistence:
    def test_model_file_created(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_make_df())
        assert _model_path(tmp_path, p.model_id).exists()

    def test_model_file_is_loadable(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_make_df())
        loaded = LogisticRegressionModel.load(_model_path(tmp_path, p.model_id))
        assert isinstance(loaded, LogisticRegressionModel)

    def test_loaded_model_can_predict(self, tmp_path):
        df = _make_df()
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(df)
        loaded = LogisticRegressionModel.load(_model_path(tmp_path, p.model_id))
        X = df.drop(columns=["target"])
        preds = loaded.predict(X)
        assert preds.shape == (len(X),)

    def test_each_run_saves_separate_model(self, tmp_path):
        cfg = _make_config(tmp_path)
        p1 = Pipeline(cfg).run(_make_df())
        p2 = Pipeline(cfg).run(_make_df())
        assert _model_path(tmp_path, p1.model_id).exists()
        assert _model_path(tmp_path, p2.model_id).exists()
        assert p1.model_id != p2.model_id


# ---------------------------------------------------------------------------
# Pipeline integration: models.csv logr-specific columns
# ---------------------------------------------------------------------------


class TestPipelineModelsCsvLogr:
    def _csv_rows(self, tmp_path, name="logr_run"):
        csv_path = Path(tmp_path) / name / "models" / "models.csv"
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def test_model_type_is_logr(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_make_df())
        rows = self._csv_rows(tmp_path)
        assert rows[0]["model_type"] == "logr"

    def test_test_score_is_recorded(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_make_df())
        rows = self._csv_rows(tmp_path)
        assert rows[0]["test_score"] != ""

    def test_test_score_is_numeric(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_make_df())
        rows = self._csv_rows(tmp_path)
        score = float(rows[0]["test_score"])
        assert 0.0 <= score <= 1.0

    def test_test_score_matches_pipeline_property(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_make_df())
        rows = self._csv_rows(tmp_path)
        assert float(rows[0]["test_score"]) == pytest.approx(round(p.test_score, 4))


# ---------------------------------------------------------------------------
# Pipeline integration: logr hyperparameter passthrough
# ---------------------------------------------------------------------------


class TestLogisticRegressionConfigPassthrough:
    def test_custom_max_iter_accepted(self, tmp_path):
        cfg = PipelineConfig(
            base=BaseConfig(name="custom", output_dir=str(tmp_path), seed=42),
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
            model=ModelConfig(type="logr", logr=LogisticRegressionConfig(max_iter=500)),
        )
        p = Pipeline(cfg).run(_make_df())
        assert p.test_score is not None

    def test_custom_cv_accepted(self, tmp_path):
        cfg = PipelineConfig(
            base=BaseConfig(name="custom_cv", output_dir=str(tmp_path), seed=42),
            data=DataConfig(target_column="target"),
            split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
            model=ModelConfig(type="logr", logr=LogisticRegressionConfig(cv=3)),
        )
        p = Pipeline(cfg).run(_make_df())
        assert p.test_score is not None
