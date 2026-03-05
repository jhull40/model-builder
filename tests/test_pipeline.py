"""Tests for Pipeline: timestamp, model ID auto-increment, models.csv, scaler persistence."""

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_builder.config.schema import (
    BaseConfig,
    DataConfig,
    PipelineConfig,
    TrainTestSplitConfig,
)
from model_builder.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, *, name="myrun"):
    return PipelineConfig(
        base=BaseConfig(name=name, output_dir=str(tmp_path), seed=42),
        data=DataConfig(target_column="target"),
        split=TrainTestSplitConfig(train_size=0.8, test_size=0.2),
    )


def _iris_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.exponential(2, n),
            "c": rng.uniform(0, 1, n),
            "target": rng.integers(0, 3, n),
        }
    )


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


class TestTimestamp:
    def test_timestamp_set_after_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        assert p.timestamp is not None

    def test_timestamp_format(self, tmp_path):
        from datetime import datetime

        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        # Should parse without error
        datetime.strptime(p.timestamp, "%Y-%m-%d %H:%M:%S")

    def test_timestamp_none_before_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert Pipeline(cfg).timestamp is None


# ---------------------------------------------------------------------------
# Model ID auto-increment
# ---------------------------------------------------------------------------


class TestModelId:
    def test_first_run_is_1(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        assert p.model_id == 1

    def test_second_run_increments(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_iris_df())
        p2 = Pipeline(cfg).run(_iris_df())
        assert p2.model_id == 2

    def test_multiple_runs_increment_sequentially(self, tmp_path):
        cfg = _make_config(tmp_path)
        ids = [Pipeline(cfg).run(_iris_df()).model_id for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_model_id_none_before_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert Pipeline(cfg).model_id is None


# ---------------------------------------------------------------------------
# models.csv
# ---------------------------------------------------------------------------


class TestModelsCsv:
    def _csv_rows(self, tmp_path, name="myrun"):
        csv_path = Path(tmp_path) / name / "models" / "models.csv"
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def test_csv_created_after_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_iris_df())
        csv_path = Path(tmp_path) / "myrun" / "models" / "models.csv"
        assert csv_path.exists()

    def test_csv_has_header(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path)
        assert set(rows[0].keys()) == {
            "model_id",
            "name",
            "model_type",
            "test_score",
            "timestamp",
        }

    def test_csv_name_matches_config(self, tmp_path):
        cfg = _make_config(tmp_path, name="experiment1")
        Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path, name="experiment1")
        assert rows[0]["name"] == "experiment1"

    def test_csv_model_id_recorded(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path)
        assert rows[0]["model_id"] == str(p.model_id)

    def test_csv_timestamp_recorded(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path)
        assert rows[0]["timestamp"] == p.timestamp

    def test_csv_appends_across_runs(self, tmp_path):
        cfg = _make_config(tmp_path)
        Pipeline(cfg).run(_iris_df())
        Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path)
        assert len(rows) == 2

    def test_csv_header_appears_once(self, tmp_path):
        cfg = _make_config(tmp_path)
        for _ in range(3):
            Pipeline(cfg).run(_iris_df())
        csv_path = Path(tmp_path) / "myrun" / "models" / "models.csv"
        lines = csv_path.read_text().splitlines()
        header_count = sum(1 for line in lines if line.startswith("model_id"))
        assert header_count == 1

    def test_csv_rows_match_run_count(self, tmp_path):
        cfg = _make_config(tmp_path)
        n_runs = 4
        for _ in range(n_runs):
            Pipeline(cfg).run(_iris_df())
        assert len(self._csv_rows(tmp_path)) == n_runs

    def test_csv_model_ids_are_sequential(self, tmp_path):
        cfg = _make_config(tmp_path)
        for _ in range(3):
            Pipeline(cfg).run(_iris_df())
        rows = self._csv_rows(tmp_path)
        ids = [int(r["model_id"]) for r in rows]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Scaler persistence
# ---------------------------------------------------------------------------


class TestScalerPersistence:
    def _scaler_path(self, tmp_path, model_id, name="myrun"):
        return Path(tmp_path) / name / "models" / str(model_id) / f"scaler_{model_id}"

    def test_scaler_file_exists_after_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        assert self._scaler_path(tmp_path, p.model_id).exists()

    def test_scaler_file_loadable(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        scalers = joblib.load(self._scaler_path(tmp_path, p.model_id))
        assert isinstance(scalers, dict)

    def test_scaler_keys_are_feature_columns(self, tmp_path):
        cfg = _make_config(tmp_path)
        p = Pipeline(cfg).run(_iris_df())
        scalers = joblib.load(self._scaler_path(tmp_path, p.model_id))
        # target is excluded; non-binary numeric columns should have scalers
        assert "target" not in scalers
        assert len(scalers) > 0

    def test_each_run_has_own_scaler_dir(self, tmp_path):
        cfg = _make_config(tmp_path)
        p1 = Pipeline(cfg).run(_iris_df())
        p2 = Pipeline(cfg).run(_iris_df())
        assert self._scaler_path(tmp_path, p1.model_id).exists()
        assert self._scaler_path(tmp_path, p2.model_id).exists()
        assert p1.model_id != p2.model_id

    def test_scaler_isolated_per_run(self, tmp_path):
        """Scalers from two runs fitted on different data should differ."""
        cfg = _make_config(tmp_path)
        df1 = _iris_df(seed=0)
        df2 = _iris_df(seed=99)
        p1 = Pipeline(cfg).run(df1)
        p2 = Pipeline(cfg).run(df2)
        s1 = joblib.load(self._scaler_path(tmp_path, p1.model_id))
        s2 = joblib.load(self._scaler_path(tmp_path, p2.model_id))
        # At least one scaler's mean/scale should differ
        col = next(iter(s1))
        if hasattr(s1[col], "mean_"):
            assert not np.allclose(s1[col].mean_, s2[col].mean_)
        else:
            assert not np.allclose(s1[col].data_min_, s2[col].data_min_)
