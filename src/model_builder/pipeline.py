import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from model_builder.config.schema import PipelineConfig
from model_builder.preprocessing.preprocessor import Preprocessor

_MODELS_CSV_COLUMNS = ["model_id", "name", "timestamp"]


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.preprocessor = Preprocessor(config)
        self._model_id: Optional[int] = None
        self._timestamp: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> "Pipeline":
        self._timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._model_id = self._next_model_id()

        train_df, test_df, val_df = self.preprocessor.split(df)
        self.preprocessor.fit(train_df)
        train_df = self.preprocessor.transform(train_df)
        test_df = self.preprocessor.transform(test_df)
        if val_df is not None:
            val_df = self.preprocessor.transform(val_df)

        self._write_models_csv()
        self._save_scaler()

        return self

    @property
    def model_id(self) -> Optional[int]:
        return self._model_id

    @property
    def timestamp(self) -> Optional[str]:
        return self._timestamp

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _models_dir(self) -> Path:
        return Path(self.config.base.output_dir) / self.config.base.name / "models"

    def _model_dir(self) -> Path:
        assert self._model_id is not None
        return self._models_dir() / str(self._model_id)

    def _models_csv_path(self) -> Path:
        return self._models_dir() / "models.csv"

    # ------------------------------------------------------------------
    # Model ID (auto-increment)
    # ------------------------------------------------------------------

    def _next_model_id(self) -> int:
        csv_path = self._models_csv_path()
        if not csv_path.exists():
            return 1
        with open(csv_path, newline="") as f:
            ids = [int(row["model_id"]) for row in csv.DictReader(f)]
        return max(ids) + 1 if ids else 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_models_csv(self) -> None:
        assert self._model_id is not None and self._timestamp is not None
        csv_path = self._models_csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_MODELS_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "model_id": self._model_id,
                    "name": self.config.base.name,
                    "timestamp": self._timestamp,
                }
            )

    def _save_scaler(self) -> None:
        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor._scalers, model_dir / f"scaler_{self._model_id}")
