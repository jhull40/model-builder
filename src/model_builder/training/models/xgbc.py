from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from model_builder.training.base import Model


class XGBClassifierModel(Model):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        seed: int = 524,
    ) -> None:
        self._clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBClassifierModel":
        self._clf.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return float(self._clf.score(X, y))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: Path) -> "XGBClassifierModel":
        instance = cls.__new__(cls)
        instance._clf = joblib.load(Path(path))
        return instance
