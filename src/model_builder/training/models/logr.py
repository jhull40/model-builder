from pathlib import Path
from typing import List, Literal, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

from model_builder.training.base import Model


class LogisticRegressionModel(Model):
    def __init__(
        self,
        Cs: Union[int, List[float]] = 10,
        cv: int = 5,
        solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "lbfgs",
        max_iter: int = 100,
        l1_ratios: List[float] = [0.0],
        seed: int = 524,
    ) -> None:
        self._clf = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            solver=solver,
            max_iter=max_iter,
            l1_ratios=l1_ratios,
            n_jobs=-1,
            random_state=seed,
            use_legacy_attributes=False,  # type: ignore[call-arg]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
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
    def load(cls, path: Path) -> "LogisticRegressionModel":
        instance = cls.__new__(cls)
        instance._clf = joblib.load(Path(path))
        return instance
