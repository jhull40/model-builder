from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Model(ABC):
    """Framework-agnostic model interface."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probability for the positive class, shape (n,)."""
        ...

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model": ...
