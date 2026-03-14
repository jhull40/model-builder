"""Feature importance extraction and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_builder.config.schema import PipelineConfig
from model_builder.training.base import Model
from model_builder.training.models.logr import LogisticRegressionModel
from model_builder.training.models.xgbc import XGBClassifierModel


def _extract_importances(model: Model, feature_names: List[str]) -> pd.DataFrame:
    """Return a DataFrame with columns [feature, importance] sorted descending.

    For XGBClassifier: uses ``feature_importances_`` (gain-based).
    For LogisticRegression: uses the value of ``coef_[0]``.
    """
    if isinstance(model, XGBClassifierModel):
        importances = model._clf.feature_importances_
    elif isinstance(model, LogisticRegressionModel):
        importances = model._clf.coef_[0]
    else:
        raise TypeError(
            f"Feature importance is not supported for model type {type(model).__name__}. "
            "Supported types: XGBClassifierModel, LogisticRegressionModel."
        )

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


class FeatureImportanceAnalyzer:
    """Compute, save, and plot feature importances for a fitted model."""

    def __init__(self, config: PipelineConfig, model_id: int) -> None:
        self._config = config
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, model: Model, feature_names: List[str]
    ) -> "FeatureImportanceAnalyzer":
        """Extract importances and write CSV + bar-plot artifacts.

        Parameters
        ----------
        model:
            Fitted model (XGBClassifierModel or LogisticRegressionModel).
        feature_names:
            Ordered list of feature column names used during training.
        """
        df = _extract_importances(model, feature_names)
        out = self._fi_dir()
        out.mkdir(parents=True, exist_ok=True)
        self._write_csv(df, out)
        self._write_plot(df, out)
        return self

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _fi_dir(self) -> Path:
        return (
            Path(self._config.base.output_dir)
            / self._config.base.name
            / "feature_importance"
            / str(self._model_id)
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _write_csv(self, df: pd.DataFrame, out: Path) -> None:
        df.to_csv(out / "feature_importance.csv", index=False)

    def _write_plot(self, df: pd.DataFrame, out: Path) -> None:
        n = len(df)
        fig_height = max(4, n * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, n))  # type: ignore[attr-defined]
        ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(
            f"Feature Importance — {self._config.base.name} (model {self._model_id})"
        )
        ax.tick_params(axis="y", labelsize=max(6, min(10, 200 // n)))

        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
