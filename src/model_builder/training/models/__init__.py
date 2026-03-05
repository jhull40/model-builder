from __future__ import annotations

from typing import TYPE_CHECKING

from model_builder.training.base import Model
from model_builder.training.models.logr import LogisticRegressionModel

if TYPE_CHECKING:
    from model_builder.config.schema import ModelConfig


def build_model(config: ModelConfig, seed: int = 524) -> Model:
    """Instantiate a Model from config."""
    if config.type == "logr":
        c = config.logr
        return LogisticRegressionModel(
            Cs=c.Cs,
            cv=c.cv,
            solver=c.solver,
            max_iter=c.max_iter,
            l1_ratios=c.l1_ratios,
            n_jobs=c.n_jobs,
            seed=seed,
        )
    raise ValueError(f"Unknown model type: {config.type!r}")


__all__ = ["build_model", "LogisticRegressionModel"]
