import os
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, field_validator, model_validator


def _nullify_none_string(v: object) -> object:
    """Convert YAML literal 'none' string to Python None."""
    if isinstance(v, str) and v.lower() == "none":
        return None
    return v


class BaseConfig(BaseModel):
    name: str
    seed: int = 524
    output_dir: str = "output"


class LogisticRegressionConfig(BaseModel):
    Cs: Union[int, List[float]] = 10
    cv: int = 5
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    max_iter: int = 100
    l1_ratios: List[float] = [0.0]


class XGBClassifierConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0


class ModelConfig(BaseModel):
    type: Literal["logr", "xgbc"] = "logr"
    logr: LogisticRegressionConfig = LogisticRegressionConfig()
    xgbc: XGBClassifierConfig = XGBClassifierConfig()


class DataConfig(BaseModel):
    target_column: Optional[str] = None
    date_column: Optional[str] = None
    impute_strategy: Literal["mean", "median", "drop"] = "median"

    @field_validator("target_column", "date_column", mode="before")
    @classmethod
    def _nullify(cls, v: object) -> object:
        return _nullify_none_string(v)


class TrainTestSplitConfig(BaseModel):
    train_size: float = 0.8
    test_size: float = 0.2
    val_size: float = 0.0
    shuffle: bool = True
    stratified_split: bool = True
    split_column: Optional[str] = None
    start_train_date: Optional[str] = None
    stop_train_date: Optional[str] = None
    start_test_date: Optional[str] = None
    stop_test_date: Optional[str] = None
    start_val_date: Optional[str] = None
    stop_val_date: Optional[str] = None

    @field_validator(
        "split_column",
        "start_train_date",
        "stop_train_date",
        "start_test_date",
        "stop_test_date",
        "start_val_date",
        "stop_val_date",
        mode="before",
    )
    @classmethod
    def _nullify(cls, v: object) -> object:
        return _nullify_none_string(v)


class PipelineConfig(BaseModel):
    base: BaseConfig
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    split: TrainTestSplitConfig = TrainTestSplitConfig()

    @model_validator(mode="after")
    def _setup(self) -> "PipelineConfig":
        output_path = os.path.join(self.base.output_dir, self.base.name)
        os.makedirs(output_path, exist_ok=True)
        return self
