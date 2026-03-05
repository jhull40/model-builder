import os
from typing import Literal, Optional
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
    data: DataConfig = DataConfig()
    split: TrainTestSplitConfig = TrainTestSplitConfig()

    @model_validator(mode="after")
    def _setup(self) -> "PipelineConfig":
        output_path = os.path.join(self.base.output_dir, self.base.name)
        os.makedirs(output_path, exist_ok=True)
        return self
