import os
from pydantic import BaseModel, model_validator


class BaseConfig(BaseModel):
    name: str
    seed: int = 524
    output_dir: str = "output"


class PipelineConfig(BaseModel):
    base: BaseConfig

    @model_validator(mode="after")
    def _setup(self) -> "PipelineConfig":
        output_path = os.path.join(self.base.output_dir, self.base.name)
        os.makedirs(output_path, exist_ok=True)
        return self
