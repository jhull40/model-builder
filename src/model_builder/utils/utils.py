import yaml
from model_builder.config.schema import PipelineConfig


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return PipelineConfig(**config_dict)
