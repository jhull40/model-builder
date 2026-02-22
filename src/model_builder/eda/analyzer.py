from model_builder.config.schema import PipelineConfig


class DataAnalyzer:
    def __init__(self, config: PipelineConfig):
        self.config = config
