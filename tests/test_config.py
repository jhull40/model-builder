import os
import textwrap

import pytest
from pydantic import ValidationError

from model_builder.config.schema import BaseConfig, PipelineConfig
from model_builder.utils.utils import load_config


# ---------------------------------------------------------------------------
# BaseConfig
# ---------------------------------------------------------------------------


class TestBaseConfig:
    def test_required_field_name(self):
        with pytest.raises(ValidationError):
            BaseConfig()  # type: ignore[call-arg]

    def test_default_seed(self):
        cfg = BaseConfig(name="run1")
        assert cfg.seed == 524

    def test_default_output_dir(self):
        cfg = BaseConfig(name="run1")
        assert cfg.output_dir == "output"

    def test_custom_values(self):
        cfg = BaseConfig(name="exp", seed=42, output_dir="results")
        assert cfg.name == "exp"
        assert cfg.seed == 42
        assert cfg.output_dir == "results"

    def test_name_must_be_string(self):
        # pydantic will coerce an int to str, but a dict should fail
        with pytest.raises(ValidationError):
            BaseConfig(name={"bad": "value"})


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_requires_base(self):
        with pytest.raises(ValidationError):
            PipelineConfig()  # type: ignore[call-arg]

    def test_valid_creation(self, tmp_path):
        cfg = PipelineConfig(base={"name": "test", "output_dir": str(tmp_path)})
        assert isinstance(cfg.base, BaseConfig)
        assert cfg.base.name == "test"

    def test_setup_creates_output_directory(self, tmp_path):
        run_name = "my_run"
        PipelineConfig(base={"name": run_name, "output_dir": str(tmp_path)})
        expected = tmp_path / run_name
        assert expected.is_dir(), f"Expected directory {expected} to be created"

    def test_setup_output_dir_path_components(self, tmp_path):
        cfg = PipelineConfig(base={"name": "nested", "output_dir": str(tmp_path)})
        assert os.path.join(str(tmp_path), "nested") == os.path.join(
            cfg.base.output_dir, cfg.base.name
        )

    def test_nested_base_config_defaults(self, tmp_path):
        cfg = PipelineConfig(
            base={"name": "defaults_check", "output_dir": str(tmp_path)}
        )
        assert cfg.base.seed == 524

    def test_base_accepts_base_config_instance(self, tmp_path):
        base = BaseConfig(name="direct", output_dir=str(tmp_path))
        cfg = PipelineConfig(base=base)
        assert cfg.base.name == "direct"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        yaml_content = textwrap.dedent(
            f"""\
            base:
              name: loaded
              output_dir: {tmp_path}
              seed: 7
        """
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert isinstance(cfg, PipelineConfig)
        assert cfg.base.name == "loaded"
        assert cfg.base.seed == 7

    def test_loads_default_config_yaml(self, tmp_path):
        """The committed configs/config.yaml should parse successfully."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(repo_root, "configs", "config.yaml")
        # Redirect output_dir to tmp_path to avoid side effects
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)
        data["base"]["output_dir"] = str(tmp_path)

        # Write a patched copy and load it
        patched = tmp_path / "patched_config.yaml"
        patched.write_text(yaml.dump(data))

        cfg = load_config(str(patched))
        assert isinstance(cfg, PipelineConfig)
        assert cfg.base.name == "test"
        assert cfg.base.seed == 524

    def test_missing_required_field_raises(self, tmp_path):
        """A YAML missing the required 'name' field should raise ValidationError."""
        yaml_content = textwrap.dedent(
            f"""\
            base:
              output_dir: {tmp_path}
        """
        )
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValidationError):
            load_config(str(config_file))

    def test_missing_base_key_raises(self, tmp_path):
        """A YAML with no 'base' key should raise ValidationError."""
        yaml_content = "name: standalone\n"
        config_file = tmp_path / "no_base.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValidationError):
            load_config(str(config_file))

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
