"""Tests for Config class."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from expsetup import Config


def test_config_from_dict():
    """Test Config creation from dictionary."""
    config_dict = {"lr": 0.001, "batch_size": 32}
    config = Config(config_dict)

    assert config.lr == 0.001
    assert config.batch_size == 32
    assert config["lr"] == 0.001
    assert "lr" in config


def test_config_attribute_access():
    """Test attribute-style access to config values."""
    config = Config({"learning_rate": 0.001})
    assert config.learning_rate == 0.001

    config.learning_rate = 0.01
    assert config.learning_rate == 0.01


def test_config_dict_access():
    """Test dictionary-style access to config values."""
    config = Config({"param": "value"})
    assert config["param"] == "value"

    config["param"] = "new_value"
    assert config["param"] == "new_value"


def test_config_save_yaml():
    """Test saving config to YAML file."""
    config = Config({"lr": 0.001, "epochs": 10}, name="test_config")

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        config.save(yaml_path)

        assert yaml_path.exists()

        with open(yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded["config"]["lr"] == 0.001
        assert loaded["config"]["epochs"] == 10


def test_config_save_json():
    """Test saving config to JSON file."""
    config = Config({"lr": 0.001, "epochs": 10})

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "config.json"
        config.save(json_path)

        assert json_path.exists()

        with open(json_path, 'r') as f:
            loaded = json.load(f)

        assert loaded["config"]["lr"] == 0.001


def test_config_load_yaml():
    """Test loading config from YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"

        # Create a YAML file
        config_data = {"lr": 0.001, "batch_size": 64}
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load it
        config = Config(yaml_path)
        assert config.lr == 0.001
        assert config.batch_size == 64


def test_config_get_method():
    """Test get method with default values."""
    config = Config({"existing": "value"})

    assert config.get("existing") == "value"
    assert config.get("nonexistent", "default") == "default"
    assert config.get("nonexistent") is None


def test_config_update():
    """Test updating configuration."""
    config = Config({"lr": 0.001})
    config.update({"lr": 0.01, "batch_size": 32})

    assert config.lr == 0.01
    assert config.batch_size == 32
