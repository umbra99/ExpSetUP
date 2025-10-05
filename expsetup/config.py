"""
Configuration management for experiments.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


class Config:
    """
    Configuration manager for experiment settings.

    Supports loading from dict, JSON, or YAML files, and provides
    easy access to configuration parameters.

    Args:
        config: Dictionary, path to JSON/YAML file, or None
        name: Optional name for this configuration

    Example:
        >>> config = Config({"lr": 0.001, "batch_size": 32})
        >>> config.lr
        0.001
        >>> config.save("config.yaml")
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        name: Optional[str] = None
    ):
        self._config: Dict[str, Any] = {}
        self._name = name
        self._created_at = datetime.now().isoformat()

        if config is not None:
            if isinstance(config, (str, Path)):
                self.load(config)
            elif isinstance(config, dict):
                self._config = config.copy()
            else:
                raise ValueError(f"Invalid config type: {type(config)}")

    def load(self, path: Union[str, Path]) -> None:
        """Load configuration from JSON or YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                self._config = yaml.safe_load(f)
            elif path.suffix == '.json':
                self._config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Output file path
            format: 'json' or 'yaml'. If None, inferred from file extension
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = 'yaml' if path.suffix in ['.yaml', '.yml'] else 'json'

        with open(path, 'w') as f:
            if format == 'yaml':
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with metadata."""
        return {
            'name': self._name,
            'created_at': self._created_at,
            'config': self._config
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._config.get(key, default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-style setting of config values."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._config[name] = value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to config values."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style setting of config values."""
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config

    def __repr__(self) -> str:
        return f"Config(name={self._name}, items={len(self._config)})"

    def __str__(self) -> str:
        return json.dumps(self._config, indent=2)
