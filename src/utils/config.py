"""Configuration management for the copula framework."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class Config:
    """Configuration manager for the copula framework."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            # Default to config/config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'bybit.api_key')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'bybit.api_key')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save config. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path

        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    @property
    def bybit_api_key(self) -> str:
        """Get Bybit API key."""
        return self.get("bybit.api_key", "")

    @property
    def bybit_api_secret(self) -> str:
        """Get Bybit API secret."""
        return self.get("bybit.api_secret", "")

    @property
    def bybit_testnet(self) -> bool:
        """Check if using testnet."""
        return self.get("bybit.testnet", False)

    @property
    def data_interval(self) -> str:
        """Get data interval."""
        return self.get("data.interval", "15")

    @property
    def default_days(self) -> int:
        """Get default number of days to fetch."""
        return self.get("data.default_days", 30)

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get("data.cache_enabled", True)

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        project_root = Path(__file__).parent.parent.parent
        cache_path = self.get("data.cache_dir", "data/cache")
        return project_root / cache_path

    @property
    def symbols(self) -> List[str]:
        """Get list of symbols to analyze."""
        return self.get("data.symbols", ["BTCUSDT", "ETHUSDT"])

    @property
    def pairs(self) -> List[List[str]]:
        """Get list of pairs to analyze."""
        return self.get("data.pairs", [["BTCUSDT", "ETHUSDT"]])

    @property
    def copula_types(self) -> List[str]:
        """Get list of copula types to fit."""
        return self.get("copulas.types", ["gaussian", "student_t"])

    @property
    def selection_criterion(self) -> str:
        """Get model selection criterion (aic or bic)."""
        return self.get("copulas.selection_criterion", "aic")

    @property
    def transformation_method(self) -> str:
        """Get transformation method."""
        return self.get("transformations.method", "empirical")

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.get("logging.level", "INFO")

    @property
    def log_file(self) -> Path:
        """Get log file path."""
        project_root = Path(__file__).parent.parent.parent
        log_path = self.get("logging.file", "logs/copula.log")
        return project_root / log_path

    @property
    def results_dir(self) -> Path:
        """Get results directory."""
        project_root = Path(__file__).parent.parent.parent
        results_path = self.get("output.results_dir", "results")
        return project_root / results_path


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """
    Set global configuration instance.

    Args:
        config: Config instance to use globally
    """
    global _config
    _config = config
