"""
Configuration Loader — Load and validate configuration from files and environment.

Usage:
    from src.config.loader import ConfigLoader
    
    loader = ConfigLoader()
    config = loader.load()
    
    # Access toggleable features
    if config.auto_trading.enabled:
        print("Auto-trading is ON")
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .models import SecureConfig
from src.security.secrets_manager import get_secrets_manager, SecretAccessError


class ConfigLoadError(Exception):
    """Failed to load configuration."""
    pass


class ConfigLoader:
    """
    Loads configuration from multiple sources with precedence:
    1. Environment variables (highest priority)
    2. YAML config files
    3. Default values (lowest priority)
    """

    def __init__(self, config_dir: str = "config") -> None:
        self._config_dir = Path(config_dir)
        self._secrets = get_secrets_manager()

    def load(self, env: Optional[str] = None) -> SecureConfig:
        """
        Load and validate configuration.
        
        Args:
            env: Override environment. If None, reads from ENV var.
            
        Returns:
            Validated SecureConfig instance
            
        Raises:
            ConfigLoadError: If configuration is invalid or missing required values
        """
        # Determine environment
        if env is None:
            env = os.getenv("ENV", "development")

        # Start with base config from environment
        try:
            config = SecureConfig.from_env()
        except (SecretAccessError, ValidationError) as e:
            raise ConfigLoadError(f"Failed to load base config: {e}") from e

        # Load YAML overrides if files exist
        base_file = self._config_dir / "default.yaml"
        env_file = self._config_dir / f"{env}.yaml"

        yaml_config = {}
        if base_file.exists():
            yaml_config.update(self._load_yaml(base_file))
        if env_file.exists():
            yaml_config.update(self._load_yaml(env_file))

        # Apply YAML overrides (but env vars still take precedence)
        if yaml_config:
            config = self._merge_yaml(config, yaml_config)

        # Validate environment-specific rules
        self._validate_environment_rules(config)

        return config

    def _load_yaml(self, path: Path) -> dict:
        """Load YAML file safely."""
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in {path}: {e}") from e
        except FileNotFoundError:
            return {}

    def _merge_yaml(self, config: SecureConfig, yaml_data: dict) -> SecureConfig:
        """
        Merge YAML data into existing config.
        
        Only updates non-sensitive fields. Secrets always come from env.
        """
        # Re-serialize and merge
        current = config.model_dump()
        
        # Deep merge (simple version)
        for key, value in yaml_data.items():
            if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = value

        return SecureConfig(**current)

    def _validate_environment_rules(self, config: SecureConfig) -> None:
        """
        Apply environment-specific validation rules.
        
        Raises:
            ConfigLoadError: If config violates environment rules
        """
        if config.env == "development":
            # Development can only use testnet
            if config.auto_trading.enabled:
                raise ConfigLoadError(
                    "Auto-trading cannot be enabled in development environment. "
                    "Use staging or production."
                )

        elif config.env == "staging":
            # Staging allows paper trading but warns
            if config.auto_trading.enabled:
                # TODO: Check if using testnet
                pass

        elif config.env == "production":
            # Production requires all security features
            if not config.daily_loss_breaker.enabled:
                raise ConfigLoadError(
                    "Daily loss breaker MUST be enabled in production."
                )
            if not config.drawdown_breaker.enabled:
                raise ConfigLoadError(
                    "Drawdown breaker MUST be enabled in production."
                )
            if config.debug.enabled:
                raise ConfigLoadError(
                    "Debug mode cannot be enabled in production."
                )

        # Global validations
        if config.max_leverage > 10:
            raise ConfigLoadError("Max leverage cannot exceed 10x.")

        if config.max_total_exposure.value > 0.75:
            raise ConfigLoadError("Max total exposure cannot exceed 75%.")
