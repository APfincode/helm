"""
Secrets Manager — Centralized, secure secret access.

ALL secrets are loaded from environment variables ONLY.
This module ensures secrets never appear in logs, error messages, or serialized output.

Usage:
    from src.security.secrets_manager import SecretsManager
    secrets = SecretsManager()
    api_key = secrets.get("HYPERLIQUID_API_KEY")
"""

import os
import re
from typing import Optional


class SecretAccessError(Exception):
    """Raised when a secret cannot be accessed or is missing."""
    pass


class SecretsManager:
    """
    Centralized secrets access with masking for logs.
    
    Design principles:
    1. Secrets ONLY from environment variables
    2. No caching of raw secret values
    3. Automatic masking in string representations
    4. Explicit failure on missing secrets (no defaults for sensitive values)
    """

    # Patterns that indicate a secret value
    _SENSITIVE_KEYS = {
        "KEY", "SECRET", "TOKEN", "PASSWORD", "PRIVATE_KEY",
        "WALLET", "API_KEY", "AUTH", "CREDENTIAL",
    }

    # Regex to detect private key formats
    _PRIVATE_KEY_PATTERN = re.compile(r"^(0x)?[a-fA-F0-9]{64}$")
    _API_KEY_PATTERN = re.compile(r"^(sk-[a-zA-Z0-9]+|AKIA[0-9A-Z]{16})$")

    def __init__(self) -> None:
        """Initialize secrets manager. Verifies critical secrets exist."""
        self._verified: set[str] = set()

    def get(self, key: str, required: bool = True) -> Optional[str]:
        """
        Retrieve a secret from environment variables.
        
        Args:
            key: Environment variable name
            required: If True, raises SecretAccessError when missing
            
        Returns:
            The secret value, or None if not required and missing
            
        Raises:
            SecretAccessError: If required=True and secret is missing
        """
        value = os.getenv(key)
        
        if value is None or value.strip() == "":
            if required:
                raise SecretAccessError(
                    f"Required secret '{key}' is missing from environment. "
                    f"Check your .env file."
                )
            return None
            
        # Verify the secret doesn't look like a placeholder
        if self._is_placeholder(value):
            if required:
                raise SecretAccessError(
                    f"Secret '{key}' appears to be a placeholder value: "
                    f"'{self.mask(value)}'. Update your .env file."
                )
            return None
            
        self._verified.add(key)
        return value

    def get_masked(self, key: str, required: bool = True) -> Optional[str]:
        """
        Retrieve a secret but always return masked version.
        Useful for logging or displaying config status.
        """
        raw = self.get(key, required=required)
        if raw is None:
            return None
        return self.mask(raw)

    @classmethod
    def mask(cls, value: str, visible_prefix: int = 4, visible_suffix: int = 4) -> str:
        """
        Mask a secret value for safe display in logs.
        
        Args:
            value: The secret to mask
            visible_prefix: Number of characters to show at start
            visible_suffix: Number of characters to show at end
            
        Returns:
            Masked string like "sk-...abcd"
        """
        if len(value) <= visible_prefix + visible_suffix + 3:
            return "***"
            
        prefix = value[:visible_prefix]
        suffix = value[-visible_suffix:]
        return f"{prefix}...{suffix}"

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a key name indicates it holds sensitive data."""
        upper_key = key.upper()
        return any(sensitive in upper_key for sensitive in self._SENSITIVE_KEYS)

    @classmethod
    def _is_placeholder(cls, value: str) -> bool:
        """Detect if a value is a placeholder/example."""
        placeholders = [
            "your_", "example_", "placeholder", "changeme",
            "insert_", "put_your", "xxx", "123456789",
            "abcdef", "sample", "demo", "test",
        ]
        lower_val = value.lower()
        return any(ph in lower_val for ph in placeholders)

    def verify_all_required(self, required_keys: list[str]) -> None:
        """
        Verify all required secrets are present and valid.
        
        Args:
            required_keys: List of environment variable names
            
        Raises:
            SecretAccessError: If any required secret is missing
        """
        missing = []
        for key in required_keys:
            try:
                self.get(key, required=True)
            except SecretAccessError:
                missing.append(key)
                
        if missing:
            raise SecretAccessError(
                f"Missing required secrets: {', '.join(missing)}. "
                f"Please check your .env file."
            )


# Singleton instance for application-wide use
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the singleton SecretsManager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
