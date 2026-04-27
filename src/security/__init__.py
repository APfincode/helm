"""
Security module — Centralized security controls for Hyper-Alpha-Arena V2.

Modules:
    auth: HMAC token generation/validation, Telegram whitelist
    secrets_manager: Centralized env-only secret access with masking
    input_validator: Pydantic schemas, prompt injection detection
    audit_logger: Append-only signed audit trail
    rate_limiter: Token bucket per operation
"""

from .auth import AuthManager, TokenScope, AuthError
from .secrets_manager import SecretsManager, SecretAccessError
from .input_validator import (
    LLMResponseValidator,
    TelegramCommandValidator,
    ConfigChangeValidator,
    ExchangeResponseValidator,
    PromptInjectionDetector,
    validate_llm_output,
)
from .audit_logger import AuditLogger, AuditEntry, EventType, Severity
from .rate_limiter import RateLimiter, RateLimit, RateLimitError

__all__ = [
    "AuthManager",
    "TokenScope",
    "AuthError",
    "SecretsManager",
    "SecretAccessError",
    "LLMResponseValidator",
    "TelegramCommandValidator",
    "ConfigChangeValidator",
    "ExchangeResponseValidator",
    "PromptInjectionDetector",
    "validate_llm_output",
    "AuditLogger",
    "AuditEntry",
    "EventType",
    "Severity",
    "RateLimiter",
    "RateLimit",
    "RateLimitError",
]
