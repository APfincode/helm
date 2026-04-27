"""
Input Validation — Pydantic schemas and injection detection.

ALL external input passes through here before touching business logic:
- LLM JSON output
- Telegram commands
- Exchange API responses
- Config file contents

Usage:
    from src.security.input_validator import LLMResponseValidator, validate_llm_output
    from src.security.input_validator import PromptInjectionDetector
    
    # Validate LLM output
    signal = LLMResponseValidator.model_validate_json(raw_json)
    
    # Check for prompt injection
    detector = PromptInjectionDetector()
    if detector.is_suspicious(user_input):
        raise ValueError("Potential prompt injection detected")
"""

import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ValidationError(Exception):
    """Input failed validation."""
    pass


class PromptInjectionError(ValidationError):
    """Potential prompt injection detected."""
    pass


# =============================================================================
# LLM OUTPUT VALIDATION
# =============================================================================

class RiskParams(BaseModel):
    """Risk parameters from LLM signal."""
    stop_loss_pct: float = Field(..., ge=0.5, le=50.0)
    take_profit_pct: float = Field(..., ge=0.5, le=100.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @field_validator("take_profit_pct")
    @classmethod
    def tp_greater_than_sl(cls, v: float, info) -> float:
        """Ensure take profit is greater than stop loss."""
        data = info.data
        if "stop_loss_pct" in data and v <= data["stop_loss_pct"]:
            raise ValueError("take_profit_pct must be greater than stop_loss_pct")
        return v


class LLMResponseValidator(BaseModel):
    """
    Strict schema for LLM-generated trading signals.
    
    Any field that fails validation causes the entire signal to be rejected,
    defaulting to NEUTRAL.
    """
    signal: Literal["LONG", "SHORT", "NEUTRAL"] = Field(
        ..., description="Trading direction signal"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    reasoning: str = Field(
        ..., max_length=2000, description="Human-readable reasoning"
    )
    regime: Literal[
        "trending_up",
        "trending_down",
        "ranging",
        "volatile",
        "accumulation",
        "distribution",
        "unknown",
    ] = Field(..., description="Detected market regime")
    risk_params: RiskParams = Field(..., description="Risk management parameters")
    
    @field_validator("reasoning")
    @classmethod
    def no_control_chars(cls, v: str) -> str:
        """Strip control characters that could be used for injection."""
        # Allow normal whitespace (space, tab, newline) but block others
        cleaned = "".join(
            c for c in v if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) < 127)
        )
        if len(cleaned) != len(v):
            raise ValueError("Control characters not allowed in reasoning")
        return cleaned.strip()
    
    @field_validator("reasoning")
    @classmethod
    def no_code_markers(cls, v: str) -> str:
        """Detect and reject code execution markers."""
        dangerous_patterns = [
            r"```\s*(python|js|javascript|bash|sh|cmd|powershell)",
            r"<script",
            r"eval\s*\(",
            r"exec\s*\(",
            r"os\.system",
            r"subprocess\.run",
            r"subprocess\.call",
            r"subprocess\.Popen",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Code execution markers not allowed: {pattern}")
        return v


# =============================================================================
# TELEGRAM COMMAND VALIDATION
# =============================================================================

class TelegramCommandValidator(BaseModel):
    """Validates Telegram bot commands."""
    chat_id: int = Field(..., gt=0)
    command: str = Field(..., max_length=100)
    args: list[str] = Field(default_factory=list)
    
    @field_validator("command")
    @classmethod
    def valid_command_format(cls, v: str) -> str:
        """Commands must start with / and contain only alphanumeric and underscores."""
        if not v.startswith("/"):
            raise ValueError("Command must start with /")
        if not re.match(r"^/[a-zA-Z0-9_]+$", v):
            raise ValueError("Command contains invalid characters")
        return v.lower()
    
    @field_validator("args")
    @classmethod
    def no_injection_in_args(cls, v: list[str]) -> list[str]:
        """Sanitize command arguments."""
        sanitized = []
        for arg in v:
            # Remove shell metacharacters
            clean = re.sub(r"[;|&$`\\<>\"\']", "", arg)
            if clean != arg:
                raise ValueError(f"Argument contains forbidden characters: {arg}")
            sanitized.append(clean)
        return sanitized


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

class ConfigChangeValidator(BaseModel):
    """Validates configuration change requests."""
    key: str = Field(..., max_length=100)
    old_value: Optional[str] = Field(None, max_length=1000)
    new_value: str = Field(..., max_length=1000)
    actor: str = Field(..., max_length=100)
    
    @field_validator("key")
    @classmethod
    def no_nested_keys(cls, v: str) -> str:
        """Prevent path traversal in config keys."""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Config key cannot contain path separators")
        return v
    
    @field_validator("new_value")
    @classmethod
    def no_secrets_in_value(cls, v: str) -> str:
        """Detect attempts to set secrets via config changes."""
        secret_patterns = [
            r"0x[a-fA-F0-9]{64}",  # Private key
            r"sk-[a-zA-Z0-9]{20,}",  # API key
        ]
        for pattern in secret_patterns:
            if re.search(pattern, v):
                raise ValueError("Config value appears to contain a secret. Use .env file.")
        return v


# =============================================================================
# PROMPT INJECTION DETECTION
# =============================================================================

class PromptInjectionDetector:
    """
    Detects potential prompt injection attempts in text inputs.
    
    Based on known attack patterns:
    https://owasp.org/www-project-top-10-for-large-language-model-applications/
    """
    
    # Patterns that suggest prompt injection
    _SUSPICIOUS_PATTERNS = [
        # Direct instruction override
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(the\s+)?(above|prior|earlier)",
        r"disregard\s+(all\s+)?instructions",
        
        # System prompt extraction
        r"system\s+prompt",
        r"your\s+instructions\s+are",
        r"your\s+system\s+message",
        
        # Role play / persona override
        r"you\s+are\s+now\s+",
        r"act\s+as\s+(if\s+)?you\s+are",
        r"pretend\s+to\s+be",
        r"from\s+now\s+on\s+you\s+are",
        
        # Delimiter manipulation
        r"```\s*system",
        r"<\|system\|>",
        r"<\|im_start\|>\s*system",
        r"\[\s*SYSTEM\s*\]",
        
        # Obfuscation attempts
        r"i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s",
        r"\bignore\b.*\binstructions\b",
        
        # Jailbreak patterns
        r"DAN\s+mode",
        r"jailbreak",
        r"developer\s+mode",
    ]
    
    # Compiled regex patterns
    _compiled_patterns: list[re.Pattern]
    
    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._SUSPICIOUS_PATTERNS
        ]
    
    def is_suspicious(self, text: str) -> bool:
        """
        Check if text contains potential prompt injection patterns.
        
        Returns:
            True if suspicious patterns detected
        """
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def get_matches(self, text: str) -> list[str]:
        """
        Return all suspicious patterns found in text.
        
        Returns:
            List of matched pattern descriptions
        """
        matches = []
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                matches.append(self._SUSPICIOUS_PATTERNS[i])
        return matches
    
    def sanitize(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Sanitize text by replacing suspicious patterns.
        
        Note: This is a last resort. Prefer rejecting suspicious input entirely.
        """
        sanitized = text
        for pattern in self._compiled_patterns:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


# =============================================================================
# EXCHANGE RESPONSE VALIDATION
# =============================================================================

class ExchangeResponseValidator(BaseModel):
    """Validates exchange API responses before processing."""
    status: Literal["success", "error", "pending"]
    data: Optional[dict] = None
    error_message: Optional[str] = Field(None, max_length=1000)
    request_id: Optional[str] = Field(None, max_length=100)
    
    @field_validator("error_message")
    @classmethod
    def sanitize_error(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize error messages before logging."""
        if v is None:
            return None
        # Remove potential secrets from error messages
        sanitized = re.sub(r"0x[a-fA-F0-9]{64}", "[PRIVATE_KEY_REDACTED]", v)
        sanitized = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[API_KEY_REDACTED]", sanitized)
        return sanitized[:500]  # Truncate long errors


def validate_llm_output(raw_json: str) -> LLMResponseValidator:
    """
    Validate raw LLM JSON output.
    
    Args:
        raw_json: JSON string from LLM
        
    Returns:
        Validated LLMResponseValidator
        
    Raises:
        ValidationError: If JSON is invalid or schema fails
    """
    try:
        return LLMResponseValidator.model_validate_json(raw_json)
    except Exception as e:
        raise ValidationError(f"LLM output validation failed: {e}") from e
