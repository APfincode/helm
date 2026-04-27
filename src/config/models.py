"""
Configuration Models — Pydantic-validated config with fail-safe defaults.

Every toggleable feature has:
- enabled: bool — ON/OFF switch
- description: str — Hover tooltip / Telegram help text

Usage:
    from src.config.models import SecureConfig
    config = SecureConfig.from_env()
    
    if config.auto_trading_enabled.enabled:
        # Execute trade
        pass
"""

import os
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, SecretStr

from src.security.secrets_manager import SecretsManager, get_secrets_manager


class ToggleableFeature(BaseModel):
    """Base class for toggleable features with description."""
    enabled: bool = False
    description: str = ""


class CircuitBreakerConfig(ToggleableFeature):
    """Circuit breaker with threshold and cooldown."""
    enabled: bool = True  # Breakers ON by default for safety
    threshold: float = 0.0
    cooldown_minutes: int = 60


class LimitConfig(ToggleableFeature):
    """Configurable limit with value."""
    enabled: bool = True
    value: float = 0.0


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    description: str = ""
    requests: int = 100
    window_minutes: int = 1


class TradeFrequencyConfig(BaseModel):
    """Trade frequency limit."""
    description: str = "Minimum time between trades. Prevents over-trading and exchange spam."
    min_seconds: int = 30


class ManualApprovalConfig(ToggleableFeature):
    """Manual approval gate for live trades."""
    enabled: bool = True
    description: str = "Require manual /approve for each live trade. Safety net until bot is trusted."
    trades_until_auto: int = 50
    approval_window_seconds: int = 300


class SecureConfig(BaseModel):
    """
    Master configuration model with fail-safe defaults.
    
    ALL secrets loaded from environment variables.
    """

    # Environment
    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment. Controls available features.",
    )

    debug: ToggleableFeature = ToggleableFeature(
        enabled=False,
        description="Enable verbose logging and debug endpoints. Increases disk usage.",
    )

    auto_trading: ToggleableFeature = ToggleableFeature(
        enabled=False,
        description="Enable automatic trade execution. When OFF, bot generates signals only.",
    )

    # Exchange credentials (loaded from env)
    hyperliquid_wallet: str = Field(
        default="",
        min_length=42,
        description="Hyperliquid wallet address",
    )
    hyperliquid_private_key: SecretStr = Field(
        default=SecretStr(""),
        description="Hyperliquid private key. NEVER log or serialize this.",
    )

    # Security settings
    telegram_whitelist: list[int] = Field(
        default_factory=list,
        description="Only these Telegram chat IDs can control the bot.",
    )

    max_leverage: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum allowed leverage per position. Higher = higher liquidation risk.",
    )

    # Circuit Breakers
    daily_loss_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Stops trading if daily loss exceeds threshold. Protects against cascading losses during bad days.",
        threshold=5.0,
        cooldown_minutes=1440,
    )
    weekly_loss_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Stops trading if weekly loss exceeds threshold. Prevents a bad week from bleeding into the next.",
        threshold=10.0,
        cooldown_minutes=10080,
    )
    drawdown_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Stops trading if drawdown from peak capital exceeds threshold. Prevents ruin. Requires manual /resume.",
        threshold=15.0,
        cooldown_minutes=-1,
    )
    volatility_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Pauses new entries during extreme volatility (ATR > 5x average). Avoids getting chopped in whipsaws.",
        threshold=5.0,
        cooldown_minutes=240,
    )
    exchange_error_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Pauses trading after consecutive exchange API errors. Prevents order spam during outages.",
        threshold=5.0,
        cooldown_minutes=30,
    )
    llm_error_breaker: CircuitBreakerConfig = CircuitBreakerConfig(
        enabled=True,
        description="Switches to rule-based fallback after consecutive LLM failures. Prevents executing bad AI signals.",
        threshold=3.0,
        cooldown_minutes=60,
    )

    # Position Sizing Limits
    max_concurrent_positions: LimitConfig = LimitConfig(
        enabled=True,
        description="Maximum number of open positions at once. Prevents over-concentration.",
        value=3,
    )
    max_exposure_per_asset: LimitConfig = LimitConfig(
        enabled=True,
        description="Maximum capital allocated to a single asset. Reduces single-asset risk.",
        value=0.25,
    )
    max_total_exposure: LimitConfig = LimitConfig(
        enabled=True,
        description="Maximum total capital deployed across all positions. Prevents over-leverage.",
        value=0.50,
    )
    max_leverage_limit: LimitConfig = LimitConfig(
        enabled=True,
        description="Maximum allowed leverage per position. Higher leverage = higher liquidation risk.",
        value=3,
    )

    # Manual Approval
    manual_approval: ManualApprovalConfig = ManualApprovalConfig()

    # Rate Limiting
    rate_limits: dict[str, RateLimitConfig] = Field(
        default_factory=lambda: {
            "exchange_api": RateLimitConfig(
                description="Max exchange API calls per window. Prevents hitting Hyperliquid's 1200/10min limit.",
                requests=1200,
                window_minutes=10,
            ),
            "llm_api": RateLimitConfig(
                description="Max LLM requests per minute. Controls API costs.",
                requests=60,
                window_minutes=1,
            ),
            "telegram_commands": RateLimitConfig(
                description="Max Telegram commands per minute per user. Prevents spam.",
                requests=10,
                window_minutes=1,
            ),
        },
    )
    trade_frequency: TradeFrequencyConfig = TradeFrequencyConfig()

    @field_validator("hyperliquid_private_key")
    @classmethod
    def validate_key_format(cls, v: SecretStr) -> SecretStr:
        """Ensure private key format is valid."""
        raw = v.get_secret_value()
        if raw and not raw.startswith("0x"):
            raise ValueError("Private key must start with 0x")
        return v

    @classmethod
    def from_env(cls) -> "SecureConfig":
        """
        Load configuration from environment variables.
        
        Returns:
            SecureConfig instance with values from environment
        """
        secrets = get_secrets_manager()

        # Parse Telegram whitelist
        whitelist_str = secrets.get("TELEGRAM_WHITELIST", required=False)
        whitelist = []
        if whitelist_str:
            whitelist = [int(x.strip()) for x in whitelist_str.split(",") if x.strip()]

        return cls(
            env=secrets.get("ENV") or "development",
            debug=ToggleableFeature(
                enabled=(secrets.get("DEBUG", required=False) or "").lower() == "true",
                description="Enable verbose logging and debug endpoints.",
            ),
            hyperliquid_wallet=secrets.get("HYPERLIQUID_WALLET") or "",
            hyperliquid_private_key=SecretStr(
                secrets.get("HYPERLIQUID_PRIVATE_KEY") or ""
            ),
            telegram_whitelist=whitelist,
        )

    def to_safe_dict(self) -> dict:
        """
        Export config as dict with secrets masked.
        
        Safe for logging or displaying to users.
        """
        data = self.model_dump()
        # Mask sensitive fields
        if "hyperliquid_private_key" in data:
            data["hyperliquid_private_key"] = "***REDACTED***"
        return data
