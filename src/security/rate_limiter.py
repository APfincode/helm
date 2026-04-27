"""
Rate Limiter — Token bucket per operation.

Prevents:
- Exchange API spam (respects Hyperliquid 1200/10min limit)
- LLM API cost overruns
- Telegram command spam
- Trade execution spam

Usage:
    from src.security.rate_limiter import RateLimiter, RateLimit
    
    limiter = RateLimiter()
    
    # Check if operation is allowed
    if await limiter.is_allowed("exchange_api", key="user_123"):
        # Execute API call
        pass
    else:
        # Rate limited
        pass
"""

import time
import asyncio
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict


class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass


@dataclass
class RateLimit:
    """Rate limit configuration."""
    name: str
    description: str
    max_requests: int
    window_seconds: int


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    Thread-safe using asyncio locks.
    """

    def __init__(self, max_tokens: int, refill_rate: float) -> None:
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate  # tokens per second
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get seconds to wait before enough tokens are available.
        
        Returns:
            0.0 if tokens are available now, otherwise wait time
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate
            )
            self._last_refill = now

            if self._tokens >= tokens:
                return 0.0

            needed = tokens - self._tokens
            return needed / self._refill_rate


class RateLimiter:
    """
    Multi-bucket rate limiter for different operation types.
    
    Default limits:
    - exchange_api: 1200 per 10 min (Hyperliquid limit)
    - llm_api: 60 per min
    - telegram_commands: 10 per min per chat
    - trade_execution: 1 per 30 sec
    """

    DEFAULT_LIMITS: dict[str, RateLimit] = {
        "exchange_api": RateLimit(
            name="exchange_api",
            description="Max exchange API calls per window. Prevents hitting Hyperliquid's 1200/10min limit.",
            max_requests=1200,
            window_seconds=600,
        ),
        "llm_api": RateLimit(
            name="llm_api",
            description="Max LLM requests per minute. Controls API costs.",
            max_requests=60,
            window_seconds=60,
        ),
        "telegram_commands": RateLimit(
            name="telegram_commands",
            description="Max Telegram commands per minute per user. Prevents spam.",
            max_requests=10,
            window_seconds=60,
        ),
        "trade_execution": RateLimit(
            name="trade_execution",
            description="Minimum time between trades. Prevents over-trading and exchange spam.",
            max_requests=1,
            window_seconds=30,
        ),
    }

    def __init__(self, limits: Optional[dict[str, RateLimit]] = None) -> None:
        self._limits = limits or self.DEFAULT_LIMITS.copy()
        # buckets[limit_name][key] = TokenBucket
        self._buckets: dict[str, dict[str, TokenBucket]] = defaultdict(dict)

    def _get_bucket(self, limit_name: str, key: str) -> TokenBucket:
        """Get or create a token bucket for a limit/key pair."""
        if limit_name not in self._buckets:
            self._buckets[limit_name] = {}

        if key not in self._buckets[limit_name]:
            limit = self._limits[limit_name]
            refill_rate = limit.max_requests / limit.window_seconds
            self._buckets[limit_name][key] = TokenBucket(
                max_tokens=limit.max_requests,
                refill_rate=refill_rate,
            )

        return self._buckets[limit_name][key]

    async def is_allowed(
        self,
        limit_name: str,
        key: str = "default",
        tokens: int = 1,
    ) -> bool:
        """
        Check if an operation is allowed under rate limit.
        
        Args:
            limit_name: Name of the rate limit rule
            key: Identifier for this bucket (e.g., chat_id, user_id)
            tokens: Number of tokens to consume
            
        Returns:
            True if operation is allowed
            
        Raises:
            RateLimitError: If limit_name is not configured
        """
        if limit_name not in self._limits:
            raise RateLimitError(f"Unknown rate limit: {limit_name}")

        bucket = self._get_bucket(limit_name, key)
        return await bucket.consume(tokens)

    async def check_or_raise(
        self,
        limit_name: str,
        key: str = "default",
        tokens: int = 1,
    ) -> None:
        """
        Check rate limit and raise if exceeded.
        
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        if not await self.is_allowed(limit_name, key, tokens):
            limit = self._limits[limit_name]
            bucket = self._get_bucket(limit_name, key)
            wait_time = await bucket.get_wait_time(tokens)
            raise RateLimitError(
                f"Rate limit exceeded for '{limit_name}'. "
                f"Limit: {limit.max_requests} per {limit.window_seconds}s. "
                f"Retry after {wait_time:.1f}s."
            )

    async def get_wait_time(
        self,
        limit_name: str,
        key: str = "default",
        tokens: int = 1,
    ) -> float:
        """
        Get seconds to wait before operation is allowed.
        
        Returns:
            0.0 if allowed now, otherwise wait time in seconds
        """
        if limit_name not in self._limits:
            return 0.0

        bucket = self._get_bucket(limit_name, key)
        return await bucket.get_wait_time(tokens)

    def get_limit_info(self, limit_name: str) -> Optional[RateLimit]:
        """Get rate limit configuration by name."""
        return self._limits.get(limit_name)

    def add_limit(self, limit: RateLimit) -> None:
        """Add a custom rate limit."""
        self._limits[limit.name] = limit
