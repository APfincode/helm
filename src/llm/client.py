"""
LLM Client — Multi-provider async LLM client with failover and safety.

Design goals:
1. NEVER let a raw LLM output touch trading logic
2. On ANY error (API, validation, rate limit), return NEUTRAL
3. Track every token and dollar spent
4. Automatic provider failover
5. Response caching to reduce costs

Usage:
    client = LLMClient()
    
    # Simple: text prompt → validated Signal object
    signal = await client.generate_signal(
        system_prompt="You are a crypto analyst...",
        user_prompt="Here is the market data: ...",
        temperature=0.2,
    )
    
    # Result is ALWAYS a Signal — even on failure it's NEUTRAL
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Any, AsyncIterator
from enum import Enum

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
)
import logging

from src.security.secrets_manager import SecretsManager, get_secrets_manager
from src.security.input_validator import (
    LLMResponseValidator,
    PromptInjectionDetector,
    ValidationError,
    validate_llm_output,
)
from src.security.rate_limiter import RateLimiter, RateLimitError
from src.llm.tracker import TokenTracker, TokenUsage


logger = logging.getLogger(__name__)


# =============================================================================
# Error Types
# =============================================================================

class LLMError(Exception):
    """Base LLM error — all failures are caught and converted to NEUTRAL."""
    pass


class LLMResponseError(LLMError):
    """LLM response is malformed or dangerous."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limited by LLM provider."""
    pass


class LLMBudgetExceeded(LLMError):
    """Monthly/weekly budget exceeded."""
    pass


class LLMPromptInjectionError(LLMError):
    """Prompt injection detected before sending."""
    pass


class ProviderError(LLMError):
    """Provider-specific error."""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


# =============================================================================
# Models
# =============================================================================

class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class Signal:
    """
    Trading signal output from LLM.
    
    This is the ONLY thing that the trading engine sees.
    Raw LLM output never leaves this module.
    """
    direction: str = "NEUTRAL"  # LONG, SHORT, or NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""
    regime: str = "unknown"
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    
    # Metadata (not used for trading, only for logging)
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    error: str = ""  # If non-empty, signal was generated as fallback
    
    @classmethod
    def neutral(cls, reason: str = "No signal generated") -> "Signal":
        """Create a safe NEUTRAL signal."""
        return cls(
            direction="NEUTRAL",
            confidence=0.0,
            reasoning=reason,
            error="",
        )
    
    @classmethod
    def error(cls, error: str, context: str = "") -> "Signal":
        """
        Create a NEUTRAL signal with error info.
        
        This is the CRITICAL failsafe: every error becomes NEUTRAL.
        """
        return cls(
            direction="NEUTRAL",
            confidence=0.0,
            reasoning=f"LLM error prevented signal generation: {error}",
            error=f"{error} | context: {context}",
        )
    
    def is_neutral(self) -> bool:
        return self.direction.upper() == "NEUTRAL"
    
    def is_valid(self) -> bool:
        """Extra validation above Pydantic schema."""
        if self.direction not in ("LONG", "SHORT", "NEUTRAL"):
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if not (0.1 <= self.stop_loss_pct <= 50.0):
            return False
        if not (0.1 <= self.take_profit_pct <= 100.0):
            return False
        # Stop must be less than take profit for non-neutral signals
        if self.direction != "NEUTRAL" and self.stop_loss_pct >= self.take_profit_pct:
            return False
        return True


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """
    Multi-provider async LLM client with maximum failsafes.
    
    Every method that could fail is wrapped to return Signal.error()
    instead of raising exceptions to the trading engine.
    """

    # Provider priority (failover order)
    DEFAULT_PROVIDERS = [Provider.OPENROUTER, Provider.OPENAI, Provider.ANTHROPIC]
    
    # Cost per 1K tokens (approximate)
    COST_PER_1K_TOKENS = {
        "openrouter": {
            "input": 0.0005,
            "output": 0.0015,
        },
        "openai": {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        },
    }

    def __init__(
        self,
        primary_provider: Optional[Provider] = None,
        fallback_providers: Optional[list[Provider]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        token_tracker: Optional[TokenTracker] = None,
        max_monthly_cost: float = 100.0,
        request_timeout: float = 60.0,
    ) -> None:
        """
        Initialize LLM client.
        
        Args:
            primary_provider: Primary LLM provider
            fallback_providers: Ordered list of fallback providers
            rate_limiter: Rate limiter for LLM requests
            token_tracker: Token usage tracker
            max_monthly_cost: Maximum monthly budget in USD
            request_timeout: HTTP request timeout in seconds
        """
        self._secrets = get_secrets_manager()
        
        self._provider = primary_provider or Provider.OPENROUTER
        self._fallbacks = fallback_providers or self.DEFAULT_PROVIDERS.copy()
        self._fallbacks = [p for p in self._fallbacks if p != self._provider]
        
        self._rate_limiter = rate_limiter or RateLimiter()
        self._tracker = token_tracker or TokenTracker()
        self._max_monthly_cost = max_monthly_cost
        self._timeout = request_timeout
        
        self._injection_detector = PromptInjectionDetector()
        
        # Response cache (simple in-memory LRU)
        self._response_cache: dict[str, tuple[Signal, float]] = {}
        self._cache_ttl_seconds = 30.0  # Very short for signals
        self._cache_max_size = 100
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ========================================================================
    # Public API
    # ========================================================================

    async def generate_signal(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
        model: Optional[str] = None,
        use_cache: bool = False,
    ) -> Signal:
        """
        Generate a trading signal from LLM.
        
        This is the ONLY entry point for signal generation.
        All errors are caught and converted to Signal.error() (NEUTRAL).
        
        Args:
            system_prompt: System instructions to the LLM
            user_prompt: User message with market data
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response tokens
            model: Specific model to use
            use_cache: Whether to use response caching
            
        Returns:
            Signal object (NEUTRAL on any failure)
        """
        # CRITICAL: Check budget BEFORE any API call
        if self._tracker.get_monthly_cost() >= self._max_monthly_cost:
            logger.warning("Monthly budget exceeded — returning NEUTRAL")
            return Signal.error(
                error="Monthly budget exceeded",
                context=f"Limit: ${self._max_monthly_cost}",
            )
        
        # CRITICAL: Scan for prompt injection
        injection_check = self._check_prompt_safety(system_prompt, user_prompt)
        if injection_check:
            logger.warning(f"Prompt injection detected: {injection_check}")
            return Signal.error(
                error=f"Prompt injection detected: {injection_check}",
                context="INJECTION_BLOCKED",
            )
        
        # Check cache
        if use_cache:
            cached = self._get_cached(system_prompt, user_prompt)
            if cached:
                cached.cached = True
                return cached
        
        # Build provider chain (primary + fallbacks)
        providers = [self._provider] + self._fallbacks
        
        last_error = ""
        
        for provider in providers:
            try:
                # Check rate limit
                try:
                    await self._rate_limiter.check_or_raise("llm_api", key=provider.value)
                except RateLimitError:
                    logger.warning(f"Rate limit hit for {provider}, trying next")
                    continue
                
                # Call provider
                start_time = time.monotonic()
                raw_response = await self._call_provider(
                    provider=provider,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                )
                latency_ms = (time.monotonic() - start_time) * 1000
                
                # Parse response
                parsed = self._parse_llm_response(raw_response)
                parsed.provider = provider.value
                parsed.latency_ms = latency_ms
                
                # Track usage
                if raw_response.get("usage"):
                    self._tracker.record(
                        provider=provider.value,
                        model=raw_response.get("model", "unknown"),
                        input_tokens=raw_response["usage"].get("prompt_tokens", 0),
                        output_tokens=raw_response["usage"].get("completion_tokens", 0),
                        cost_usd=self._estimate_cost(raw_response, provider),
                        latency_ms=latency_ms,
                    )
                
                # Validate signal
                if not parsed.is_valid():
                    logger.warning(f"Signal validation failed: {parsed}")
                    return Signal.error(
                        error="Signal validation failed",
                        context=f"direction={parsed.direction}, confidence={parsed.confidence}",
                    )
                
                # Cache result
                if use_cache:
                    self._set_cache(system_prompt, user_prompt, parsed)
                
                return parsed
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider} failed: {e}")
                continue
        
        # All providers failed
        return Signal.error(
            error=f"All providers failed. Last error: {last_error}",
            context=f"Tried: {', '.join(p.value for p in providers)}",
        )

    # ========================================================================
    # Safety Checks
    # ========================================================================

    def _check_prompt_safety(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Check prompts for injection attempts.
        
        Returns:
            Error message if injection detected, None if safe
        """
        combined = system_prompt + " " + user_prompt
        
        if self._injection_detector.is_suspicious(combined):
            matches = self._injection_detector.get_matches(combined)
            return f"Detected patterns: {', '.join(matches[:3])}"
        
        return None

    # ========================================================================
    # Provider Calls
    # ========================================================================

    async def _call_provider(
        self,
        provider: Provider,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
    ) -> dict:
        """Call a specific LLM provider."""
        if provider == Provider.OPENROUTER:
            return await self._call_openrouter(
                system_prompt, user_prompt, temperature, max_tokens, model
            )
        elif provider == Provider.OPENAI:
            return await self._call_openai(
                system_prompt, user_prompt, temperature, max_tokens, model
            )
        elif provider == Provider.ANTHROPIC:
            return await self._call_anthropic(
                system_prompt, user_prompt, temperature, max_tokens, model
            )
        else:
            raise ProviderError(f"Unknown provider: {provider}", provider=provider.value)

    async def _call_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
    ) -> dict:
        """Call OpenRouter API."""
        api_key = self._secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY not set", provider="openrouter")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/APfincode/helm",
            "X-Title": "Helm",
        }
        
        payload = {
            "model": model or "meta-llama/llama-3.1-70b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        
        if self._client is None:
            raise LLMError("HTTP client not initialized")
        
        response = await self._client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        
        response.raise_for_status()
        return response.json()

    async def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
    ) -> dict:
        """Call OpenAI API."""
        api_key = self._secrets.get("OPENAI_API_KEY", required=False)
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set", provider="openai")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        
        if self._client is None:
            raise LLMError("HTTP client not initialized")
        
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        
        response.raise_for_status()
        return response.json()

    async def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
    ) -> dict:
        """Call Anthropic API."""
        api_key = self._secrets.get("ANTHROPIC_API_KEY", required=False)
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set", provider="anthropic")
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": model or "claude-3-sonnet-20240229",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        
        if self._client is None:
            raise LLMError("HTTP client not initialized")
        
        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        
        response.raise_for_status()
        return response.json()

    # ========================================================================
    # Response Parsing
    # ========================================================================

    def _parse_llm_response(self, raw_response: dict) -> Signal:
        """
        Parse raw API response into validated Signal.
        
        Handles different provider response formats.
        """
        content = ""
        
        # Extract content based on provider format
        if "choices" in raw_response:
            # OpenAI / OpenRouter format
            message = raw_response["choices"][0].get("message", {})
            content = message.get("content", "")
        elif "content" in raw_response:
            # Anthropic format
            content_blocks = raw_response["content"]
            if isinstance(content_blocks, list) and content_blocks:
                content = content_blocks[0].get("text", "")
        
        if not content:
            return Signal.error("Empty LLM response", context="NO_CONTENT")
        
        # Try to parse JSON
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    return Signal.error(
                        f"Invalid JSON in response: {e}",
                        context=f"content_preview={content[:200]}",
                    )
            else:
                return Signal.error(
                    f"Invalid JSON in response: {e}",
                    context=f"content_preview={content[:200]}",
                )
        
        # Validate against Pydantic schema
        try:
            validated = validate_llm_output(json.dumps(parsed_json))
            
            return Signal(
                direction=validated.signal,
                confidence=validated.confidence,
                reasoning=validated.reasoning,
                regime=validated.regime,
                stop_loss_pct=validated.risk_params.stop_loss_pct,
                take_profit_pct=validated.risk_params.take_profit_pct,
                model=raw_response.get("model", "unknown"),
                tokens_used=raw_response.get("usage", {}).get("total_tokens", 0),
            )
        except ValidationError as e:
            return Signal.error(
                f"Schema validation failed: {e}",
                context=f"json={json.dumps(parsed_json)[:500]}",
            )

    # ========================================================================
    # Cost Estimation
    # ========================================================================

    def _estimate_cost(self, response: dict, provider: Provider) -> float:
        """Estimate API cost from response."""
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        model = response.get("model", "")
        
        provider_costs = self.COST_PER_1K_TOKENS.get(provider.value, {})
        
        if provider == Provider.OPENROUTER:
            input_cost = (input_tokens / 1000) * provider_costs.get("input", 0.0005)
            output_cost = (output_tokens / 1000) * provider_costs.get("output", 0.0015)
        else:
            model_costs = provider_costs.get(model, {"input": 0.005, "output": 0.015})
            input_cost = (input_tokens / 1000) * model_costs["input"]
            output_cost = (output_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost

    # ========================================================================
    # Cache
    # ========================================================================

    def _get_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        import hashlib
        combined = system_prompt + "|||" + user_prompt
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached(self, system_prompt: str, user_prompt: str) -> Optional[Signal]:
        """Get cached response if valid."""
        key = self._get_cache_key(system_prompt, user_prompt)
        if key not in self._response_cache:
            return None
        
        signal, cached_at = self._response_cache[key]
        if time.time() - cached_at > self._cache_ttl_seconds:
            del self._response_cache[key]
            return None
        
        return signal

    def _set_cache(self, system_prompt: str, user_prompt: str, signal: Signal) -> None:
        """Cache a response."""
        # Enforce max size (simple LRU eviction)
        if len(self._response_cache) >= self._cache_max_size:
            oldest_key = min(
                self._response_cache,
                key=lambda k: self._response_cache[k][1]
            )
            del self._response_cache[oldest_key]
        
        key = self._get_cache_key(system_prompt, user_prompt)
        self._response_cache[key] = (signal, time.time())

    # ========================================================================
    # Utility
    # ========================================================================

    def get_stats(self) -> dict:
        """Get client usage statistics."""
        return {
            "total_requests": self._tracker.total_requests,
            "total_tokens": self._tracker.total_tokens,
            "total_cost_usd": self._tracker.total_cost_usd,
            "monthly_cost_usd": self._tracker.get_monthly_cost(),
            "average_latency_ms": self._tracker.average_latency_ms,
            "error_rate": self._tracker.error_rate,
            "cache_size": len(self._response_cache),
        }
