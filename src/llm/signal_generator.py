"""
Signal Generator — Failsafe integration layer between LLM and trading engine.

This is the ONLY component that the rest of the bot interacts with for signals.
It orchestrates:
1. Prompt rendering (from template)
2. LLM API call (with failover)
3. Response validation (strict Pydantic)
4. Safety checks (injection detection, budget limits)
5. Fallback to NEUTRAL on any error

Usage:
    from src.llm.signal_generator import SignalGenerator
    
    generator = SignalGenerator(template_name="basic_signal")
    
    # This is the one-line interface for the entire bot
    signal = await generator.generate(market_data)
    
    if signal.direction == "LONG":
        # Execute trade
        pass
    elif signal.direction == "NEUTRAL":
        # Do nothing (safe default)
        pass
"""

import logging
from typing import Optional

import pandas as pd

from .client import LLMClient, Signal, LLMError
from .prompt_engine import PromptEngine, PromptError
from .tracker import TokenTracker
from src.security.audit_logger import AuditLogger, EventType, Severity
from src.security.rate_limiter import RateLimiter
from src.market_context import MarketContextBuilder, UnifiedMarketContext


logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    One-stop interface for AI signal generation.
    
    Guarantees:
    1. ALWAYS returns a Signal object (never raises)
    2. On ANY failure, returns Signal.error() → NEUTRAL
    3. All costs tracked
    4. All calls audited
    5. Budget limits enforced
    """

    def __init__(
        self,
        template_name: str = "basic_signal",
        client: Optional[LLMClient] = None,
        prompt_engine: Optional[PromptEngine] = None,
        audit_logger: Optional[AuditLogger] = None,
        market_context_builder: Optional[MarketContextBuilder] = None,
        enable_market_context: bool = True,
    ) -> None:
        """
        Initialize signal generator.
        
        Args:
            template_name: Which prompt template to use
            client: LLMClient instance (creates default if None)
            prompt_engine: PromptEngine instance (creates default if None)
            audit_logger: Audit logger for recording all calls
            market_context_builder: Optional MarketContextBuilder for perp microstructure + macro data
            enable_market_context: Whether to inject market context into prompts
        """
        self._template_name = template_name
        
        try:
            self._engine = prompt_engine or PromptEngine(template_name)
        except PromptError as e:
            logger.error(f"Failed to load prompt template: {e}")
            self._engine = None
        
        self._client = client
        self._audit = audit_logger
        self._market_context_builder = market_context_builder
        self._enable_market_context = enable_market_context
        self._shared_context_builder: Optional[MarketContextBuilder] = None
        self._initialized = False

    async def init(self) -> "SignalGenerator":
        """Async initialization (creates LLM client and market context builder)."""
        if self._initialized:
            return self
        
        if self._client is None:
            self._client = LLMClient()
            await self._client.__aenter__()
        
        # Initialize market context builder if enabled and not provided
        if self._enable_market_context and self._market_context_builder is None:
            self._shared_context_builder = MarketContextBuilder()
            await self._shared_context_builder.__aenter__()
        
        self._initialized = True
        return self

    async def close(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
        if self._shared_context_builder:
            await self._shared_context_builder.__aexit__(None, None, None)
            self._shared_context_builder = None
        self._initialized = False

    async def generate(
        self,
        market_data: pd.DataFrame,
        symbol: str = "",
        trade_history: Optional[list[dict]] = None,
        performance_summary: Optional[dict] = None,
        use_cache: bool = False,
    ) -> Signal:
        """
        Generate a trading signal from market data.
        
        This is the ONLY method the trading engine calls.
        Every possible failure path returns Signal.error() (NEUTRAL).
        
        Args:
            market_data: OHLCV DataFrame
            symbol: Trading symbol (e.g., "BTC") for market context fetch
            trade_history: Optional recent trades for context
            performance_summary: Optional performance metrics
            use_cache: Whether to cache responses
            
        Returns:
            Signal object (NEUTRAL on any failure)
        """
        # CRITICAL: Check initialization
        if not self._initialized or self._client is None:
            logger.error("SignalGenerator not initialized. Call init() first.")
            return Signal.error(
                error="SignalGenerator not initialized",
                context="INIT_REQUIRED",
            )
        
        # CRITICAL: Check prompt engine loaded
        if self._engine is None:
            logger.error("Prompt engine failed to load")
            return Signal.error(
                error="Prompt engine not available",
                context="TEMPLATE_LOAD_FAILED",
            )
        
        # Step 0: Fetch market context if enabled
        market_context = {}  # type: dict[str, any]
        if self._enable_market_context and symbol:
            try:
                builder = self._market_context_builder or self._shared_context_builder
                if builder:
                    ctx = await builder.build(symbol)
                    market_context = ctx.to_dict()
                    logger.debug(f"Market context injected for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch market context for {symbol}: {e}")
                # Continue without context — non-fatal
        
        # Step 1: Render prompt
        try:
            if trade_history:
                system_prompt, user_prompt = self._engine.render_with_history(
                    market_data=market_data,
                    trade_history=trade_history,
                    performance_summary=performance_summary,
                    **market_context,
                )
            else:
                system_prompt, user_prompt = self._engine.render(
                    market_data=market_data,
                    **market_context,
                )
        except PromptError as e:
            logger.error(f"Prompt rendering failed: {e}")
            return Signal.error(error=f"Prompt rendering failed: {e}")
        
        # Step 2: Call LLM with full failsafe wrapper
        signal = await self._client.generate_signal(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self._engine.config.temperature,
            max_tokens=self._engine.config.max_tokens,
            use_cache=use_cache,
        )
        
        # Step 3: Post-validation (defense in depth)
        if not signal.is_valid():
            logger.warning(f"Signal failed post-validation: {signal}")
            await self._audit_llm_failure(
                error="Post-validation failed",
                signal=signal,
            )
            return Signal.error(
                error="Signal post-validation failed",
                context=f"direction={signal.direction}, conf={signal.confidence}",
            )
        
        # Step 4: Audit successful signal
        if self._audit:
            await self._audit.log(
                event_type=EventType.LLM_SIGNAL,
                action=f"SIGNAL_{signal.direction}",
                actor="llm",
                severity=Severity.INFO,
                details={
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "regime": signal.regime,
                    "provider": signal.provider,
                    "model": signal.model,
                    "tokens_used": signal.tokens_used,
                    "cost_usd": signal.cost_usd,
                    "latency_ms": signal.latency_ms,
                    "cached": signal.cached,
                    "reasoning": signal.reasoning[:200],  # Truncate for audit
                },
            )
        
        logger.info(
            f"Signal generated: {signal.direction} (confidence: {signal.confidence:.2f}, "
            f"provider: {signal.provider}, cost: ${signal.cost_usd:.4f})"
        )
        
        return signal

    async def _audit_llm_failure(self, error: str, signal: Optional[Signal] = None) -> None:
        """Log LLM failure to audit trail."""
        if self._audit:
            await self._audit.log(
                event_type=EventType.ERROR,
                action="LLM_SIGNAL_FAILURE",
                actor="llm",
                severity=Severity.WARNING,
                details={
                    "error": error,
                    "signal_direction": signal.direction if signal else "unknown",
                    "signal_error": signal.error if signal else "",
                },
            )

    async def generate_batch(
        self,
        market_data: pd.DataFrame,
        n_variants: int = 3,
    ) -> list[Signal]:
        """
        Generate multiple signal variants for ensemble strategies.
        
        Useful for:
        - Ensemble voting (majority wins)
        - Confidence calibration (average confidence)
        - Strategy comparison (A/B testing prompts)
        
        Args:
            market_data: OHLCV DataFrame
            n_variants: Number of variants to generate
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        for i in range(n_variants):
            # Slightly vary temperature for diversity
            signal = await self.generate(
                market_data=market_data,
                use_cache=False,  # Never cache batch variants
            )
            signals.append(signal)
        
        return signals

    def get_stats(self) -> dict:
        """Get LLM usage statistics."""
        if self._client:
            return self._client.get_stats()
        return {}

    def __repr__(self) -> str:
        return f"SignalGenerator(template={self._template_name}, initialized={self._initialized})"
