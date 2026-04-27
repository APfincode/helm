"""
Market Context Builder

Aggregates perp microstructure + macro context into a single structured object
that gets injected into the LLM prompt.

This is the single interface the SignalGenerator calls to get context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .perp_microstructure import PerpMicrostructureFetcher, PerpMicrostructure
from .macro_context import MacroContextFetcher, MacroContext

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMarketContext:
    """
    Complete market context for LLM prompt injection.

    Combines exchange-specific microstructure with macro sentiment.
    """

    symbol: str
    timestamp: datetime

    # Perp microstructure (HL-specific)
    funding_rate_8h: str = ""
    funding_annualized: str = ""
    funding_percentile: str = ""
    open_interest: str = ""
    oi_change_24h: str = ""
    perp_premium: str = ""
    day_return: str = ""
    volume_24h: str = ""
    predominant_bias: str = ""
    extreme_funding: bool = False

    # Macro context
    fear_greed: str = "Unavailable"
    fear_greed_trend: str = ""
    breaking_news: str = ""

    def to_prompt_section(self) -> str:
        """
        Render as a markdown section for LLM prompt.

        This is injected into the user prompt as context.
        """
        lines = [
            "## Hyperliquid Perp Market Context",
            "",
            f"Funding Rate (8h): {self.funding_rate_8h}",
            f"Funding Annualized: {self.funding_annualized}",
            f"Funding Percentile (24h): {self.funding_percentile}",
            f"Open Interest: {self.open_interest}",
            f"OI Change (24h): {self.oi_change_24h}",
            f"Perp Premium vs Spot: {self.perp_premium}",
            f"24h Return: {self.day_return}",
            f"24h Volume: {self.volume_24h}",
            f"Predominant Bias: {self.predominant_bias}",
            "",
            "## Macro Context",
            "",
            f"Fear & Greed Index: {self.fear_greed}",
            f"F&G Trend: {self.fear_greed_trend}",
        ]

        if self.breaking_news:
            lines.extend([
                "",
                "## Breaking News",
                self.breaking_news,
            ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to plain dict for prompt template variables."""
        return {
            "market_context_section": self.to_prompt_section(),
            "funding_rate_8h": self.funding_rate_8h,
            "funding_annualized": self.funding_annualized,
            "funding_percentile": self.funding_percentile,
            "open_interest": self.open_interest,
            "oi_change_24h": self.oi_change_24h,
            "perp_premium": self.perp_premium,
            "day_return": self.day_return,
            "volume_24h": self.volume_24h,
            "predominant_bias": self.predominant_bias,
            "extreme_funding": self.extreme_funding,
            "fear_greed": self.fear_greed,
            "fear_greed_trend": self.fear_greed_trend,
            "breaking_news": self.breaking_news,
        }


class MarketContextBuilder:
    """
    Orchestrates fetching and building unified market context.

    Usage:
        async with MarketContextBuilder() as builder:
            context = await builder.build("BTC")
            # context.to_prompt_section() -> inject into LLM prompt
    """

    def __init__(
        self,
        perp_fetcher: Optional[PerpMicrostructureFetcher] = None,
        macro_fetcher: Optional[MacroContextFetcher] = None,
    ) -> None:
        self._perp_fetcher = perp_fetcher
        self._macro_fetcher = macro_fetcher
        self._shared_perp_fetcher: Optional[PerpMicrostructureFetcher] = None
        self._shared_macro_fetcher: Optional[MacroContextFetcher] = None
        self._last_perp_data: dict[str, PerpMicrostructure] = {}
        self._last_macro_data: Optional[MacroContext] = None
        self._last_fetch_time: Optional[datetime] = None

    async def __aenter__(self) -> MarketContextBuilder:
        if self._perp_fetcher is None:
            self._shared_perp_fetcher = PerpMicrostructureFetcher()
            await self._shared_perp_fetcher.__aenter__()
        if self._macro_fetcher is None:
            self._shared_macro_fetcher = MacroContextFetcher()
            await self._shared_macro_fetcher.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        if self._shared_perp_fetcher:
            await self._shared_perp_fetcher.__aexit__(*args)
        if self._shared_macro_fetcher:
            await self._shared_macro_fetcher.__aexit__(*args)

    async def build(self, symbol: str) -> UnifiedMarketContext:
        """
        Build unified context for a single symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC")

        Returns:
            UnifiedMarketContext ready for prompt injection.
        """
        # Fetch perp microstructure
        perp_data: Optional[PerpMicrostructure] = None
        try:
            fetcher = self._perp_fetcher or self._shared_perp_fetcher
            if fetcher:
                perp_data = await fetcher.fetch_single(symbol)
                if perp_data:
                    self._last_perp_data[symbol] = perp_data
        except Exception as e:
            logger.warning(f"Failed to fetch perp microstructure for {symbol}: {e}")
            # Use cached data if available
            perp_data = self._last_perp_data.get(symbol)

        # Fetch macro context (shared across all symbols)
        macro_data: Optional[MacroContext] = None
        try:
            macro_fetcher = self._macro_fetcher or self._shared_macro_fetcher
            if macro_fetcher:
                macro_data = await macro_fetcher.fetch(max_news=3)
                self._last_macro_data = macro_data
        except Exception as e:
            logger.warning(f"Failed to fetch macro context: {e}")
            macro_data = self._last_macro_data

        # Build unified context
        return self._assemble(symbol, perp_data, macro_data)

    async def build_batch(self, symbols: list[str]) -> dict[str, UnifiedMarketContext]:
        """
        Build context for multiple symbols efficiently.

        Fetches perp data in one batch call. Macro context is shared.
        """
        # Batch fetch perp data
        perp_results: dict[str, PerpMicrostructure] = {}
        try:
            fetcher = self._perp_fetcher or self._shared_perp_fetcher
            if fetcher:
                perp_results = await fetcher.fetch_all(symbols)
                self._last_perp_data.update(perp_results)
        except Exception as e:
            logger.warning(f"Failed to batch fetch perp data: {e}")
            perp_results = {s: self._last_perp_data.get(s) for s in symbols}

        # Fetch macro context once
        macro_data: Optional[MacroContext] = None
        try:
            macro_fetcher = self._macro_fetcher or self._shared_macro_fetcher
            if macro_fetcher:
                macro_data = await macro_fetcher.fetch(max_news=3)
                self._last_macro_data = macro_data
        except Exception as e:
            logger.warning(f"Failed to fetch macro context: {e}")
            macro_data = self._last_macro_data

        # Assemble for each symbol
        results = {}
        for symbol in symbols:
            perp = perp_results.get(symbol) or self._last_perp_data.get(symbol)
            results[symbol] = self._assemble(symbol, perp, macro_data)

        return results

    def _assemble(
        self,
        symbol: str,
        perp: Optional[PerpMicrostructure],
        macro: Optional[MacroContext],
    ) -> UnifiedMarketContext:
        """Assemble unified context from components."""
        ctx = UnifiedMarketContext(symbol=symbol, timestamp=datetime.utcnow())

        if perp:
            pctx = perp.to_prompt_context()
            ctx.funding_rate_8h = pctx.get("funding_rate_8h", "")
            ctx.funding_annualized = pctx.get("funding_annualized", "")
            ctx.funding_percentile = pctx.get("funding_percentile", "")
            ctx.open_interest = pctx.get("open_interest", "")
            ctx.oi_change_24h = pctx.get("oi_change_24h", "")
            ctx.perp_premium = pctx.get("perp_premium", "")
            ctx.day_return = pctx.get("day_return", "")
            ctx.volume_24h = pctx.get("volume_24h", "")
            ctx.predominant_bias = pctx.get("predominant_bias", "")
            ctx.extreme_funding = pctx.get("extreme_funding", False)

        if macro:
            mctx = macro.to_prompt_context()
            ctx.fear_greed = mctx.get("fear_greed", "Unavailable")
            ctx.fear_greed_trend = mctx.get("fear_greed_trend", "")
            ctx.breaking_news = mctx.get("breaking_news", "")

        return ctx
