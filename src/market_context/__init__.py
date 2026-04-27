"""
Market Context Module

Provides perp-specific market microstructure + macro context for LLM prompts.

Usage:
    from src.market_context import MarketContextBuilder, UnifiedMarketContext
    
    async with MarketContextBuilder() as builder:
        context = await builder.build("BTC")
        prompt_section = context.to_prompt_section()
"""

from .perp_microstructure import PerpMicrostructureFetcher, PerpMicrostructure
from .macro_context import MacroContextFetcher, MacroContext, FearGreedReading, NewsItem
from .context_builder import MarketContextBuilder, UnifiedMarketContext

__all__ = [
    "PerpMicrostructureFetcher",
    "PerpMicrostructure",
    "MacroContextFetcher",
    "MacroContext",
    "FearGreedReading",
    "NewsItem",
    "MarketContextBuilder",
    "UnifiedMarketContext",
]
