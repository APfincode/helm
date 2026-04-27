"""
Perp Market Microstructure Fetcher

Fetches Hyperliquid-specific data that actually moves perp markets:
- Funding rates (positioning bias)
- Open interest (new money entering / positions closing)
- Premium (perp vs spot premium/discount)
- Order book imbalance (bid/ask ratio)

All data comes from Hyperliquid's free public API. No API key required.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Hyperliquid API endpoints
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_EXPLORER_URL = "https://api.hyperliquid.xyz/explorer"


@dataclass
class PerpMicrostructure:
    """Snapshot of perp market microstructure for a single asset."""

    symbol: str
    timestamp: datetime

    # Funding
    funding_rate: float = 0.0  # Current 8h funding rate
    funding_annualized: float = 0.0  # Approx annualized: rate * 3 * 365
    funding_percentile_24h: float = 0.0  # Where current rate sits in 24h range (0-1)

    # Open Interest
    open_interest_usd: float = 0.0
    oi_change_24h: float = 0.0  # Percentage change

    # Premium
    premium_pct: float = 0.0  # Perp premium vs oracle price

    # Price
    mark_price: float = 0.0
    oracle_price: float = 0.0
    mid_price: float = 0.0
    prev_day_price: float = 0.0
    day_return: float = 0.0

    # Volume
    day_volume_usd: float = 0.0

    # Order book (if fetched)
    bid_ask_ratio: Optional[float] = None  # Bid volume / Ask volume at impact size
    spread_pct: Optional[float] = None
    impact_bid: Optional[float] = None
    impact_ask: Optional[float] = None

    # Computed bias signal
    def __post_init__(self):
        """Auto-compute funding_annualized if not provided."""
        if self.funding_annualized == 0.0 and self.funding_rate != 0.0:
            self.funding_annualized = self.funding_rate * 3 * 365 * 100

    @property
    def predominant_bias(self) -> str:
        """
        Classify market positioning based on funding + OI + premium.

        Returns:
            'long-heavy' | 'short-heavy' | 'neutral'
        """
        score = 0

        # Funding: positive = longs pay shorts = long-heavy
        if self.funding_rate > 0.0001:  # > 0.01% per 8h
            score += 2
        elif self.funding_rate < -0.0001:
            score -= 2

        # Premium: positive = perp above spot = long-heavy
        if self.premium_pct > 0.05:  # > 0.05%
            score += 1
        elif self.premium_pct < -0.05:
            score -= 1

        # OI increasing while price up = longs entering
        if self.oi_change_24h > 5 and self.day_return > 0:
            score += 1
        elif self.oi_change_24h > 5 and self.day_return < 0:
            score -= 1

        if score >= 2:
            return "long-heavy"
        if score <= -2:
            return "short-heavy"
        return "neutral"

    @property
    def is_extreme_funding(self) -> bool:
        """Whether funding is at extreme levels (potential mean reversion)."""
        annual = abs(self.funding_annualized)
        return annual > 30.0  # > 30% annualized is extreme

    def to_prompt_context(self) -> dict:
        """Format for LLM prompt injection."""
        bias = self.predominant_bias
        bias_note = ""
        if bias == "long-heavy":
            bias_note = " (crowded longs — potential short setup)"
        elif bias == "short-heavy":
            bias_note = " (crowded shorts — potential long setup)"

        extreme_note = ""
        if self.is_extreme_funding:
            direction = "positive" if self.funding_rate > 0 else "negative"
            extreme_note = f" EXTREME {direction} funding — mean reversion likely"

        return {
            "funding_rate_8h": f"{self.funding_rate * 100:.4f}%",
            "funding_annualized": f"{self.funding_annualized:.1f}%",
            "funding_percentile": f"{self.funding_percentile_24h * 100:.0f}%",
            "open_interest": f"${self.open_interest_usd / 1e6:.2f}M",
            "oi_change_24h": f"{self.oi_change_24h:+.1f}%",
            "perp_premium": f"{self.premium_pct:.3f}%",
            "day_return": f"{self.day_return:+.2f}%",
            "volume_24h": f"${self.day_volume_usd / 1e6:.2f}M",
            "predominant_bias": f"{bias}{bias_note}{extreme_note}",
            "extreme_funding": self.is_extreme_funding,
        }


class PerpMicrostructureFetcher:
    """
    Fetches perp market microstructure from Hyperliquid's free API.

    Rate limits: ~1200 requests per minute (info endpoint is generous).
    All calls are read-only, no authentication required.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._funding_history_cache: dict[str, list[dict]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_fetch: Optional[datetime] = None

    async def __aenter__(self) -> PerpMicrostructureFetcher:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_all(self, symbols: list[str]) -> dict[str, PerpMicrostructure]:
        """
        Fetch microstructure for multiple symbols in one batch.

        Args:
            symbols: List of coin symbols (e.g., ["BTC", "ETH"])

        Returns:
            Dict mapping symbol -> PerpMicrostructure
        """
        if not self._client:
            raise RuntimeError("Fetcher not initialized. Use async context manager.")

        # Fetch meta + asset contexts (single call gets ALL assets)
        meta_ctx = await self._fetch_meta_and_asset_ctxs()
        if not meta_ctx:
            return {}

        _, asset_contexts = meta_ctx

        # Build per-symbol data
        results = {}
        for symbol in symbols:
            ctx = self._find_asset_ctx(asset_contexts, symbol)
            if ctx:
                micro = self._build_microstructure(symbol, ctx)
                results[symbol] = micro

        # Fetch funding history for percentile calculation
        await self._fetch_funding_histories(symbols)
        for symbol in symbols:
            if symbol in results:
                results[symbol].funding_percentile_24h = self._calc_funding_percentile(
                    symbol, results[symbol].funding_rate
                )

        self._last_fetch = datetime.utcnow()
        return results

    async def fetch_single(self, symbol: str) -> Optional[PerpMicrostructure]:
        """Fetch microstructure for a single symbol."""
        results = await self.fetch_all([symbol])
        return results.get(symbol)

    async def _fetch_meta_and_asset_ctxs(self) -> Optional[tuple]:
        """
        Fetch meta and asset contexts from HL info endpoint.

        Returns:
            Tuple of (metadata, asset_contexts) or None on failure.
        """
        try:
            payload = {"type": "metaAndAssetCtxs"}
            resp = await self._client.post(HL_INFO_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or len(data) < 2:
                logger.warning("Unexpected metaAndAssetCtxs response format")
                return None

            return data[0], data[1]

        except Exception as e:
            logger.error(f"Failed to fetch metaAndAssetCtxs: {e}")
            return None

    def _find_asset_ctx(self, asset_contexts: list, symbol: str) -> Optional[dict]:
        """Find asset context by symbol name."""
        # HL uses uppercase coin names like "BTC", "ETH"
        target = symbol.upper()
        for ctx in asset_contexts:
            if ctx.get("name", "").upper() == target:
                return ctx
        return None

    def _build_microstructure(self, symbol: str, ctx: dict) -> PerpMicrostructure:
        """Build PerpMicrostructure from raw HL asset context."""
        mark_px = self._parse_float(ctx.get("markPx", 0))
        oracle_px = self._parse_float(ctx.get("oraclePx", 0))
        prev_px = self._parse_float(ctx.get("prevDayPx", 0))

        day_return = 0.0
        if prev_px > 0 and mark_px > 0:
            day_return = (mark_px - prev_px) / prev_px * 100

        funding = self._parse_float(ctx.get("funding", 0))
        premium = self._parse_float(ctx.get("premium", 0))
        oi = self._parse_float(ctx.get("openInterest", 0))
        volume = self._parse_float(ctx.get("dayNtlVlm", 0))

        # OI is in base asset units, convert to USD
        oi_usd = oi * mark_px if oi and mark_px else 0.0

        # Impact prices for liquidity assessment
        impact_pxs = ctx.get("impactPxs", [])
        impact_bid = self._parse_float(impact_pxs[0]) if impact_pxs else None
        impact_ask = self._parse_float(impact_pxs[1]) if len(impact_pxs) > 1 else None

        spread_pct = None
        if impact_bid and impact_ask and impact_bid > 0:
            spread_pct = (impact_ask - impact_bid) / impact_bid * 100

        return PerpMicrostructure(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            funding_rate=funding,
            funding_annualized=funding * 3 * 365 * 100,  # 8h -> annual %
            open_interest_usd=oi_usd,
            oi_change_24h=0.0,  # Will be populated from history if available
            premium_pct=premium * 100,  # Convert to percentage
            mark_price=mark_px,
            oracle_price=oracle_px,
            mid_price=self._parse_float(ctx.get("midPx", 0)),
            prev_day_price=prev_px,
            day_return=day_return,
            day_volume_usd=volume,
            impact_bid=impact_bid,
            impact_ask=impact_ask,
            spread_pct=spread_pct,
        )

    async def _fetch_funding_histories(self, symbols: list[str]) -> None:
        """Fetch recent funding history for percentile calculation."""
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        start_ms = now_ms - 24 * 3600 * 1000  # Last 24h

        for symbol in symbols:
            try:
                payload = {
                    "type": "fundingHistory",
                    "coin": symbol.upper(),
                    "startTime": start_ms,
                    "endTime": now_ms,
                }
                resp = await self._client.post(HL_INFO_URL, json=payload)
                resp.raise_for_status()
                history = resp.json()

                if isinstance(history, list):
                    self._funding_history_cache[symbol.upper()] = history

            except Exception as e:
                logger.debug(f"Failed to fetch funding history for {symbol}: {e}")

    def _calc_funding_percentile(self, symbol: str, current_rate: float) -> float:
        """Calculate where current funding sits in 24h history (0-1)."""
        history = self._funding_history_cache.get(symbol.upper(), [])
        if not history:
            return 0.5

        rates = [self._parse_float(h.get("fundingRate", 0)) for h in history]
        rates.append(current_rate)
        rates.sort()

        try:
            idx = rates.index(current_rate)
            return idx / max(len(rates) - 1, 1)
        except ValueError:
            return 0.5

    @staticmethod
    def _parse_float(value) -> float:
        """Safely parse a float from string or number."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    async def fetch_order_book_imbalance(self, symbol: str, depth: int = 50) -> Optional[float]:
        """
        Fetch L2 order book and compute bid/ask volume ratio.

        Args:
            symbol: Coin symbol
            depth: Number of levels to aggregate

        Returns:
            Bid volume / Ask volume ratio (>1 = more bids = bullish microstructure)
        """
        try:
            payload = {"type": "l2Book", "coin": symbol.upper()}
            resp = await self._client.post(HL_INFO_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            levels = data.get("levels", [])
            if not levels or len(levels) < 2:
                return None

            bids = levels[0][:depth]
            asks = levels[1][:depth]

            bid_volume = sum(self._parse_float(b["px"]) * self._parse_float(b["sz"]) for b in bids)
            ask_volume = sum(self._parse_float(a["px"]) * self._parse_float(a["sz"]) for a in asks)

            if ask_volume == 0:
                return None

            return bid_volume / ask_volume

        except Exception as e:
            logger.warning(f"Failed to fetch order book for {symbol}: {e}")
            return None
