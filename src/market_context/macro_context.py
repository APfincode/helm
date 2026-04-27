"""
Macro Context Fetcher

Fetches macro-level market sentiment data that affects ALL perp markets:
1. Fear & Greed Index (Alternative.me — free, no API key)
2. Breaking news (cryptocurrency.cv — free, no API key)

Both sources are free forever with no rate limits that would impact a trading bot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Free API endpoints
FEAR_GREED_URL = "https://api.alternative.me/fng/"
CRYPTO_NEWS_URL = "https://cryptocurrency.cv/api/breaking"


@dataclass
class FearGreedReading:
    """Single Fear & Greed index reading."""

    value: int  # 0-100
    classification: str  # e.g., "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime

    @property
    def is_extreme(self) -> bool:
        return self.value <= 20 or self.value >= 80

    def to_prompt_line(self) -> str:
        if self.is_extreme:
            return f"{self.value} ({self.classification}) — EXTREME, contrarian signal"
        return f"{self.value} ({self.classification})"


@dataclass
class NewsItem:
    """Single breaking news item."""

    title: str
    source: str
    tickers: list[str]
    published_at: datetime
    sentiment_score: Optional[float] = None  # -1 to +1 if available
    sentiment_label: Optional[str] = None


@dataclass
class MacroContext:
    """Aggregated macro market context."""

    fear_greed: Optional[FearGreedReading] = None
    fear_greed_history: list[FearGreedReading] = None  # Last N readings
    breaking_news: list[NewsItem] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.fear_greed_history is None:
            self.fear_greed_history = []
        if self.breaking_news is None:
            self.breaking_news = []
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

    def to_prompt_context(self) -> dict:
        """Format for LLM prompt injection."""
        fg_line = "Unavailable"
        if self.fear_greed:
            fg_line = self.fear_greed.to_prompt_line()

        # Only include top 3 news items to avoid prompt bloat
        news_lines = []
        for item in self.breaking_news[:3]:
            tickers = ", ".join(item.tickers) if item.tickers else "general"
            age = ""
            delta = datetime.utcnow() - item.published_at
            if delta < timedelta(hours=1):
                age = " [JUST IN]"
            elif delta < timedelta(hours=6):
                age = " [RECENT]"
            news_lines.append(f"- [{tickers}] {item.title}{age}")

        if not news_lines:
            news_lines = ["- No major breaking news"]

        return {
            "fear_greed": fg_line,
            "fear_greed_trend": self._fg_trend(),
            "breaking_news": "\n".join(news_lines),
        }

    def _fg_trend(self) -> str:
        """Describe Fear & Greed trend over last few readings."""
        if len(self.fear_greed_history) < 2:
            return "insufficient data"

        recent = self.fear_greed_history[-5:]  # Last 5 readings
        values = [r.value for r in recent]
        if not values:
            return "insufficient data"

        first, last = values[0], values[-1]
        change = last - first

        if change > 10:
            return f"rapidly increasing greed (+{change} points)"
        if change > 5:
            return f"increasing greed (+{change} points)"
        if change < -10:
            return f"rapidly increasing fear ({change} points)"
        if change < -5:
            return f"increasing fear ({change} points)"
        return f"stable ({change:+d} points)"


class MacroContextFetcher:
    """
    Fetches macro market context from free APIs.

    No API keys required. Rate limits are generous for these endpoints.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._last_fetched: Optional[datetime] = None

    async def __aenter__(self) -> MacroContextFetcher:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, max_news: int = 5, fg_history_days: int = 7) -> MacroContext:
        """
        Fetch complete macro context.

        Args:
            max_news: Max number of breaking news items to fetch
            fg_history_days: Number of days of F&G history to fetch

        Returns:
            MacroContext with all available data.
        """
        if not self._client:
            raise RuntimeError("Fetcher not initialized. Use async context manager.")

        fg_current, fg_history = await self._fetch_fear_greed(fg_history_days)
        news = await self._fetch_breaking_news(max_news)

        self._last_fetched = datetime.utcnow()

        return MacroContext(
            fear_greed=fg_current,
            fear_greed_history=fg_history,
            breaking_news=news,
            last_updated=datetime.utcnow(),
        )

    async def _fetch_fear_greed(
        self, history_days: int
    ) -> tuple[Optional[FearGreedReading], list[FearGreedReading]]:
        """
        Fetch Fear & Greed index.

        Args:
            history_days: How many days of history to fetch.

        Returns:
            Tuple of (current reading, list of historical readings).
        """
        limit = max(history_days, 1)
        url = f"{FEAR_GREED_URL}?limit={limit}"

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()

            if not data or "data" not in data:
                logger.warning("Unexpected F&G response format")
                return None, []

            readings = []
            for item in data["data"]:
                try:
                    ts = int(item.get("timestamp", 0))
                    reading = FearGreedReading(
                        value=int(item.get("value", 50)),
                        classification=item.get("value_classification", "Neutral"),
                        timestamp=datetime.utcfromtimestamp(ts),
                    )
                    readings.append(reading)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid F&G reading: {e}")
                    continue

            if readings:
                # Sort by timestamp ascending
                readings.sort(key=lambda r: r.timestamp)
                return readings[-1], readings

            return None, []

        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed index: {e}")
            return None, []

    async def _fetch_breaking_news(self, max_items: int) -> list[NewsItem]:
        """
        Fetch breaking news from cryptocurrency.cv.

        Args:
            max_items: Maximum number of news items to return.

        Returns:
            List of NewsItem objects.
        """
        try:
            resp = await self._client.get(CRYPTO_NEWS_URL, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                logger.warning("Unexpected news response format")
                return []

            items = []
            for article in data[:max_items]:
                try:
                    # Parse publication time
                    pub_str = article.get("pub_date") or article.get("published_at") or ""
                    pub_dt = self._parse_datetime(pub_str)

                    # Extract sentiment if available
                    sentiment = article.get("sentiment", {})
                    sentiment_score = sentiment.get("score") if isinstance(sentiment, dict) else None
                    sentiment_label = sentiment.get("label") if isinstance(sentiment, dict) else None

                    item = NewsItem(
                        title=article.get("title", "Untitled"),
                        source=article.get("source", "Unknown"),
                        tickers=article.get("tickers", []),
                        published_at=pub_dt or datetime.utcnow(),
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                    )
                    items.append(item)
                except Exception as e:
                    logger.debug(f"Skipping invalid news item: {e}")
                    continue

            return items

        except Exception as e:
            logger.error(f"Failed to fetch breaking news: {e}")
            return []

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        """Parse various datetime formats."""
        if not value:
            return None
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        # Try ISO format as fallback
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
        return None
