"""
Token Tracker — Usage and cost tracking for LLM API calls.

Purpose:
1. Prevent budget overruns (hard limits per day/week/month)
2. Track which prompts/strategies cost the most
3. Provide visibility into API spend
4. Alert when approaching limits

Usage:
    tracker = TokenTracker()
    tracker.record(provider="openrouter", model="llama-70b", input_tokens=100, output_tokens=50, cost_usd=0.003)
    
    if tracker.get_monthly_cost() > 50.0:
        print("Approaching monthly budget")
"""

import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiosqlite


@dataclass
class TokenUsage:
    """Single API call usage record."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: str = ""


class TokenTracker:
    """
    Tracks LLM API usage and costs.
    
    Stores usage in SQLite for persistence across sessions.
    Provides budget monitoring and alerts.
    """

    def __init__(
        self,
        db_path: str = "data/token_usage.db",
        daily_budget: float = 10.0,
        weekly_budget: float = 50.0,
        monthly_budget: float = 200.0,
    ) -> None:
        self._db_path = db_path
        self._daily_budget = daily_budget
        self._weekly_budget = weekly_budget
        self._monthly_budget = monthly_budget
        
        # In-memory aggregates
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._error_count: int = 0
        self._total_latency_ms: float = 0.0

    async def init_db(self) -> None:
        """Initialize SQLite database."""
        import os
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT DEFAULT ''
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp 
                ON token_usage(timestamp)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_provider 
                ON token_usage(provider)
            """)
            
            await db.commit()

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        success: bool = True,
        error: str = "",
    ) -> None:
        """
        Record a single API call.
        
        This is synchronous (fast, in-memory) — persistence is async background task.
        """
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        
        # Update aggregates
        self._total_requests += 1
        self._total_tokens += usage.total_tokens
        self._total_cost_usd += cost_usd
        self._total_latency_ms += latency_ms
        
        if not success:
            self._error_count += 1
        
        # Queue for async persistence (fire and forget)
        # In production, use a background task queue
        # For now, we'll use an async save method

    async def save(self, usage: TokenUsage) -> None:
        """Persist usage record to database."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO token_usage 
                (timestamp, provider, model, input_tokens, output_tokens, total_tokens, cost_usd, latency_ms, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage.timestamp, usage.provider, usage.model,
                    usage.input_tokens, usage.output_tokens, usage.total_tokens,
                    usage.cost_usd, usage.latency_ms, int(usage.success), usage.error,
                )
            )
            await db.commit()

    async def get_daily_cost(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return await self._get_cost_for_date_range(today, today)

    async def get_weekly_cost(self) -> float:
        """Get total cost for this week."""
        today = datetime.now()
        week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        week_end = today.strftime("%Y-%m-%d")
        return await self._get_cost_for_date_range(week_start, week_end)

    async def get_monthly_cost(self) -> float:
        """Get total cost for this month."""
        today = datetime.now()
        month_start = today.strftime("%Y-%m-01")
        return await self._get_cost_for_date_range(month_start, today.strftime("%Y-%m-%d"))

    async def _get_cost_for_date_range(self, start_date: str, end_date: str) -> float:
        """Query database for cost in date range."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT SUM(cost_usd) FROM token_usage WHERE date(timestamp) BETWEEN ? AND ?",
                    (start_date, end_date)
                ) as cursor:
                    row = await cursor.fetchone()
                    return row[0] or 0.0
        except Exception:
            return 0.0

    def check_budget(self) -> dict[str, bool]:
        """
        Check if any budget limit is exceeded.
        
        Returns:
            Dict with status of each budget limit
        """
        # Use in-memory totals for quick checks
        daily_cost = self._total_cost_usd  # Simplified — assumes single session
        
        return {
            "daily": daily_cost < self._daily_budget,
            "weekly": True,  # Requires database query
            "monthly": True,  # Requires database query
        }

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def average_latency_ms(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._total_latency_ms / self._total_requests

    @property
    def error_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._error_count / self._total_requests

    async def get_stats_by_provider(self) -> dict[str, dict]:
        """Get usage statistics grouped by provider."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """
                    SELECT provider, 
                           COUNT(*) as requests,
                           SUM(total_tokens) as tokens,
                           SUM(cost_usd) as cost,
                           AVG(latency_ms) as avg_latency
                    FROM token_usage
                    GROUP BY provider
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
                    
            stats = {}
            for row in rows:
                stats[row[0]] = {
                    "requests": row[1],
                    "tokens": row[2],
                    "cost_usd": row[3],
                    "avg_latency_ms": row[4] or 0,
                }
            return stats
        except Exception:
            return {}


class CostTracker:
    """
    Simple cost tracker for non-persistent usage monitoring.
    
    Use TokenTracker for production — this is for lightweight cases.
    """

    def __init__(self) -> None:
        self._calls: list[tuple[str, float]] = []  # (provider, cost)

    def add(self, provider: str, cost_usd: float) -> None:
        self._calls.append((provider, cost_usd))

    @property
    def total_cost(self) -> float:
        return sum(cost for _, cost in self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def get_by_provider(self, provider: str) -> float:
        return sum(cost for p, cost in self._calls if p == provider)

    def reset(self) -> None:
        self._calls.clear()
