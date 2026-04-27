"""
Web UI state reader — reads bot state from multiple sources.

Provides a unified read-only interface for the dashboard.
Safe fallbacks if DB tables don't exist yet.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BotStatus:
    """Current bot status snapshot."""

    version: str = "0.1.0"
    mode: str = "paper"
    is_running: bool = False
    uptime_seconds: int = 0
    symbols: list[str] = field(default_factory=list)
    equity: float = 0.0
    equity_change_24h: float = 0.0
    open_position_count: int = 0
    total_trades_today: int = 0
    signals_generated: int = 0
    last_signal_time: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PositionDTO:
    """Position data transfer object for UI."""

    id: str
    symbol: str
    direction: str
    entry_price: float
    mark_price: float
    quantity: float
    leverage: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: Optional[datetime] = None
    time_in_trade: str = ""


@dataclass
class TradeDTO:
    """Closed trade data transfer object."""

    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    realized_pnl_pct: float
    fee: float
    exit_reason: str
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration: str = ""


@dataclass
class SignalDTO:
    """LLM signal data transfer object."""

    id: str
    timestamp: datetime
    symbol: str
    direction: str
    confidence: float
    reasoning: str
    regime: str
    provider: str
    model: str
    cost_usd: float
    latency_ms: int


@dataclass
class RiskStatus:
    """Risk guard status snapshot."""

    daily_loss_current: float = 0.0
    daily_loss_limit: float = 5.0
    drawdown_current: float = 0.0
    drawdown_max: float = 15.0
    fee_budget_current: float = 0.0
    fee_budget_warn: float = 2.0
    fee_budget_max: float = 5.0
    total_trades_today: int = 0
    total_trades_accepted: int = 0
    total_trades_rejected: int = 0
    circuit_breakers: list[dict] = field(default_factory=list)
    is_halted: bool = False
    halt_reason: str = ""


class StateReader:
    """
    Reads bot state from SQLite, YAML config, and optionally an engine object.

    Provides safe fallbacks for all operations.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        db_path: Optional[str] = None,
        config_dir: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._db_path = db_path or str(Path.home() / ".hermes" / "audit.db")
        self._config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self._start_time = datetime.utcnow()

    # ========================================================================
    # Status
    # ========================================================================

    def get_status(self) -> BotStatus:
        """Get current bot status."""
        status = BotStatus(
            mode=self._get_mode(),
            is_running=self._get_is_running(),
            uptime_seconds=int((datetime.utcnow() - self._start_time).total_seconds()),
            symbols=self._get_symbols(),
            last_updated=datetime.utcnow(),
        )

        # Try to get from engine
        if self._engine:
            try:
                stats = self._engine.stats if hasattr(self._engine, "stats") else {}
                status.equity = self._safe_float(stats.get("unrealized_pnl", 0))
                status.open_position_count = stats.get("open_positions", 0)
                status.signals_generated = stats.get("signals_generated", 0)
            except Exception as e:
                logger.debug(f"Failed to read engine stats: {e}")

        return status

    # ========================================================================
    # Positions
    # ========================================================================

    def get_positions(self) -> list[PositionDTO]:
        """Get current open positions."""
        positions = []

        if self._engine and hasattr(self._engine, "_tracker"):
            try:
                tracker = self._engine._tracker
                open_pos = tracker.get_all_open() if hasattr(tracker, "get_all_open") else []
                for pos in open_pos:
                    pnl = getattr(pos, "unrealized_pnl", 0.0)
                    entry = getattr(pos, "entry_price", 0.0)
                    pnl_pct = (pnl / entry * 100) if entry else 0.0
                    entry_time = getattr(pos, "entry_time", None)

                    positions.append(PositionDTO(
                        id=getattr(pos, "id", "unknown"),
                        symbol=getattr(pos, "coin", "?"),
                        direction=getattr(pos, "direction", "?"),
                        entry_price=entry,
                        mark_price=getattr(pos, "current_price", entry),
                        quantity=getattr(pos, "quantity", 0.0),
                        leverage=getattr(pos, "leverage", 1.0),
                        unrealized_pnl=pnl,
                        unrealized_pnl_pct=pnl_pct,
                        stop_loss=getattr(pos, "stop_loss_price", None),
                        take_profit=getattr(pos, "take_profit_price", None),
                        entry_time=entry_time,
                        time_in_trade=self._format_duration(entry_time),
                    ))
            except Exception as e:
                logger.debug(f"Failed to read positions from engine: {e}")

        # Fallback: try SQLite
        if not positions:
            positions = self._get_positions_from_db()

        return positions

    # ========================================================================
    # History
    # ========================================================================

    def get_history(self, limit: int = 50) -> list[TradeDTO]:
        """Get closed trade history."""
        trades = []

        if self._engine and hasattr(self._engine, "_tracker"):
            try:
                tracker = self._engine._tracker
                history = getattr(tracker, "_history", [])[-limit:]
                for pos in history:
                    entry = getattr(pos, "entry_price", 0.0)
                    exit_p = getattr(pos, "exit_price", entry)
                    qty = getattr(pos, "quantity", 0.0)
                    pnl = getattr(pos, "realized_pnl", 0.0)
                    pnl_pct = (pnl / (entry * qty) * 100) if entry and qty else 0.0
                    entry_time = getattr(pos, "entry_time", None)
                    exit_time = getattr(pos, "exit_time", None)

                    trades.append(TradeDTO(
                        id=getattr(pos, "id", "unknown"),
                        symbol=getattr(pos, "coin", "?"),
                        direction=getattr(pos, "direction", "?"),
                        entry_price=entry,
                        exit_price=exit_p,
                        quantity=qty,
                        realized_pnl=pnl,
                        realized_pnl_pct=pnl_pct,
                        fee=getattr(pos, "fees_paid", 0.0),
                        exit_reason=getattr(pos, "exit_reason", "unknown"),
                        entry_time=entry_time,
                        exit_time=exit_time,
                        duration=self._format_duration_between(entry_time, exit_time),
                    ))
            except Exception as e:
                logger.debug(f"Failed to read history from engine: {e}")

        if not trades:
            trades = self._get_history_from_db(limit)

        return trades

    # ========================================================================
    # Signals
    # ========================================================================

    def get_signals(self, limit: int = 20) -> list[SignalDTO]:
        """Get recent LLM signals."""
        return self._get_signals_from_db(limit)

    # ========================================================================
    # Risk
    # ========================================================================

    def get_risk(self) -> RiskStatus:
        """Get current risk guard status."""
        risk = RiskStatus()

        if self._engine and hasattr(self._engine, "_risk_manager"):
            try:
                rm = self._engine._risk_manager
                stats = getattr(rm, "stats", {})
                risk.total_trades_accepted = stats.get("total_trades_accepted", 0)
                risk.total_trades_rejected = stats.get("total_trades_rejected", 0)
            except Exception as e:
                logger.debug(f"Failed to read risk stats: {e}")

        # Build circuit breaker list
        risk.circuit_breakers = [
            {"name": "Daily Loss", "status": "armed", "current": f"{risk.daily_loss_current:.2f}%", "limit": f"{risk.daily_loss_limit:.1f}%"},
            {"name": "Max Drawdown", "status": "armed", "current": f"{risk.drawdown_current:.2f}%", "limit": f"{risk.drawdown_max:.1f}%"},
            {"name": "Fee Budget", "status": "armed", "current": f"{risk.fee_budget_current:.2f}%", "limit": f"{risk.fee_budget_max:.1f}%"},
        ]

        return risk

    # ========================================================================
    # Config
    # ========================================================================

    def get_config(self) -> dict:
        """Read current configuration from files."""
        config = {}

        # Try to load config files
        try:
            # Look for config files in config dir
            for f in self._config_dir.glob("*.yaml"):
                try:
                    with open(f, "r") as fh:
                        data = yaml.safe_load(fh)
                        if data and isinstance(data, dict):
                            config[f.stem] = data
                except Exception:
                    continue
        except Exception:
            pass

        # Add runtime config from engine
        if self._engine and hasattr(self._engine, "_config"):
            try:
                cfg = self._engine._config
                config["runtime"] = {
                    "mode": getattr(cfg, "mode", "paper"),
                    "symbols": getattr(cfg, "symbols", ["BTC", "ETH"]),
                    "signal_interval_minutes": getattr(cfg, "signal_interval_minutes", 60),
                }
            except Exception:
                pass

        return config

    # ========================================================================
    # Logs
    # ========================================================================

    def get_logs(self, limit: int = 100) -> list[dict]:
        """Get recent log lines from SQLite audit log."""
        return self._get_logs_from_db(limit)

    # ========================================================================
    # DB Helpers
    # ========================================================================

    def _get_positions_from_db(self) -> list[PositionDTO]:
        """Fallback: read positions from audit DB."""
        return []

    def _get_history_from_db(self, limit: int) -> list[TradeDTO]:
        """Fallback: read trade history from audit DB."""
        rows = self._query_db(
            "SELECT timestamp, action, details FROM audit_log WHERE event_type = 'TRADE' ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        trades = []
        for row in rows:
            try:
                details = json.loads(row[2]) if row[2] else {}
                trades.append(TradeDTO(
                    id=details.get("order_id", "unknown"),
                    symbol=details.get("symbol", "?"),
                    direction=details.get("direction", "?"),
                    entry_price=details.get("entry_price", 0.0),
                    exit_price=details.get("exit_price", 0.0),
                    quantity=details.get("quantity", 0.0),
                    realized_pnl=details.get("pnl", 0.0),
                    realized_pnl_pct=0.0,
                    fee=details.get("fee", 0.0),
                    exit_reason=details.get("reason", "unknown"),
                ))
            except Exception:
                continue
        return trades

    def _get_signals_from_db(self, limit: int) -> list[SignalDTO]:
        """Read LLM signals from audit DB."""
        rows = self._query_db(
            "SELECT timestamp, action, details FROM audit_log WHERE event_type = 'LLM_SIGNAL' ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        signals = []
        for row in rows:
            try:
                details = json.loads(row[2]) if row[2] else {}
                ts = datetime.fromisoformat(row[0]) if row[0] else datetime.utcnow()
                signals.append(SignalDTO(
                    id=f"sig_{len(signals)}",
                    timestamp=ts,
                    symbol=details.get("symbol", "?"),
                    direction=details.get("direction", "NEUTRAL"),
                    confidence=details.get("confidence", 0.0),
                    reasoning=details.get("reasoning", "")[:200],
                    regime=details.get("regime", "unknown"),
                    provider=details.get("provider", "unknown"),
                    model=details.get("model", "unknown"),
                    cost_usd=details.get("cost_usd", 0.0),
                    latency_ms=details.get("latency_ms", 0),
                ))
            except Exception:
                continue
        return signals

    def _get_logs_from_db(self, limit: int) -> list[dict]:
        """Read recent log entries from audit DB."""
        rows = self._query_db(
            "SELECT timestamp, severity, actor, action, details FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        logs = []
        severity_map = {"INFO": "info", "WARNING": "warning", "ERROR": "error", "CRITICAL": "critical"}
        for row in rows:
            try:
                logs.append({
                    "timestamp": row[0] if row[0] else "",
                    "severity": severity_map.get(row[1], "info"),
                    "actor": row[2] if row[2] else "",
                    "action": row[3] if row[3] else "",
                    "details": row[4] if row[4] else "",
                })
            except Exception:
                continue
        return logs

    def _query_db(self, sql: str, params: tuple = ()) -> list:
        """Safely query SQLite DB with fallback."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            return [tuple(r) for r in rows]
        except Exception as e:
            logger.debug(f"DB query failed: {e}")
            return []

    # ========================================================================
    # Engine Helpers
    # ========================================================================

    def _get_mode(self) -> str:
        if self._engine and hasattr(self._engine, "_config"):
            try:
                return str(self._engine._config.mode.value)
            except Exception:
                pass
        return "paper"

    def _get_is_running(self) -> bool:
        if self._engine and hasattr(self._engine, "is_running"):
            try:
                return bool(self._engine.is_running)
            except Exception:
                pass
        return False

    def _get_symbols(self) -> list[str]:
        if self._engine and hasattr(self._engine, "_config"):
            try:
                return list(self._engine._config.symbols)[:10]
            except Exception:
                pass
        return ["BTC", "ETH"]

    # ========================================================================
    # Formatting
    # ========================================================================

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_duration(start: Optional[datetime]) -> str:
        if not start:
            return ""
        delta = datetime.utcnow() - start
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        if delta.seconds > 3600:
            return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
        return f"{delta.seconds // 60}m"

    @staticmethod
    def _format_duration_between(start: Optional[datetime], end: Optional[datetime]) -> str:
        if not start or not end:
            return ""
        delta = end - start
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        if delta.seconds > 3600:
            return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
        return f"{delta.seconds // 60}m"
