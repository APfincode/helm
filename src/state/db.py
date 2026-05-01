"""Helm State — SQLite-backed bot state for cross-component coordination.

Provides a read/write interface for the trading engine and a read-only
interface for the web UI. All state is normalized into SQLite tables.

Tables:
  - bot_status: current equity, mode, P&L, uptime
  - positions: open + closed position records
  - trades: executed trade log with fees and P&L
  - signals: LLM signal history
  - risk_events: circuit breaker triggers, guard decisions
  - logs: structured log lines
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class StateDB:
    """Thread-safe SQLite state database for Helm."""

    _DB_PATH = Path(__file__).parent.parent.parent / "data" / "helm_state.db"

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._path = Path(db_path) if db_path else self._DB_PATH
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_tables()

    # ------------------------------------------------------------------
    # Internal helpers (thread-safe via _lock)
    # ------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_status (
                    id          INTEGER PRIMARY KEY CHECK(id=1),
                    updated_at  TEXT NOT NULL,
                    mode        TEXT NOT NULL DEFAULT 'PAPER',
                    running     INTEGER NOT NULL DEFAULT 0,
                    equity      REAL NOT NULL DEFAULT 0.0,
                    day_pnl     REAL NOT NULL DEFAULT 0.0,
                    day_pnl_pct REAL NOT NULL DEFAULT 0.0,
                    open_positions INTEGER NOT NULL DEFAULT 0,
                    total_exposure REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    signals_generated INTEGER NOT NULL DEFAULT 0,
                    trades_executed  INTEGER NOT NULL DEFAULT 0,
                    trades_rejected  INTEGER NOT NULL DEFAULT 0,
                    last_error       TEXT DEFAULT ''
                );
                INSERT OR IGNORE INTO bot_status (id,updated_at,mode)
                VALUES (1, datetime('now'), 'PAPER');

                CREATE TABLE IF NOT EXISTS positions (
                    id        TEXT PRIMARY KEY,
                    coin      TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time  TEXT NOT NULL,
                    mark_price  REAL,
                    quantity    REAL NOT NULL,
                    leverage  REAL NOT NULL DEFAULT 1.0,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    risk_usd  REAL,
                    margin_used REAL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    realized_pnl   REAL DEFAULT 0.0,
                    fees_paid      REAL DEFAULT 0.0,
                    status    TEXT NOT NULL DEFAULT 'OPEN',
                    close_price REAL,
                    close_time  TEXT,
                    is_paper  INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    time        TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    direction   TEXT NOT NULL,
                    entry_price REAL,
                    exit_price  REAL,
                    size        REAL,
                    pnl         REAL,
                    pnl_pct     REAL,
                    fee         REAL,
                    reason      TEXT,
                    mode        TEXT
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    time      TEXT NOT NULL,
                    symbol    TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    reasoning TEXT,
                    regime    TEXT,
                    error     TEXT,
                    cost_usd  REAL
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    time      TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity   TEXT NOT NULL,
                    details    TEXT,
                    triggered  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    time      TEXT NOT NULL,
                    level     TEXT NOT NULL,
                    source    TEXT,
                    message   TEXT NOT NULL
                );
                """
            )
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Write API (used by ExecutionEngine)
    # ------------------------------------------------------------------
    def update_status(self, **fields: Any) -> None:
        """Update one or more columns in bot_status row 1."""
        allowed = {
            "mode", "running", "equity", "day_pnl", "day_pnl_pct",
            "open_positions", "total_exposure", "unrealized_pnl",
            "signals_generated", "trades_executed", "trades_rejected",
            "last_error",
        }
        update = {k: v for k, v in fields.items() if k in allowed}
        if not update:
            return
        update["updated_at"] = datetime.utcnow().isoformat()
        cols = ", ".join(f"{k}=?" for k in update)
        vals = tuple(update.values())
        with self._lock:
            conn = self._conn()
            conn.execute(f"UPDATE bot_status SET {cols} WHERE id=1", vals)
            conn.commit()
            conn.close()

    def insert_position(self, position: dict[str, Any]) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO positions (
                    id, coin, direction, entry_price, entry_time, quantity,
                    leverage, stop_loss_price, take_profit_price,
                    risk_usd, margin_used, status, is_paper
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    position["id"],
                    position["coin"],
                    position["direction"],
                    position.get("entry_price", 0.0),
                    position.get("entry_time", datetime.utcnow().isoformat()),
                    position.get("quantity", 0.0),
                    position.get("leverage", 1.0),
                    position.get("stop_loss_price"),
                    position.get("take_profit_price"),
                    position.get("risk_usd"),
                    position.get("margin_used"),
                    position.get("status", "OPEN"),
                    int(position.get("is_paper", True)),
                ),
            )
            conn.commit()
            conn.close()

    def close_position(self, position_id: str, close_price: float,
                       realized_pnl: float, fees: float) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """UPDATE positions
                   SET status='CLOSED', close_price=?, close_time=?,
                       realized_pnl=?, fees_paid=?
                   WHERE id=?""",
                (close_price, datetime.utcnow().isoformat(), realized_pnl, fees, position_id),
            )
            conn.commit()
            conn.close()

    def insert_trade(self, trade: dict[str, Any]) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """INSERT INTO trades
                   (time, symbol, direction, entry_price, exit_price, size,
                    pnl, pnl_pct, fee, reason, mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.get("time", datetime.utcnow().isoformat()),
                    trade["symbol"],
                    trade["direction"],
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("size"),
                    trade.get("pnl"),
                    trade.get("pnl_pct"),
                    trade.get("fee"),
                    trade.get("reason"),
                    trade.get("mode", "PAPER"),
                ),
            )
            conn.commit()
            conn.close()

    def insert_signal(self, signal: dict[str, Any]) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """INSERT INTO signals
                   (time, symbol, direction, confidence, reasoning, regime, error, cost_usd)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    signal.get("time", datetime.utcnow().isoformat()),
                    signal.get("symbol", ""),
                    signal.get("direction", "NEUTRAL"),
                    signal.get("confidence"),
                    signal.get("reasoning", "")[:500],
                    signal.get("regime", ""),
                    signal.get("error", ""),
                    signal.get("cost_usd"),
                ),
            )
            conn.commit()
            conn.close()

    def log_risk_event(self, event_type: str, severity: str,
                       details: Optional[dict] = None, triggered: bool = False) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """INSERT INTO risk_events (time, event_type, severity, details, triggered)
                   VALUES (?,?,?,?,?)""",
                (datetime.utcnow().isoformat(), event_type, severity,
                 json.dumps(details or {}), int(triggered)),
            )
            conn.commit()
            conn.close()

    def append_log(self, level: str, message: str, source: str = "helm") -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO logs (time, level, source, message) VALUES (?,?,?,?)",
                (datetime.utcnow().isoformat(), level, source, message),
            )
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Read API (used by Web UI and Telegram)
    # ------------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM bot_status WHERE id=1"
            ).fetchone()
            conn.close()
            if not row:
                return {}
            return {
                "mode": row["mode"],
                "running": bool(row["running"]),
                "equity": row["equity"],
                "day_pnl": row["day_pnl"],
                "day_pnl_pct": row["day_pnl_pct"],
                "open_positions": row["open_positions"],
                "total_exposure": row["total_exposure"],
                "unrealized_pnl": row["unrealized_pnl"],
                "signals_generated": row["signals_generated"],
                "trades_executed": row["trades_executed"],
                "trades_rejected": row["trades_rejected"],
            }

    def get_positions(self, open_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM positions"
        if open_only:
            sql += " WHERE status='OPEN'"
        sql += " ORDER BY entry_time DESC"
        with self._lock:
            conn = self._conn()
            rows = conn.execute(sql).fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def get_signals(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def get_risk(self) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            events = conn.execute(
                "SELECT * FROM risk_events ORDER BY time DESC LIMIT 20"
            ).fetchall()
            conn.close()
        return {
            "halted": any(bool(r["triggered"]) for r in events),
            "halt_reason": "",
            "circuit_breakers": [
                {"name": r["event_type"],
                 "limit": "",
                 "current": json.loads(r["details"]).get("current", ""),
                 "status": "tripped" if r["triggered"] else "armed"}
                for r in events[:5]
            ],
        }

    def get_logs(self, limit: int = 50) -> list[str]:
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT time, level, message FROM logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        return [f"[{r['time']}] {r['level']:8} {r['message']}" for r in rows]


_state_db_instance: Optional[StateDB] = None


def get_state_db(db_path: Optional[str | Path] = None) -> StateDB:
    """Get the global singleton StateDB instance."""
    global _state_db_instance
    if _state_db_instance is None:
        _state_db_instance = StateDB(db_path)
    return _state_db_instance
