"""
Web UI Server — Hermes-inspired lightweight dashboard.

No build step, no bundler, no React. Pure FastAPI + Jinja2 + vanilla JS + ~250 LOC CSS.

Routes:
  GET /              → Overview (default)
  GET /positions     → Open positions
  GET /history       → Trade history
  GET /signals       → LLM signal log
  GET /risk          → Risk guard status
  GET /config        → Configuration editor
  GET /logs          → Recent log lines

API:
  GET /api/status    → JSON (equity, positions, mode)
  POST /api/stop     → Emergency stop
  GET /api/config    → JSON (current config)
  POST /api/config   → JSON (update config)
  GET /api/events    → SSE (live updates)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict:
    """Safely load a YAML file."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def _save_yaml(path: Path, data: dict) -> bool:
    """Safely save a YAML file."""
    try:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        return False


class BotStateProvider:
    """
    Reads bot state from SQLite and config files.
    
    This is a read-only interface for the web UI. The trading bot
    writes to these stores; the web UI only reads.
    """

    def __init__(self) -> None:
        self._db_path = PROJECT_ROOT / "data" / "bot_state.db"

    def get_status(self) -> dict:
        """Get current bot status."""
        # In production, this reads from SQLite. For now, return placeholder.
        return {
            "mode": "PAPER",
            "running": True,
            "equity": 12450.0,
            "day_pnl": 234.0,
            "day_pnl_pct": 1.9,
            "open_positions": 2,
            "total_exposure": 8500.0,
            "unrealized_pnl": 165.0,
            "signals_generated": 42,
            "trades_executed": 15,
            "trades_rejected": 8,
        }

    def get_positions(self) -> list[dict]:
        """Get open positions."""
        return [
            {
                "symbol": "BTC",
                "direction": "LONG",
                "size": 0.10,
                "entry": 94500.0,
                "mark": 95400.0,
                "unrealized": 90.0,
                "leverage": 5.0,
                "stop": 92000.0,
                "tp": 98000.0,
                "time_in_trade": "2h 15m",
            },
            {
                "symbol": "ETH",
                "direction": "SHORT",
                "size": 1.50,
                "entry": 3250.0,
                "mark": 3200.0,
                "unrealized": 75.0,
                "leverage": 3.0,
                "stop": 3400.0,
                "tp": 3000.0,
                "time_in_trade": "45m",
            },
        ]

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get closed trade history."""
        return [
            {
                "time": "14:32",
                "symbol": "BTC",
                "direction": "LONG",
                "entry": 94000.0,
                "exit": 95300.0,
                "pnl": 130.0,
                "pnl_pct": 1.38,
                "reason": "take_profit",
            },
            {
                "time": "11:15",
                "symbol": "ETH",
                "direction": "SHORT",
                "entry": 3300.0,
                "exit": 3250.0,
                "pnl": 75.0,
                "pnl_pct": 1.52,
                "reason": "stop_loss",
            },
        ]

    def get_signals(self, limit: int = 10) -> list[dict]:
        """Get LLM signal log."""
        return [
            {
                "time": "14:30",
                "symbol": "BTC",
                "direction": "LONG",
                "confidence": 0.82,
                "reasoning": "Bullish engulfing on 4h with volume confirmation. Funding slightly positive but not extreme.",
                "regime": "trending_up",
            },
            {
                "time": "13:00",
                "symbol": "ETH",
                "direction": "NEUTRAL",
                "confidence": 0.45,
                "reasoning": "Consolidating near support. Wait for breakout confirmation.",
                "regime": "ranging",
            },
        ]

    def get_risk(self) -> dict:
        """Get risk guard status."""
        return {
            "halted": False,
            "halt_reason": "",
            "max_drawdown_pct": 15.0,
            "current_drawdown_pct": 3.2,
            "daily_loss_limit_pct": 5.0,
            "current_daily_loss_pct": 1.1,
            "fee_budget_pct": 2.0,
            "current_fee_pct": 0.8,
            "circuit_breakers": [
                {"name": "Max Drawdown", "limit": "15%", "current": "3.2%", "status": "armed"},
                {"name": "Daily Loss", "limit": "5%", "current": "1.1%", "status": "armed"},
                {"name": "Fee Budget", "limit": "2%", "current": "0.8%", "status": "armed"},
                {"name": "Min Trade Interval", "limit": "4h", "current": "OK", "status": "armed"},
                {"name": "Max Positions", "limit": "3", "current": "2/3", "status": "armed"},
            ],
        }

    def get_config(self) -> dict:
        """Get current configuration from YAML files."""
        config = {}
        for yaml_file in CONFIG_DIR.glob("*.yaml"):
            key = yaml_file.stem
            config[key] = _load_yaml(yaml_file)
        return config

    def update_config(self, key: str, data: dict) -> bool:
        """Update a configuration file."""
        path = CONFIG_DIR / f"{key}.yaml"
        return _save_yaml(path, data)

    def get_logs(self, limit: int = 50) -> list[str]:
        """Get recent log lines."""
        log_path = PROJECT_ROOT / "logs" / "bot.log"
        if not log_path.exists():
            return ["No log file found."]
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            return lines[-limit:] if lines else ["Log file is empty."]
        except Exception as e:
            return [f"Error reading logs: {e}"]


# Global state provider
_state = BotStateProvider()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Hyper-Alpha-Arena Dashboard", version="0.1.0")

    # Static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Templates
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    templates.env.filters["min"] = lambda v, arg: min(v, arg)

    def _render(request: Request, template: str, active_tab: str, **ctx) -> HTMLResponse:
        """Helper to render a template with common context."""
        status = _state.get_status()
        return templates.TemplateResponse(
            request,
            template,
            context={
                "active_tab": active_tab,
                "status": status,
                **ctx,
            },
        )

    # =====================================================================
    # HTML Pages
    # =====================================================================

    @app.get("/", response_class=HTMLResponse)
    async def page_overview(request: Request) -> HTMLResponse:
        status = _state.get_status()
        positions = _state.get_positions()
        history = _state.get_history(limit=5)
        return _render(
            request, "overview.html", "overview",
            positions=positions,
            history=history,
            equity=status.get("equity", 0),
            day_pnl=status.get("day_pnl", 0),
            day_pnl_pct=status.get("day_pnl_pct", 0),
            open_count=status.get("open_positions", 0),
            win_rate=58.0,  # placeholder
        )

    @app.get("/positions", response_class=HTMLResponse)
    async def page_positions(request: Request) -> HTMLResponse:
        positions = _state.get_positions()
        return _render(request, "positions.html", "positions", positions=positions)

    @app.get("/history", response_class=HTMLResponse)
    async def page_history(request: Request) -> HTMLResponse:
        history = _state.get_history(limit=20)
        return _render(request, "history.html", "history", history=history)

    @app.get("/signals", response_class=HTMLResponse)
    async def page_signals(request: Request) -> HTMLResponse:
        signals = _state.get_signals(limit=10)
        return _render(request, "signals.html", "signals", signals=signals)

    @app.get("/risk", response_class=HTMLResponse)
    async def page_risk(request: Request) -> HTMLResponse:
        risk = _state.get_risk()
        return _render(request, "risk.html", "risk", risk=risk)

    @app.get("/config", response_class=HTMLResponse)
    async def page_config(request: Request) -> HTMLResponse:
        config = _state.get_config()
        return _render(request, "config.html", "config", config=config)

    @app.get("/logs", response_class=HTMLResponse)
    async def page_logs(request: Request) -> HTMLResponse:
        logs = _state.get_logs(limit=100)
        return _render(request, "logs.html", "logs", logs=logs)

    # =====================================================================
    # API Endpoints
    # =====================================================================

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(_state.get_status())

    @app.get("/api/positions")
    async def api_positions() -> JSONResponse:
        return JSONResponse({"positions": _state.get_positions()})

    @app.get("/api/history")
    async def api_history(limit: int = 20) -> JSONResponse:
        return JSONResponse({"history": _state.get_history(limit)})

    @app.get("/api/signals")
    async def api_signals(limit: int = 10) -> JSONResponse:
        return JSONResponse({"signals": _state.get_signals(limit)})

    @app.get("/api/risk")
    async def api_risk() -> JSONResponse:
        return JSONResponse(_state.get_risk())

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        return JSONResponse(_state.get_config())

    @app.post("/api/config")
    async def api_config_update(request: Request) -> JSONResponse:
        data = await request.json()
        key = data.get("key", "")
        value = data.get("value", {})
        success = _state.update_config(key, value)
        return JSONResponse({"success": success})

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        # In production, this would call engine.emergency_stop()
        logger.critical("Emergency stop requested via web UI")
        return JSONResponse({"success": True, "message": "Emergency stop triggered"})

    return app
