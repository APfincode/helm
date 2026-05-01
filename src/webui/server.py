"""Helm Web UI Server — FastAPI + Jinja2 dark-theme dashboard.

Connects to the live `StateDB` so data is real (not placeholders).
No build step, no bundler, no React. Pure Python + vanilla JS.

Routes:
  GET /              → Overview
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
  GET /api/events    → SSE stub (live updates)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

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

try:
    from src.state.db import get_state_db, StateDB
except ImportError:
    get_state_db = None  # type: ignore
    StateDB = None  # type: ignore


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
    """Reads/writes live bot state from the shared SQLite database."""

    _instance: Optional[BotStateProvider] = None

    def __new__(cls, *args, **kwargs):
        """Singleton so the UI and engine share one cache handle."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._db: Optional["StateDB"] = None
        return cls._instance

    def connect(self) -> None:
        """Wire to the live StateDB.  Safe to call multiple times."""
        if self._db is None and get_state_db is not None:
            self._db = get_state_db()

    def connected(self) -> bool:
        return self._db is not None

    # ------------------------------------------------------------------
    # Read API (delegating to StateDB)
    # ------------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        if self._db:
            return self._db.get_status()
        return {}

    def get_positions(self) -> list[dict[str, Any]]:
        if self._db:
            try:
                return self._db.get_positions(open_only=True)
            except Exception:
                return []
        return []

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        if self._db:
            try:
                return self._db.get_history(limit)
            except Exception:
                return []
        return []

    def get_signals(self, limit: int = 10) -> list[dict[str, Any]]:
        if self._db:
            try:
                return self._db.get_signals(limit)
            except Exception:
                return []
        return []

    def get_risk(self) -> dict[str, Any]:
        if self._db:
            try:
                return self._db.get_risk()
            except Exception:
                return {"halted": False, "circuit_breakers": []}
        return {"halted": False, "circuit_breakers": []}

    def get_config(self) -> dict[str, Any]:
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
        if self._db:
            try:
                return self._db.get_logs(limit)
            except Exception:
                return ["No logs available."]
        return ["Helm not running — no logs yet."]


# Global singleton ─ lazily connected in create_app()
_state = BotStateProvider()


def create_app(
    on_stop: Optional[Callable[[], None]] = None,
    on_status: Optional[Callable[[], dict]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    _state.connect()
    app = FastAPI(title="Helm Dashboard", version="0.1.0")

    # Static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Templates
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    def _render(request: Request, template: str, active_tab: str, **ctx) -> HTMLResponse:
        """Helper to render a template with common context."""
        status = _state.get_status() or {}
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
        status = _state.get_status() or {}
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
            win_rate=status.get("win_rate", 0.0),
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
        status = on_status() if on_status else _state.get_status()
        return JSONResponse(status or {})

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
        """Emergency stop endpoint: invoke provided callback."""
        logger.critical("Emergency stop triggered via web UI")
        if on_stop:
            await asyncio.to_thread(on_stop) if not asyncio.iscoroutinefunction(on_stop) else on_stop()
        return JSONResponse({"success": True, "message": "Emergency stop triggered"})

    return app
