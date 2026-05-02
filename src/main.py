"""
Helm — Integrated main orchestrator.

Launches the full trading stack with a single command:
    python -m src.main --mode paper
    python -m src.main --mode backtest
    python -m src.main --mode live --confirm-live

Coordinated components (concurrent async tasks):
1. ExecutionEngine — main trading loop
2. TelegramBot (optional) — alerts + commands
3. WebUI Server (optional) — FastAPI dashboard

Graceful shutdown via asyncio.Event — closes positions, flushes audit trail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env BEFORE any project imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Inject src into PYTHONPATH at runtime
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.loader import ConfigLoader, ConfigLoadError
from src.execution.execution_engine import ExecutionEngine, ExecutionConfig, ExecutionMode
from src.llm.signal_generator import SignalGenerator
from src.risk.manager import RiskManager
from src.data.fetcher import DataFetcher
from src.security.secrets_manager import get_secrets_manager, SecretAccessError
from src.security.audit_logger import AuditLogger, EventType, Severity
from src.telegram.bot import TelegramBot, TelegramConfig
from src.state.db import get_state_db, StateDB


try:
    from src.webui.server import create_app
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False


logger = logging.getLogger("helm")


# =============================================================================
# Configurable constants
# =============================================================================
SIGNAL_INTERVAL_MINUTES = 60     # Check for signals every hour
RISK_CHECK_INTERVAL_S = 30     # Check stops / P&L every 30s
PAPER_INITIAL_EQUITY = 10000.0  # Starting capital for paper mode

DEFAULT_SYMBOLS = ["BTC", "ETH"]


# =============================================================================
# Helpers
# =============================================================================
def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/helm.log", encoding="utf-8"),
        ],
    )


def get_hardware_pid() -> str:
    """Return approximate machine identifier (for BASIC state-machine feel)."""
    return f"{datetime.now().strftime('%H:%M')}"


def _build_stats(engine: ExecutionEngine, engine_name: str = "Helm") -> dict:
    """Return a JSON-safe snapshot for Telegram / Web UI."""
    return {
        "mode":        engine._config.mode.value,
        "running":     engine._running,
        "emergency":   engine._emergency_stop,
        "signals":     engine._signal_count,
        "trades":      engine._trade_count,
        "rejected":    engine._reject_count,
        "positions":   len(engine._tracker.get_all_open()),
        "equity":      0.0,  # populated asynchronously later
        "pid":         get_hardware_pid(),
    }


# =============================================================================
# Telegram helpers (initialised only if token present)
# =============================================================================
def _init_telegram(secret_mgr, on_stats, on_stop) -> Optional[TelegramBot]:
    token = secret_mgr.get("TELEGRAM_BOT_TOKEN")
    chat  = secret_mgr.get("TELEGRAM_CHAT_ID")
    if not token:
        return None
    config = TelegramConfig(
        token=token.raw if hasattr(token, "raw") else str(token),
        chat_id=chat.raw if hasattr(chat, "raw") else str(chat),
        enabled=True,
        alert_on_trade=True,
        alert_on_stop=True,
        alert_on_risk=True,
        alert_on_mode_change=True,
    )
    return TelegramBot(
        config=config,
        get_engine_stats=lambda: on_stats(),
        on_emergency_stop=lambda: on_stop(),
    )


# =============================================================================
# Web UI helper
# =============================================================================
def _launch_webui(stop_event: threading.Event, on_stats, port: int = 8080) -> None:
    """Run Uvicorn in a daemon thread so the trading engine isn't blocked."""
    import uvicorn
    app = create_app(
        on_stop=lambda: stop_event.set(),
        on_status=on_stats,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)


# =============================================================================
# Main orchestrator
# =============================================================================
async def main(mode: str, confirm_live: bool = False,
               symbols: Optional[list[str]] = None,
               enable_telegram: bool = True,
               enable_webui: bool = True,
               webui_port: int = 8080,
               ) -> int:
    """
    Bootstrap and run Helm.

    Returns:  exit code (0 = clean, 1 = error)
    """
    print("=" * 60)
    print(f"{'  HELM  ':=^60}")
    print("  AI-Driven Perpetual Futures Trading Bot")
    print("  https://github.com/APfincode/helm")
    print("=" * 60)

    # 0. Logging & directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    setup_logging()

    # --- Phase 0: Secrets & config --------------------------------------
    try:
        secrets = get_secrets_manager()
        secrets.verify_all_required(["HYPERLIQUID_WALLET",
                                     "HYPERLIQUID_PRIVATE_KEY",
                                     "HMAC_SECRET_KEY"])
        logger.info("✅ Secrets verified")
    except SecretAccessError as e:
        logger.error(f"❌ Secret error: {e}")
        print("\nPlease copy .env.example to .env and fill in your credentials.")
        return 1

    try:
        loader = ConfigLoader()
        _ = loader.load()   # returns SecureConfig; kept for future use
        logger.info("✅ Config loaded")
    except ConfigLoadError as e:
        logger.warning(f"Config warning (non-fatal): {e}")

    # --- Phase 1: Initialise StateDB ------------------------------------
    state_db = get_state_db()
    state_db.update_status(
        mode="PAPER",
        running=False,
        equity=PAPER_INITIAL_EQUITY,
        day_pnl=0.0,
        day_pnl_pct=0.0,
        open_positions=0,
        last_error="",
    )
    logger.info("✅ StateDB ready")

    # --- Phase 2: Choose mode -------------------------------------------
    exec_mode = ExecutionMode.PAPER
    if mode.lower() == "live":
        if confirm_live:
            exec_mode = ExecutionMode.LIVE
            logger.critical("🚨 LIVE MODE CONFIRMED — REAL MONEY AT RISK")
        else:
            logger.warning("LIVE requested but --confirm-live absent; falling back to PAPER")
    elif mode.lower() == "backtest":
        exec_mode = ExecutionMode.BACKTEST
        logger.info("📊 Backtest mode selected")

    # --- Phase 3: Build components --------------------------------------
    exec_cfg = ExecutionConfig(
        mode=exec_mode,
        symbols=symbols or DEFAULT_SYMBOLS,
        signal_interval_minutes=SIGNAL_INTERVAL_MINUTES,
        risk_check_interval_seconds=RISK_CHECK_INTERVAL_S,
        paper_initial_equity=PAPER_INITIAL_EQUITY,
        require_live_confirmation=not confirm_live,
    )

    sig_gen = SignalGenerator(template_name="basic_signal")
    risk_mgr = RiskManager()
    audit = AuditLogger()
    await audit.__aenter__()

    # Telegram (best-effort)
    tgbot: Optional[TelegramBot] = None
    if enable_telegram:
        try:
            tgbot = _init_telegram(secrets, on_stats=dict, on_stop=lambda: None)
            if tgbot:
                await tgbot.__aenter__()
                await tgbot.start()
                logger.info("✅ Telegram bot polling")
        except Exception as e:
            logger.warning(f"Telegram init failed (continuing): {e}")

    # --- Phase 4: Engine assembly ---------------------------------------
    engine = ExecutionEngine(
        config=exec_cfg,
        signal_generator=sig_gen,
        risk_manager=risk_mgr,
        audit_logger=audit,
        telegram_bot=tgbot,
    )
    await engine.init()

    # Refresh UI stats function bound to the live engine
    def _ui_stats():
        return _build_stats(engine)

    # --- Phase 5: Web UI ------------------------------------------------
    webui_thread: Optional[threading.Thread] = None
    stop_event = threading.Event()
    if enable_webui and HAS_UVICORN:
        webui_thread = threading.Thread(
            target=_launch_webui,
            args=(stop_event, _ui_stats, webui_port),
            daemon=True,
            name="HelmWebUI",
        )
        webui_thread.start()
        logger.info(f"🔗 Web UI http://localhost:{webui_port}")
    elif enable_webui:
        logger.info("🪓 Web UI disabled (uvicorn missing)")
    else:
        logger.info("🔗 Web UI disabled by flag")

    # --- Phase 6: Graceful shutdown hooks ------------------------------
    shutdown_requested = asyncio.Event()

    async def _on_signal(_s=None):
        logger.info("SIGINT / SIGTERM received")
        shutdown_requested.set()
        await engine.stop()

    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda _s=s: asyncio.create_task(_on_signal(_s)))
        except AttributeError:
            pass   # Windows — graceful shutdown via KeyboardInterrupt

    # --- Phase 7: Launch trading loop -----------------------------------
    state_db.update_status(mode=exec_mode.value, running=True)

    try:
        logger.info(f"🚀 Helm started | {exec_mode.value.upper()} | symbols: {exec_cfg.symbols}")
        # The engine runs its own loops; we wait until shutdown.
        engine_task = asyncio.create_task(engine.start())
        await shutdown_requested.wait()
        logger.info("Shutting down …")
        await engine.stop()
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping")
        await engine.stop()
    except Exception as e:
        logger.exception("Critical engine failure")
        state_db.update_status(running=False, last_error=str(e)[:500])
        return 1

    # --- Phase 8: Cleanup ----------------------------------------------
    state_db.update_status(running=False)
    if tgbot:
        try:
            await tgbot.stop()
            await tgbot.__aexit__(None, None, None)
        except Exception:
            pass
    await audit.__aexit__(None, None, None)
    await sig_gen.close()

    logger.info("✅ Helm stopped cleanly")
    print("\n\nHelm stopped. See logs/helm.log for details.")
    return 0


# =============================================================================
# CLI entry point
# =============================================================================
def cli():
    parser = argparse.ArgumentParser(
        prog="helm",
        description="AI-driven perpetual futures trading bot",
    )
    parser.add_argument(
        "--mode", default="paper", choices=["paper", "live", "backtest"],
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Must be set explicitly for LIVE mode",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to trade (default: BTC ETH)",
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Skip Telegram bot",
    )
    parser.add_argument(
        "--no-webui", action="store_true",
        help="Skip Web UI server",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Web UI port (default: 8080)",
    )
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(main(
            mode=args.mode,
            confirm_live=args.confirm_live,
            symbols=args.symbols,
            enable_telegram=not args.no_telegram,
            enable_webui=not args.no_webui,
            webui_port=args.port,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
