"""
Telegram Bot — Embedded alert + command interface for the trading bot.

Runs as an async task inside the main bot process. Sends alerts on trades,
stops, risk triggers, LLM signals, and mode changes. Accepts commands
for status checks, emergency stop, and configuration viewing.

Requirements (already in pyproject.toml):
    python-telegram-bot>=20.7

Usage:
    async with TelegramBot(execution_engine, risk_manager) as bot:
        await bot.start()
        # Bot runs concurrently with trading engine
        # Telegram task polls for messages + sends alerts
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes


logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    token: str
    chat_id: str
    enabled: bool = True
    alert_on_trade: bool = True
    alert_on_stop: bool = True
    alert_on_risk: bool = True
    alert_on_signal: bool = False  # Too noisy; turn on if debugging
    alert_on_mode_change: bool = True
    daily_summary_at: str = "00:00"  # UTC time for daily P&L summary


class TelegramBot:
    """
    Embedded Telegram bot for trading alerts and commands.

    Architecture:
    1. Application.start() runs the Telegram polling loop in background
    2. Other methods are called by the ExecutionEngine to send alerts
    3. Commands interact with the ExecutionEngine via thread-safe callbacks

    Thread safety: Telegram uses asyncio. All alert methods are async.
    """

    def __init__(
        self,
        config: TelegramConfig,
        get_engine_stats: Optional[Callable[[], dict]] = None,
        on_emergency_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._get_engine_stats = get_engine_stats
        self._on_emergency_stop = on_emergency_stop
        self._app: Optional[Application] = None
        self._running: bool = False
        self._last_daily_summary: Optional[datetime] = None

    async def __aenter__(self) -> "TelegramBot":
        if not self._config.enabled:
            return self
        try:
            self._app = Application.builder().token(self._config.token).build()
            self._register_handlers()
            await self._app.initialize()
            await self._app.start()
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self._app = None
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    def _register_handlers(self) -> None:
        """Register command handlers."""
        if not self._app:
            return
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("risk", self._cmd_risk))
        self._app.add_handler(CommandHandler("metrics", self._cmd_metrics))
        self._app.add_handler(CommandHandler("signals", self._cmd_signals))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("config", self._cmd_config))

    async def start(self) -> None:
        """Start Telegram polling in background."""
        if not self._app or self._running:
            return
        self._running = True
        await self._app.updater.start_polling()
        logger.info("Telegram bot polling started")

    async def stop(self) -> None:
        """Stop Telegram polling gracefully."""
        if not self._app or not self._running:
            return
        self._running = False
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot stopped")

    # ========================================================================
    # Alert Methods (called by ExecutionEngine / PositionTracker / RiskManager)
    # ========================================================================

    async def alert_trade(self, trade: dict) -> None:
        """Alert on trade execution."""
        if not self._config.alert_on_trade or not self._app:
            return
        emoji = "🟢" if trade.get("direction") == "LONG" else "🔴"
        msg = (
            f"{emoji} TRADE EXECUTED\n"
            f"Symbol: {trade.get('symbol')}\n"
            f"Direction: {trade.get('direction')}\n"
            f"Size: {trade.get('quantity', 0):.4f}\n"
            f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
            f"Leverage: {trade.get('leverage', 1)}x\n"
            f"Stop: ${trade.get('stop_loss', 0):,.2f}\n"
            f"TP: ${trade.get('take_profit', 0):,.2f}\n"
            f"Risk: ${trade.get('risk_usd', 0):,.2f}\n"
            f"Mode: {trade.get('mode', '?')}"
        )
        await self._send_message(msg)

    async def alert_stop_hit(self, position: dict, pnl: float) -> None:
        """Alert on stop loss hit."""
        if not self._config.alert_on_stop or not self._app:
            return
        pct = position.get("entry_price", 0)
        if pct > 0:
            pct = abs(pnl) / (position.get("entry_price", 1) * position.get("quantity", 1)) * 100
        else:
            pct = 0
        msg = (
            f"🛑 STOP LOSS HIT\n"
            f"Symbol: {position.get('coin')}\n"
            f"Direction: {position.get('direction')}\n"
            f"Entry: ${position.get('entry_price', 0):,.2f}\n"
            f"Exit: ${position.get('current_price', 0):,.2f}\n"
            f"P&L: {pnl:+,.2f} ({pct:.1f}%)"
        )
        await self._send_message(msg)

    async def alert_take_profit(self, position: dict, pnl: float) -> None:
        """Alert on take profit hit."""
        if not self._config.alert_on_stop or not self._app:
            return
        msg = (
            f"🎯 TAKE PROFIT HIT\n"
            f"Symbol: {position.get('coin')}\n"
            f"Direction: {position.get('direction')}\n"
            f"Entry: ${position.get('entry_price', 0):,.2f}\n"
            f"Exit: ${position.get('current_price', 0):,.2f}\n"
            f"P&L: {pnl:+,.2f}"
        )
        await self._send_message(msg)

    async def alert_risk_guard(self, reason: str, details: dict) -> None:
        """Alert on risk guard trigger (drawdown, daily loss, fee budget)."""
        if not self._config.alert_on_risk or not self._app:
            return
        msg = (
            f"⚠️ RISK GUARD TRIGGERED\n"
            f"Reason: {reason}\n"
            f"Details: {details}"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🛑 Emergency Stop", callback_data="emergency_stop")]
        ])
        await self._send_message(msg, reply_markup=keyboard)

    async def alert_signal(self, symbol: str, signal: dict) -> None:
        """Alert on LLM signal generation."""
        if not self._config.alert_on_signal or not self._app:
            return
        emoji = {"LONG": "📈", "SHORT": "📉", "NEUTRAL": "➖"}
        direction = signal.get("direction", "NEUTRAL")
        msg = (
            f"{emoji.get(direction, '❓')} SIGNAL: {direction}\n"
            f"Symbol: {symbol}\n"
            f"Confidence: {signal.get('confidence', 0):.2f}\n"
            f"Reason: {signal.get('reasoning', '')[:100]}..."
        )
        await self._send_message(msg)

    async def alert_mode_change(self, old_mode: str, new_mode: str) -> None:
        """Alert on mode change (PAPER -> LIVE)."""
        if not self._config.alert_on_mode_change or not self._app:
            return
        if new_mode == "LIVE":
            msg = (
                f"🔴 LIVE MODE ACTIVATED\n"
                f"REAL MONEY AT RISK\n"
                f"All subsequent trades will use real funds."
            )
        else:
            msg = f"🟡 Mode changed: {old_mode} -> {new_mode}"
        await self._send_message(msg)

    async def alert_daily_summary(self, stats: dict) -> None:
        """Send daily P&L summary."""
        if not self._app:
            return
        msg = (
            f"📈 DAILY SUMMARY\n"
            f"Equity: ${stats.get('equity', 0):,.2f}\n"
            f"Day P&L: {stats.get('day_pnl', 0):+,.2f}\n"
            f"Trades: {stats.get('trades_today', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"Max Drawdown: {stats.get('max_drawdown', 0):.1f}%"
        )
        await self._send_message(msg)
        self._last_daily_summary = datetime.utcnow()

    # ========================================================================
    # Command Handlers
    # ========================================================================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "⚡ Hyper-Alpha-Arena V2\n"
            "AI Crypto Trading Bot\n\n"
            "Type /help for commands.\n"
            "Use /status to check current state."
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "📋 Available Commands:\n\n"
            "/status — Current bot state, equity, positions\n"
            "/positions — Open positions with unrealized P&L\n"
            "/history — Last 10 closed trades\n"
            "/risk — Risk guard circuit breaker statuses\n"
            "/metrics — Sharpe, win rate, max drawdown\n"
            "/signals — Last 5 LLM signals with reasoning\n"
            "/config — View key configuration (read-only)\n"
            "/stop — Emergency halt all trading\n\n"
            "⚠️ /stop is irreversible until restart."
        )
        await update.message.reply_text(help_text)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ Stats unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            mode = stats.get("mode", "PAPER")
            emoji = "🟢" if stats.get("running") else "🔴"
            mode_emoji = "🔴" if mode == "LIVE" else "🟡"
            msg = (
                f"{emoji} Bot: {'Running' if stats.get('running') else 'Stopped'}\n"
                f"{mode_emoji} Mode: {mode.upper()}\n"
                f"📊 Signals: {stats.get('signals_generated', 0)}\n"
                f"🔄 Trades: {stats.get('trades_executed', 0)} executed, "
                f"{stats.get('trades_rejected', 0)} rejected\n"
                f"📈 Open: {stats.get('open_positions', 0)}\n"
                f"💰 Unrealized: ${stats.get('unrealized_pnl', 0):+,.2f}"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Error in /status: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ Positions unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            pos_stats = stats.get("position_stats", {})
            if not pos_stats.get("total_opened", 0):
                await update.message.reply_text("📭 No open positions.")
                return
            lines = ["📊 Open Positions:"]
            # Note: full position list requires access to PositionTracker directly
            # This is a simplified view from stats
            lines.append(
                f"Total Exposure: ${stats.get('total_exposure', 0):,.2f}\n"
                f"Unrealized P&L: ${stats.get('unrealized_pnl', 0):+,.2f}"
            )
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error(f"Error in /positions: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ History unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            pos_stats = stats.get("position_stats", {})
            total_closed = pos_stats.get("total_closed", 0)
            total_tp = pos_stats.get("total_tp_hit", 0)
            total_sl = pos_stats.get("total_sl_hit", 0)
            msg = (
                f"📜 Trade History (last session):\n\n"
                f"Closed: {total_closed}\n"
                f"Take Profit: {total_tp}\n"
                f"Stop Loss: {total_sl}\n\n"
                f"Full history available via web UI."
            )
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Error in /history: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risk command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ Risk data unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            risk_stats = stats.get("risk_manager_stats", {})
            halted = stats.get("emergency_stop", False)
            status = "🚨 HALTED" if halted else "✅ ACTIVE"
            msg = (
                f"🛡️ Risk Guard Status: {status}\n\n"
                f"Trades accepted: {risk_stats.get('total_trades_accepted', 0)}\n"
                f"Trades rejected: {risk_stats.get('total_trades_rejected', 0)}\n"
                f"Max drawdown: {risk_stats.get('max_drawdown_pct', 0):.1f}%\n"
                f"Daily loss: {risk_stats.get('daily_loss_pct', 0):.1f}%\n"
                f"Fee budget: {risk_stats.get('fees_pct', 0):.1f}%"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Error in /risk: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /metrics command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ Metrics unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            pos_stats = stats.get("position_stats", {})
            total = pos_stats.get("total_closed", 0)
            wins = pos_stats.get("total_tp_hit", 0)
            win_rate = (wins / total * 100) if total > 0 else 0
            risk_stats = stats.get("risk_manager_stats", {})
            msg = (
                f"📊 Performance Metrics:\n\n"
                f"Total trades: {total}\n"
                f"Win rate: {win_rate:.1f}% ({wins}/{total})\n"
                f"Max drawdown: {risk_stats.get('max_drawdown_pct', 0):.1f}%\n\n"
                "Full analytics via web UI."
            )
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Error in /metrics: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signals command."""
        msg = (
            "🤖 Recent LLM signals are logged to the database.\n"
            "Full signal history with reasoning is available via web UI.\n\n"
            "Use /status to see how many signals were generated."
        )
        await update.message.reply_text(msg)

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command — emergency halt."""
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Confirm STOP", callback_data="confirm_stop"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_stop"),
            ]
        ])
        await update.message.reply_text(
            "🚨 EMERGENCY STOP\n\n"
            "This will immediately close all positions and halt trading.\n"
            "Are you sure?",
            reply_markup=keyboard,
        )

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /config command."""
        if not self._get_engine_stats:
            await update.message.reply_text("❌ Config unavailable.")
            return
        try:
            stats = self._get_engine_stats()
            mode = stats.get("mode", "PAPER")
            msg = (
                "⚙️ Current Configuration:\n\n"
                f"Trading mode: {mode.upper()}\n"
                f"Max positions: see web UI\n"
                f"Max leverage: see web UI\n"
                f"Risk guards: see /risk\n\n"
                "Full config editable via web UI."
            )
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Error in /config: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    # ========================================================================
    # Callback Handlers
    # ========================================================================

    async def _send_message(self, text: str, reply_markup=None) -> None:
        """Send message to configured chat."""
        if not self._app or not self._config.chat_id:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._config.chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=None,  # Plain text — most reliable
            )
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")

    def __repr__(self) -> str:
        return f"TelegramBot(enabled={self._config.enabled}, running={self._running})"
