"""
Execution Engine — Main orchestrator for live trading.

Ties together:
- RiskManager (Phase 5): Validates and sizes positions
- SignalGenerator (Phase 3): LLM signal source
- HyperliquidExecutor: Live exchange execution
- PaperTradingExecutor: Paper trading simulation
- PositionTracker: Tracks positions and stop/take-profit

Alpha Arena Lesson: Bots that traded every 2-3 minutes died from fees.
Our engine trades on SIGNALS, not on a fixed schedule.
Only trades when LLM generates a non-NEUTRAL signal.

Modes:
- PAPER: Paper trading for validation (default)
- LIVE: Real money execution (requires explicit activation)
- BACKTEST: Use backtest engine (Phase 2)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Literal

from src.llm.client import Signal, LLMClient
from src.llm.signal_generator import SignalGenerator
from src.risk.manager import RiskManager, RiskDecision
from src.data.fetcher import DataFetcher
from src.execution.hyperliquid_executor import HyperliquidExecutor, ExecutorConfig, OrderResult
from src.execution.paper_trading import PaperTradingExecutor
from src.execution.position_tracker import PositionTracker, PositionState, PositionStatus
from src.security.audit_logger import AuditLogger, EventType, Severity
from src.telegram.bot import TelegramBot as _TelegramBot, TelegramConfig as _TelegramConfig
TelegramBot = _TelegramBot


logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    PAPER = "paper"       # Simulated execution
    LIVE = "live"         # Real money (requires explicit activation)
    BACKTEST = "backtest" # Use backtest engine


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    mode: ExecutionMode = ExecutionMode.PAPER
    
    # Symbols to trade
    symbols: list[str] = None
    primary_timeframe: str = "1h"
    
    # Signal generation
    signal_interval_minutes: int = 60  # Check for signals every N minutes
    
    # Risk management
    risk_check_interval_seconds: int = 30  # Check stops every 30s
    
    # Paper trading
    paper_initial_equity: float = 10000.0
    
    # Activation
    require_live_confirmation: bool = True  # Must explicitly confirm LIVE mode
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC", "ETH"]


# =============================================================================
# Execution Engine
# =============================================================================

class ExecutionEngine:
    """
    Main live trading engine.
    
    Lifecycle:
    1. init() — initialize all components
    2. start() — begin trading loop
    3. On each tick:
       a. Fetch market data
       b. Generate signal (LLM)
       c. Validate signal (RiskManager)
       d. Execute trade (if approved)
       e. Update positions (check stops)
    4. stop() — gracefully shutdown
    
    Safety:
    - Always starts in PAPER mode
    - LIVE mode requires explicit confirmation
    - All trades logged to audit trail
    - Emergency stop available
    """
    
    def __init__(
        self,
        config: ExecutionConfig,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        audit_logger: Optional[AuditLogger] = None,
        telegram_bot: Optional[TelegramBot] = None,
    ):
        self._config = config
        self._signal_generator = signal_generator
        self._risk_manager = risk_manager
        self._audit = audit_logger
        self._telegram = telegram_bot
        
        # Components (initialized in init())
        self._executor: Optional[HyperliquidExecutor | PaperTradingExecutor] = None
        self._fetcher: Optional[DataFetcher] = None
        self._tracker = PositionTracker()
        
        # State
        self._running: bool = False
        self._shutdown_event = asyncio.Event()
        self._emergency_stop: bool = False
        
        # Stats
        self._signal_count: int = 0
        self._trade_count: int = 0
        self._reject_count: int = 0
    
    async def init(self) -> "ExecutionEngine":
        """Initialize all components."""
        logger.info(f"Initializing execution engine in {self._config.mode.value} mode")
        
        # Validate mode
        if self._config.mode == ExecutionMode.LIVE and self._config.require_live_confirmation:
            logger.warning("LIVE mode requested but not confirmed — falling back to PAPER")
            self._config.mode = ExecutionMode.PAPER
        
        # Initialize executor
        if self._config.mode == ExecutionMode.PAPER:
            self._executor = PaperTradingExecutor(self._config.paper_initial_equity)
            logger.info(f"Paper trading with ${self._config.paper_initial_equity:.2f}")
        elif self._config.mode == ExecutionMode.LIVE:
            self._executor = HyperliquidExecutor(ExecutorConfig())
            logger.warning("LIVE TRADING ACTIVATED — REAL MONEY AT RISK")
        
        # Initialize data fetcher
        self._fetcher = DataFetcher()
        await self._fetcher.__aenter__()
        
        # Initialize signal generator
        await self._signal_generator.init()
        
        # Initialize executor (if async)
        if hasattr(self._executor, '__aenter__'):
            await self._executor.__aenter__()
        
        logger.info("Execution engine initialized")
        return self
    
    async def start(self) -> None:
        """Start the main trading loop."""
        if self._running:
            logger.warning("Engine already running")
            return
        
        self._running = True
        logger.info("Trading loop started")
        
        # Start concurrent tasks
        signal_task = asyncio.create_task(self._signal_loop())
        risk_task = asyncio.create_task(self._risk_check_loop())
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cancel tasks
        signal_task.cancel()
        risk_task.cancel()
        
        try:
            await signal_task
            await risk_task
        except asyncio.CancelledError:
            pass
        
        self._running = False
        logger.info("Trading loop stopped")
    
    async def stop(self) -> None:
        """Gracefully stop the trading loop."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()
        
        # Close positions (reduce-only)
        await self._close_all_positions(reason="shutdown")
    
    async def emergency_stop(self) -> None:
        """Emergency stop — close all positions immediately."""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self._emergency_stop = True
        self._shutdown_event.set()
        
        await self._close_all_positions(reason="emergency_stop")
        
        if self._telegram:
            await self._telegram.alert_risk_guard(
                "Emergency stop triggered manually",
                {"reason": "Manual emergency stop", "equity": await self._get_account_equity()},
            )
        
        if self._audit:
            await self._audit.log(
                event_type=EventType.SYSTEM,
                action="EMERGENCY_STOP",
                actor="system",
                severity=Severity.CRITICAL,
                details={"reason": "Manual emergency stop triggered"},
            )
    
    # ========================================================================
    # Main Loops
    # ========================================================================
    
    async def _signal_loop(self) -> None:
        """
        Main signal generation and execution loop.
        
        Runs every signal_interval_minutes.
        """
        while self._running and not self._emergency_stop:
            try:
                await self._evaluate_signals()
            except Exception as e:
                logger.exception("Error in signal loop")
                await asyncio.sleep(60)  # Wait a minute on error
                continue
            
            # Wait for next cycle
            await asyncio.sleep(self._config.signal_interval_minutes * 60)
    
    async def _risk_check_loop(self) -> None:
        """
        Risk monitoring loop.
        
        Checks stops and updates P&L more frequently than signal loop.
        """
        while self._running and not self._emergency_stop:
            try:
                await self._check_risk_conditions()
            except Exception as e:
                logger.exception("Error in risk check loop")
            
            await asyncio.sleep(self._config.risk_check_interval_seconds)
    
    # ========================================================================
    # Signal Evaluation
    # ========================================================================
    
    async def _evaluate_signals(self) -> None:
        """Evaluate signals for all configured symbols."""
        for symbol in self._config.symbols:
            if self._emergency_stop:
                break
            
            try:
                await self._evaluate_symbol(symbol)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
    
    async def _evaluate_symbol(self, symbol: str) -> None:
        """
        Evaluate a single symbol:
        1. Fetch market data
        2. Generate LLM signal
        3. Validate via RiskManager
        4. Execute if approved
        """
        # 1. Fetch data
        try:
            data = await self._fetcher.get_ohlcv(
                symbol=symbol,
                timeframe=self._config.primary_timeframe,
                use_cache=True,
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return
        
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            return
        
        # 2. Generate signal
        try:
            signal = await self._signal_generator.generate(
                data, symbol=symbol
            )
            self._signal_count += 1
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return
        
        logger.info(f"Signal: {symbol} → {signal.direction} (conf: {signal.confidence:.2f})")
        
        # Skip NEUTRAL
        if signal.direction == "NEUTRAL":
            return
        
        # 3. Get current price for execution
        current_price = float(data["close"].iloc[-1])
        
        # 4. Get account equity
        account_equity = await self._get_account_equity()
        
        # 5. Build trade history for risk manager
        trade_history = [
            {"pnl": pos.realized_pnl}
            for pos in self._tracker._history[-20:]  # Last 20 trades
        ]
        
        # 6. Validate via RiskManager
        open_positions = [pos.to_dict() for pos in self._tracker.get_all_open()]
        
        decision = self._risk_manager.validate(
            signal=signal,
            account_equity=account_equity,
            entry_price=current_price,
            market_data=data,
            open_positions=open_positions,
            trade_history=trade_history,
        )
        
        if not decision.approved:
            self._reject_count += 1
            logger.info(f"Trade REJECTED: {decision.rejection_reason}")
            
            if self._audit:
                await self._audit.log(
                    event_type=EventType.SYSTEM,
                    action="TRADE_REJECTED",
                    actor="risk_manager",
                    severity=Severity.INFO,
                    details={
                        "symbol": symbol,
                        "direction": signal.direction,
                        "reason": decision.rejection_reason,
                        "confidence": signal.confidence,
                    },
                )
            return
        
        # 7. Execute trade
        await self._execute_trade(symbol, decision)
    
    # ========================================================================
    # Trade Execution
    # ========================================================================
    
    async def _execute_trade(self, symbol: str, decision: RiskDecision) -> None:
        """Execute an approved trade decision."""
        if not self._executor:
            logger.error("Executor not initialized")
            return
        
        # Determine reduce-only
        existing = self._tracker.get_positions_by_coin(symbol)
        reduce_only = len(existing) > 0 and any(
            p.direction != decision.direction for p in existing
        )
        
        try:
            # Place order
            result = await self._executor.place_market_order(
                coin=symbol,
                side=decision.direction,
                quantity=decision.quantity,
                reduce_only=reduce_only,
            )
            
            if result.success:
                self._trade_count += 1
                
                # Track position
                position = PositionState(
                    id=result.order_id or f"pos_{self._trade_count}",
                    coin=symbol,
                    direction=decision.direction,
                    entry_price=result.avg_fill_price or decision.entry_price,
                    entry_time=datetime.now(),
                    quantity=result.filled_quantity,
                    leverage=decision.leverage,
                    stop_loss_price=decision.stop_price,
                    take_profit_price=decision.take_profit_price,
                    risk_usd=decision.risk_usd,
                    margin_used=decision.margin_required,
                    is_paper=(self._config.mode == ExecutionMode.PAPER),
                )
                
                self._tracker.add_position(position)
                self._risk_manager.record_position_opened(position.to_dict())
                
                # Audit
                if self._audit:
                    await self._audit.log(
                        event_type=EventType.TRADE,
                        action=f"TRADE_{decision.direction}",
                        actor="execution_engine",
                        severity=Severity.INFO,
                        details={
                            "symbol": symbol,
                            "direction": decision.direction,
                            "quantity": decision.quantity,
                            "entry_price": result.avg_fill_price,
                            "stop_loss": decision.stop_price,
                            "take_profit": decision.take_profit_price,
                            "risk_usd": decision.risk_usd,
                            "leverage": decision.leverage,
                            "fee": result.fee_paid,
                            "mode": self._config.mode.value,
                        },
                    )
                
                logger.info(
                    f"Trade EXECUTED: {symbol} {decision.direction} "
                    f"{result.filled_quantity:.4f} @ {result.avg_fill_price:.2f}"
                )
                
                # Telegram alert
                if self._telegram:
                    await self._telegram.alert_trade({
                        "symbol": symbol,
                        "direction": decision.direction,
                        "quantity": result.filled_quantity,
                        "entry_price": result.avg_fill_price,
                        "leverage": decision.leverage,
                        "stop_loss": decision.stop_price,
                        "take_profit": decision.take_profit_price,
                        "risk_usd": decision.risk_usd,
                        "mode": self._config.mode.value,
                    })
                
            else:
                logger.error(f"Trade execution failed: {result.error_message}")
                
        except Exception as e:
            logger.exception("Trade execution error")
    
    # ========================================================================
    # Risk Checks
    # ========================================================================
    
    async def _check_risk_conditions(self) -> None:
        """Check stops and update P&L."""
        if not self._executor:
            return
        
        # Get current prices for all open positions
        open_positions = self._tracker.get_all_open()
        if not open_positions:
            return
        
        prices = {}
        for symbol in self._config.symbols:
            try:
                if hasattr(self._executor, 'get_current_price'):
                    price = await self._executor.get_current_price(symbol)
                else:
                    # For paper trading, use stored prices or fetch
                    price = await self._fetcher.get_ohlcv(
                        symbol=symbol, 
                        timeframe=self._config.primary_timeframe,
                        use_cache=True
                    )
                    price = float(price["close"].iloc[-1]) if not price.empty else 0.0
                
                if price > 0:
                    prices[symbol] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        
        # Update prices and check triggers
        triggers = self._tracker.update_prices(prices)
        
        # Execute stop/take-profit orders
        for trigger in triggers:
            await self._execute_close(
                position_id=trigger["position_id"],
                price=trigger["price"],
                reason=trigger["trigger"],
            )
        
        # Update RiskManager with closed positions
        for trigger in triggers:
            pos = self._tracker.get_position(trigger["position_id"])
            if pos and pos.status == PositionStatus.CLOSED:
                self._risk_manager.record_trade_result(
                    pnl=pos.realized_pnl,
                    fees=pos.fees_paid,
                    position_id=pos.id,
                )
    
    async def _execute_close(
        self,
        position_id: str,
        price: float,
        reason: str,
    ) -> None:
        """Close a position at given price."""
        pos = self._tracker.get_position(position_id)
        if not pos:
            return
        
        # Determine close side (opposite of position direction)
        close_side = "SHORT" if pos.direction == "LONG" else "LONG"
        
        try:
            if self._config.mode == ExecutionMode.PAPER:
                result = await self._executor.place_market_order(
                    coin=pos.coin,
                    side=close_side,
                    quantity=pos.quantity,
                    current_price=price,
                    reduce_only=True,
                )
            else:
                result = await self._executor.place_market_order(
                    coin=pos.coin,
                    side=close_side,
                    quantity=pos.quantity,
                    reduce_only=True,
                )
            
            if result.success:
                self._tracker.close_position(
                    position_id=position_id,
                    exit_price=result.avg_fill_price or price,
                    reason=reason,
                    fees=result.fee_paid,
                )
                
                logger.info(
                    f"Position closed: {pos.coin} {pos.direction} "
                    f"P&L: ${pos.realized_pnl:.2f} ({reason})"
                )
                
                # Telegram alert for stop/tp
                if self._telegram:
                    pos_dict = pos.to_dict()
                    if reason == "stop_loss":
                        await self._telegram.alert_stop_hit(pos_dict, pos.realized_pnl)
                    elif reason == "take_profit":
                        await self._telegram.alert_take_profit(pos_dict, pos.realized_pnl)
                
        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
    
    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        open_positions = self._tracker.get_all_open()
        
        for pos in open_positions:
            # Get current price
            try:
                if hasattr(self._executor, 'get_current_price'):
                    price = await self._executor.get_current_price(pos.coin)
                else:
                    price = pos.current_price or pos.entry_price
                
                await self._execute_close(pos.id, price, reason)
            except Exception as e:
                logger.error(f"Failed to close {pos.id}: {e}")
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    async def _get_account_equity(self) -> float:
        """Get current account equity."""
        try:
            if hasattr(self._executor, 'get_account_summary'):
                summary = await self._executor.get_account_summary()
                return summary.get("equity", 0.0)
            elif hasattr(self._executor, 'current_equity'):
                return self._executor.current_equity
            else:
                return 10000.0  # Fallback
        except Exception as e:
            logger.error(f"Failed to get equity: {e}")
            return 0.0
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def stats(self) -> dict:
        """Current trading statistics."""
        return {
            "mode": self._config.mode.value,
            "running": self._running,
            "emergency_stop": self._emergency_stop,
            "signals_generated": self._signal_count,
            "trades_executed": self._trade_count,
            "trades_rejected": self._reject_count,
            "open_positions": self._tracker.open_position_count,
            "total_exposure": self._tracker.total_exposure,
            "unrealized_pnl": self._tracker.total_unrealized_pnl,
            "risk_manager_stats": self._risk_manager.stats,
            "position_stats": self._tracker.stats,
        }
    
    def enable_live_mode(self, confirmed: bool = False) -> None:
        """Switch from PAPER to LIVE mode."""
        if not confirmed:
            raise ValueError("LIVE mode requires explicit confirmation (confirmed=True)")
        
        if self._running:
            raise RuntimeError("Cannot switch modes while running — stop first")
        
        old_mode = self._config.mode.value
        self._config.mode = ExecutionMode.LIVE
        self._config.require_live_confirmation = False
        logger.critical("LIVE MODE ENABLED — REAL MONEY AT RISK")
        
        # Fire-and-forget Telegram alert
        if self._telegram and self._telegram._app:
            asyncio.create_task(self._telegram.alert_mode_change(old_mode, "LIVE"))
    
    def __repr__(self) -> str:
        return f"ExecutionEngine(mode={self._config.mode.value}, running={self._running})"
