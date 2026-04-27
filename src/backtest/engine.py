"""
Backtest Engine — Event-driven backtesting with realistic execution.

Features:
- Event-driven loop (tick-by-tick simulation)
- Realistic fee modeling (Hyperliquid: 0.035% taker)
- Slippage estimation based on position size
- Position tracking with leverage
- P&L calculation (realized + unrealized)
- Result signing for tamper detection

Usage:
    from src.backtest.engine import BacktestEngine
    from src.backtest.models import BacktestConfig
    
    config = BacktestConfig(initial_capital=10000, taker_fee_pct=0.035)
    engine = BacktestEngine(config)
    
    # Synchronous:
    results = engine.run_sync(data, strategy_signals)
    
    # Async:
    results = await engine.run(data, strategy_signals)
    print(f"Return: {results.total_return_pct:.2f}%")
"""

import hmac
import hashlib
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Callable

import pandas as pd
import numpy as np

from .models import (
    BacktestConfig,
    BacktestResult,
    TradeRecord,
    Position,
    TradeDirection,
)
from .fees import HyperliquidFeeModel
from src.security.audit_logger import AuditLogger, EventType, Severity
from src.security.secrets_manager import get_secrets_manager


class BacktestEngine:
    """
    Event-driven backtest engine.
    
    Simulates trades on historical data with realistic execution costs.
    """

    def __init__(
        self,
        config: BacktestConfig,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._config = config
        self._audit = audit_logger
        self._fee_model = HyperliquidFeeModel(
            taker_fee_pct=config.taker_fee_pct,
            maker_fee_pct=config.maker_fee_pct,
            base_slippage_pct=config.slippage_pct,
        )
        
        # State
        self._capital: float = config.initial_capital
        self._peak_capital: float = config.initial_capital
        self._positions: dict[str, Position] = {}
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._daily_pnl: dict[str, float] = {}
        
        # Circuit breaker state
        self._halted: bool = False
        self._halt_reason: Optional[str] = None

    # ========================================================================
    # Public API (sync + async)
    # ========================================================================

    def run_sync(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """Synchronous wrapper for backtest run."""
        return asyncio.run(self.run(data, signals))

    async def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame with index=datetime
            signals: DataFrame with columns: signal, confidence, stop_loss, take_profit
            
        Returns:
            BacktestResult with full trade history and metrics
        """
        result = BacktestResult(config=self._config)
        result.initial_capital = self._config.initial_capital
        result.data_points = len(data)
        
        # Align signals with data
        aligned = self._align_data_signals(data, signals)
        
        # Event loop
        for timestamp, row in aligned.iterrows():
            if self._halted:
                break
                
            current_price = row["close"]
            
            # Update unrealized P&L
            self._update_unrealized_pnl(current_price)
            
            # Check circuit breakers
            if self._check_circuit_breakers(timestamp, current_price):
                continue
            
            # Process signal
            if "signal" in row and pd.notna(row["signal"]):
                await self._process_signal(
                    timestamp=timestamp,
                    signal=row["signal"],
                    price=current_price,
                    confidence=row.get("confidence", 0.5),
                    stop_loss=row.get("stop_loss", self._config.default_stop_loss_pct),
                    take_profit=row.get("take_profit", self._config.default_take_profit_pct),
                )
            
            # Check stop losses and take profits
            self._check_exit_conditions(timestamp, current_price)
            
            # Record equity
            total_equity = self._capital + self._total_unrealized_pnl()
            self._equity_curve.append((timestamp, total_equity))
            
            # Update peak capital
            if total_equity > self._peak_capital:
                self._peak_capital = total_equity
        
        # Close any remaining positions at last price
        if self._positions:
            last_price = data["close"].iloc[-1]
            last_time = data.index[-1]
            for pos in list(self._positions.values()):
                self._close_position_sync(pos, last_time, last_price, "end_of_data")
        
        # Build result
        result = self._build_result(result)
        
        # Sign result
        result.signature = self._sign_result(result)
        result.end_time = datetime.now()
        
        # Audit log
        if self._audit:
            await self._audit.log(
                event_type=EventType.SYSTEM,
                action="BACKTEST_COMPLETE",
                actor="system",
                severity=Severity.INFO,
                details={
                    "symbol": self._config.symbol,
                    "timeframe": self._config.timeframe,
                    "total_return_pct": result.total_return_pct,
                    "total_trades": result.total_trades,
                    "max_drawdown_pct": result.max_drawdown_pct,
                },
            )
        
        return result

    # ========================================================================
    # Signal Processing
    # ========================================================================

    async def _process_signal(
        self,
        timestamp: datetime,
        signal: str,
        price: float,
        confidence: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Process a trading signal."""
        symbol = self._config.symbol
        
        # Check max positions
        if len(self._positions) >= self._config.max_concurrent_positions:
            return
        
        # Determine direction
        if signal.upper() == "LONG":
            direction = TradeDirection.LONG
        elif signal.upper() == "SHORT":
            direction = TradeDirection.SHORT
        else:
            return  # NEUTRAL or invalid
        
        # Calculate position size
        position_size = self._calculate_position_size(confidence)
        
        # Check exposure limits
        if not self._check_exposure_limits(position_size, price):
            return
        
        # Calculate stop loss and take profit prices
        if direction == TradeDirection.LONG:
            sl_price = price * (1 - stop_loss / 100)
            tp_price = price * (1 + take_profit / 100)
        else:
            sl_price = price * (1 + stop_loss / 100)
            tp_price = price * (1 - take_profit / 100)
        
        # Open position (sync - no async needed)
        self._open_position(
            symbol=symbol,
            direction=direction,
            price=price,
            size=position_size,
            timestamp=timestamp,
            stop_loss=sl_price,
            take_profit=tp_price,
        )

    def _open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        size: float,
        timestamp: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Open a new position."""
        # Calculate fees
        fee = self._fee_model.calculate_fee(size, price, is_taker=True)
        
        # Deduct fees from capital
        self._capital -= fee.total_cost
        
        # Create position
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=direction,
            entry_price=price,
            size=size,
            leverage=self._config.max_leverage,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        self._positions[position.id] = position

    def _close_position_sync(
        self,
        position: Position,
        timestamp: datetime,
        price: float,
        reason: str,
    ) -> None:
        """
        Synchronously close an existing position.
        
        Called from the sync event loop - no async here.
        """
        if position.id not in self._positions:
            return  # Already closed
            
        # Calculate P&L
        if position.direction == TradeDirection.LONG:
            price_diff = price - position.entry_price
        else:
            price_diff = position.entry_price - price
        
        raw_pnl = price_diff * position.size * position.leverage
        
        # Calculate exit fees
        fee = self._fee_model.calculate_fee(position.size, price, is_taker=True)
        
        # Net P&L
        net_pnl = raw_pnl - fee.total_cost
        
        # Update capital
        self._capital += net_pnl
        
        # Record trade
        trade = TradeRecord(
            id=position.id,
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            leverage=position.leverage,
            pnl=raw_pnl,
            pnl_pct=(raw_pnl / (position.size * position.entry_price)) * 100 if position.size * position.entry_price > 0 else 0,
            fees=fee.trading_fee,
            slippage=fee.slippage_cost,
            exit_reason=reason,
        )
        
        self._trades.append(trade)
        
        # Track daily P&L
        date_key = timestamp.strftime("%Y-%m-%d")
        self._daily_pnl[date_key] = self._daily_pnl.get(date_key, 0) + net_pnl
        
        # Remove position
        del self._positions[position.id]

    def _check_exit_conditions(self, timestamp: datetime, price: float) -> None:
        """
        Check stop loss and take profit conditions.
        
        Uses sync close since we're inside the event loop iteration.
        """
        positions_to_close = []
        
        for position in list(self._positions.values()):
            should_close = False
            reason = ""
            
            if position.direction == TradeDirection.LONG:
                if position.stop_loss and price <= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
                elif position.take_profit and price >= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            else:  # SHORT
                if position.stop_loss and price >= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
                elif position.take_profit and price <= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, reason))
        
        # Close positions synchronously
        for position, reason in positions_to_close:
            self._close_position_sync(position, timestamp, price, reason)

    def _update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L for all open positions."""
        for position in self._positions.values():
            position.calculate_unrealized_pnl(current_price)

    def _total_unrealized_pnl(self) -> float:
        """Sum of all unrealized P&L."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    def _check_circuit_breakers(self, timestamp: datetime, price: float) -> bool:
        """
        Check circuit breaker conditions.
        
        Returns:
            True if trading should be halted
        """
        total_equity = self._capital + self._total_unrealized_pnl()
        
        # Daily loss limit
        date_key = timestamp.strftime("%Y-%m-%d")
        daily_pnl = self._daily_pnl.get(date_key, 0)
        daily_loss_pct = (-daily_pnl / self._config.initial_capital) * 100
        
        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            self._halted = True
            self._halt_reason = f"Daily loss limit: {daily_loss_pct:.2f}%"
            return True
        
        # Max drawdown
        if self._peak_capital > 0:
            drawdown_pct = ((self._peak_capital - total_equity) / self._peak_capital) * 100
            if drawdown_pct >= self._config.max_drawdown_pct:
                self._halted = True
                self._halt_reason = f"Max drawdown: {drawdown_pct:.2f}%"
                return True
        
        return False

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and config."""
        base_size = (self._capital * self._config.position_size_pct) / self._config.max_leverage
        
        if self._config.use_confidence_sizing:
            # Scale by confidence (0.5-1.0 → 0.5x-1.0x)
            confidence_multiplier = max(0.5, min(1.0, confidence))
            return base_size * confidence_multiplier
        
        return base_size

    def _check_exposure_limits(self, size: float, price: float) -> bool:
        """Check if new position would exceed exposure limits."""
        notional = size * price
        total_exposure = sum(p.size * p.entry_price for p in self._positions.values())
        total_exposure += notional
        
        max_exposure = self._capital * self._config.max_leverage
        return total_exposure <= max_exposure

    def _align_data_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align signals with OHLCV data."""
        # Merge on timestamp index
        aligned = data.copy()
        
        if not signals.empty:
            for col in ["signal", "confidence", "stop_loss", "take_profit"]:
                if col in signals.columns:
                    aligned[col] = signals[col]
        
        # Fill missing signals with NEUTRAL
        if "signal" not in aligned.columns:
            aligned["signal"] = "NEUTRAL"
        aligned["signal"] = aligned["signal"].fillna("NEUTRAL")
        
        return aligned

    def _build_result(self, result: BacktestResult) -> BacktestResult:
        """Build final backtest result from state."""
        result.trades = self._trades
        result.equity_curve = self._equity_curve
        result.final_capital = self._capital
        result.peak_capital = self._peak_capital
        
        # Calculate metrics
        if self._config.initial_capital > 0:
            result.total_return_pct = (
                (self._capital - self._config.initial_capital) / self._config.initial_capital
            ) * 100
        
        result.total_trades = len(self._trades)
        result.winning_trades = sum(1 for t in self._trades if t.net_pnl > 0)
        result.losing_trades = sum(1 for t in self._trades if t.net_pnl < 0)
        
        if result.total_trades > 0:
            result.win_rate = (result.winning_trades / result.total_trades) * 100
        
        # Profit factor
        gross_profit = sum(t.net_pnl for t in self._trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self._trades if t.net_pnl < 0))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        
        # Max drawdown
        if self._equity_curve:
            equity_values = [e[1] for e in self._equity_curve]
            peak = equity_values[0]
            max_dd = 0
            for eq in equity_values:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd
        
        # Sharpe ratio (simplified)
        if len(self._equity_curve) > 1:
            returns = []
            for i in range(1, len(self._equity_curve)):
                prev_eq = self._equity_curve[i-1][1]
                curr_eq = self._equity_curve[i][1]
                if prev_eq > 0:
                    returns.append((curr_eq - prev_eq) / prev_eq)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    # Annualized (assuming hourly data)
                    result.sharpe_ratio = (avg_return / std_return) * np.sqrt(365 * 24)
        
        return result

    def _sign_result(self, result: BacktestResult) -> str:
        """Create HMAC signature for result integrity."""
        secrets = get_secrets_manager()
        key = secrets.get("HMAC_SECRET_KEY", required=True)
        
        payload = f"{result.config.symbol}:{result.config.timeframe}:{result.total_trades}:{result.total_return_pct:.4f}"
        
        return hmac.new(
            key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
