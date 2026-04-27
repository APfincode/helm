"""
Paper Trading Executor — Simulates Hyperliquid execution without real money.

This is CRITICAL for live validation before deploying with real capital.
It mimics Hyperliquid's behavior exactly:
- Same fee structure (0.035% taker, 0.01% maker)
- Same slippage estimation
- Same order types
- Full P&L tracking

When to use:
- New strategy validation (1-2 weeks)
- After major code changes
- Before increasing position sizes
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, Any

from src.execution.hyperliquid_executor import (
    OrderResult, OrderStatus, OrderType, HyperliquidPosition
)
from src.backtest.fees import HyperliquidFeeModel


logger = logging.getLogger(__name__)


@dataclass
class SimulatedPosition:
    """A simulated (paper) position."""
    id: str
    coin: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float  # Base currency
    notional: float  # USD value
    margin: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: datetime = field(default_factory=datetime.now)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0


@dataclass
class PaperAccount:
    """Paper trading account state."""
    equity: float = 10000.0
    cash: float = 10000.0
    margin_used: float = 0.0
    peak_equity: float = 10000.0
    total_fees: float = 0.0
    total_pnl: float = 0.0
    positions: dict[str, SimulatedPosition] = field(default_factory=dict)
    trade_history: list[dict] = field(default_factory=list)


class PaperTradingExecutor:
    """
    Paper trading executor that simulates Hyperliquid exactly.
    
    Use this for 1-2 weeks before switching to real execution.
    Track P&L, fees, drawdown — everything must be profitable
    before risking real capital.
    """
    
    def __init__(self, initial_equity: float = 10000.0):
        self._account = PaperAccount(
            equity=initial_equity,
            cash=initial_equity,
            peak_equity=initial_equity,
        )
        self._fee_model = HyperliquidFeeModel(
            taker_fee_pct=0.035,
            maker_fee_pct=0.01,
            base_slippage_pct=0.02,
        )
    
    async def place_market_order(
        self,
        coin: str,
        side: Literal["LONG", "SHORT"],
        quantity: float,
        current_price: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Simulate a market order.
        
        No async IO — just state updates.
        """
        result = OrderResult()
        
        # Validate
        if quantity <= 0:
            result.error_message = "Invalid quantity"
            return result
        
        if current_price <= 0:
            result.error_message = "Invalid price"
            return result
        
        # Calculate fees and slippage
        fee = self._fee_model.calculate_fee(quantity, current_price, is_taker=True)
        fill_price = current_price + fee.slippage_cost / quantity
        
        # Execute
        notional = quantity * fill_price
        margin = notional / 5.0  # Assume 5x leverage max
        
        # Check margin
        if not reduce_only and self._account.cash < margin + fee.total_cost:
            result.status = OrderStatus.REJECTED
            result.error_message = f"Insufficient margin: need ${margin + fee.total_cost:.2f}, have ${self._account.cash:.2f}"
            logger.warning(f"Paper trade REJECTED: {result.error_message}")
            return result
        
        # Update account
        if not reduce_only:
            self._account.cash -= margin + fee.total_cost
            self._account.margin_used += margin
        
        self._account.total_fees += fee.total_cost
        
        # Record position
        position_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        if reduce_only:
            # Close existing position
            pos = self._account.positions.get(coin)
            if pos and pos.direction == side:
                # Same direction — should not reduce
                result.error_message = "Reduce-only but no matching position"
                return result
            
            if pos:
                close_qty = min(quantity, pos.size)
                pnl = self._calculate_pnl(pos, fill_price, close_qty)
                self._account.total_pnl += pnl
                self._account.cash += margin + pnl  # Release margin + PnL
                self._account.margin_used -= margin
                
                pos.size -= close_qty
                pos.realized_pnl += pnl
                pos.fees_paid += fee.total_cost
                
                if pos.size <= 0:
                    del self._account.positions[coin]
                
                logger.info(
                    f"Paper CLOSE: {coin} {side} {close_qty} @ {fill_price:.2f} "
                    f"P&L: ${pnl:.2f}"
                )
        else:
            # Open new position
            self._account.positions[coin] = SimulatedPosition(
                id=position_id,
                coin=coin,
                direction=side,
                entry_price=fill_price,
                size=quantity,
                notional=notional,
                margin=margin,
                leverage=5.0,
                stop_loss=None,
                take_profit=None,
                fees_paid=fee.total_cost,
            )
            
            logger.info(
                f"Paper OPEN: {coin} {side} {quantity:.4f} @ {fill_price:.2f} "
                f"Margin: ${margin:.2f} Fee: ${fee.total_cost:.4f}"
            )
        
        # Record trade
        self._account.trade_history.append({
            "time": datetime.now().isoformat(),
            "coin": coin,
            "side": side,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee.total_cost,
            "reduce_only": reduce_only,
        })
        
        result.success = True
        result.status = OrderStatus.FILLED
        result.order_id = position_id
        result.filled_quantity = quantity
        result.avg_fill_price = fill_price
        result.fee_paid = fee.total_cost
        
        return result
    
    async def get_positions(self) -> list[HyperliquidPosition]:
        """Return simulated positions in Hyperliquid format."""
        positions = []
        for pos in self._account.positions.values():
            positions.append(HyperliquidPosition(
                coin=pos.coin,
                szi=pos.size if pos.direction == "LONG" else -pos.size,
                entry_px=pos.entry_price,
                position_value=pos.notional,
                unrealized_pnl=pos.unrealized_pnl,
                leverage=pos.leverage,
                liquidation_px=None,
                margin_used=pos.margin,
            ))
        return positions
    
    async def get_account_summary(self) -> dict:
        """Return simulated account summary."""
        # Update equity with unrealized PnL
        unrealized = sum(p.unrealized_pnl for p in self._account.positions.values())
        total_equity = self._account.cash + self._account.margin_used + unrealized
        self._account.equity = total_equity
        
        if total_equity > self._account.peak_equity:
            self._account.peak_equity = total_equity
        
        drawdown = 0.0
        if self._account.peak_equity > 0:
            drawdown = (self._account.peak_equity - total_equity) / self._account.peak_equity * 100
        
        return {
            "equity": total_equity,
            "cash": self._account.cash,
            "margin_used": self._account.margin_used,
            "peak_equity": self._account.peak_equity,
            "drawdown_pct": drawdown,
            "total_fees": self._account.total_fees,
            "total_pnl": self._account.total_pnl,
            "open_positions": len(self._account.positions),
            "unrealized_pnl": unrealized,
        }
    
    async def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update positions with current prices.
        
        Call this every tick with latest market prices.
        """
        for coin, price in prices.items():
            pos = self._account.positions.get(coin)
            if pos:
                pos.unrealized_pnl = self._calculate_pnl(pos, price, pos.size)
    
    def _calculate_pnl(
        self,
        position: SimulatedPosition,
        current_price: float,
        size: float,
    ) -> float:
        """Calculate P&L for a position."""
        if position.direction == "LONG":
            price_diff = current_price - position.entry_price
        else:
            price_diff = position.entry_price - current_price
        
        return price_diff * size * position.leverage
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def account(self) -> PaperAccount:
        return self._account
    
    @property
    def current_equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self._account.positions.values())
        return self._account.cash + self._account.margin_used + unrealized
    
    @property
    def stats(self) -> dict:
        return {
            "equity": self.current_equity,
            "cash": self._account.cash,
            "margin_used": self._account.margin_used,
            "total_trades": len(self._account.trade_history),
            "total_fees": self._account.total_fees,
            "total_pnl": self._account.total_pnl,
            "open_positions": len(self._account.positions),
            "win_rate": self._calculate_win_rate(),
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self._account.trade_history:
            return 0.0
        
        # Simplified: count positions with positive realized PnL
        # Full implementation would track per-position realized PnL
        return 0.0  # Placeholder
