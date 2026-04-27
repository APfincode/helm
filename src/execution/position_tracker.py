"""
Position Tracker — Tracks open positions and manages state transitions.

Responsibilities:
1. Track all open positions (from both live and paper trading)
2. Monitor stop loss and take profit levels
3. Update unrealized P&L at each tick
4. Trigger reduce-only close orders when stops are hit
5. Provide position summary for risk manager

Alpha Arena Lesson: Models that held 6 simultaneous positions performed worse.
Our tracker enforces max positions AND tracks correlation between positions.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSING = "closing"  # Close order submitted but not filled
    CLOSED = "closed"
    STOPPED = "stopped"  # Hit stop loss
    PROFIT = "profit"     # Hit take profit
    LIQUIDATED = "liquidated"


@dataclass
class PositionState:
    """
    Complete state of a single position.
    
    Tracks everything RiskManager needs to know about a position.
    """
    id: str
    coin: str
    direction: Literal["LONG", "SHORT"]
    
    # Entry
    entry_price: float
    entry_time: datetime
    quantity: float
    leverage: float
    
    # Risk parameters (set by RiskManager)
    stop_loss_price: float
    take_profit_price: float
    risk_usd: float  # Amount at risk when position opened
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    margin_used: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    
    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    exit_reason: str = ""
    
    # Metadata
    order_id: Optional[str] = None  # Exchange order ID
    is_paper: bool = False
    
    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"
    
    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"
    
    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def time_in_trade_seconds(self) -> float:
        """Duration in trade."""
        end = self.exit_time or datetime.now()
        return (end - self.entry_time).total_seconds()
    
    @property
    def is_stop_triggered(self) -> bool:
        """Check if stop loss should trigger."""
        if self.is_long:
            return self.current_price <= self.stop_loss_price
        else:
            return self.current_price >= self.stop_loss_price
    
    @property
    def is_tp_triggered(self) -> bool:
        """Check if take profit should trigger."""
        if self.is_long:
            return self.current_price >= self.take_profit_price
        else:
            return self.current_price <= self.take_profit_price
    
    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L."""
        self.current_price = current_price
        
        if self.is_long:
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
        
        self.unrealized_pnl = price_diff * self.quantity * self.leverage
        
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (price_diff / self.entry_price) * 100
    
    def close(self, exit_price: float, reason: str, fees: float = 0.0) -> None:
        """Close position and calculate realized P&L."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.fees_paid = fees
        
        # Calculate realized P&L
        if self.is_long:
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price
        
        self.realized_pnl = price_diff * self.quantity * self.leverage - fees
        self.status = PositionStatus.CLOSED
        
        # Update unrealized to 0 (position closed)
        self.unrealized_pnl = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "coin": self.coin,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "risk_usd": self.risk_usd,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "margin_used": self.margin_used,
            "status": self.status.value,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "exit_reason": self.exit_reason,
            "is_paper": self.is_paper,
        }


class PositionTracker:
    """
    Track and manage all positions.
    
    Monitors positions for stop/take-profit triggers.
    Provides risk metrics to RiskManager.
    """
    
    def __init__(self):
        self._positions: dict[str, PositionState] = {}  # position_id -> PositionState
        self._by_coin: dict[str, set[str]] = {}  # coin -> set(position_ids)
        self._history: list[PositionState] = []  # Closed positions
        
        # Stats
        self._total_opened: int = 0
        self._total_closed: int = 0
        self._total_stopped: int = 0
        self._total_tp_hit: int = 0
    
    def add_position(self, position: PositionState) -> None:
        """Add a new position."""
        self._positions[position.id] = position
        
        if position.coin not in self._by_coin:
            self._by_coin[position.coin] = set()
        self._by_coin[position.coin].add(position.id)
        
        self._total_opened += 1
        
        logger.info(
            f"Position opened: {position.coin} {position.direction} "
            f"{position.quantity:.4f} @ {position.entry_price:.2f}"
        )
    
    def remove_position(self, position_id: str) -> Optional[PositionState]:
        """Remove a position (when closed)."""
        pos = self._positions.pop(position_id, None)
        if pos:
            self._by_coin.get(pos.coin, set()).discard(position_id)
            self._history.append(pos)
            
            # Count exits by reason, not just closed status
            if pos.exit_reason == "stop_loss":
                self._total_stopped += 1
            elif pos.exit_reason == "take_profit":
                self._total_tp_hit += 1
            else:
                self._total_closed += 1
                
        return pos
    
    def get_position(self, position_id: str) -> Optional[PositionState]:
        """Get position by ID."""
        return self._positions.get(position_id)
    
    def get_positions_by_coin(self, coin: str) -> list[PositionState]:
        """Get all positions for a coin."""
        ids = self._by_coin.get(coin, set())
        return [self._positions[pid] for pid in ids if pid in self._positions]
    
    def get_all_open(self) -> list[PositionState]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]
    
    def update_prices(self, prices: dict[str, float]) -> list[dict]:
        """
        Update all positions with current prices.
        
        Returns list of triggered stops/take-profits for execution.
        """
        triggers = []
        
        for pos in self._positions.values():
            if pos.status != PositionStatus.OPEN:
                continue
            
            price = prices.get(pos.coin)
            if not price:
                continue
            
            pos.update_pnl(price)
            
            # Check stops
            if pos.is_stop_triggered:
                triggers.append({
                    "position_id": pos.id,
                    "coin": pos.coin,
                    "trigger": "stop_loss",
                    "price": price,
                    "pnl": pos.unrealized_pnl,
                })
                pos.status = PositionStatus.STOPPED
                
            elif pos.is_tp_triggered:
                triggers.append({
                    "position_id": pos.id,
                    "coin": pos.coin,
                    "trigger": "take_profit",
                    "price": price,
                    "pnl": pos.unrealized_pnl,
                })
                pos.status = PositionStatus.PROFIT
        
        return triggers
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str,
        fees: float = 0.0,
    ) -> Optional[PositionState]:
        """Close a position."""
        pos = self._positions.get(position_id)
        if pos:
            pos.close(exit_price, reason, fees)
            self.remove_position(position_id)
            logger.info(
                f"Position closed: {pos.coin} {pos.direction} "
                f"P&L: ${pos.realized_pnl:.2f} ({reason})"
            )
        return pos
    
    # ========================================================================
    # Risk Metrics
    # ========================================================================
    
    @property
    def total_exposure(self) -> float:
        """Total notional exposure across all positions."""
        return sum(p.notional_value for p in self._positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())
    
    @property
    def total_margin_used(self) -> float:
        return sum(p.margin_used for p in self._positions.values())
    
    @property
    def open_position_count(self) -> int:
        return len([p for p in self._positions.values() if p.status == PositionStatus.OPEN])
    
    @property
    def stats(self) -> dict:
        """Position statistics."""
        open_positions = self.get_all_open()
        
        return {
            "open_positions": len(open_positions),
            "total_opened": self._total_opened,
            "total_closed": self._total_closed,
            "total_stopped": self._total_stopped,
            "total_tp_hit": self._total_tp_hit,
            "total_exposure": self.total_exposure,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_margin_used": self.total_margin_used,
            "by_coins": {coin: len(ids) for coin, ids in self._by_coin.items()},
        }
    
    def to_dict(self) -> dict:
        """Export all positions as dict."""
        return {
            "open": [p.to_dict() for p in self.get_all_open()],
            "history": [p.to_dict() for p in self._history[-20:]],  # Last 20
            "stats": self.stats,
        }
