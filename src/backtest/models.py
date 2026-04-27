"""
Backtest models — Data structures for backtest configuration and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Literal


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 10000.0
    max_leverage: float = 3.0
    taker_fee_pct: float = 0.035  # Hyperliquid taker fee
    maker_fee_pct: float = 0.01   # Hyperliquid maker fee
    slippage_pct: float = 0.02    # Estimated slippage
    use_slippage: bool = True
    
    # Position sizing
    position_size_pct: float = 0.1  # 10% of capital per trade
    use_confidence_sizing: bool = False
    
    # Risk management
    max_concurrent_positions: int = 3
    max_drawdown_pct: float = 15.0
    daily_loss_limit_pct: float = 5.0
    
    # Signal defaults
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0
    
    # Run metadata
    symbol: str = "BTC"
    timeframe: str = "1h"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "max_leverage": self.max_leverage,
            "taker_fee_pct": self.taker_fee_pct,
            "maker_fee_pct": self.maker_fee_pct,
            "slippage_pct": self.slippage_pct,
            "use_slippage": self.use_slippage,
            "position_size_pct": self.position_size_pct,
            "use_confidence_sizing": self.use_confidence_sizing,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "default_stop_loss_pct": self.default_stop_loss_pct,
            "default_take_profit_pct": self.default_take_profit_pct,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


@dataclass
class Position:
    """Open position tracking."""
    id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    size: float  # Position size in base currency
    leverage: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    unrealized_pnl: float = 0.0
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.direction == TradeDirection.LONG:
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
        
        self.unrealized_pnl = price_diff * self.size * self.leverage
        return self.unrealized_pnl


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    id: str
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    exit_reason: str  # "stop_loss", "take_profit", "signal", "forced"
    
    @property
    def net_pnl(self) -> float:
        """Net P&L after fees and slippage."""
        return self.pnl - self.fees - self.slippage


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    
    # Performance metrics
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # candles
    
    # Capital tracking
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    
    # Trade history
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    data_points: int = 0
    signature: str = ""  # HMAC signature for tamper detection
    
    @property
    def avg_trade_return(self) -> float:
        """Average return per trade."""
        if not self.trades:
            return 0.0
        return sum(t.net_pnl for t in self.trades) / len(self.trades)
    
    @property
    def avg_win(self) -> float:
        """Average winning trade."""
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade."""
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0
    
    def to_dict(self) -> dict:
        """Export results as dictionary."""
        return {
            "config": self.config.to_dict(),
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "peak_capital": self.peak_capital,
            "avg_trade_return": self.avg_trade_return,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "data_points": self.data_points,
        }
