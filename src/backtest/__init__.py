"""
Backtest module — Event-driven backtest engine with realistic fee modeling.

Usage:
    from src.backtest.engine import BacktestEngine
    engine = BacktestEngine(config)
    results = await engine.run(data, strategy)
"""

from .engine import BacktestEngine
from .models import BacktestConfig, BacktestResult, TradeRecord, Position
from .fees import FeeModel, HyperliquidFeeModel

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "Position",
    "FeeModel",
    "HyperliquidFeeModel",
]
