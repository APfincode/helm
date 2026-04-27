"""
Strategy module — Abstract strategy interface and example implementations.

All strategies inherit from BaseStrategy and produce signals that the
backtest engine consumes.

Usage:
    from src.strategy.base import BaseStrategy
    from src.strategy.examples import MovingAverageCrossover
    
    strategy = MovingAverageCrossover(fast=14, slow=50)
    signals = strategy.generate_signals(data)
"""

from .base import BaseStrategy, Signal, SignalType
from .examples import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "MovingAverageCrossover",
    "RSIStrategy",
    "BollingerBandsStrategy",
]
