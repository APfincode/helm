"""
Strategy Base — Abstract interface for all trading strategies.

A strategy takes OHLCV data and produces signals (LONG, SHORT, NEUTRAL).
Each signal includes:
- direction: LONG, SHORT, or NEUTRAL
- confidence: 0.0 to 1.0
- stop_loss_pct: Recommended stop loss as percentage
- take_profit_pct: Recommended take profit as percentage
- reasoning: Human-readable explanation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

from src.security.input_validator import PromptInjectionDetector


class SignalType(str, Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class Signal:
    """
    Immutable trading signal.
    
    Frozen dataclass prevents accidental mutation after creation.
    """
    timestamp: pd.Timestamp
    signal: SignalType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    stop_loss_pct: float = Field(default=2.0, ge=0.1, le=50.0)
    take_profit_pct: float = Field(default=4.0, ge=0.1, le=100.0)
    reasoning: str = ""
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        # Validate signal value
        if isinstance(self.signal, str):
            object.__setattr__(self, 'signal', SignalType(self.signal))


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Subclasses must implement:
    1. generate_signals(data) -> pd.DataFrame
    2. (optional) validate_parameters()
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._injection_detector = PromptInjectionDetector()
        self._validate_parameters()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            data: DataFrame with columns: open, high, low, close, volume
                  Index: datetime
        
        Returns:
            DataFrame with columns:
                - signal: "LONG", "SHORT", or "NEUTRAL"
                - confidence: float 0.0-1.0
                - stop_loss: float (percentage)
                - take_profit: float (percentage)
                - reasoning: str
        """
        pass

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        Override in subclass to add custom validation.
        """
        pass

    def _sanitize_reasoning(self, text: str) -> str:
        """
        Sanitize strategy reasoning text.
        
        Removes control characters and injection patterns.
        """
        if not text:
            return ""
        
        # Remove control characters
        cleaned = "".join(
            c for c in text 
            if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) < 127)
        )
        
        # Check for injection
        if self._injection_detector.is_suspicious(cleaned):
            return "[REDACTED - suspicious content detected]"
        
        return cleaned.strip()

    def _create_signals_df(
        self,
        data: pd.DataFrame,
        signals: list[Signal],
    ) -> pd.DataFrame:
        """
        Convert list of Signal objects to DataFrame aligned with data index.
        
        Args:
            data: Original OHLCV data
            signals: List of Signal objects
            
        Returns:
            DataFrame with signal columns, indexed same as data
        """
        # Create empty signals DataFrame with same index
        result = pd.DataFrame(index=data.index)
        result["signal"] = "NEUTRAL"
        result["confidence"] = 0.0
        result["stop_loss"] = 2.0
        result["take_profit"] = 4.0
        result["reasoning"] = ""
        
        # Fill in actual signals
        for sig in signals:
            if sig.timestamp in result.index:
                result.at[sig.timestamp, "signal"] = sig.signal.value
                result.at[sig.timestamp, "confidence"] = sig.confidence
                result.at[sig.timestamp, "stop_loss"] = sig.stop_loss_pct
                result.at[sig.timestamp, "take_profit"] = sig.take_profit_pct
                result.at[sig.timestamp, "reasoning"] = sig.reasoning
        
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# Need to import Field for dataclass validation
from pydantic import Field
