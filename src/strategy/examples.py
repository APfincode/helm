"""
Example Strategies — Simple rule-based strategies for testing the backtest engine.

These are NOT meant to be profitable — they're for:
1. Testing backtest engine correctness
2. Comparing LLM strategy performance against baselines
3. Validating signal-to-trade pipeline
"""

from typing import Optional

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalType


class MovingAverageCrossover(BaseStrategy):
    """
    Simple moving average crossover strategy.
    
    LONG when fast MA crosses above slow MA.
    SHORT when fast MA crosses below slow MA.
    """

    def __init__(
        self,
        fast_period: int = 14,
        slow_period: int = 50,
        name: Optional[str] = None,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(name=name or f"MA_Cross_{fast_period}_{slow_period}")

    def _validate_parameters(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if self.fast_period < 2:
            raise ValueError("fast_period must be at least 2")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals."""
        close = data["close"]
        
        # Calculate moving averages
        fast_ma = close.rolling(window=self.fast_period).mean()
        slow_ma = close.rolling(window=self.slow_period).mean()
        
        # Previous values for crossover detection
        fast_ma_prev = fast_ma.shift(1)
        slow_ma_prev = slow_ma.shift(1)
        
        signals = []
        
        for timestamp in data.index:
            if pd.isna(fast_ma[timestamp]) or pd.isna(slow_ma[timestamp]):
                continue
                
            fast = fast_ma[timestamp]
            slow = slow_ma[timestamp]
            fast_prev = fast_ma_prev[timestamp]
            slow_prev = slow_ma_prev[timestamp]
            
            # Skip if previous values are NaN
            if pd.isna(fast_prev) or pd.isna(slow_prev):
                continue
            
            # Check for crossover
            golden_cross = fast > slow and fast_prev <= slow_prev
            death_cross = fast < slow and fast_prev >= slow_prev
            
            if golden_cross:
                # Calculate confidence based on trend strength
                ma_distance = abs(fast - slow) / slow
                confidence = min(0.95, 0.5 + ma_distance * 10)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.LONG,
                    confidence=confidence,
                    stop_loss_pct=2.0,
                    take_profit_pct=4.0,
                    reasoning=f"Golden cross: {self.fast_period}MA ({fast:.2f}) crossed above {self.slow_period}MA ({slow:.2f})",
                )
                signals.append(sig)
                
            elif death_cross:
                ma_distance = abs(fast - slow) / slow
                confidence = min(0.95, 0.5 + ma_distance * 10)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.SHORT,
                    confidence=confidence,
                    stop_loss_pct=2.0,
                    take_profit_pct=4.0,
                    reasoning=f"Death cross: {self.fast_period}MA ({fast:.2f}) crossed below {self.slow_period}MA ({slow:.2f})",
                )
                signals.append(sig)
        
        return self._create_signals_df(data, signals)


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    LONG when RSI < oversold threshold.
    SHORT when RSI > overbought threshold.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: Optional[str] = None,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        super().__init__(name=name or f"RSI_{period}_{oversold}_{overbought}")

    def _validate_parameters(self) -> None:
        if not (0 < self.oversold < self.overbought < 100):
            raise ValueError("Must have: 0 < oversold < overbought < 100")

    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI signals."""
        close = data["close"]
        rsi = self._calculate_rsi(close, self.period)
        rsi_prev = rsi.shift(1)
        
        signals = []
        
        for timestamp in data.index:
            if pd.isna(rsi[timestamp]):
                continue
            
            current_rsi = rsi[timestamp]
            prev_rsi = rsi_prev[timestamp]
            
            # Entry: RSI crosses into oversold/overbought region
            if prev_rsi > self.oversold and current_rsi <= self.oversold:
                # Entering oversold → LONG (mean reversion)
                confidence = min(0.9, 0.5 + (self.oversold - current_rsi) / 20)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.LONG,
                    confidence=confidence,
                    stop_loss_pct=3.0,
                    take_profit_pct=5.0,
                    reasoning=f"RSI oversold: {current_rsi:.1f} (threshold: {self.oversold})",
                )
                signals.append(sig)
                
            elif prev_rsi < self.overbought and current_rsi >= self.overbought:
                # Entering overbought → SHORT (mean reversion)
                confidence = min(0.9, 0.5 + (current_rsi - self.overbought) / 20)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.SHORT,
                    confidence=confidence,
                    stop_loss_pct=3.0,
                    take_profit_pct=5.0,
                    reasoning=f"RSI overbought: {current_rsi:.1f} (threshold: {self.overbought})",
                )
                signals.append(sig)
        
        return self._create_signals_df(data, signals)


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy.
    
    LONG when price touches lower band.
    SHORT when price touches upper band.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        name: Optional[str] = None,
    ) -> None:
        self.period = period
        self.std_dev = std_dev
        super().__init__(name=name or f"BB_{period}_{std_dev}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        close = data["close"]
        
        # Calculate Bollinger Bands
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        # Previous values
        close_prev = close.shift(1)
        
        signals = []
        
        for timestamp in data.index:
            if pd.isna(upper[timestamp]) or pd.isna(lower[timestamp]):
                continue
            
            current_close = close[timestamp]
            prev_close = close_prev[timestamp]
            current_upper = upper[timestamp]
            current_lower = lower[timestamp]
            
            # LONG: Price crosses below lower band
            if prev_close >= current_lower and current_close < current_lower:
                band_width = (current_upper - current_lower) / current_lower
                confidence = min(0.9, 0.5 + band_width)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.LONG,
                    confidence=confidence,
                    stop_loss_pct=2.5,
                    take_profit_pct=4.0,
                    reasoning=f"Price below lower BB: {current_close:.2f} < {current_lower:.2f}",
                )
                signals.append(sig)
            
            # SHORT: Price crosses above upper band
            elif prev_close <= current_upper and current_close > current_upper:
                band_width = (current_upper - current_lower) / current_lower
                confidence = min(0.9, 0.5 + band_width)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=SignalType.SHORT,
                    confidence=confidence,
                    stop_loss_pct=2.5,
                    take_profit_pct=4.0,
                    reasoning=f"Price above upper BB: {current_close:.2f} > {current_upper:.2f}",
                )
                signals.append(sig)
        
        return self._create_signals_df(data, signals)


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and hold baseline strategy.
    
    Buys at first candle, holds until end.
    Used as benchmark for comparing strategy performance.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name or "BuyAndHold")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate single buy signal at start."""
        if len(data) < 2:
            return self._create_signals_df(data, [])
        
        # Only signal at first candle
        first_timestamp = data.index[0]
        
        sig = Signal(
            timestamp=first_timestamp,
            signal=SignalType.LONG,
            confidence=1.0,
            stop_loss_pct=50.0,  # Very wide
            take_profit_pct=100.0,
            reasoning="Buy and hold benchmark",
        )
        
        return self._create_signals_df(data, [sig])


class RandomStrategy(BaseStrategy):
    """
    Random signal generator for testing.
    
    Generates random LONG/SHORT signals at random intervals.
    NOT for production use — only for testing engine mechanics.
    """

    def __init__(
        self,
        signal_probability: float = 0.02,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self.signal_probability = signal_probability
        self.seed = seed
        super().__init__(name=name or "Random")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate random signals."""
        rng = np.random.RandomState(self.seed)
        
        signals = []
        
        for timestamp in data.index:
            if rng.random() < self.signal_probability:
                # Random direction
                direction = SignalType.LONG if rng.random() > 0.5 else SignalType.SHORT
                confidence = rng.uniform(0.5, 0.9)
                
                sig = Signal(
                    timestamp=timestamp,
                    signal=direction,
                    confidence=confidence,
                    stop_loss_pct=2.0,
                    take_profit_pct=4.0,
                    reasoning="Random test signal",
                )
                signals.append(sig)
        
        return self._create_signals_df(data, signals)
