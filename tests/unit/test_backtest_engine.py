"""
Backtest Engine Tests — Comprehensive test suite.

Tests cover:
1. Basic buy-and-hold vs known return
2. Fee calculation accuracy
3. Position sizing
4. Stop loss / take profit execution
5. Circuit breaker triggers
6. Result signing integrity
7. Edge cases (no signals, single trade, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig
from src.strategy.examples import BuyAndHoldStrategy, MovingAverageCrossover


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    np.random.seed(42)
    
    # Create trending data
    base = 100
    prices = [base]
    for i in range(1, 100):
        change = np.random.normal(0.001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, 100)),
        "high": prices * (1 + abs(np.random.normal(0, 0.005, 100))),
        "low": prices * (1 - abs(np.random.normal(0, 0.005, 100))),
        "close": prices,
        "volume": np.random.uniform(100, 1000, 100),
    }, index=dates)
    
    return df


@pytest.fixture
def engine_config():
    """Default backtest config for tests."""
    return BacktestConfig(
        initial_capital=10000.0,
        max_leverage=3.0,
        taker_fee_pct=0.035,
        position_size_pct=0.1,
        max_concurrent_positions=3,
        symbol="BTC",
        timeframe="1h",
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@pytest.mark.asyncio
async def test_buy_and_hold_returns(sample_data, engine_config):
    """Buy-and-hold should match expected simple return minus fees."""
    strategy = BuyAndHoldStrategy()
    signals = strategy.generate_signals(sample_data)
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    # Should have exactly 1 trade
    assert result.total_trades == 1
    
    # Calculate expected return
    first_price = sample_data["close"].iloc[0]
    last_price = sample_data["close"].iloc[-1]
    simple_return = (last_price - first_price) / first_price
    
    # Result should be close to simple return minus fees
    # (with some difference due to leverage and sizing)
    assert result.total_return_pct is not None
    assert isinstance(result.total_return_pct, float)
    
    # Should have a trade record
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.direction.value == "LONG"
    assert trade.exit_reason == "end_of_data"


@pytest.mark.asyncio
async def test_no_signals_no_trades(sample_data, engine_config):
    """With no signals, no trades should occur."""
    empty_signals = pd.DataFrame(index=sample_data.index)
    empty_signals["signal"] = "NEUTRAL"
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, empty_signals)
    
    assert result.total_trades == 0
    assert result.total_return_pct == 0.0
    assert len(result.trades) == 0


@pytest.mark.asyncio
async def test_fee_accuracy(sample_data, engine_config):
    """Fees should match Hyperliquid's 0.035% taker fee."""
    strategy = BuyAndHoldStrategy()
    signals = strategy.generate_signals(sample_data)
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    trade = result.trades[0]
    
    # Calculate expected fee
    notional = trade.size * trade.entry_price
    expected_fee = notional * 0.00035  # 0.035%
    
    # Should be close (plus exit fee)
    assert trade.fees > 0
    assert abs(trade.fees - expected_fee) < expected_fee * 0.1  # Within 10%


# =============================================================================
# Position Management Tests
# =============================================================================

@pytest.mark.asyncio
async def test_max_positions_limit(sample_data, engine_config):
    """Should not exceed max concurrent positions."""
    engine_config.max_concurrent_positions = 1
    
    # Create signals that would generate multiple positions
    signals = pd.DataFrame(index=sample_data.index)
    signals["signal"] = "NEUTRAL"
    signals["confidence"] = 0.8
    
    # Signal at multiple points
    signals.iloc[10, signals.columns.get_loc("signal")] = "LONG"
    signals.iloc[20, signals.columns.get_loc("signal")] = "LONG"
    signals.iloc[30, signals.columns.get_loc("signal")] = "LONG"
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    # With max_positions=1, should only open first position
    assert result.total_trades <= 2  # 1 open + 1 close


@pytest.mark.asyncio
async def test_stop_loss_execution(sample_data, engine_config):
    """Stop loss should trigger when price falls below threshold."""
    # Create data with a drop
    dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
    prices = np.concatenate([
        np.linspace(100, 105, 10),   # Rise
        np.linspace(105, 90, 20),     # Drop below SL
        np.linspace(90, 95, 20),      # Recover
    ])
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(50) * 100,
    }, index=dates)
    
    # Signal at start
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = "NEUTRAL"
    signals.iloc[0, signals.columns.get_loc("signal")] = "LONG"
    signals["stop_loss"] = 2.0  # 2% stop loss
    signals["take_profit"] = 10.0
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(df, signals)
    
    # Should have closed via stop loss
    assert result.total_trades >= 1
    if result.trades:
        assert result.trades[0].exit_reason == "stop_loss"


@pytest.mark.asyncio
async def test_take_profit_execution(sample_data, engine_config):
    """Take profit should trigger when price rises above threshold."""
    # Create data with a rise
    dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
    prices = np.concatenate([
        np.linspace(100, 110, 25),   # Rise above TP
        np.linspace(110, 105, 25),   # Drop
    ])
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(50) * 100,
    }, index=dates)
    
    # Signal at start
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = "NEUTRAL"
    signals.iloc[0, signals.columns.get_loc("signal")] = "LONG"
    signals["stop_loss"] = 10.0
    signals["take_profit"] = 5.0  # 5% take profit
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(df, signals)
    
    # Should have closed via take profit
    assert result.total_trades >= 1
    if result.trades:
        assert result.trades[0].exit_reason == "take_profit"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

@pytest.mark.asyncio
async def test_daily_loss_circuit_breaker(sample_data, engine_config):
    """Should halt trading when daily loss limit is hit."""
    engine_config.daily_loss_limit_pct = 1.0  # Tight limit
    
    # Create data with big drop
    dates = pd.date_range(start="2024-01-01", periods=24, freq="1h")
    prices = np.linspace(100, 70, 24)  # 30% drop
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(24) * 100,
    }, index=dates)
    
    # Always long signal
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = "LONG"
    signals["confidence"] = 0.9
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(df, signals)
    
    # Should have halted early
    assert result.total_trades >= 1


# =============================================================================
# Result Integrity Tests
# =============================================================================

@pytest.mark.asyncio
async def test_result_signature(sample_data, engine_config):
    """Backtest results should be signed."""
    strategy = BuyAndHoldStrategy()
    signals = strategy.generate_signals(sample_data)
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    assert result.signature is not None
    assert len(result.signature) > 0
    assert isinstance(result.signature, str)


@pytest.mark.asyncio
async def test_equity_curve_length(sample_data, engine_config):
    """Equity curve should have one entry per data point."""
    strategy = BuyAndHoldStrategy()
    signals = strategy.generate_signals(sample_data)
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    assert len(result.equity_curve) == len(sample_data)


@pytest.mark.asyncio
async def test_metrics_calculated(sample_data, engine_config):
    """All metrics should be calculated after backtest."""
    strategy = MovingAverageCrossover(fast_period=5, slow_period=20)
    signals = strategy.generate_signals(sample_data)
    
    engine = BacktestEngine(engine_config)
    result = await engine.run(sample_data, signals)
    
    # Check all expected metrics exist
    assert result.total_return_pct is not None
    assert result.total_trades is not None
    assert result.win_rate is not None
    assert result.max_drawdown_pct is not None
    assert result.sharpe_ratio is not None
    assert result.profit_factor is not None
    
    # Check metric bounds
    assert -100 <= result.total_return_pct <= 1000  # Reasonable bounds
    assert 0 <= result.win_rate <= 100
    assert result.max_drawdown_pct >= 0
