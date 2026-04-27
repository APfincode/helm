#!/usr/bin/env python3
"""
Example backtest runner.

Demonstrates how to:
1. Generate sample data
2. Run a strategy (e.g., Moving Average Crossover)
3. Compare against Buy-and-Hold benchmark
4. Print results

Usage:
    python scripts/run_backtest.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.models import BacktestConfig
from strategy.examples import MovingAverageCrossover, BuyAndHoldStrategy


def generate_sample_data(n: int = 500) -> pd.DataFrame:
    """Generate realistic-looking OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Random walk with trend
    returns = np.random.normal(0.0002, 0.008, n)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, n)),
        "high": prices * (1 + abs(np.random.normal(0, 0.004, n))),
        "low": prices * (1 - abs(np.random.normal(0, 0.004, n))),
        "close": prices,
        "volume": np.random.uniform(1000, 10000, n),
    }, index=dates)
    
    return df


async def run_comparison():
    """Run strategy vs buy-and-hold comparison."""
    print("=" * 60)
    print("Hyper-Alpha-Arena V2 - Backtest Demo")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating sample data...")
    data = generate_sample_data(n=500)
    print(f"    Data points: {len(data)}")
    print(f"    Date range: {data.index[0]} to {data.index[-1]}")
    print(f"    Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Config
    config = BacktestConfig(
        initial_capital=10000.0,
        max_leverage=3.0,
        taker_fee_pct=0.035,
        position_size_pct=0.1,
        max_concurrent_positions=2,
        symbol="BTC",
        timeframe="1h",
    )
    
    # Run Buy-and-Hold benchmark
    print("\n[2] Running Buy-and-Hold benchmark...")
    benchmark_strategy = BuyAndHoldStrategy()
    benchmark_signals = benchmark_strategy.generate_signals(data)
    
    benchmark_engine = BacktestEngine(config)
    benchmark_result = await benchmark_engine.run(data, benchmark_signals)
    
    print(f"    Trades: {benchmark_result.total_trades}")
    print(f"    Return: {benchmark_result.total_return_pct:.2f}%")
    print(f"    Max Drawdown: {benchmark_result.max_drawdown_pct:.2f}%")
    
    # Run MA Crossover strategy
    print("\n[3] Running Moving Average Crossover strategy...")
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
    ma_signals = ma_strategy.generate_signals(data)
    
    ma_engine = BacktestEngine(config)
    ma_result = await ma_engine.run(data, ma_signals)
    
    print(f"    Trades: {ma_result.total_trades}")
    print(f"    Win Rate: {ma_result.win_rate:.1f}%")
    print(f"    Return: {ma_result.total_return_pct:.2f}%")
    print(f"    Max Drawdown: {ma_result.max_drawdown_pct:.2f}%")
    print(f"    Sharpe Ratio: {ma_result.sharpe_ratio:.2f}")
    print(f"    Profit Factor: {ma_result.profit_factor:.2f}")
    
    # Comparison
    print("\n[4] Comparison")
    print("-" * 40)
    print(f"{'Metric':<20} {'Buy&Hold':>12} {'MA Cross':>12}")
    print("-" * 40)
    print(f"{'Total Return':<20} {benchmark_result.total_return_pct:>11.2f}% {ma_result.total_return_pct:>11.2f}%")
    print(f"{'Max Drawdown':<20} {benchmark_result.max_drawdown_pct:>11.2f}% {ma_result.max_drawdown_pct:>11.2f}%")
    print(f"{'Total Trades':<20} {benchmark_result.total_trades:>12} {ma_result.total_trades:>12}")
    print(f"{'Sharpe Ratio':<20} {benchmark_result.sharpe_ratio:>12.2f} {ma_result.sharpe_ratio:>12.2f}")
    
    alpha = ma_result.total_return_pct - benchmark_result.total_return_pct
    print(f"\n{'Alpha vs Benchmark':<20} {alpha:>+11.2f}%")
    
    # Show last few trades
    if ma_result.trades:
        print(f"\n[5] Last 3 trades")
        print("-" * 80)
        for trade in ma_result.trades[-3:]:
            print(f"  {trade.direction.value:>5} | Entry: ${trade.entry_price:,.2f} | "
                  f"Exit: ${trade.exit_price:,.2f} | P&L: ${trade.net_pnl:+.2f} | "
                  f"Reason: {trade.exit_reason}")
    
    print("\n" + "=" * 60)
    print("Backtest complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_comparison())
