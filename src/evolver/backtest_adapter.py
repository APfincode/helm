"""
Backtest Adapter — Connects prompt variants to the backtest engine.

For each prompt variant, this adapter:
1. Runs the variant's prompt through the LLM to generate signals at intervals
2. Feeds those signals into the backtest engine
3. Runs on BOTH Hyperliquid and Binance data
4. Computes cross-exchange consistency metrics

KEY DESIGN DECISIONS:
- We pre-generate ALL signals before running backtest (deterministic, testable)
- LLM calls are budget-tracked and rate-limited
- Signals are cached per-variant to avoid re-generation
- Any LLM error generates NEUTRAL (safe fallback)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from .models import PromptVariant, FitnessScore
from src.llm.client import LLMClient, Signal, LLMError
from src.llm.prompt_engine import PromptEngine, PromptError
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, BacktestResult
from src.data.fetcher import DataFetcher
from src.data.models import DataSource


logger = logging.getLogger(__name__)


class BacktestEvaluator:
    """
    Evaluates a single prompt variant on dual-exchange historical data.
    
    Workflow:
    1. Load HL and BN data for the backtest window
    2. Generate signals using the variant's prompt text
    3. Run backtest on HL data -> result
    4. Run backtest on BN data -> result
    5. Compute cross-exchange metrics
    6. Return FitnessScore
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        hl_config: BacktestConfig,
        bn_config: BacktestConfig,
        signal_interval: int = 24,  # hours
    ):
        self._llm_client = llm_client
        self._hl_config = hl_config
        self._bn_config = bn_config
        self._signal_interval = signal_interval  # Generate signal every N hours
        self._total_llm_cost = 0.0
        self._total_llm_calls = 0
    
    async def evaluate(
        self,
        variant: PromptVariant,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "BTC",
        timeframe: str = "1h",
    ) -> FitnessScore:
        """
        Evaluate a prompt variant on real historical data.
        
        This is the MAIN entry point for fitness evaluation.
        """
        logger.info(f"Evaluating variant {variant.id} (gen {variant.generation}) on {symbol} {timeframe}")
        
        # Step 1: Fetch data from both exchanges
        hl_data, bn_data = await self._fetch_dual_data(
            symbol, timeframe, start_date, end_date
        )
        
        if hl_data.empty or bn_data.empty:
            logger.error("Failed to fetch data for one or both exchanges")
            return self._error_fitness("Data fetch failed")
        
        # Step 2: Generate signals for each dataset
        hl_signals, hl_cost = await self._generate_signals_for_data(
            variant, hl_data
        )
        bn_signals, bn_cost = await self._generate_signals_for_data(
            variant, bn_data
        )
        
        total_cost = hl_cost + bn_cost
        
        # Step 3: Run backtests
        hl_start = time.monotonic()
        hl_result = await self._run_backtest(hl_data, hl_signals, self._hl_config)
        hl_duration = time.monotonic() - hl_start
        
        bn_start = time.monotonic()
        bn_result = await self._run_backtest(bn_data, bn_signals, self._bn_config)
        bn_duration = time.monotonic() - bn_start
        
        # Step 4: Compute consistency metrics
        agreement_rate, return_corr = self._compute_consistency(
            hl_signals, hl_data,
            bn_signals, bn_data
        )
        
        # Step 5: Build fitness score
        fitness = FitnessScore()
        
        # Hyperliquid metrics
        fitness.hyperliquid_return_pct = hl_result.total_return_pct
        fitness.hyperliquid_sharpe = hl_result.sharpe_ratio
        fitness.hyperliquid_max_dd_pct = hl_result.max_drawdown_pct
        fitness.hyperliquid_trades = hl_result.total_trades
        fitness.hyperliquid_win_rate = hl_result.win_rate
        
        # Binance metrics
        fitness.binance_return_pct = bn_result.total_return_pct
        fitness.binance_sharpe = bn_result.sharpe_ratio
        fitness.binance_max_dd_pct = bn_result.max_drawdown_pct
        fitness.binance_trades = bn_result.total_trades
        fitness.binance_win_rate = bn_result.win_rate
        
        # Consistency
        fitness.direction_agreement_rate = agreement_rate
        fitness.return_correlation = return_corr
        
        # Cost tracking
        fitness.total_llm_calls = self._total_llm_calls
        fitness.total_llm_cost_usd = total_cost
        
        # Timing
        fitness.hl_backtest_duration_sec = hl_duration
        fitness.bn_backtest_duration_sec = bn_duration
        
        # Compute composite (via CrossExchangeFitness)
        from .fitness import CrossExchangeFitness, EvolutionConfig
        evo_config = EvolutionConfig()  # Default, could be injected
        calc = CrossExchangeFitness(evo_config)
        fitness.composite_score = calc._compute_composite(fitness)
        
        logger.info(
            f"Variant {variant.id}: HL return={hl_result.total_return_pct:.2f}%, "
            f"BN return={bn_result.total_return_pct:.2f}%, "
            f"composite={fitness.composite_score:.4f}"
        )
        
        return fitness
    
    # ========================================================================
    # Data Fetching
    # ========================================================================
    
    async def _fetch_dual_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch historical data from both Hyperliquid and Binance."""
        fetcher = DataFetcher()
        
        hl_data = pd.DataFrame()
        bn_data = pd.DataFrame()
        
        try:
            async with fetcher:
                # Fetch from both exchanges in parallel
                hl_task = asyncio.create_task(
                    fetcher.get_ohlcv(symbol, timeframe, start, end, use_cache=True)
                )
                bn_task = asyncio.create_task(
                    fetcher.get_ohlcv_binance(symbol, timeframe, start, end, use_cache=True)
                )
                
                hl_data, bn_data = await asyncio.gather(hl_task, bn_task)
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
        
        return hl_data, bn_data
    
    # ========================================================================
    # Signal Generation
    # ========================================================================
    
    async def _generate_signals_for_data(
        self,
        variant: PromptVariant,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        """
        Generate trading signals for each interval in the dataset.
        
        Returns:
            (signals DataFrame, total_llm_cost)
        """
        if data.empty:
            return pd.DataFrame(), 0.0
        
        signals = []
        cost = 0.0
        
        # Generate signal every N hours
        interval_hours = self._signal_interval
        
        # Group timestamps into intervals
        data['interval_group'] = pd.cut(
            pd.RangeIndex(len(data)), 
            bins=max(1, len(data) // interval_hours),
            labels=False
        )
        
        for group_id, group_df in data.groupby('interval_group', sort=True):
            timestamp = group_df.index[-1]  # Use last timestamp of group
            
            # Create market data context (last 20 candles up to this point)
            # For backtesting, we use historical data up to but NOT including future data
            historical_window = data.loc[:timestamp].tail(50)
            
            # Generate signal using variant's prompt
            signal = await self._call_llm_with_variant(variant, historical_window)
            
            signals.append({
                "timestamp": timestamp,
                "signal": signal.direction,
                "confidence": signal.confidence,
                "stop_loss": signal.stop_loss_pct,
                "take_profit": signal.take_profit_pct,
            })
            
            cost += signal.cost_usd
            self._total_llm_calls += 1
        
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df.set_index("timestamp", inplace=True)
        
        return signals_df, cost
    
    async def _call_llm_with_variant(
        self,
        variant: PromptVariant,
        market_data: pd.DataFrame,
    ) -> Signal:
        """
        Call LLM using the variant's prompt text.
        
        Bypasses PromptEngine (which requires a YAML file) and uses raw text.
        """
        try:
            # Format market data for the prompt
            data_summary = self._format_market_data_simple(market_data)
            
            # Inject market data into user template
            user_prompt = variant.user_template.replace("{{market_data}}", data_summary)
            
            # Call LLM directly
            signal = await self._llm_client.generate_signal(
                system_prompt=variant.system_prompt,
                user_prompt=user_prompt,
                temperature=variant.temperature,
                max_tokens=variant.max_tokens,
                use_cache=False,  # Never cache evolutionary signals
            )
            
            return signal
            
        except Exception as e:
            logger.warning(f"LLM call failed for variant {variant.id}: {e}")
            return Signal.neutral(reason=f"LLM error during evolution: {e}")
    
    @staticmethod
    def _format_market_data_simple(data: pd.DataFrame, max_candles: int = 20) -> str:
        """Format market data as simple text (not structured JSON) to save tokens."""
        recent = data.tail(max_candles)
        
        lines = [
            f"Symbol: BTC/USDT",
            f"Timeframe: {max_candles} most recent candles",
            f"Latest close: {recent['close'].iloc[-1]:.2f}",
            f"Period high: {recent['high'].max():.2f}",
            f"Period low: {recent['low'].min():.2f}",
            f"Volume: {recent['volume'].sum():.4f}",
            f"Change: {((recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1) * 100:.2f}%",
            "",
            "Recent candles:",
        ]
        
        for ts, row in recent.iterrows():
            lines.append(
                f"{ts.strftime('%m-%d %H:%M')} O:{row['open']:.1f} H:{row['high']:.1f} "
                f"L:{row['low']:.1f} C:{row['close']:.1f} V:{row['volume']:.0f}"
            )
        
        return "\n".join(lines)
    
    # ========================================================================
    # Backtest Execution
    # ========================================================================
    
    async def _run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run backtest engine on signals."""
        try:
            engine = BacktestEngine(config)
            result = await engine.run(data, signals)
            return result
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            # Return empty result on error
            return BacktestResult(config=config)
    
    # ========================================================================
    # Cross-Exchange Consistency
    # ========================================================================
    
    @staticmethod
    def _compute_consistency(
        hl_signals: pd.DataFrame,
        hl_data: pd.DataFrame,
        bn_signals: pd.DataFrame,
        bn_data: pd.DataFrame,
    ) -> tuple[float, float]:
        """
        Compute consistency metrics between HL and BN signals.
        
        Returns:
            (direction_agreement_rate, return_correlation)
        """
        if hl_signals.empty or bn_signals.empty:
            return 0.0, 0.0
        
        # Align signals by timestamp
        combined = pd.merge(
            hl_signals[['signal']].rename(columns={'signal': 'hl_signal'}),
            bn_signals[['signal']].rename(columns={'signal': 'bn_signal'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        if combined.empty:
            return 0.0, 0.0
        
        # Direction agreement
        agreements = (combined['hl_signal'] == combined['bn_signal']).sum()
        direction_agreement = agreements / len(combined) if len(combined) > 0 else 0.0
        
        # Return correlation (simplified: correlation of price changes around signal times)
        # For now, just use direction agreement as proxy for return correlation
        return_correlation = direction_agreement  # Placeholder for more sophisticated calc
        
        return direction_agreement, return_correlation
    
    # ========================================================================
    # Error Handling
    # ========================================================================
    
    def _error_fitness(self, error_msg: str) -> FitnessScore:
        """Create a zero-fitness score for evaluation errors."""
        fitness = FitnessScore()
        fitness.composite_score = -999.0  # Marker for failure
        fitness.evaluation_error = error_msg
        return fitness
