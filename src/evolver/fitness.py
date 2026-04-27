"""
Cross-Exchange Fitness Calculator

Evaluates prompt variants on REAL historical data from both Hyperliquid and Binance,
computing a weighted composite fitness score.

The composite score rewards:
- Profitability on BOTH exchanges (not just one)
- Risk-adjusted returns (Sharpe ratio)
- Cross-exchange consistency (signals that agree)
- Penalizes: excessive drawdowns, too few trades (overfitting risk)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import FitnessScore, EvolutionConfig
from src.backtest.models import BacktestResult


@dataclass
class FitnessWeights:
    """Configurable weights for the composite fitness formula."""
    return_weight: float = 0.40
    sharpe_weight: float = 0.25
    drawdown_weight: float = 0.20
    consistency_weight: float = 0.15

    def __post_init__(self):
        total = self.return_weight + self.sharpe_weight + self.drawdown_weight + self.consistency_weight
        assert abs(total - 1.0) < 0.001, f"Fitness weights must sum to 1, got {total}"


class CrossExchangeFitness:
    """
    Calculates fitness scores by combining backtest results from both exchanges.
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        weights: Optional[FitnessWeights] = None,
    ):
        self._evolution_config = config
        self._weights = weights or FitnessWeights()
    
    def calculate(
        self,
        hl_result: BacktestResult,
        bn_result: BacktestResult,
        direction_agreement_rate: float,
        return_correlation: float,
        llm_calls: int,
        llm_cost: float,
    ) -> FitnessScore:
        """
        Calculate composite fitness from dual-exchange backtest results.
        
        Formula (all terms are normalized to [0, 1] before weighting):
        - Return component: weighted average of HL and BN return, clipped to [-50%, +50%] per month
        - Sharpe component: weighted average of HL and BN Sharpe, clipped to [-2, +4]
        - Drawdown penalty: penalizes max drawdown, clipped to [0%, 30%]
        - Consistency bonus: rewards cross-exchange agreement
        """
        
        fitness = FitnessScore()
        
        # Raw metrics from backtest
        fitness.hyperliquid_return_pct = hl_result.total_return_pct
        fitness.hyperliquid_sharpe = hl_result.sharpe_ratio
        fitness.hyperliquid_max_dd_pct = hl_result.max_drawdown_pct
        fitness.hyperliquid_trades = hl_result.total_trades
        fitness.hyperliquid_win_rate = hl_result.win_rate
        
        fitness.binance_return_pct = bn_result.total_return_pct
        fitness.binance_sharpe = bn_result.sharpe_ratio
        fitness.binance_max_dd_pct = bn_result.max_drawdown_pct
        fitness.binance_trades = bn_result.total_trades
        fitness.binance_win_rate = bn_result.win_rate
        
        # Cross-market metrics
        fitness.direction_agreement_rate = direction_agreement_rate
        fitness.return_correlation = return_correlation
        
        # Cost tracking
        fitness.total_llm_calls = llm_calls
        fitness.total_llm_cost_usd = llm_cost
        
        # Compute composite
        fitness.composite_score = self._compute_composite(fitness)
        
        return fitness
    
    def _compute_composite(self, fitness: FitnessScore) -> float:
        """
        Core fitness formula.
        
        All components normalized to [0, 1] range where 1 = best.
        """
        hl_weight = self._evolution_config.hl_weight
        bn_weight = self._evolution_config.bn_weight
        
        # === Return Component (40% weight) ===
        # Normalize: assume monthly return range [-50%, +50%] = [0, 1]
        hl_return_norm = self._normalize_return(fitness.hyperliquid_return_pct)
        bn_return_norm = self._normalize_return(fitness.binance_return_pct)
        return_component = hl_weight * hl_return_norm + bn_weight * bn_return_norm
        
        # === Sharpe Component (25% weight) ===
        # Normalize: Sharpe [-2, +4] = [0, 1]
        hl_sharpe_norm = self._normalize_sharpe(fitness.hyperliquid_sharpe)
        bn_sharpe_norm = self._normalize_sharpe(fitness.binance_sharpe)
        sharpe_component = hl_weight * hl_sharpe_norm + bn_weight * bn_sharpe_norm
        
        # === Drawdown Penalty (20% weight) ===
        # Lower drawdown = higher fitness. Max at 30% drawdown = 0
        hl_dd_component = self._normalize_drawdown(fitness.hyperliquid_max_dd_pct)
        bn_dd_component = self._normalize_drawdown(fitness.binance_max_dd_pct)
        dd_component = hl_weight * hl_dd_component + bn_weight * bn_dd_component
        
        # === Consistency Component (15% weight) ===
        # Direction agreement rate already in [0, 1]
        consistency_component = min(1.0, fitness.direction_agreement_rate)
        
        # === Trade Count Penalty ===
        # Too few trades = overfitting risk, penalize heavily
        total_trades = fitness.hyperliquid_trades + fitness.binance_trades
        min_trades = self._evolution_config.min_trades_for_fitness
        if total_trades < min_trades * 2:  # Need meaningful sample on both exchanges
            trade_penalty = 0.5 + 0.5 * (total_trades / (min_trades * 2))
        else:
            trade_penalty = 1.0
        
        # === Composite ===
        raw_score = (
            self._weights.return_weight * return_component +
            self._weights.sharpe_weight * sharpe_component +
            self._weights.drawdown_weight * dd_component +
            self._weights.consistency_weight * consistency_component
        )
        
        # Apply penalties
        score = raw_score * trade_penalty
        
        # Ensure strictly increasing with Sharpe (no "high return, terrible risk" pathologies)
        if fitness.avg_sharpe < 0 and fitness.avg_return_pct > 0:
            # Negative Sharpe with positive return = lucky, not skilled
            score *= max(0.1, 1 + fitness.avg_sharpe * 0.5)
        
        return max(0.0, score)
    
    @staticmethod
    def _normalize_return(return_pct: float) -> float:
        """Normalize return to [0, 1]. -50% -> 0, 0% -> 0.5, +50% -> 1"""
        return max(0.0, min(1.0, (return_pct + 50.0) / 100.0))
    
    @staticmethod
    def _normalize_sharpe(sharpe: float) -> float:
        """Normalize Sharpe to [0, 1]. -2 -> 0, 0 -> 0.33, +4 -> 1"""
        return max(0.0, min(1.0, (sharpe + 2.0) / 6.0))
    
    @staticmethod
    def _normalize_drawdown(dd_pct: float) -> float:
        """Normalize drawdown to [0, 1 fitness] where 0 drawdown = 1, 30% = 0."""
        return max(0.0, 1.0 - (dd_pct / 30.0))
