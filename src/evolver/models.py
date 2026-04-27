"""
Evolution models — Data structures for the Darwinian prompt evolver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class MutationType(str, Enum):
    """Types of prompt mutations."""
    INSTRUCTION_MODIFY = "instruction_modify"      # Change wording in instructions
    INSTRUCTION_ADD = "instruction_add"             # Add new instruction line
    INSTRUCTION_REMOVE = "instruction_remove"       # Remove instruction line
    EMPHASIS_WEIGHT = "emphasis_weight"             # Add emphasis markers (e.g., **IMPORTANT**)
    EXAMPLE_ADD = "example_add"                     # Add an example to system prompt
    EXAMPLE_REMOVE = "example_remove"               # Remove an example
    SCHEMA_MODIFY = "schema_modify"                 # Modify output schema guidance
    ROLE_REFRAME = "role_reframe"                   # Change the system role description
    TEMPERATURE = "temperature"                     # Adjust temperature gene
    CROSSOVER = "crossover"                         # Combine genes from two parents


@dataclass
class PromptVariant:
    """
    A single prompt variant in the population.
    
    Design: system_prompt and user_template are the "genes" that can be independently
    mutated, crossed, and composited. The lineage tracks ancestry for contrastive analysis.
    """
    id: str
    generation: int
    system_prompt: str
    user_template: str
    temperature: float = 0.2
    max_tokens: int = 800
    
    # Evolution metadata
    parent_ids: list[str] = field(default_factory=list)
    mutations_applied: list[MutationType] = field(default_factory=list)
    mutation_descriptions: list[str] = field(default_factory=list)
    
    # Fitness (filled in after evaluation)
    fitness: Optional[FitnessScore] = None
    evaluated: bool = False
    evaluation_error: Optional[str] = None
    
    # Audit
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def full_prompt_text(self) -> str:
        """Combined prompt text for comparison and hashing."""
        return f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{self.user_template}"
    
    @property
    def prompt_hash(self) -> str:
        """Content-addressable hash for deduplication."""
        return hashlib.sha256(self.full_prompt_text.encode()).hexdigest()[:16]
    
    @property
    def fitness_value(self) -> float:
        """Scalar fitness for sorting (negative infinity if not evaluated)."""
        if self.fitness is None:
            return float("-inf")
        return self.fitness.composite_score
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "parent_ids": self.parent_ids,
            "mutations_applied": [m.value for m in self.mutations_applied],
            "mutation_descriptions": self.mutation_descriptions,
            "fitness": self.fitness.to_dict() if self.fitness else None,
            "evaluated": self.evaluated,
            "evaluation_error": self.evaluation_error,
            "created_at": self.created_at.isoformat(),
            "prompt_hash": self.prompt_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVariant:
        fitness = None
        if data.get("fitness"):
            fitness = FitnessScore.from_dict(data["fitness"])
        
        return cls(
            id=data["id"],
            generation=data["generation"],
            system_prompt=data["system_prompt"],
            user_template=data["user_template"],
            temperature=data.get("temperature", 0.2),
            max_tokens=data.get("max_tokens", 800),
            parent_ids=data.get("parent_ids", []),
            mutations_applied=[MutationType(m) for m in data.get("mutations_applied", [])],
            mutation_descriptions=data.get("mutation_descriptions", []),
            fitness=fitness,
            evaluated=data.get("evaluated", False),
            evaluation_error=data.get("evaluation_error"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    
    def __lt__(self, other: PromptVariant) -> bool:
        return self.fitness_value > other.fitness_value  # Higher fitness = "less than" for sorting
    
    def __repr__(self) -> str:
        fitness_str = f"fitness={self.fitness_value:.4f}" if self.fitness else "unevaluated"
        return f"PromptVariant({self.id}, gen={self.generation}, {fitness_str})"


@dataclass
class FitnessScore:
    """
    Multi-objective fitnessscore evaluated on REAL exchange data.
    
    Components:
    - hyperliquid_score: Performance on Hyperliquid historical data
    - binance_score: Performance on Binance Futures historical data
    - consistency_bonus: Signals agree across exchanges (penalizes overfitting)
    - composite_score: Weighted combination used for selection
    """
    # Per-exchange metrics
    hyperliquid_return_pct: float = 0.0
    hyperliquid_sharpe: float = 0.0
    hyperliquid_max_dd_pct: float = 0.0
    hyperliquid_trades: int = 0
    hyperliquid_win_rate: float = 0.0
    
    binance_return_pct: float = 0.0
    binance_sharpe: float = 0.0
    binance_max_dd_pct: float = 0.0
    binance_trades: int = 0
    binance_win_rate: float = 0.0
    
    # Cross-market metrics
    direction_agreement_rate: float = 0.0  # % of time signals match across exchanges
    return_correlation: float = 0.0
    
    # Composite (computed)
    composite_score: float = 0.0
    
    # Metadata
    hl_backtest_duration_sec: float = 0.0
    bn_backtest_duration_sec: float = 0.0
    total_llm_calls: int = 0
    total_llm_cost_usd: float = 0.0
    
    @property
    def avg_return_pct(self) -> float:
        return (self.hyperliquid_return_pct + self.binance_return_pct) / 2
    
    @property
    def avg_sharpe(self) -> float:
        return (self.hyperliquid_sharpe + self.binance_sharpe) / 2
    
    @property
    def avg_max_dd_pct(self) -> float:
        return (self.hyperliquid_max_dd_pct + self.binance_max_dd_pct) / 2
    
    @property
    def avg_win_rate(self) -> float:
        total_trades = self.hyperliquid_trades + self.binance_trades
        if total_trades == 0:
            return 0.0
        total_wins = int(self.hyperliquid_trades * self.hyperliquid_win_rate / 100) + \
                     int(self.binance_trades * self.binance_win_rate / 100)
        return (total_wins / total_trades) * 100
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperliquid_return_pct": self.hyperliquid_return_pct,
            "hyperliquid_sharpe": self.hyperliquid_sharpe,
            "hyperliquid_max_dd_pct": self.hyperliquid_max_dd_pct,
            "hyperliquid_trades": self.hyperliquid_trades,
            "hyperliquid_win_rate": self.hyperliquid_win_rate,
            "binance_return_pct": self.binance_return_pct,
            "binance_sharpe": self.binance_sharpe,
            "binance_max_dd_pct": self.binance_max_dd_pct,
            "binance_trades": self.binance_trades,
            "binance_win_rate": self.binance_win_rate,
            "direction_agreement_rate": self.direction_agreement_rate,
            "return_correlation": self.return_correlation,
            "composite_score": self.composite_score,
            "hl_backtest_duration_sec": self.hl_backtest_duration_sec,
            "bn_backtest_duration_sec": self.bn_backtest_duration_sec,
            "total_llm_calls": self.total_llm_calls,
            "total_llm_cost_usd": self.total_llm_cost_usd,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitnessScore:
        fs = cls()
        for key, value in data.items():
            if hasattr(fs, key):
                setattr(fs, key, value)
        return fs


@dataclass
class Population:
    """
    A generation of prompt variants.
    
    Lineage tracking enables contrastive analysis: by comparing generations,
    we can isolate which mutations caused fitness improvements.
    """
    generation: int
    variants: list[PromptVariant] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    evaluated_at: Optional[datetime] = None
    
    # Cross-generation metadata
    best_fitness_ever: float = float("-inf")
    best_variant_id: Optional[str] = None
    avg_fitness: float = 0.0
    diversity_score: float = 0.0  # Average hash distance between variants
    
    def sort_by_fitness(self) -> None:
        """Sort variants descending by fitness (best first)."""
        self.variants.sort(key=lambda v: v.fitness_value, reverse=True)
    
    def get_best(self) -> Optional[PromptVariant]:
        """Return the highest-fitness variant."""
        if not self.variants:
            return None
        self.sort_by_fitness()
        return self.variants[0]
    
    def get_top_k(self, k: int) -> list[PromptVariant]:
        """Return top k variants by fitness."""
        self.sort_by_fitness()
        return self.variants[:k]
    
    def compute_diversity(self) -> float:
        """
        Measure average hamming-like distance between prompt hashes.
        High diversity = more exploration = less local maxima risk.
        """
        if len(self.variants) < 2:
            return 0.0
        
        hash_pairs = [(v1.prompt_hash, v2.prompt_hash) for i, v1 in enumerate(self.variants) for v2 in self.variants[i+1:]]
        if not hash_pairs:
            return 0.0
        
        distances = []
        for h1, h2 in hash_pairs:
            # Simple: count differing hex digits
            diff = sum(c1 != c2 for c1, c2 in zip(h1, h2)) / len(h1)
            distances.append(diff)
        
        self.diversity_score = sum(distances) / len(distances)
        return self.diversity_score
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "variants": [v.to_dict() for v in self.variants],
            "created_at": self.created_at.isoformat(),
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "best_fitness_ever": self.best_fitness_ever,
            "best_variant_id": self.best_variant_id,
            "avg_fitness": self.avg_fitness,
            "diversity_score": self.diversity_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Population:
        pop = cls(
            generation=data["generation"],
            variants=[PromptVariant.from_dict(v) for v in data.get("variants", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            evaluated_at=datetime.fromisoformat(data["evaluated_at"]) if data.get("evaluated_at") else None,
            best_fitness_ever=data.get("best_fitness_ever", float("-inf")),
            best_variant_id=data.get("best_variant_id"),
            avg_fitness=data.get("avg_fitness", 0.0),
            diversity_score=data.get("diversity_score",  0.0),
        )
        return pop


@dataclass
class EvolutionConfig:
    """
    Configuration for the evolutionary process.
    
    All parameters are validated on init to prevent misconfiguration.
    """
    # Population genetics
    population_size: int = 10
    elite_count: int = 2  # Top N automatically survive
    mutation_rate: float = 0.7  # Probability of mutation per gene
    crossover_rate: float = 0.3  # Probability of crossover (vs asexual reproduction)
    
    # Selection
    tournament_size: int = 3
    
    # Evaluation
    backtest_window_days: int = 90  # Days of historical data to test on
    signal_interval_hours: int = 24  # Generate signal every N hours
    
    # Exchanges to evaluate on
    hl_weight: float = 0.6  # Hyperliquid fitness weight
    bn_weight: float = 0.4  # Binance fitness weight
    consistency_weight: float = 0.15  # Bonus for cross-exchange agreement
    
    # Fitness normalization
    min_trades_for_fitness: int = 5  # Variants with fewer trades get penalty
    max_drawdown_penalty: float = 2.0  # Multiplier: fitness /= (1 + max_dd * penalty)
    
    # Termination
    max_generations: int = 20
    convergence_generations: int = 5  # Stop if best fitness doesn't improve for N gens
    convergence_threshold: float = 0.001  # Improvement must exceed this
    
    # Budget
    max_evolution_cost_usd: float = 50.0  # Hard ceiling for entire evolution run
    
    # Safety
    require_injection_check: bool = True
    max_prompt_length: int = 8000
    
    # Output
    log_interval: int = 1  # Log every N generations
    save_population_interval: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.population_size >= 3, "Population must be >= 3"
        assert self.elite_count < self.population_size, "Elite must be < population"
        assert 0 <= self.mutation_rate <= 1, "Mutation rate must be in [0,1]"
        assert 0 <= self.crossover_rate <= 1, "Crossover rate must be in [0,1]"
        assert self.hl_weight + self.bn_weight == 1.0, "Exchange weights must sum to 1"
        assert self.backtest_window_days >= 7, "Backtest must be at least 7 days"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "backtest_window_days": self.backtest_window_days,
            "signal_interval_hours": self.signal_interval_hours,
            "hl_weight": self.hl_weight,
            "bn_weight": self.bn_weight,
            "consistency_weight": self.consistency_weight,
            "min_trades_for_fitness": self.min_trades_for_fitness,
            "max_drawdown_penalty": self.max_drawdown_penalty,
            "max_generations": self.max_generations,
            "convergence_generations": self.convergence_generations,
            "convergence_threshold": self.convergence_threshold,
            "max_evolution_cost_usd": self.max_evolution_cost_usd,
            "require_injection_check": self.require_injection_check,
            "max_prompt_length": self.max_prompt_length,
            "log_interval": self.log_interval,
            "save_population_interval": self.save_population_interval,
        }
