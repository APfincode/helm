"""
Prompt Pool — SQLite-backed persistence for evolutionary populations.

Stores:
- All populations (generations)
- All variants with full prompts and fitness
- Lineage tracking (parent-child relationships)
- Best variant promotion history

This enables:
- Resuming evolution runs after interruption
- Analyzing which mutations led to improvements
- Rolling back to previous populations
- Selecting best-ever variant for production
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import PromptVariant, Population, EvolutionConfig


logger = logging.getLogger(__name__)


class PromptPool:
    """
    SQLite persistence for the prompt evolution system.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or Path("data/prompt_evolution.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create tables with schema version tracking."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (1)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS populations (
                    generation INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    evaluated_at TEXT,
                    best_fitness REAL,
                    best_variant_id TEXT,
                    avg_fitness REAL,
                    diversity_score REAL,
                    config_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS variants (
                    id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_template TEXT NOT NULL,
                    temperature REAL,
                    max_tokens INTEGER,
                    parent_ids TEXT,  -- JSON list
                    mutations_applied TEXT,  -- JSON list
                    mutation_descriptions TEXT,  -- JSON list
                    fitness_json TEXT,
                    evaluated INTEGER DEFAULT 0,
                    evaluation_error TEXT,
                    created_at TEXT,
                    prompt_hash TEXT,
                    FOREIGN KEY (generation) REFERENCES populations(generation)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_variants_gen 
                ON variants(generation)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_variants_hash 
                ON variants(prompt_hash)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS best_variants (
                    promoted_at TEXT PRIMARY KEY,
                    variant_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    fitness_score REAL,
                    promotion_reason TEXT
                )
            """)
    
    # ========================================================================
    # Population CRUD
    # ========================================================================
    
    def save_population(self, population: Population, config: Optional[EvolutionConfig] = None) -> None:
        """Save a complete population with all variants."""
        with sqlite3.connect(str(self._db_path)) as conn:
            # Delete old population if exists
            conn.execute("DELETE FROM populations WHERE generation = ?", (population.generation,))
            conn.execute("DELETE FROM variants WHERE generation = ?", (population.generation,))
            
            # Insert population metadata
            best = population.get_best()
            conn.execute(
                """
                INSERT INTO populations (
                    generation, created_at, evaluated_at, best_fitness,
                    best_variant_id, avg_fitness, diversity_score, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    population.generation,
                    population.created_at.isoformat(),
                    population.evaluated_at.isoformat() if population.evaluated_at else None,
                    best.fitness_value if best else None,
                    best.id if best else None,
                    population.avg_fitness,
                    population.diversity_score,
                    json.dumps(config.to_dict()) if config else None,
                )
            )
            
            # Insert all variants
            for variant in population.variants:
                self._insert_variant(conn, variant)
            
            conn.commit()
        
        logger.info(f"Saved population gen={population.generation} with {len(population.variants)} variants")
    
    def load_population(self, generation: int) -> Optional[Population]:
        """Load a population by generation number."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Load population metadata
            row = conn.execute(
                "SELECT * FROM populations WHERE generation = ?",
                (generation,)
            ).fetchone()
            
            if not row:
                return None
            
            # Load variants
            variant_rows = conn.execute(
                "SELECT * FROM variants WHERE generation = ?",
                (generation,)
            ).fetchall()
            
            variants = [self._row_to_variant(dict(v)) for v in variant_rows]
            
            return Population(
                generation=generation,
                variants=variants,
                created_at=datetime.fromisoformat(row["created_at"]),
                evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
                best_fitness_ever=row["best_fitness"] or float("-inf"),
                best_variant_id=row["best_variant_id"],
                avg_fitness=row["avg_fitness"] or 0.0,
                diversity_score=row["diversity_score"] or 0.0,
            )
    
    def load_latest_population(self) -> Optional[Population]:
        """Load the most recently saved population."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT generation FROM populations ORDER BY generation DESC LIMIT 1"
            ).fetchone()
            
            if row:
                return self.load_population(row[0])
            return None
    
    def get_best_variant_ever(self) -> Optional[PromptVariant]:
        """Return the highest-fitness variant across all generations."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            row = conn.execute("""
                SELECT * FROM variants v
                JOIN populations p ON v.generation = p.generation
                WHERE v.evaluated = 1
                ORDER BY json_extract(v.fitness_json, '$.composite_score') DESC
                LIMIT 1
            """).fetchone()
            
            if row:
                return self._row_to_variant(dict(row))
            return None
    
    def save_best_variant(self, variant: PromptVariant, reason: str = "") -> None:
        """Record that a variant was promoted to production."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO best_variants (
                    promoted_at, variant_id, generation, fitness_score, promotion_reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    variant.id,
                    variant.generation,
                    variant.fitness_value if variant.fitness else None,
                    reason,
                )
            )
    
    # ========================================================================
    # Lineage & Analysis
    # ========================================================================
    
    def get_lineage(self, variant_id: str) -> list[dict]:
        """Get ancestry chain for a variant."""
        lineage = []
        current_id = variant_id
        
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            max_depth = 10
            for _ in range(max_depth):
                row = conn.execute(
                    "SELECT * FROM variants WHERE id = ?",
                    (current_id,)
                ).fetchone()
                
                if not row:
                    break
                
                lineage.append(self._row_to_variant(dict(row)).to_dict())
                
                # Move to first parent
                parent_ids = json.loads(row["parent_ids"] or "[]")
                if not parent_ids:
                    break
                current_id = parent_ids[0]
        
        return lineage
    
    def get_mutation_effectiveness(self) -> dict[str, dict]:
        """
        Analyze which mutations tend to improve or hurt fitness.
        
        Returns:
            Dict mapping mutation_type -> {avg_delta, count, improvement_rate}
        """
        stats = {}
        
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            rows = conn.execute("""
                SELECT v.id, v.mutations_applied, v.fitness_json, p.parent_ids
                FROM variants v
                LEFT JOIN variants p ON json_extract(v.parent_ids, '$[0]') = p.id
                WHERE v.evaluated = 1 AND p.evaluated = 1
            """).fetchall()
            
            for row in rows:
                child_fitness = json.loads(row["fitness_json"] or "{}")
                child_score = child_fitness.get("composite_score", 0)
                
                parent_ids = json.loads(row["parent_ids"] or "[]")
                if not parent_ids:
                    continue
                
                parent_row = conn.execute(
                    "SELECT fitness_json FROM variants WHERE id = ?",
                    (parent_ids[0],)
                ).fetchone()
                
                if not parent_row:
                    continue
                
                parent_fitness = json.loads(parent_row["fitness_json"] or "{}")
                parent_score = parent_fitness.get("composite_score", 0)
                
                delta = child_score - parent_score
                mutations = json.loads(row["mutations_applied"] or "[]")
                
                for mutation in mutations:
                    if mutation not in stats:
                        stats[mutation] = {"deltas": [], "count": 0, "improvements": 0}
                    
                    stats[mutation]["deltas"].append(delta)
                    stats[mutation]["count"] += 1
                    if delta > 0:
                        stats[mutation]["improvements"] += 1
        
        # Summarize
        result = {}
        for mutation, data in stats.items():
            result[mutation] = {
                "avg_delta": sum(data["deltas"]) / len(data["deltas"]) if data["deltas"] else 0,
                "count": data["count"],
                "improvement_rate": data["improvements"] / data["count"] if data["count"] else 0,
            }
        
        return result
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _insert_variant(self, conn: sqlite3.Connection, variant: PromptVariant) -> None:
        """Insert a variant into the database."""
        conn.execute(
            """
            INSERT OR REPLACE INTO variants (
                id, generation, system_prompt, user_template, temperature,
                max_tokens, parent_ids, mutations_applied, mutation_descriptions,
                fitness_json, evaluated, evaluation_error, created_at, prompt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                variant.id,
                variant.generation,
                variant.system_prompt,
                variant.user_template,
                variant.temperature,
                variant.max_tokens,
                json.dumps(variant.parent_ids),
                json.dumps([m.value for m in variant.mutations_applied]),
                json.dumps(variant.mutation_descriptions),
                json.dumps(variant.fitness.to_dict()) if variant.fitness else None,
                1 if variant.evaluated else 0,
                variant.evaluation_error,
                variant.created_at.isoformat(),
                variant.prompt_hash,
            )
        )
    
    def _row_to_variant(self, row: dict) -> PromptVariant:
        """Convert database row to PromptVariant."""
        from .models import FitnessScore, MutationType
        
        fitness = None
        if row.get("fitness_json"):
            try:
                fitness = FitnessScore.from_dict(json.loads(row["fitness_json"]))
            except Exception:
                pass
        
        return PromptVariant(
            id=row["id"],
            generation=row["generation"],
            system_prompt=row["system_prompt"],
            user_template=row["user_template"],
            temperature=row.get("temperature", 0.2),
            max_tokens=row.get("max_tokens", 800),
            parent_ids=json.loads(row.get("parent_ids", "[]")),
            mutations_applied=[MutationType(m) for m in json.loads(row.get("mutations_applied", "[]"))],
            mutation_descriptions=json.loads(row.get("mutation_descriptions", "[]")),
            fitness=fitness,
            evaluated=bool(row.get("evaluated", 0)),
            evaluation_error=row.get("evaluation_error"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
