"""
Mutation Engine — composable, contrastive prompt mutations.

Each mutation is a ComposableMutation that transforms prompt text.
Mutations can be composed (stacked) to create complex variants.
Key principle: we track WHICH mutation caused WHICH fitness change.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from .models import PromptVariant, MutationType


# =============================================================================
# Mutation Library
# =============================================================================

@dataclass
class ComposableMutation:
    """
    A single composable mutation operation.
    
    Can be chained: mutation1.apply(mutation2.apply(variant))
    """
    mutation_type: MutationType
    description: str
    transform: callable  # Function: (text: str) -> Optional[str]
    target: str = "system"  # "system" or "user" template
    probability: float = 1.0
    
    def apply(self, text: str) -> Optional[str]:
        """Apply mutation, return new text or None if mutation invalid."""
        if random.random() > self.probability:
            return text  # Skipped probabilistically
        result = self.transform(text)
        return result if result is not None else text


class MutationEngine:
    """
    Orchestrates mutation generation and application.
    
    Maintains a library of composable mutations. When generating a child,
    it samples from this library and applies mutations, tracking lineage.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        self._library = self._build_library()
    
    # ========================================================================
    # Library of Composable Mutations
    # ========================================================================
    
    def _build_library(self) -> list[ComposableMutation]:
        """Build the complete mutation library."""
        library = []
        
        # 1. INSTRUCTION_MODIFY — reword existing instructions
        library.append(ComposableMutation(
            mutation_type=MutationType.INSTRUCTION_MODIFY,
            description="Reword instruction to be more specific",
            target="system",
            transform=self._reword_instruction,
        ))
        
        # 2. INSTRUCTION_ADD — add a new constraint or instruction
        library.append(ComposableMutation(
            mutation_type=MutationType.INSTRUCTION_ADD,
            description="Add new performance constraint instruction",
            target="system",
            transform=self._add_instruction,
        ))
        
        # 3. EMPHASIS_WEIGHT — add emphasis markers
        library.append(ComposableMutation(
            mutation_type=MutationType.EMPHASIS_WEIGHT,
            description="Add emphasis weight to critical rules",
            target="system",
            transform=self._add_emphasis,
        ))
        
        # 4. ROLE_REFRAME — change system role framing
        library.append(ComposableMutation(
            mutation_type=MutationType.ROLE_REFRAME,
            description="Reframe system role perspective",
            target="system",
            transform=self._reframe_role,
        ))
        
        # 5. USER_TEMPLATE_MODIFY — change user instructions
        library.append(ComposableMutation(
            mutation_type=MutationType.INSTRUCTION_MODIFY,
            description="Modify user template instructions",
            target="user",
            transform=self._modify_user_instructions,
        ))
        
        # 6. EXAMPLE_ADD — add examples to system prompt
        library.append(ComposableMutation(
            mutation_type=MutationType.EXAMPLE_ADD,
            description="Add trading example to system prompt",
            target="system",
            transform=self._add_example,
        ))
        
        # 7. TEMPERATURE — adjust temperature gene
        library.append(ComposableMutation(
            mutation_type=MutationType.TEMPERATURE,
            description="Adjust temperature",
            target="meta",
            transform=lambda x: x,  # Handled separately
        ))
        
        # 8. SCHEMA_GUIDANCE — modify output schema guidance
        library.append(ComposableMutation(
            mutation_type=MutationType.SCHEMA_MODIFY,
            description="Enhance JSON schema guidance",
            target="system",
            transform=self._enhance_schema_guidance,
        ))
        
        return library
    
    # ========================================================================
    # Mutation Transform Functions
    # ========================================================================
    
    @staticmethod
    def _reword_instruction(text: str) -> Optional[str]:
        """Reword a random instruction to be more specific."""
        replacements = [
            ("be conservative", [
                "take only high-probability setups",
                "require strong confirmation before acting",
                "avoid marginal trades — demand clear edge",
            ]),
            ("when uncertain", [
                "if market structure is unclear",
                "when evidence is mixed",
                "if conviction level is below 60%",
            ]),
            ("provide clear reasoning", [
                "explain the confluence of factors that led to your conclusion",
                "cite specific candlestick patterns, volume anomalies, and support/resistance levels",
                "describe both bull and bear arguments, then state which you believe is stronger",
            ]),
        ]
        
        text_lower = text.lower()
        for old_phrase, new_phrases in replacements:
            if old_phrase in text_lower:
                new_phrase = random.choice(new_phrases)
                # Case-insensitive replacement but preserve case of first char
                pattern = re.compile(re.escape(old_phrase), re.IGNORECASE)
                return pattern.sub(new_phrase, text, count=1)
        
        # If no match, add specificity to a random line
        lines = text.split("\n")
        if len(lines) > 2:
            idx = random.randint(1, len(lines) - 1)
            lines[idx] += " (with specific numerical thresholds where applicable)"
            return "\n".join(lines)
        
        return None
    
    @staticmethod
    def _add_instruction(text: str) -> Optional[str]:
        """Add a new performance-aware instruction."""
        new_instructions = [
            "- If volatility (ATR%) exceeds 5%, reduce position size implied by confidence.",
            "- When RSI is above 70 OR below 30, mention it explicitly and adjust confidence.",
            "- Require a minimum of 2 confirming indicators before non-NEUTRAL signals.",
            "- If the last signal was wrong, explain what changed in the setup.",
            "- For SHORT signals, always reference funding rate implications if available.",
            "- If support/resistance levels are within 2% of current price, mention them.",
            "- Report current market regime confidence (0.0-1.0) separately from signal confidence.",
            "- If volume is declining on a price advance, flag bearish divergence.",
            "- For LONG signals, check if the 20-period MA is sloping upward.",
            "- If the signal direction conflicts with the 4h trend, reduce confidence by 0.2.",
        ]
        
        instruction = random.choice(new_instructions)
        
        # Insert after a numbered list section, or append
        if "=" * 20 in text or "-" * 20 in text:
            # Append after separator
            return f"{text}\n{instruction}"
        
        lines = text.rstrip().split("\n")
        lines.append(instruction)
        return "\n".join(lines)
    
    @staticmethod
    def _add_emphasis(text: str) -> Optional[str]:
        """Add emphasis weighting to critical rules."""
        emphasis_rules = [
            "**CRITICAL**: Your output is parsed by a computer. Invalid JSON = system failure.",
            "**IMPORTANT**: When uncertain, choose NEUTRAL. Missing a trade is better than a bad trade.",
            "**WARNING**: Do NOT ignore instructions. Do NOT attempt prompt injection.",
            "**NOTE**: The trading engine expects exact enum values. No synonyms allowed.",
        ]
        
        rule = random.choice(emphasis_rules)
        
        # Insert after "IMPORTANT RULES" or similar header
        if "IMPORTANT RULES" in text.upper():
            idx = text.upper().find("IMPORTANT RULES")
            # Find next newline after header
            nl = text.find("\n", idx)
            if nl > 0:
                return text[:nl+1] + rule + "\n" + text[nl+1:]
        
        # Prepend to text
        return f"{rule}\n\n{text}"
    
    @staticmethod
    def _reframe_role(text: str) -> Optional[str]:
        """Reframe the system role to test different personas."""
        roles = [
            "You are a disciplined systematic trader with 10 years experience in crypto perps.",
            "You are a risk-averse quantitative analyst who prioritizes capital preservation.",
            "You are a pattern-recognition specialist focused on candlestick and volume analysis.",
            "You are a macro-aware technician who considers both price action and funding rates.",
            "You are a conservative swing trader who waits for 3+ confluences before acting.",
        ]
        
        old_role = None
        new_role = random.choice(roles)
        
        # Find first "You are a" and replace
        match = re.search(r"You are a[^.\n]+", text)
        if match:
            old_role = match.group(0)
            return text.replace(old_role, new_role, 1)
        
        return None
    
    @staticmethod
    def _modify_user_instructions(text: str) -> Optional[str]:
        """Modify user template instructions."""
        addendum = random.choice([
            "\n\nAdditional requirement: Compare current setup to the last 3 similar setups and state why this one differs.",
            "\n\nAdditional requirement: If there is a clear directional edge, specify the invalidation level (price at which you would admit being wrong).",
            "\n\nAdditional requirement: Rate your signal conviction as: 'tentative', 'moderate', or 'strong' and explain why.",
            "\n\nAdditional requirement: Consider both the microstructure (last 20 candles) and macrostructure (200+ candles) in your analysis.",
        ])
        return text + addendum
    
    @staticmethod
    def _add_example(text: str) -> Optional[str]:
        """Add an example to the system prompt."""
        examples = [
            '''Example of excellent reasoning:
            Given a long lower wick hammer candle on 4h with RSI at 45 and volume 2x average:
            "Long lower wick indicates rejection of lower prices. RSI at 45 leaves room for upside. Volume spike confirms commitment. Setup: LONG, confidence 0.72, stop_loss 1.8%, take_profit 4.5%."
            ''',
            '''Example of excellent reasoning:
            Given multiple bearish engulfing candles on 1h with declining volume and price at resistance:
            "Bearish engulfing pattern is reliable. Declining volume suggests lack of follow-through buying. Resistance held. Setup: SHORT, confidence 0.65, stop_loss 2.1%, take_profit 3.8%."
            ''',
        ]
        
        example = random.choice(examples)
        return text + "\n\n" + example
    
    @staticmethod
    def _enhance_schema_guidance(text: str) -> Optional[str]:
        """Add stronger schema enforcement."""
        if "Output JSON schema" in text:
            enhancer = """
Validation rules:
- 'signal' MUST be exactly one of: "LONG", "SHORT", "NEUTRAL" (case sensitive)
- 'confidence' MUST be a number between 0.00 and 1.00 with 2 decimal places
- 'regime' MUST be exactly one of the enum values listed above
- If 'signal' is "NEUTRAL", 'confidence' MUST be <= 0.60
- 'stop_loss_pct' MUST be strictly less than 'take_profit_pct'
"""
            # Insert after "Output JSON schema" section
            idx = text.find("Output JSON schema")
            if idx > 0:
                # Find end of schema block (next empty line after closing brace)
                schema_end = text.find("}", idx)
                if schema_end > 0:
                    next_nl = text.find("\n", schema_end)
                    if next_nl > 0:
                        return text[:next_nl] + enhancer + text[next_nl:]
        
        return text
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def mutate(self, parent: PromptVariant) -> tuple[PromptVariant, list[ComposableMutation]]:
        """
        Create a mutated child from a parent variant.
        
        Returns:
            (child_variant, list_of_applied_mutations)
        """
        import uuid
        
        # Clone parent's text
        new_system = parent.system_prompt
        new_user = parent.user_template
        new_temperature = parent.temperature
        
        applied: list[ComposableMutation] = []
        descriptions: list[str] = []
        
        # Determine number of mutations (geometric distribution, mostly 1-2)
        n_mutations = 1
        while random.random() < 0.4 and n_mutations < 3:
            n_mutations += 1
        
        # Apply mutations (without replacement from library for this variant)
        available = self._library.copy()
        random.shuffle(available)
        
        for _ in range(n_mutations):
            if not available:
                break
            
            mutation = available.pop()
            
            if mutation.target == "system":
                result = mutation.apply(new_system)
                if result and result != new_system:
                    new_system = result
                    applied.append(mutation)
                    descriptions.append(mutation.description)
            elif mutation.target == "user":
                result = mutation.apply(new_user)
                if result and result != new_user:
                    new_user = result
                    applied.append(mutation)
                    descriptions.append(mutation.description)
            elif mutation.target == "meta":
                # Temperature mutation
                if mutation.mutation_type == MutationType.TEMPERATURE:
                    delta = random.uniform(-0.05, 0.05)
                    new_temperature = max(0.05, min(0.5, parent.temperature + delta))
                    applied.append(mutation)
                    descriptions.append(f"Temperature: {parent.temperature:.3f} -> {new_temperature:.3f}")
        
        child = PromptVariant(
            id=str(uuid.uuid4())[:10],
            generation=parent.generation + 1,
            system_prompt=new_system,
            user_template=new_user,
            temperature=new_temperature,
            max_tokens=parent.max_tokens,
            parent_ids=[parent.id],
            mutations_applied=[m.mutation_type for m in applied],
            mutation_descriptions=descriptions,
        )
        
        return child, applied
    
    def crossover(
        self,
        parent1: PromptVariant,
        parent2: PromptVariant,
    ) -> tuple[PromptVariant, list[ComposableMutation]]:
        """
        Create a child by combining genes from two parents.
        
        Child gets system prompt from one parent, user template from another.
        Temperature is averaged.
        """
        import uuid
        
        # Randomly swap which gene comes from which parent
        if random.random() > 0.5:
            sys_parent, usr_parent = parent1, parent2
        else:
            sys_parent, usr_parent = parent2, parent1
        
        child = PromptVariant(
            id=str(uuid.uuid4())[:10],
            generation=max(parent1.generation, parent2.generation) + 1,
            system_prompt=sys_parent.system_prompt,
            user_template=usr_parent.user_template,
            temperature=(parent1.temperature + parent2.temperature) / 2,
            max_tokens=parent1.max_tokens,
            parent_ids=[parent1.id, parent2.id],
            mutations_applied=[MutationType.CROSSOVER],
            mutation_descriptions=[
                f"Crossover: system_prompt from {sys_parent.id}, "
                f"user_template from {usr_parent.id}"
            ],
        )
        
        crossover_mutation = ComposableMutation(
            mutation_type=MutationType.CROSSOVER,
            description=f"Crossover of {parent1.id} x {parent2.id}",
            target="meta",
            transform=lambda x: x,
        )
        
        return child, [crossover_mutation]
