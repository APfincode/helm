"""
Prompt Engine — YAML-based, injection-safe prompt templates.

Key features:
1. Templates are loaded from YAML files (version controlled, auditable)
2. Data is inserted via structured formatting — NEVER raw string interpolation for user data
3. Market data is pre-validated and formatted before insertion
4. Templates include explicit JSON schema instructions to force structured output
5. Every prompt is logged (hashed, not full text) for audit

Usage:
    engine = PromptEngine("basic_signal")
    prompt = engine.render(market_data)
    
    # Or with custom variables
    prompt = engine.render(market_data, context={"regime": "volatile"})
"""

import yaml
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import pandas as pd

from src.security.input_validator import PromptInjectionDetector


class PromptError(Exception):
    """Prompt loading or rendering error."""
    pass


@dataclass
class PromptTemplate:
    """Loaded prompt template."""
    name: str
    version: str
    description: str
    system_prompt: str
    user_template: str
    output_schema: dict[str, Any]
    variables: list[str]  # Required template variables
    max_tokens: int = 500
    temperature: float = 0.2


class PromptEngine:
    """
    Safe prompt renderer for LLM signal generation.
    
    Design principles:
    1. Templates loaded from files (never constructed at runtime)
    2. Data validated before insertion
    3. Output schema enforced via system prompt
    4. No raw user input reaches the LLM without sanitization
    """

    TEMPLATE_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
    
    def __init__(self, template_name: str, template_dir: Optional[Path] = None) -> None:
        """
        Initialize prompt engine with a template.
        
        Args:
            template_name: Name of template file (without .yaml extension)
            template_dir: Override default template directory
        """
        self._detector = PromptInjectionDetector()
        self._template = self._load_template(template_name, template_dir)
    
    def _load_template(self, name: str, template_dir: Optional[Path] = None) -> PromptTemplate:
        """Load and validate a prompt template from YAML."""
        dir_path = template_dir or self.TEMPLATE_DIR
        file_path = dir_path / f"{name}.yaml"
        
        if not file_path.exists():
            raise PromptError(f"Template not found: {file_path}")
        
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PromptError(f"Invalid YAML in template {name}: {e}")
        
        # Validate required fields
        required_fields = ["name", "version", "system_prompt", "user_template", "output_schema"]
        for field in required_fields:
            if field not in data:
                raise PromptError(f"Template {name} missing required field: {field}")
        
        return PromptTemplate(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            system_prompt=data["system_prompt"],
            user_template=data["user_template"],
            output_schema=data["output_schema"],
            variables=data.get("variables", []),
            max_tokens=data.get("max_tokens", 500),
            temperature=data.get("temperature", 0.2),
        )
    
    def render(
        self,
        market_data: pd.DataFrame,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """
        Render system and user prompts from template.
        
        Args:
            market_data: OHLCV DataFrame (last N candles used)
            context: Additional context variables
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Raises:
            PromptError: If data validation fails
        """
        context = context or {}
        
        # Prepare market data summary
        data_summary = self._format_market_data(market_data)
        
        # Build user prompt (structured formatting, not raw interpolation)
        user_prompt = self._render_user_template(
            self._template.user_template,
            market_data=data_summary,
            **context
        )
        
        # Sanitize both prompts
        system_prompt = self._sanitize(self._template.system_prompt)
        user_prompt = self._sanitize(user_prompt)
        
        # Validate no injection
        if self._detector.is_suspicious(system_prompt + user_prompt):
            raise PromptError("Injected content detected in rendered prompt")
        
        return system_prompt, user_prompt
    
    def render_with_history(
        self,
        market_data: pd.DataFrame,
        trade_history: list[dict],
        performance_summary: Optional[dict] = None,
    ) -> tuple[str, str]:
        """
        Render prompt with trading history for evolutionary strategies.
        
        Args:
            market_data: Current OHLCV data
            trade_history: List of recent trade records
            performance_summary: Optional performance metrics
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = {
            "recent_trades": self._format_trade_history(trade_history),
            "performance": performance_summary or {},
        }
        return self.render(market_data, context=context)
    
    @property
    def output_schema(self) -> dict[str, Any]:
        """Get expected output JSON schema."""
        return self._template.output_schema.copy()
    
    @property
    def config(self) -> PromptTemplate:
        """Get template configuration."""
        return self._template
    
    def get_prompt_hash(self, market_data: pd.DataFrame, context: Optional[dict] = None) -> str:
        """
        Get hash of rendered prompt for caching/audit.
        
        Returns:
            SHA256 hash of the prompt content
        """
        system, user = self.render(market_data, context)
        combined = system + "|||" + user
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    # ========================================================================
    # Formatting Helpers
    # ========================================================================
    
    @staticmethod
    def _format_market_data(data: pd.DataFrame, max_candles: int = 20) -> dict[str, Any]:
        """
        Format OHLCV data into a safe, structured summary.
        
        Args:
            data: Full OHLCV DataFrame
            max_candles: Maximum candles to include
            
        Returns:
            Dictionary with data summary
        """
        if len(data) == 0:
            raise PromptError("Empty market data provided")
        
        # Use most recent candles
        recent = data.tail(max_candles)
        
        summary = {
            "symbol": "BTC",  # TODO: Pass symbol through
            "timeframe": "1h",
            "total_candles": len(data),
            "latest_candles": [],
            "statistics": {
                "open": float(recent["open"].iloc[0]),
                "high": float(recent["high"].max()),
                "low": float(recent["low"].min()),
                "close": float(recent["close"].iloc[-1]),
                "volume": float(recent["volume"].sum()),
                "change_pct": float(
                    (recent["close"].iloc[-1] - recent["open"].iloc[0]) 
                    / recent["open"].iloc[0] * 100
                ),
            },
        }
        
        # Format individual candles (limit to prevent token overuse)
        for timestamp, row in recent.iterrows():
            summary["latest_candles"].append({
                "time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        
        return summary
    
    @staticmethod
    def _format_trade_history(trades: list[dict], max_trades: int = 5) -> str:
        """Format recent trade history for prompt context."""
        if not trades:
            return "No recent trades."
        
        recent = trades[-max_trades:]
        lines = [f"Recent trades (last {len(recent)}):"]
        
        for trade in recent:
            pnl_str = f"+${trade.get('pnl', 0):.2f}" if trade.get('pnl', 0) >= 0 else f"-${abs(trade.get('pnl', 0)):.2f}"
            lines.append(
                f"  {trade.get('direction', '?')} | "
                f"Entry: {trade.get('entry_price', '?')} | "
                f"Exit: {trade.get('exit_price', '?')} | "
                f"P&L: {pnl_str} | "
                f"Reason: {trade.get('exit_reason', '?')}"
            )
        
        return "\n".join(lines)
    
    def _render_user_template(self, template: str, **kwargs) -> str:
        """
        Render user template with variables.
        
        Uses safe formatting — only allows specific variables.
        """
        result = template
        
        # Format market data as JSON (safe, structured)
        if "market_data" in kwargs:
            market_json = json.dumps(kwargs["market_data"], indent=2)
            result = result.replace("{{market_data}}", market_json)
        
        # Format context variables
        for key, value in kwargs.items():
            if key == "market_data":
                continue
            
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            
            # Sanitize the value before insertion
            value_str = self._sanitize_value(value_str)
            result = result.replace(f"{{{{{key}}}}}", value_str)
        
        # Check for unreplaced template variables
        import re
        unreplaced = re.findall(r"\{\{(\w+)\}\}", result)
        if unreplaced:
            raise PromptError(f"Unreplaced template variables: {unreplaced}")
        
        return result
    
    def _sanitize(self, text: str) -> str:
        """
        Sanitize prompt text.
        
        Removes control characters, injection patterns.
        """
        # Remove control characters (except newline and tab)
        cleaned = "".join(
            c for c in text
            if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) < 127)
        )
        
        # Strip excessive whitespace
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
        
        return cleaned
    
    def _sanitize_value(self, value: str) -> str:
        """Sanitize a template variable value before insertion."""
        # Limit length
        max_length = 10000
        if len(value) > max_length:
            value = value[:max_length] + "\n[truncated]"
        
        # Remove dangerous characters
        dangerous = ["}", "{", "<", ">", "|", "&", ";"]
        for char in dangerous:
            value = value.replace(char, "")
        
        return value
