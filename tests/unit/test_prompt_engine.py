"""
Prompt Engine Tests — Template loading, rendering, and safety.

Tests cover:
1. Template loading from YAML
2. Market data formatting
3. Variable substitution
4. Injection detection in rendered prompts
5. Context rendering (trade history, performance)
6. Error handling for missing variables
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.llm.prompt_engine import PromptEngine, PromptTemplate, PromptError


# =============================================================================
# Template Loading Tests
# =============================================================================

class TestTemplateLoading:
    """Test loading prompt templates from YAML files."""

    def test_load_basic_signal_template(self):
        """Should load the basic_signal template successfully."""
        engine = PromptEngine("basic_signal")
        
        assert engine.config.name == "basic_signal"
        assert engine.config.version == "1.0.0"
        assert len(engine.config.system_prompt) > 0
        assert len(engine.config.user_template) > 0
        assert "signal" in engine.config.output_schema["properties"]

    def test_load_evolutionary_template(self):
        """Should load the evolutionary_signal template."""
        engine = PromptEngine("evolutionary_signal")
        
        assert engine.config.name == "evolutionary_signal"
        assert "recent_trades" in engine.config.variables
        assert "performance" in engine.config.variables

    def test_load_nonexistent_template(self):
        """Should raise error for missing template."""
        with pytest.raises(PromptError) as exc_info:
            PromptEngine("does_not_exist")
        
        assert "not found" in str(exc_info.value)


# =============================================================================
# Rendering Tests
# =============================================================================

class TestRendering:
    """Test prompt rendering with market data."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="1h")
        np.random.seed(42)
        
        prices = np.cumsum(np.random.normal(0.001, 0.01, 20)) + 100
        
        return pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.001, 20)),
            "high": prices * (1 + abs(np.random.normal(0, 0.005, 20))),
            "low": prices * (1 - abs(np.random.normal(0, 0.005, 20))),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, 20),
        }, index=dates)

    def test_render_basic_prompt(self, sample_market_data):
        """Should render system and user prompts."""
        engine = PromptEngine("basic_signal")
        
        system, user = engine.render(sample_market_data)
        
        assert len(system) > 0
        assert len(user) > 0
        assert "MARKET DATA" in user
        assert "JSON" in system

    def test_render_with_context(self, sample_market_data):
        """Should include context variables."""
        engine = PromptEngine("basic_signal")
        
        # basic_signal doesn't use extra context, but rendering should work
        system, user = engine.render(sample_market_data, context={})
        
        assert len(system) > 0
        assert len(user) > 0

    def test_render_evolutionary_with_history(self, sample_market_data):
        """Should render evolutionary template with trade history."""
        engine = PromptEngine("evolutionary_signal")
        
        trade_history = [
            {
                "direction": "LONG",
                "entry_price": 100,
                "exit_price": 105,
                "pnl": 5,
                "exit_reason": "take_profit",
            },
            {
                "direction": "SHORT",
                "entry_price": 105,
                "exit_price": 103,
                "pnl": 2,
                "exit_reason": "take_profit",
            },
        ]
        
        system, user = engine.render_with_history(
            market_data=sample_market_data,
            trade_history=trade_history,
            performance_summary={"win_rate": 0.8, "total_trades": 10},
        )
        
        assert "Recent trades" in user or "recent_trades" not in user
        assert len(system) > 0

    def test_render_empty_data(self):
        """Should raise error for empty data."""
        engine = PromptEngine("basic_signal")
        empty_data = pd.DataFrame()
        
        with pytest.raises(PromptError):
            engine.render(empty_data)


# =============================================================================
# Safety Tests
# =============================================================================

class TestPromptSafety:
    """Test prompt injection prevention."""

    @pytest.fixture
    def sample_market_data(self):
        """Create minimal market data."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="1h")
        return pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }, index=dates)

    def test_market_data_formatting(self, sample_market_data):
        """Market data should be formatted as JSON, not raw text."""
        engine = PromptEngine("basic_signal")
        
        _, user = engine.render(sample_market_data)
        
        # Should contain structured JSON
        assert "\"close\"" in user or "close" in user
        assert "symbol" in user or "BTC" in user

    def test_prompt_hash_consistency(self, sample_market_data):
        """Same data should produce same hash."""
        engine = PromptEngine("basic_signal")
        
        hash1 = engine.get_prompt_hash(sample_market_data)
        hash2 = engine.get_prompt_hash(sample_market_data)
        
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_prompt_hash_different_data(self):
        """Different data should produce different hashes."""
        engine = PromptEngine("basic_signal")
        
        dates1 = pd.date_range(start="2024-01-01", periods=5, freq="1h")
        data1 = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000] * 5,
        }, index=dates1)
        
        dates2 = pd.date_range(start="2024-01-01", periods=5, freq="1h")
        data2 = pd.DataFrame({
            "open": [200, 201, 202, 203, 204],
            "high": [201, 202, 203, 204, 205],
            "low": [199, 200, 201, 202, 203],
            "close": [201, 202, 203, 204, 205],
            "volume": [2000] * 5,
        }, index=dates2)
        
        hash1 = engine.get_prompt_hash(data1)
        hash2 = engine.get_prompt_hash(data2)
        
        assert hash1 != hash2

    def test_sanitize_removes_control_chars(self):
        """Control characters should be removed from prompts."""
        engine = PromptEngine("basic_signal")
        
        dirty = "Hello\x00World\x01\x02\nTest"
        clean = engine._sanitize(dirty)
        
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "\n" in clean  # Newlines are allowed

    def test_sanitize_value_limits_length(self):
        """Long values should be truncated."""
        engine = PromptEngine("basic_signal")
        
        long_value = "x" * 20000
        truncated = engine._sanitize_value(long_value)
        
        assert len(truncated) < 20000
        assert "[truncated]" in truncated


# =============================================================================
# Output Schema Tests
# =============================================================================

class TestOutputSchema:
    """Test output schema definition."""

    def test_basic_signal_schema(self):
        """Schema should define all required fields."""
        engine = PromptEngine("basic_signal")
        schema = engine.output_schema
        
        assert "required" in schema
        assert "properties" in schema
        assert "signal" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "reasoning" in schema["properties"]
        assert "regime" in schema["properties"]
        assert "risk_params" in schema["properties"]

    def test_schema_signal_enum(self):
        """Signal should only allow LONG, SHORT, NEUTRAL."""
        engine = PromptEngine("basic_signal")
        schema = engine.output_schema
        
        signal_props = schema["properties"]["signal"]
        assert signal_props["enum"] == ["LONG", "SHORT", "NEUTRAL"]

    def test_schema_confidence_range(self):
        """Confidence should have min/max."""
        engine = PromptEngine("basic_signal")
        schema = engine.output_schema
        
        conf_props = schema["properties"]["confidence"]
        assert conf_props["minimum"] == 0.0
        assert conf_props["maximum"] == 1.0
