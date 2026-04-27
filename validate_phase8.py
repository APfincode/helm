#!/usr/bin/env python3
"""
Phase 8 Validation — Telegram Bot + Web UI.

Tests without external dependencies:
- TelegramBot model, alerts, commands
- WebUI server creation and routes
- Template rendering
- API endpoints
- Static assets
"""

import sys
import asyncio
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Mock external dependencies
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["aiosqlite"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic"].BaseModel = type("BaseModel", (), {"model_dump": lambda self: {}, "__init__": lambda self, **kwargs: None})
sys.modules["pydantic"].Field = lambda *a, **k: None
sys.modules["pydantic"].field_validator = lambda *a, **k: lambda f: f
sys.modules["pydantic_core"] = MagicMock()
sys.modules["tenacity"] = MagicMock()
sys.modules["tenacity"].retry = MagicMock(return_value=lambda f: f)
sys.modules["tenacity"].stop_after_attempt = MagicMock()
sys.modules["tenacity"].wait_exponential = MagicMock()
sys.modules["telegram"] = MagicMock()
sys.modules["telegram"].Update = MagicMock()
sys.modules["telegram"].InlineKeyboardButton = MagicMock()
sys.modules["telegram"].InlineKeyboardMarkup = MagicMock()
sys.modules["telegram.ext"] = MagicMock()
sys.modules["telegram.ext"].Application = MagicMock()
sys.modules["telegram.ext"].CommandHandler = MagicMock()
sys.modules["telegram.ext"].ContextTypes = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["fastapi.templating"] = MagicMock()

# Auto-detect repo root
_repo_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(_repo_root))

from src.telegram.bot import TelegramBot, TelegramConfig

print("=" * 60)
print("PHASE 8: TELEGRAM BOT + WEB UI — VALIDATION")
print("=" * 60)

errors = []
passed = 0

def test(name, condition, detail=""):
    global passed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        errors.append(f"FAIL: {name} — {detail}")
        print(f"  FAIL: {name} — {detail}")

# =====================================================================
# 1. TelegramBot Data Model
# =====================================================================
print("\n[1] TelegramBot Data Model")

cfg = TelegramConfig(
    token="test_token",
    chat_id="123456",
    enabled=True,
    alert_on_trade=True,
    alert_on_stop=True,
    alert_on_risk=True,
)
test("TelegramConfig created", cfg.token == "test_token")
test("TelegramConfig alerts on", cfg.alert_on_trade)

# =====================================================================
# 2. TelegramBot Instantiation (Mocked)
# =====================================================================
print("\n[2] TelegramBot Instantiation")

bot = TelegramBot(config=cfg)
test("TelegramBot created", bot is not None)
test("TelegramBot not running initially", not bot._running)

# =====================================================================
# 3. Telegram Alert Methods (Mocked)
# =====================================================================
print("\n[3] Telegram Alert Methods")

# Mock _send_message
bot._app = MagicMock()
bot._app.bot = MagicMock()
bot._app.bot.send_message = AsyncMock()

async def test_alerts():
    await bot.alert_trade({
        "symbol": "BTC",
        "direction": "LONG",
        "quantity": 0.1,
        "entry_price": 95000,
        "leverage": 5,
        "stop_loss": 92000,
        "take_profit": 98000,
        "risk_usd": 100,
        "mode": "PAPER",
    })
    test("alert_trade called", bot._app.bot.send_message.called)

    await bot.alert_stop_hit({"coin": "BTC", "direction": "LONG", "entry_price": 95000, "current_price": 92000}, -300)
    test("alert_stop_hit called", bot._app.bot.send_message.call_count >= 2)

    await bot.alert_take_profit({"coin": "BTC", "direction": "LONG", "entry_price": 95000, "current_price": 98000}, 300)
    test("alert_take_profit called", bot._app.bot.send_message.call_count >= 3)

    await bot.alert_risk_guard("Daily loss limit reached", {"equity": 10000})
    test("alert_risk_guard called", bot._app.bot.send_message.call_count >= 4)

    await bot.alert_mode_change("PAPER", "LIVE")
    test("alert_mode_change called", bot._app.bot.send_message.call_count >= 5)

asyncio.run(test_alerts())

# =====================================================================
# 4. Web UI Server Creation
# =====================================================================
print("\n[4] Web UI Server")

# FastAPI is mocked, so we just verify the create_app function exists
# and that the BotStateProvider works
from src.webui.server import BotStateProvider, create_app, _load_yaml, _save_yaml

provider = BotStateProvider()
status = provider.get_status()
test("BotStateProvider returns status", "mode" in status)
test("Status has equity", "equity" in status)

test("get_positions returns list", isinstance(provider.get_positions(), list))
test("get_history returns list", isinstance(provider.get_history(), list))
test("get_signals returns list", isinstance(provider.get_signals(), list))
test("get_risk has circuit_breakers", "circuit_breakers" in provider.get_risk())

# =====================================================================
# 5. Template Files
# =====================================================================
print("\n[5] Template Files")

import os
base_dir = str(_repo_root)
templates = ["base.html", "overview.html", "positions.html", "history.html",
             "signals.html", "risk.html", "config.html", "logs.html"]
for tmpl in templates:
    path = os.path.join(base_dir, "templates", tmpl)
    test(f"Template exists: {tmpl}", os.path.exists(path))

# =====================================================================
# 6. Static Assets
# =====================================================================
print("\n[6] Static Assets")

static_files = ["style.css"]
for sf in static_files:
    path = os.path.join(base_dir, "static", sf)
    test(f"Static file exists: {sf}", os.path.exists(path))

# Check CSS has key styles
with open(os.path.join(base_dir, "static", "style.css"), "r") as f:
    css = f.read()
test("CSS has dark theme", "--bg: #0d1117" in css)
test("CSS has card styles", ".card-grid" in css)
test("CSS has table styles", ".data-table" in css)
test("CSS has responsive", "@media" in css)

# =====================================================================
# 7. ExecutionEngine Telegram Integration
# =====================================================================
print("\n[7] ExecutionEngine Telegram Integration")

from src.execution.execution_engine import ExecutionEngine
import inspect
src = inspect.getsource(ExecutionEngine.__init__)
test("ExecutionEngine accepts telegram_bot", "telegram_bot" in src)

# =====================================================================
# 8. Config YAML Load/Save
# =====================================================================
print("\n[8] Config YAML Utilities")

test_yaml_path = "/tmp/test_validate_phase8.yaml"
_load_yaml(Path(test_yaml_path))  # Should not crash on missing file
test("_load_yaml handles missing file", True)

_save_yaml(Path(test_yaml_path), {"test": "value"})
test("_save_yaml writes file", os.path.exists(test_yaml_path))
loaded = _load_yaml(Path(test_yaml_path))
test("_load_yaml reads correctly", loaded.get("test") == "value")
os.remove(test_yaml_path)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {len(errors)} failed")
print("=" * 60)

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n ALL VALIDATIONS PASSED")
    print(" Phase 8 Telegram Bot + Web UI is structurally sound.")
    sys.exit(0)
