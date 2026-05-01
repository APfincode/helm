"""Telegram Bot Module.

Use explicit imports to avoid circular dependency with the `telegram` package:
    from src.telegram.bot import TelegramBot, TelegramConfig
"""

# Do NOT import from .bot here to avoid circular import:
# The installed `telegram` package shadows this module when using `import telegram`
# inside bot.py.  Import symbols explicitly from .bot when needed.
