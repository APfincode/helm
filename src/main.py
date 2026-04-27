"""
Hyper-Alpha-Arena V2 — Main entry point.

Usage:
    python -m src.main
    hyper-alpha
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.loader import ConfigLoader, ConfigLoadError
from src.security.secrets_manager import SecretAccessError, get_secrets_manager
from src.security.audit_logger import AuditLogger, EventType, Severity


async def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("=" * 60)
    print("Hyper-Alpha-Arena V2")
    print("AI Crypto Trading Bot")
    print("=" * 60)

    # Phase 0: Security foundation check
    print("\n[Phase 0] Security Foundation Check")
    print("-" * 40)

    try:
        # Load secrets
        secrets = get_secrets_manager()
        print("✓ Secrets manager initialized")

        # Verify required secrets exist
        required = [
            "HYPERLIQUID_WALLET",
            "HYPERLIQUID_PRIVATE_KEY",
            "HMAC_SECRET_KEY",
        ]
        secrets.verify_all_required(required)
        print("✓ Required secrets verified")

        # Load configuration
        loader = ConfigLoader()
        config = loader.load()
        print(f"✓ Configuration loaded (env: {config.env})")

        # Initialize audit logger
        async with AuditLogger() as logger:
            await logger.log(
                event_type=EventType.SYSTEM,
                action="BOT_STARTUP",
                actor="system",
                severity=Severity.INFO,
                details={
                    "env": config.env,
                    "version": "0.1.0",
                    "auto_trading": config.auto_trading.enabled,
                },
            )
            print("✓ Audit logger initialized")

        # Display config status (safe version)
        safe_config = config.to_safe_dict()
        print(f"\nConfiguration Summary:")
        print(f"  Environment: {safe_config['env']}")
        print(f"  Debug: {safe_config['debug']['enabled']}")
        print(f"  Auto-trading: {safe_config['auto_trading']['enabled']}")
        print(f"  Daily loss breaker: {safe_config['daily_loss_breaker']['enabled']}")
        print(f"  Drawdown breaker: {safe_config['drawdown_breaker']['enabled']}")

        print("\n" + "=" * 60)
        print("Phase 0 complete. Security foundation is solid.")
        print("Ready for Phase 1: Hardened Foundation.")
        print("=" * 60)

        return 0

    except SecretAccessError as e:
        print(f"\n✗ Secret access error: {e}")
        print("  Please check your .env file and ensure all required variables are set.")
        return 1

    except ConfigLoadError as e:
        print(f"\n✗ Configuration error: {e}")
        print("  Please check your config files and environment variables.")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Goodbye.")
        sys.exit(0)
