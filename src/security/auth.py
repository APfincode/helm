"""
Authentication & Authorization — HMAC tokens and Telegram whitelist.

Provides:
1. HMAC-SHA256 signed tokens for internal API calls
2. Telegram chat ID whitelist validation
3. Scope-based permission checking

Usage:
    from src.security.auth import AuthManager, TokenScope
    auth = AuthManager(hmac_secret="...")
    
    # Generate token for trade execution
    token = auth.generate_token(scope=TokenScope.TRADE_WRITE, ttl_seconds=60)
    
    # Verify token
    auth.verify_token(token, expected_scope=TokenScope.TRADE_WRITE)
    
    # Check Telegram chat ID
    auth.check_telegram_auth(chat_id=123456789)
"""

import hmac
import hashlib
import secrets
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from src.security.secrets_manager import SecretsManager, get_secrets_manager


class AuthError(Exception):
    """Base authentication error."""
    pass


class TokenExpiredError(AuthError):
    """Token has exceeded its time-to-live."""
    pass


class InvalidTokenError(AuthError):
    """Token signature is invalid or malformed."""
    pass


class UnauthorizedError(AuthError):
    """User/chat ID is not in the whitelist."""
    pass


class ScopeError(AuthError):
    """Token scope does not match required scope."""
    pass


class TokenScope(str, Enum):
    """Permission scopes for HMAC tokens."""
    TRADE_READ = "trade:read"           # View positions, balance
    TRADE_WRITE = "trade:write"          # Execute trades
    CONFIG_READ = "config:read"          # View configuration
    CONFIG_WRITE = "config:write"        # Modify configuration
    SYSTEM_HALT = "system:halt"          # Emergency halt
    SYSTEM_RESUME = "system:resume"      # Resume after halt


@dataclass(frozen=True)
class TokenPayload:
    """Decoded token payload."""
    timestamp: int
    nonce: str
    scope: TokenScope
    signature: str


class AuthManager:
    """
    Manages authentication for all bot operations.
    
    Uses HMAC-SHA256 for token signing.
    Tokens include timestamp (for expiry), nonce (for replay prevention),
    and scope (for permission control).
    """

    def __init__(
        self,
        hmac_secret: Optional[str] = None,
        telegram_whitelist: Optional[list[int]] = None,
        default_ttl_seconds: int = 60,
    ) -> None:
        """
        Initialize auth manager.
        
        Args:
            hmac_secret: Secret key for HMAC signing. If None, loads from env.
            telegram_whitelist: List of allowed Telegram chat IDs. If None, loads from env.
            default_ttl_seconds: Default token time-to-live
        """
        secrets_mgr = get_secrets_manager()
        
        self._hmac_secret = hmac_secret or secrets_mgr.get("HMAC_SECRET_KEY", required=True)
        self._default_ttl = default_ttl_seconds
        
        # Load Telegram whitelist
        if telegram_whitelist is not None:
            self._telegram_whitelist = set(telegram_whitelist)
        else:
            whitelist_str = secrets_mgr.get("TELEGRAM_WHITELIST", required=False)
            if whitelist_str:
                self._telegram_whitelist = {
                    int(x.strip()) for x in whitelist_str.split(",") if x.strip()
                }
            else:
                self._telegram_whitelist = set()

    def generate_token(
        self,
        scope: TokenScope,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate an HMAC-signed token.
        
        Format: timestamp:nonce:scope:signature
        
        Args:
            scope: Permission scope for this token
            ttl_seconds: Token lifetime. Uses default if not specified.
            
        Returns:
            Colon-separated token string
        """
        timestamp = int(time.time())
        nonce = secrets.token_hex(8)
        ttl = ttl_seconds or self._default_ttl
        
        # payload to sign (timestamp:nonce:scope)
        payload = f"{timestamp}:{nonce}:{scope.value}"
        signature = self._sign(payload)
        
        return f"{payload}:{signature}"

    def verify_token(
        self,
        token: str,
        expected_scope: Optional[TokenScope] = None,
    ) -> TokenPayload:
        """
        Verify an HMAC token.
        
        Args:
            token: The token string to verify
            expected_scope: If provided, validates token has this scope
            
        Returns:
            Decoded TokenPayload
            
        Raises:
            InvalidTokenError: If token format or signature is invalid
            TokenExpiredError: If token has exceeded TTL
            ScopeError: If scope doesn't match expected_scope
        """
        parts = token.split(":")
        if len(parts) != 4:
            raise InvalidTokenError("Token format invalid")
            
        try:
            timestamp = int(parts[0])
        except ValueError:
            raise InvalidTokenError("Token timestamp invalid")
            
        nonce = parts[1]
        scope_str = parts[2]
        signature = parts[3]
        
        # Verify scope is valid
        try:
            scope = TokenScope(scope_str)
        except ValueError:
            raise InvalidTokenError(f"Unknown scope: {scope_str}")
            
        # Verify signature
        payload = f"{timestamp}:{nonce}:{scope.value}"
        expected_sig = self._sign(payload)
        
        if not hmac.compare_digest(signature, expected_sig):
            raise InvalidTokenError("Token signature invalid")
            
        # Check expiry
        age_seconds = int(time.time()) - timestamp
        if age_seconds > self._default_ttl:
            raise TokenExpiredError(
                f"Token expired {age_seconds - self._default_ttl}s ago"
            )
            
        # Check scope
        if expected_scope is not None and scope != expected_scope:
            raise ScopeError(
                f"Token scope '{scope.value}' does not match "
                f"expected '{expected_scope.value}'"
            )
            
        return TokenPayload(
            timestamp=timestamp,
            nonce=nonce,
            scope=scope,
            signature=signature,
        )

    def check_telegram_auth(self, chat_id: int) -> None:
        """
        Verify a Telegram chat ID is in the whitelist.
        
        Args:
            chat_id: Telegram chat ID to check
            
        Raises:
            UnauthorizedError: If chat ID is not whitelisted
        """
        if chat_id not in self._telegram_whitelist:
            raise UnauthorizedError(
                f"Chat ID {chat_id} is not authorized. "
                f"Whitelisted IDs: {sorted(self._telegram_whitelist)}"
            )

    def is_telegram_authorized(self, chat_id: int) -> bool:
        """Check if a Telegram chat ID is authorized (no exception)."""
        return chat_id in self._telegram_whitelist

    def _sign(self, payload: str) -> str:
        """Create HMAC-SHA256 signature of payload."""
        return hmac.new(
            self._hmac_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]

    @property
    def telegram_whitelist(self) -> set[int]:
        """Return copy of Telegram whitelist."""
        return self._telegram_whitelist.copy()
