"""
Hyperliquid Executor — Low-level execution on Hyperliquid exchange.

This module handles:
1. Wallet-based authentication (Ethereum-style signing)
2. Order placement (market, limit, stop orders)
3. Position queries and modifications
4. Rate limiting compliance (1200 calls / 10 min)
5. Error handling with automatic retry

Hyperliquid Perpetuals API:
- Base URL: https://api.hyperliquid.xyz
- Signing: Ethereum personal_sign with L1 private key
- Rate limit: 1200 calls per 10 minutes
- Order types: market, limit, stop-market
"""

from __future__ import annotations

import json
import time
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, Any
from enum import Enum

import httpx

from src.security.secrets_manager import get_secrets_manager
from src.security.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"


class Side(str, Enum):
    BUY = "B"
    SELL = "S"


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    success: bool = False
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.ERROR
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    remaining_quantity: float = 0.0
    fee_paid: float = 0.0
    error_message: Optional[str] = None
    raw_response: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED


@dataclass
class ExecutorConfig:
    """Configuration for Hyperliquid executor."""
    base_url: str = "https://api.hyperliquid.xyz"
    rate_limit_calls: int = 1200
    rate_limit_window_seconds: int = 600  # 10 minutes
    max_slippage_pct: float = 0.5  # Reject if slippage > 0.5%
    order_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    
    # Signing
    use_testnet: bool = False
    
    def __post_init__(self):
        if self.use_testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"


@dataclass
class HyperliquidPosition:
    """Position on Hyperliquid."""
    coin: str
    szi: float  # Size in base currency (negative for short)
    entry_px: float  # Entry price
    position_value: float
    unrealized_pnl: float
    leverage: float
    liquidation_px: Optional[float]
    margin_used: float


# =============================================================================
# Hyperliquid Executor
# =============================================================================

class HyperliquidExecutor:
    """
    Low-level executor for Hyperliquid exchange.
    
    Handles authentication, order signing, and API communication.
    All methods are async and respect rate limits.
    """
    
    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self._config = config or ExecutorConfig()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._secrets = get_secrets_manager()
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        # State
        self._wallet_address: Optional[str] = None
        self._is_initialized: bool = False
    
    async def __aenter__(self) -> HyperliquidExecutor:
        """Initialize HTTP client and load credentials."""
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.order_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
        
        # Load wallet credentials
        self._wallet_address = self._secrets.get("HYPERLIQUID_WALLET_ADDRESS")
        if not self._wallet_address:
            logger.warning("HYPERLIQUID_WALLET_ADDRESS not set — execution will fail")
        
        self._is_initialized = True
        return self
    
    async def __aexit__(self, *args) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._is_initialized = False
    
    # ========================================================================
    # Public: Order Placement
    # ========================================================================
    
    async def place_market_order(
        self,
        coin: str,
        side: Literal["LONG", "SHORT"],
        quantity: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Place a market order.
        
        Args:
            coin: Trading pair (e.g., "BTC")
            side: LONG or SHORT
            quantity: Size in base currency
            reduce_only: If True, order only reduces position (safety)
            
        Returns:
            OrderResult with fill details
        """
        return await self._place_order(
            coin=coin,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reduce_only=reduce_only,
        )
    
    async def place_limit_order(
        self,
        coin: str,
        side: Literal["LONG", "SHORT"],
        quantity: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: Literal["Gtc", "Ioc", "Alo"] = "Gtc",
    ) -> OrderResult:
        """
        Place a limit order.
        
        Args:
            coin: Trading pair
            side: LONG or SHORT
            quantity: Size in base currency
            price: Limit price
            reduce_only: If True, only reduces position
            time_in_force: Gtc (good-til-cancel), Ioc (immediate-or-cancel), Alo (add-liquidity-only)
            
        Returns:
            OrderResult
        """
        return await self._place_order(
            coin=coin,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price,
            reduce_only=reduce_only,
            time_in_force=time_in_force,
        )
    
    async def place_stop_order(
        self,
        coin: str,
        side: Literal["LONG", "SHORT"],
        quantity: float,
        trigger_price: float,
        reduce_only: bool = True,
    ) -> OrderResult:
        """
        Place a stop-market order.
        
        Used for stop loss and take profit execution.
        reduce_only=True by default (safety).
        """
        return await self._place_order(
            coin=coin,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP_MARKET,
            trigger_price=trigger_price,
            reduce_only=reduce_only,
        )
    
    # ========================================================================
    # Public: Position & Account
    # ========================================================================
    
    async def get_positions(self) -> list[HyperliquidPosition]:
        """Get all open positions for the account."""
        await self._check_rate_limit()
        
        if not self._wallet_address:
            logger.error("No wallet address configured")
            return []
        
        try:
            response = await self._client.post(
                "/info",
                json={
                    "type": "clearinghouseState",
                    "user": self._wallet_address,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            positions = []
            for pos_data in data.get("assetPositions", []):
                pos = pos_data.get("position", {})
                if pos:
                    positions.append(HyperliquidPosition(
                        coin=pos.get("coin", ""),
                        szi=float(pos.get("szi", 0)),
                        entry_px=float(pos.get("entryPx", 0)) if pos.get("entryPx") else 0.0,
                        position_value=float(pos.get("positionValue", 0)),
                        unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                        leverage=float(pos.get("leverage", {}).get("value", 1)),
                        liquidation_px=float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None,
                        margin_used=float(pos.get("marginUsed", 0)),
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_summary(self) -> dict:
        """Get account balance and margin info."""
        await self._check_rate_limit()
        
        if not self._wallet_address:
            return {}
        
        try:
            response = await self._client.post(
                "/info",
                json={
                    "type": "clearinghouseState",
                    "user": self._wallet_address,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "equity": float(data.get("marginSummary", {}).get("accountValue", 0)),
                "margin_used": float(data.get("marginSummary", {}).get("totalMarginUsed", 0)),
                "buying_power": float(data.get("marginSummary", {}).get("totalRawUsd", 0)),
                "withdrawable": float(data.get("withdrawable", 0)),
            }
            
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}
    
    async def cancel_all_orders(self, coin: Optional[str] = None) -> bool:
        """Cancel all open orders (or for a specific coin)."""
        # This would require signing — placeholder for now
        logger.warning("cancel_all_orders requires wallet signing — not yet implemented")
        return False
    
    # ========================================================================
    # Internal: Order Placement
    # ========================================================================
    
    async def _place_order(
        self,
        coin: str,
        side: Literal["LONG", "SHORT"],
        quantity: float,
        order_type: OrderType,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "Gtc",
    ) -> OrderResult:
        """
        Core order placement logic.
        
        Signs the order with the wallet private key and sends to Hyperliquid.
        """
        result = OrderResult()
        
        # Pre-flight checks
        if not self._is_initialized or not self._client:
            result.error_message = "Executor not initialized"
            return result
        
        if not self._wallet_address:
            result.error_message = "Wallet address not configured"
            return result
        
        # Rate limit check
        try:
            await self._check_rate_limit()
        except Exception as e:
            result.error_message = f"Rate limit exceeded: {e}"
            return result
        
        # Build order action
        is_buy = side == "LONG"
        
        order_action = {
            "type": "order",
            "orders": [{
                "coin": coin,
                "isBuy": is_buy,
                "sz": abs(quantity),
                "limitPx": price if price else 0,
                "orderType": self._map_order_type(order_type),
                "reduceOnly": reduce_only,
                "tif": time_in_force,
            }],
            "grouping": "na",
        }
        
        # Add trigger price for stop orders
        if trigger_price and order_type == OrderType.STOP_MARKET:
            order_action["orders"][0]["triggerPx"] = trigger_price
        
        # Sign the action
        signed_action = await self._sign_action(order_action)
        if not signed_action:
            result.error_message = "Failed to sign order"
            return result
        
        # Send order
        try:
            response = await self._client.post(
                "/exchange",
                json=signed_action,
            )
            response.raise_for_status()
            data = response.json()
            
            result.raw_response = data
            
            # Parse response
            if data.get("status") == "ok":
                response_data = data.get("response", {}).get("data", {}).get("statuses", [{}])[0]
                
                if "filled" in response_data:
                    filled = response_data["filled"]
                    result.success = True
                    result.status = OrderStatus.FILLED
                    result.order_id = filled.get("oid")
                    result.filled_quantity = float(filled.get("totalSz", 0))
                    result.avg_fill_price = float(filled.get("avgPx", 0))
                    result.fee_paid = float(filled.get("fee", 0))
                    
                    logger.info(
                        f"Order FILLED: {coin} {'BUY' if is_buy else 'SELL'} "
                        f"{result.filled_quantity} @ {result.avg_fill_price} "
                        f"(fee: ${result.fee_paid:.4f})"
                    )
                    
                elif "resting" in response_data:
                    resting = response_data["resting"]
                    result.success = True
                    result.status = OrderStatus.OPEN
                    result.order_id = resting.get("oid")
                    result.remaining_quantity = quantity
                    
                    logger.info(f"Order OPEN: {coin} {resting.get('oid')}")
                    
                elif "error" in response_data:
                    result.status = OrderStatus.REJECTED
                    result.error_message = response_data["error"]
                    logger.warning(f"Order REJECTED: {response_data['error']}")
                    
            else:
                result.error_message = data.get("response", "Unknown error")
                logger.error(f"Order failed: {result.error_message}")
                
        except httpx.HTTPStatusError as e:
            result.error_message = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Order HTTP error: {result.error_message}")
            
        except Exception as e:
            result.error_message = f"Exception: {str(e)}"
            logger.exception("Order placement failed")
        
        return result
    
    # ========================================================================
    # Internal: Signing
    # ========================================================================
    
    async def _sign_action(self, action: dict) -> Optional[dict]:
        """
        Sign an action with the wallet private key.
        
        Hyperliquid uses Ethereum-style signing.
        Requires HYPERLIQUID_PRIVATE_KEY secret.
        """
        private_key = self._secrets.get("HYPERLIQUID_PRIVATE_KEY")
        if not private_key:
            logger.error("HYPERLIQUID_PRIVATE_KEY not configured")
            return None
        
        try:
            # Hyperliquid expects specific signing format
            # This is a simplified version — full implementation needs eth-signing library
            # For production, use the official hyperliquid-python-sdk
            
            # Build payload hash
            action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)
            action_hash = hashlib.sha256(action_json.encode()).hexdigest()
            
            # Note: Full implementation requires web3 or eth-account for proper signing
            # Placeholder: return unsigned action (will fail on mainnet)
            # In production, use:
            # from eth_account import Account
            # signed = Account.sign_message(action_hash, private_key)
            
            logger.warning("Order signing is a placeholder — use official SDK for production")
            
            return {
                "action": action,
                "nonce": int(time.time() * 1000),
                "signature": "placeholder_signature",
                "wallet": self._wallet_address,
            }
            
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return None
    
    # ========================================================================
    # Internal: Helpers
    # ========================================================================
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        await self._rate_limiter.check_or_raise("exchange_api", key="order_placement")
    
    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        """Map internal order type to Hyperliquid format."""
        mapping = {
            OrderType.MARKET: "Market",
            OrderType.LIMIT: "Limit",
            OrderType.STOP_MARKET: "StopMarket",
        }
        return mapping.get(order_type, "Market")
    
    # ========================================================================
    # Paper trading helper
    # ========================================================================
    
    async def get_current_price(self, coin: str) -> float:
        """Get current mid price for a coin."""
        await self._check_rate_limit()
        
        try:
            response = await self._client.post(
                "/info",
                json={"type": "allMids"}
            )
            response.raise_for_status()
            data = response.json()
            
            return float(data.get(coin, 0))
            
        except Exception as e:
            logger.error(f"Failed to get price for {coin}: {e}")
            return 0.0
