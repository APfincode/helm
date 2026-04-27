"""
Fee models — Realistic trading fee and slippage modeling.

Hyperliquid fees:
- Taker: 0.035%
- Maker: 0.01%

Usage:
    from src.backtest.fees import HyperliquidFeeModel
    fee_model = HyperliquidFeeModel(taker_fee=0.00035, maker_fee=0.0001)
    fee = fee_model.calculate_fee(size=1000, price=65000, is_taker=True)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FeeBreakdown:
    """Detailed fee breakdown for a trade."""
    trading_fee: float
    slippage_cost: float
    total_cost: float
    fee_pct: float


class FeeModel(ABC):
    """Abstract base class for fee models."""

    @abstractmethod
    def calculate_fee(
        self,
        size: float,
        price: float,
        is_taker: bool = True,
    ) -> FeeBreakdown:
        """Calculate fees for a trade."""
        pass

    @abstractmethod
    def calculate_slippage(
        self,
        size: float,
        price: float,
        direction: str,
        market_depth: float = 1.0,
    ) -> float:
        """Calculate expected slippage."""
        pass


class HyperliquidFeeModel(FeeModel):
    """
    Hyperliquid perpetual futures fee model.
    
    Uses actual Hyperliquid fee structure with configurable
    slippage estimation based on position size relative to market depth.
    """

    def __init__(
        self,
        taker_fee_pct: float = 0.035,
        maker_fee_pct: float = 0.01,
        base_slippage_pct: float = 0.02,
        slippage_scaling: float = 0.5,
    ) -> None:
        """
        Initialize fee model.
        
        Args:
            taker_fee_pct: Taker fee percentage (0.035 = 0.035%)
            maker_fee_pct: Maker fee percentage (0.01 = 0.01%)
            base_slippage_pct: Base slippage estimate (0.02 = 0.02%)
            slippage_scaling: How much slippage scales with position size
        """
        self._taker_fee = taker_fee_pct / 100
        self._maker_fee = maker_fee_pct / 100
        self._base_slippage = base_slippage_pct / 100
        self._slippage_scaling = slippage_scaling

    def calculate_fee(
        self,
        size: float,
        price: float,
        is_taker: bool = True,
    ) -> FeeBreakdown:
        """
        Calculate trading fees.
        
        Args:
            size: Position size in base currency
            price: Entry/exit price
            is_taker: Whether order is taker (market) or maker (limit)
            
        Returns:
            FeeBreakdown with trading fee, slippage, total cost
        """
        notional = size * price
        fee_rate = self._taker_fee if is_taker else self._maker_fee
        trading_fee = notional * fee_rate
        
        # Slippage is only applied to taker orders
        slippage_cost = 0.0
        if is_taker:
            slippage_cost = self.calculate_slippage(size, price, "buy")
        
        total_cost = trading_fee + slippage_cost
        fee_pct = (total_cost / notional) * 100 if notional > 0 else 0

        return FeeBreakdown(
            trading_fee=trading_fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            fee_pct=fee_pct,
        )

    def calculate_slippage(
        self,
        size: float,
        price: float,
        direction: str,
        market_depth: float = 1.0,
    ) -> float:
        """
        Estimate slippage based on position size.
        
        Args:
            size: Position size
            price: Current price
            direction: "buy" or "sell"
            market_depth: Normalized market depth (1.0 = average)
            
        Returns:
            Estimated slippage in base currency
        """
        notional = size * price
        
        # Base slippage scaled by position size relative to market
        # Larger positions = more slippage
        size_factor = 1.0 + (notional / 100000) * self._slippage_scaling
        adjusted_slippage = self._base_slippage * size_factor / market_depth
        
        slippage_cost = notional * adjusted_slippage
        return slippage_cost

    def calculate_funding(
        self,
        size: float,
        price: float,
        funding_rate: float,
        hours_held: float,
    ) -> float:
        """
        Calculate funding fee for held position.
        
        Hyperliquid funding is paid every 8 hours.
        
        Args:
            size: Position size
            price: Mark price
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
            hours_held: Hours position was held
            
        Returns:
            Funding fee (positive = paid, negative = received)
        """
        notional = abs(size) * price
        # Funding periods (every 8 hours)
        periods = hours_held / 8
        return notional * funding_rate * periods


class SimpleFeeModel(FeeModel):
    """Simple flat fee model for quick testing."""

    def __init__(self, fee_pct: float = 0.1) -> None:
        self._fee = fee_pct / 100

    def calculate_fee(
        self,
        size: float,
        price: float,
        is_taker: bool = True,
    ) -> FeeBreakdown:
        notional = size * price
        fee = notional * self._fee
        return FeeBreakdown(
            trading_fee=fee,
            slippage_cost=0,
            total_cost=fee,
            fee_pct=self._fee * 100,
        )

    def calculate_slippage(
        self,
        size: float,
        price: float,
        direction: str,
        market_depth: float = 1.0,
    ) -> float:
        return 0.0
