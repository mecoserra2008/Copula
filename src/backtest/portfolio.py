"""Portfolio management for backtesting."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Portfolio:
    """
    Portfolio manager for tracking positions, cash, and performance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital in USD
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Current state
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # {symbol: quantity}
        self.prices: Dict[str, float] = {}  # Current prices

        # History
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict] = []
        self.position_history: List[Dict] = []

    def update_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """
        Update current market prices.

        Args:
            prices: Dictionary of {symbol: price}
            timestamp: Current timestamp
        """
        self.prices.update(prices)

        # Record equity
        equity = self.get_equity()
        self.equity_history.append((timestamp, equity))

    def get_position(self, symbol: str) -> float:
        """
        Get current position in a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position size (positive = long, negative = short, 0 = no position)
        """
        return self.positions.get(symbol, 0.0)

    def get_equity(self) -> float:
        """
        Calculate total portfolio equity.

        Returns:
            Total equity (cash + position values)
        """
        position_value = sum(
            qty * self.prices.get(symbol, 0) for symbol, qty in self.positions.items()
        )
        return self.cash + position_value

    def get_position_value(self, symbol: str) -> float:
        """
        Get current value of a position.

        Args:
            symbol: Trading symbol

        Returns:
            Position value in USD
        """
        qty = self.positions.get(symbol, 0)
        price = self.prices.get(symbol, 0)
        return qty * price

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        timestamp: datetime,
        trade_type: str = "market",
    ) -> bool:
        """
        Execute a trade.

        Args:
            symbol: Trading symbol
            quantity: Quantity to trade (positive = buy, negative = sell)
            timestamp: Trade timestamp
            trade_type: Type of trade ('market', 'limit', etc.)

        Returns:
            True if trade executed successfully, False otherwise
        """
        if symbol not in self.prices:
            logger.warning(f"No price available for {symbol}")
            return False

        price = self.prices[symbol]

        # Apply slippage
        if quantity > 0:  # Buy
            execution_price = price * (1 + self.slippage)
        else:  # Sell
            execution_price = price * (1 - self.slippage)

        # Calculate trade value
        trade_value = abs(quantity * execution_price)

        # Calculate transaction cost
        cost = trade_value * self.transaction_cost

        # Check if we have enough cash for buys
        if quantity > 0:
            required_cash = trade_value + cost
            if required_cash > self.cash:
                logger.warning(
                    f"Insufficient cash: need ${required_cash:.2f}, "
                    f"have ${self.cash:.2f}"
                )
                return False

        # Execute trade
        self.cash -= quantity * execution_price + cost

        # Update position
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        self.positions[symbol] = new_position

        # Record trade
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": execution_price,
            "value": trade_value,
            "cost": cost,
            "type": trade_type,
            "position_before": current_position,
            "position_after": new_position,
        }
        self.trade_history.append(trade)

        logger.debug(
            f"Trade executed: {symbol} qty={quantity:.4f} @ ${execution_price:.2f}"
        )

        return True

    def close_position(self, symbol: str, timestamp: datetime) -> bool:
        """
        Close entire position in a symbol.

        Args:
            symbol: Trading symbol
            timestamp: Trade timestamp

        Returns:
            True if position closed successfully
        """
        position = self.get_position(symbol)

        if position == 0:
            return True

        # Execute opposite trade
        return self.execute_trade(symbol, -position, timestamp)

    def close_all_positions(self, timestamp: datetime) -> None:
        """
        Close all open positions.

        Args:
            timestamp: Trade timestamp
        """
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self.close_position(symbol, timestamp)

    def rebalance(
        self,
        target_weights: Dict[str, float],
        timestamp: datetime,
    ) -> None:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Dictionary of {symbol: target_weight}
            timestamp: Rebalancing timestamp
        """
        equity = self.get_equity()

        # Calculate target positions
        for symbol, weight in target_weights.items():
            if symbol not in self.prices:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            target_value = equity * weight
            current_value = self.get_position_value(symbol)
            trade_value = target_value - current_value

            # Calculate quantity to trade
            price = self.prices[symbol]
            quantity = trade_value / price

            # Execute trade if significant
            if abs(quantity * price) > 10:  # Minimum $10 trade
                self.execute_trade(symbol, quantity, timestamp)

    def get_returns(self) -> pd.Series:
        """
        Calculate portfolio returns from equity history.

        Returns:
            Series of returns
        """
        if len(self.equity_history) < 2:
            return pd.Series()

        df = pd.DataFrame(self.equity_history, columns=["timestamp", "equity"])
        df["returns"] = df["equity"].pct_change()

        return df.set_index("timestamp")["returns"].dropna()

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with timestamp and equity
        """
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history, columns=["timestamp", "equity"])
        df = df.set_index("timestamp")

        return df

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame of all trades
        """
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)

    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics.

        Returns:
            Dictionary with summary metrics
        """
        equity = self.get_equity()
        total_return = (equity - self.initial_capital) / self.initial_capital

        returns = self.get_returns()

        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # 15-min
            max_dd = self._calculate_max_drawdown()
        else:
            sharpe = 0
            max_dd = 0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "num_trades": len(self.trade_history),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "cash": self.cash,
            "num_positions": len([p for p in self.positions.values() if p != 0]),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_history) < 2:
            return 0.0

        equity_values = np.array([eq for _, eq in self.equity_history])
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max

        return np.min(drawdown)
