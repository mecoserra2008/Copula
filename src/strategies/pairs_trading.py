"""Pairs trading strategy using copulas."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque

from .base_strategy import CopulaStrategy
from ..backtest.portfolio import Portfolio
from ..data.returns import ReturnCalculator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CopulaPairsTradingStrategy(CopulaStrategy):
    """
    Copula-based pairs trading strategy.

    Uses copula to model the dependence between two assets and trades
    based on deviations from the expected relationship.

    Strategy Logic:
    1. Fit copula to historical returns
    2. Compute conditional probabilities
    3. Enter trades when conditional probability is extreme
    4. Exit when probability reverts to mean
    """

    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        lookback_period: int = 500,
        copula_type: str = "gaussian",
        entry_threshold: float = 0.05,  # Enter if prob < 0.05 or > 0.95
        exit_threshold: float = 0.5,  # Exit when prob reverts to 0.5
        refit_frequency: int = 100,  # Refit copula every N bars
        position_size: float = 0.5,  # Use 50% of equity per trade
    ):
        """
        Initialize pairs trading strategy.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            lookback_period: Number of bars for copula fitting
            copula_type: Type of copula to use
            entry_threshold: Probability threshold for entry
            exit_threshold: Probability threshold for exit
            refit_frequency: How often to refit copula
            position_size: Fraction of equity to use per trade
        """
        super().__init__(name=f"CopulaPairsTrading_{symbol1}_{symbol2}")

        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.lookback_period = lookback_period
        self.copula_type = copula_type
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.refit_frequency = refit_frequency
        self.position_size = position_size

        # State
        self.prices1 = deque(maxlen=lookback_period + 1)
        self.prices2 = deque(maxlen=lookback_period + 1)
        self.returns1 = deque(maxlen=lookback_period)
        self.returns2 = deque(maxlen=lookback_period)

        self.bars_since_refit = 0
        self.in_position = False
        self.position_type = None  # 'long_1_short_2' or 'long_2_short_1'

    def on_initialize(self) -> None:
        """Initialize strategy."""
        logger.info(f"Pairs Trading Strategy:")
        logger.info(f"  Pair: {self.symbol1} - {self.symbol2}")
        logger.info(f"  Copula: {self.copula_type}")
        logger.info(f"  Lookback: {self.lookback_period}")
        logger.info(f"  Entry threshold: {self.entry_threshold}")

    def on_data(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> None:
        """Execute strategy logic on each bar."""

        # Check if we have data for both symbols
        if self.symbol1 not in current_data or self.symbol2 not in current_data:
            return

        price1 = current_data[self.symbol1]["close"]
        price2 = current_data[self.symbol2]["close"]

        # Append prices
        self.prices1.append(price1)
        self.prices2.append(price2)

        # Calculate returns if we have enough prices
        if len(self.prices1) >= 2:
            ret1 = np.log(self.prices1[-1] / self.prices1[-2])
            ret2 = np.log(self.prices2[-1] / self.prices2[-2])
            self.returns1.append(ret1)
            self.returns2.append(ret2)

        # Need enough data to fit copula
        if len(self.returns1) < self.lookback_period:
            return

        # Refit copula periodically
        if self.copula is None or self.bars_since_refit >= self.refit_frequency:
            try:
                self._refit_copula()
                self.bars_since_refit = 0
            except Exception as e:
                logger.warning(f"Failed to fit copula: {e}")
                return

        self.bars_since_refit += 1

        # Compute current conditional probability
        try:
            u1, u2 = self._compute_uniform_values()
            prob = self.compute_conditional_probability(u1, u2, condition_on=1)
        except Exception as e:
            logger.debug(f"Failed to compute probability: {e}")
            return

        # Trading logic
        if not self.in_position:
            # Look for entry signals
            if prob < self.entry_threshold:
                # P(U1 < u1 | U2 = u2) is low
                # Asset 1 is relatively low given asset 2
                # Long asset 1, short asset 2
                self._enter_position("long_1_short_2", portfolio, timestamp)

            elif prob > (1 - self.entry_threshold):
                # P(U1 < u1 | U2 = u2) is high
                # Asset 1 is relatively high given asset 2
                # Short asset 1, long asset 2
                self._enter_position("long_2_short_1", portfolio, timestamp)

        else:
            # Look for exit signals
            if abs(prob - 0.5) < abs(self.exit_threshold - 0.5):
                # Probability has reverted toward mean
                self._exit_position(portfolio, timestamp)

    def _refit_copula(self) -> None:
        """Refit copula to recent data."""
        returns1_arr = np.array(list(self.returns1))
        returns2_arr = np.array(list(self.returns2))

        self.copula, self.transform = self.fit_copula(
            pd.Series(returns1_arr),
            pd.Series(returns2_arr),
            copula_type=self.copula_type,
        )

        logger.debug(f"Refitted copula: {self.copula}")

    def _compute_uniform_values(self) -> tuple:
        """Compute current uniform values for both assets."""
        if self.transform is None:
            raise ValueError("Transform not fitted")

        # Get current return
        curr_ret1 = self.returns1[-1]
        curr_ret2 = self.returns2[-1]

        # Transform to uniform
        u1 = self.transform.transforms_[0].transform(np.array([curr_ret1]))[0]
        u2 = self.transform.transforms_[1].transform(np.array([curr_ret2]))[0]

        return u1, u2

    def _enter_position(
        self,
        position_type: str,
        portfolio: Portfolio,
        timestamp: datetime,
    ) -> None:
        """Enter a pairs trade."""
        equity = portfolio.get_equity()
        trade_value = equity * self.position_size / 2  # Split between two positions

        price1 = self.prices1[-1]
        price2 = self.prices2[-1]

        if position_type == "long_1_short_2":
            # Long symbol1, short symbol2
            qty1 = trade_value / price1
            qty2 = -trade_value / price2

        else:  # long_2_short_1
            # Short symbol1, long symbol2
            qty1 = -trade_value / price1
            qty2 = trade_value / price2

        # Execute trades
        success1 = portfolio.execute_trade(self.symbol1, qty1, timestamp)
        success2 = portfolio.execute_trade(self.symbol2, qty2, timestamp)

        if success1 and success2:
            self.in_position = True
            self.position_type = position_type
            logger.info(f"Entered {position_type} at {timestamp}")
        else:
            # Rollback if only one trade succeeded
            if success1:
                portfolio.execute_trade(self.symbol1, -qty1, timestamp)
            if success2:
                portfolio.execute_trade(self.symbol2, -qty2, timestamp)

    def _exit_position(
        self,
        portfolio: Portfolio,
        timestamp: datetime,
    ) -> None:
        """Exit current pairs trade."""
        portfolio.close_position(self.symbol1, timestamp)
        portfolio.close_position(self.symbol2, timestamp)

        self.in_position = False
        logger.info(f"Exited position at {timestamp}")

    def on_finalize(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Close any remaining positions."""
        if self.in_position:
            self._exit_position(portfolio, timestamp)
