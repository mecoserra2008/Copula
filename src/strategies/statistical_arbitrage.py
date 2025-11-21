"""Statistical arbitrage strategy using copulas."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

from .base_strategy import CopulaStrategy
from ..backtest.portfolio import Portfolio
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CopulaStatisticalArbitrageStrategy(CopulaStrategy):
    """
    Statistical arbitrage strategy using copulas for multi-pair trading.

    Identifies mispriced relationships across multiple pairs and constructs
    a market-neutral portfolio.

    Strategy Logic:
    1. Fit copulas to multiple pairs
    2. Identify pairs with extreme conditional probabilities
    3. Construct market-neutral portfolio of overvalued/undervalued assets
    4. Rebalance periodically
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        lookback_period: int = 500,
        copula_type: str = "gaussian",
        rebalance_frequency: int = 20,  # Rebalance every N bars
        entry_threshold: float = 0.1,
        num_positions: int = 4,  # Max number of concurrent positions
        position_size: float = 0.8,  # Use 80% of equity
    ):
        """
        Initialize statistical arbitrage strategy.

        Args:
            pairs: List of (symbol1, symbol2) tuples to trade
            lookback_period: Lookback period for copula fitting
            copula_type: Type of copula to use
            rebalance_frequency: How often to rebalance
            entry_threshold: Probability threshold for identifying mispricings
            num_positions: Maximum number of positions
            position_size: Fraction of equity to use
        """
        super().__init__(name="CopulaStatArb")

        self.pairs = pairs
        self.lookback_period = lookback_period
        self.copula_type = copula_type
        self.rebalance_frequency = rebalance_frequency
        self.entry_threshold = entry_threshold
        self.num_positions = num_positions
        self.position_size = position_size

        # State for each pair
        self.pair_data = {}
        for sym1, sym2 in pairs:
            self.pair_data[(sym1, sym2)] = {
                "prices1": deque(maxlen=lookback_period + 1),
                "prices2": deque(maxlen=lookback_period + 1),
                "returns1": deque(maxlen=lookback_period),
                "returns2": deque(maxlen=lookback_period),
                "copula": None,
                "transform": None,
            }

        self.bars_since_rebalance = 0

    def on_initialize(self) -> None:
        """Initialize strategy."""
        logger.info(f"Statistical Arbitrage Strategy:")
        logger.info(f"  Pairs: {len(self.pairs)}")
        logger.info(f"  Copula: {self.copula_type}")
        logger.info(f"  Rebalance frequency: {self.rebalance_frequency}")

    def on_data(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> None:
        """Execute strategy logic."""

        # Update prices and returns for all pairs
        for (sym1, sym2), data in self.pair_data.items():
            if sym1 in current_data and sym2 in current_data:
                price1 = current_data[sym1]["close"]
                price2 = current_data[sym2]["close"]

                data["prices1"].append(price1)
                data["prices2"].append(price2)

                if len(data["prices1"]) >= 2:
                    ret1 = np.log(data["prices1"][-1] / data["prices1"][-2])
                    ret2 = np.log(data["prices2"][-1] / data["prices2"][-2])
                    data["returns1"].append(ret1)
                    data["returns2"].append(ret2)

        # Check if it's time to rebalance
        if self.bars_since_rebalance >= self.rebalance_frequency:
            self._rebalance(portfolio, timestamp)
            self.bars_since_rebalance = 0

        self.bars_since_rebalance += 1

    def _rebalance(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Rebalance portfolio based on mispricing signals."""

        # Fit copulas and compute signals for each pair
        signals = []

        for (sym1, sym2), data in self.pair_data.items():
            if len(data["returns1"]) < self.lookback_period:
                continue

            try:
                # Refit copula
                returns1 = np.array(list(data["returns1"]))
                returns2 = np.array(list(data["returns2"]))

                copula, transform = self.fit_copula(
                    pd.Series(returns1),
                    pd.Series(returns2),
                    copula_type=self.copula_type,
                )

                data["copula"] = copula
                data["transform"] = transform

                # Compute current conditional probability
                curr_ret1 = data["returns1"][-1]
                curr_ret2 = data["returns2"][-1]

                u1 = transform.transforms_[0].transform(np.array([curr_ret1]))[0]
                u2 = transform.transforms_[1].transform(np.array([curr_ret2]))[0]

                prob = copula.conditional_cdf(
                    np.array([u1]), np.array([u2]), condition_on=1
                )[0]

                # Generate signal
                if prob < self.entry_threshold:
                    # Asset 1 is undervalued relative to asset 2
                    signals.append((sym1, sym2, "long_1_short_2", 1 - prob))
                elif prob > (1 - self.entry_threshold):
                    # Asset 1 is overvalued relative to asset 2
                    signals.append((sym1, sym2, "long_2_short_1", prob))

            except Exception as e:
                logger.debug(f"Failed to process pair {sym1}-{sym2}: {e}")
                continue

        # Select top signals
        if not signals:
            # Close all positions if no signals
            portfolio.close_all_positions(timestamp)
            return

        # Sort by signal strength and take top N
        signals.sort(key=lambda x: x[3], reverse=True)
        top_signals = signals[: self.num_positions]

        # Construct target portfolio
        target_weights = {}
        equity = portfolio.get_equity()
        weight_per_position = self.position_size / (len(top_signals) * 2)

        for sym1, sym2, signal_type, strength in top_signals:
            if signal_type == "long_1_short_2":
                target_weights[sym1] = target_weights.get(sym1, 0) + weight_per_position
                target_weights[sym2] = target_weights.get(sym2, 0) - weight_per_position
            else:  # long_2_short_1
                target_weights[sym1] = target_weights.get(sym1, 0) - weight_per_position
                target_weights[sym2] = target_weights.get(sym2, 0) + weight_per_position

        # Close positions not in target
        for symbol in list(portfolio.positions.keys()):
            if symbol not in target_weights and portfolio.get_position(symbol) != 0:
                portfolio.close_position(symbol, timestamp)

        # Rebalance to target weights
        portfolio.rebalance(target_weights, timestamp)

        logger.info(f"Rebalanced with {len(top_signals)} signals")

    def on_finalize(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Close all positions."""
        portfolio.close_all_positions(timestamp)
