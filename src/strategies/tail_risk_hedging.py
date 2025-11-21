"""Tail risk hedging strategy using copulas."""

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

from .base_strategy import CopulaStrategy
from ..backtest.portfolio import Portfolio
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CopulaTailRiskHedgingStrategy(CopulaStrategy):
    """
    Tail risk hedging strategy using copulas.

    Uses copulas to identify assets with strong tail dependence and
    dynamically hedges portfolio exposure during high-risk periods.

    Strategy Logic:
    1. Fit copulas to identify tail dependencies
    2. Monitor tail risk indicators
    3. Increase hedge positions when tail risk is elevated
    4. Reduce hedges when risk subsides
    """

    def __init__(
        self,
        base_asset: str,
        hedge_assets: List[str],
        lookback_period: int = 500,
        copula_type: str = "student_t",  # Student-t for tail dependence
        rebalance_frequency: int = 20,
        tail_risk_threshold: float = 0.1,  # 10th percentile
        max_hedge_ratio: float = 0.3,  # Max 30% in hedges
        base_allocation: float = 0.7,  # 70% in base asset
    ):
        """
        Initialize tail risk hedging strategy.

        Args:
            base_asset: Main asset to hold and hedge
            hedge_assets: List of potential hedge assets
            lookback_period: Lookback for copula fitting
            copula_type: Type of copula (should have tail dependence)
            rebalance_frequency: Rebalancing frequency
            tail_risk_threshold: Percentile threshold for tail risk
            max_hedge_ratio: Maximum allocation to hedges
            base_allocation: Target allocation to base asset
        """
        super().__init__(name="CopulaTailRiskHedging")

        self.base_asset = base_asset
        self.hedge_assets = hedge_assets
        self.lookback_period = lookback_period
        self.copula_type = copula_type
        self.rebalance_frequency = rebalance_frequency
        self.tail_risk_threshold = tail_risk_threshold
        self.max_hedge_ratio = max_hedge_ratio
        self.base_allocation = base_allocation

        # State
        self.prices = {base_asset: deque(maxlen=lookback_period + 1)}
        self.returns = {base_asset: deque(maxlen=lookback_period)}

        for hedge in hedge_assets:
            self.prices[hedge] = deque(maxlen=lookback_period + 1)
            self.returns[hedge] = deque(maxlen=lookback_period)

        self.copulas = {}  # {hedge_asset: (copula, transform)}
        self.bars_since_rebalance = 0

    def on_initialize(self) -> None:
        """Initialize strategy."""
        logger.info(f"Tail Risk Hedging Strategy:")
        logger.info(f"  Base asset: {self.base_asset}")
        logger.info(f"  Hedge assets: {self.hedge_assets}")
        logger.info(f"  Copula: {self.copula_type}")

    def on_data(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> None:
        """Execute strategy logic."""

        # Update prices and returns
        all_symbols = [self.base_asset] + self.hedge_assets

        for symbol in all_symbols:
            if symbol in current_data:
                price = current_data[symbol]["close"]
                self.prices[symbol].append(price)

                if len(self.prices[symbol]) >= 2:
                    ret = np.log(
                        self.prices[symbol][-1] / self.prices[symbol][-2]
                    )
                    self.returns[symbol].append(ret)

        # Need enough data
        if len(self.returns[self.base_asset]) < self.lookback_period:
            return

        # Rebalance periodically
        if self.bars_since_rebalance >= self.rebalance_frequency:
            self._rebalance(portfolio, timestamp)
            self.bars_since_rebalance = 0

        self.bars_since_rebalance += 1

    def _rebalance(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Rebalance portfolio based on tail risk."""

        # Fit copulas for each hedge asset
        base_returns = np.array(list(self.returns[self.base_asset]))

        for hedge in self.hedge_assets:
            if len(self.returns[hedge]) < self.lookback_period:
                continue

            try:
                hedge_returns = np.array(list(self.returns[hedge]))

                copula, transform = self.fit_copula(
                    pd.Series(base_returns),
                    pd.Series(hedge_returns),
                    copula_type=self.copula_type,
                )

                self.copulas[hedge] = (copula, transform)

            except Exception as e:
                logger.debug(f"Failed to fit copula for {hedge}: {e}")

        # Estimate tail risk
        tail_risk_score = self._estimate_tail_risk()

        # Determine hedge allocation based on tail risk
        hedge_allocation = min(
            tail_risk_score * self.max_hedge_ratio,
            self.max_hedge_ratio,
        )

        # Select best hedge assets (those with negative tail dependence)
        hedge_scores = self._rank_hedge_assets()

        # Construct target portfolio
        target_weights = {self.base_asset: self.base_allocation}

        # Allocate to top hedges
        if hedge_scores:
            # Normalize scores
            total_score = sum(abs(score) for _, score in hedge_scores)

            if total_score > 0:
                for hedge, score in hedge_scores:
                    weight = (abs(score) / total_score) * hedge_allocation
                    target_weights[hedge] = weight

        # Rebalance
        portfolio.rebalance(target_weights, timestamp)

        logger.info(
            f"Rebalanced: tail_risk={tail_risk_score:.3f}, "
            f"hedge_alloc={hedge_allocation:.3f}"
        )

    def _estimate_tail_risk(self) -> float:
        """
        Estimate current tail risk level.

        Returns:
            Tail risk score between 0 (low risk) and 1 (high risk)
        """
        base_returns = np.array(list(self.returns[self.base_asset]))

        # Use recent volatility as proxy for tail risk
        recent_vol = np.std(base_returns[-50:])  # Last 50 bars
        historical_vol = np.std(base_returns)

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Current return percentile
        current_return = base_returns[-1]
        percentile = np.sum(base_returns < current_return) / len(base_returns)

        # Tail risk score (higher when volatility high or in tail)
        tail_score = 0.0

        # High volatility
        if vol_ratio > 1.5:
            tail_score += 0.5

        # In lower tail
        if percentile < self.tail_risk_threshold:
            tail_score += 0.5

        return min(tail_score, 1.0)

    def _rank_hedge_assets(self) -> List[tuple]:
        """
        Rank hedge assets by hedging effectiveness.

        Returns:
            List of (hedge_asset, score) tuples, sorted by score
        """
        hedge_scores = []

        for hedge, (copula, transform) in self.copulas.items():
            try:
                # Get tail dependence
                lower_tail, upper_tail = copula.tail_dependence()

                # Good hedges have negative correlation in lower tail
                # or low/negative dependence
                # For simplicity, use negative of lower tail dependence
                score = -lower_tail if lower_tail > 0 else 0.1

                hedge_scores.append((hedge, score))

            except Exception as e:
                logger.debug(f"Failed to score hedge {hedge}: {e}")

        # Sort by score (descending)
        hedge_scores.sort(key=lambda x: x[1], reverse=True)

        return hedge_scores[:3]  # Top 3 hedges

    def on_finalize(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Close all positions."""
        portfolio.close_all_positions(timestamp)
