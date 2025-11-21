"""Risk metrics for copula-based analysis."""

from typing import Optional, Tuple
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RiskMetrics:
    """Compute risk metrics for portfolio analysis."""

    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute Value-at-Risk (VaR).

        VaR is the maximum loss at a given confidence level.

        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value (positive = loss)
        """
        # Remove NaN
        returns_clean = returns[~np.isnan(returns)]

        if len(returns_clean) == 0:
            return np.nan

        # VaR is the negative of the quantile
        var = -np.quantile(returns_clean, 1 - confidence)

        return var

    @staticmethod
    def conditional_value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute Conditional Value-at-Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            CVaR value (positive = loss)
        """
        # Remove NaN
        returns_clean = returns[~np.isnan(returns)]

        if len(returns_clean) == 0:
            return np.nan

        # Compute VaR
        var = RiskMetrics.value_at_risk(returns_clean, confidence)

        # CVaR is the mean of returns below VaR threshold
        tail_returns = returns_clean[returns_clean < -var]

        if len(tail_returns) == 0:
            return var  # If no tail observations, return VaR

        cvar = -np.mean(tail_returns)

        return cvar

    @staticmethod
    def portfolio_return(
        returns1: np.ndarray,
        returns2: np.ndarray,
        weight1: float = 0.5,
    ) -> np.ndarray:
        """
        Compute portfolio returns with given weights.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            weight1: Weight of first asset (0 to 1)

        Returns:
            Array of portfolio returns
        """
        weight2 = 1 - weight1
        portfolio_ret = weight1 * returns1 + weight2 * returns2

        return portfolio_ret

    @staticmethod
    def portfolio_var(
        returns1: np.ndarray,
        returns2: np.ndarray,
        weight1: float = 0.5,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute portfolio VaR.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            weight1: Weight of first asset
            confidence: Confidence level

        Returns:
            Portfolio VaR
        """
        portfolio_ret = RiskMetrics.portfolio_return(returns1, returns2, weight1)
        var = RiskMetrics.value_at_risk(portfolio_ret, confidence)

        return var

    @staticmethod
    def portfolio_cvar(
        returns1: np.ndarray,
        returns2: np.ndarray,
        weight1: float = 0.5,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute portfolio CVaR.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            weight1: Weight of first asset
            confidence: Confidence level

        Returns:
            Portfolio CVaR
        """
        portfolio_ret = RiskMetrics.portfolio_return(returns1, returns2, weight1)
        cvar = RiskMetrics.conditional_value_at_risk(portfolio_ret, confidence)

        return cvar

    @staticmethod
    def diversification_benefit(
        returns1: np.ndarray,
        returns2: np.ndarray,
        weight1: float = 0.5,
        confidence: float = 0.95,
    ) -> dict:
        """
        Compute diversification benefit from portfolio formation.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            weight1: Weight of first asset
            confidence: Confidence level

        Returns:
            Dictionary with diversification metrics
        """
        # Individual VaRs
        var1 = RiskMetrics.value_at_risk(returns1, confidence)
        var2 = RiskMetrics.value_at_risk(returns2, confidence)

        # Weighted sum of individual VaRs
        weighted_var = weight1 * var1 + (1 - weight1) * var2

        # Portfolio VaR
        portfolio_var = RiskMetrics.portfolio_var(
            returns1, returns2, weight1, confidence
        )

        # Diversification benefit
        benefit = weighted_var - portfolio_var
        benefit_pct = (benefit / weighted_var * 100) if weighted_var > 0 else 0

        return {
            "var1": var1,
            "var2": var2,
            "weighted_var": weighted_var,
            "portfolio_var": portfolio_var,
            "diversification_benefit": benefit,
            "diversification_benefit_pct": benefit_pct,
        }

    @staticmethod
    def copula_var(
        copula,
        marginals,
        n_simulations: int = 10000,
        confidence: float = 0.95,
        weight1: float = 0.5,
    ) -> float:
        """
        Compute VaR using copula simulation.

        Args:
            copula: Fitted copula model
            marginals: Tuple of (transform1, transform2) for marginals
            n_simulations: Number of Monte Carlo simulations
            confidence: Confidence level
            weight1: Portfolio weight for first asset

        Returns:
            Simulated VaR
        """
        # Sample from copula
        U = copula.sample(n_simulations)

        # Transform back to returns
        transform1, transform2 = marginals
        r1 = transform1.inverse_transform(U[:, 0])
        r2 = transform2.inverse_transform(U[:, 1])

        # Compute portfolio returns
        portfolio_ret = RiskMetrics.portfolio_return(r1, r2, weight1)

        # Compute VaR
        var = RiskMetrics.value_at_risk(portfolio_ret, confidence)

        return var

    @staticmethod
    def downside_correlation(
        returns1: np.ndarray,
        returns2: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Compute correlation during downside periods.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            threshold: Threshold for downside (default 0)

        Returns:
            Downside correlation
        """
        # Remove NaN
        mask = ~(np.isnan(returns1) | np.isnan(returns2))
        r1 = returns1[mask]
        r2 = returns2[mask]

        # Filter downside periods (both below threshold)
        downside_mask = (r1 < threshold) & (r2 < threshold)

        if np.sum(downside_mask) < 10:
            logger.warning("Insufficient downside observations for correlation")
            return np.nan

        # Compute correlation
        corr = np.corrcoef(r1[downside_mask], r2[downside_mask])[0, 1]

        return corr

    @staticmethod
    def upside_correlation(
        returns1: np.ndarray,
        returns2: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Compute correlation during upside periods.

        Args:
            returns1: First asset returns
            returns2: Second asset returns
            threshold: Threshold for upside (default 0)

        Returns:
            Upside correlation
        """
        # Remove NaN
        mask = ~(np.isnan(returns1) | np.isnan(returns2))
        r1 = returns1[mask]
        r2 = returns2[mask]

        # Filter upside periods (both above threshold)
        upside_mask = (r1 > threshold) & (r2 > threshold)

        if np.sum(upside_mask) < 10:
            logger.warning("Insufficient upside observations for correlation")
            return np.nan

        # Compute correlation
        corr = np.corrcoef(r1[upside_mask], r2[upside_mask])[0, 1]

        return corr
