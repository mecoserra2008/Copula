"""Dependence metrics for copula analysis."""

from typing import Tuple
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DependenceMetrics:
    """Compute various dependence metrics between two series."""

    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Pearson linear correlation coefficient.

        Measures linear dependence.

        Args:
            x: First series
            y: Second series

        Returns:
            Pearson correlation coefficient in [-1, 1]
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return np.nan

        corr, _ = pearsonr(x_clean, y_clean)

        return corr

    @staticmethod
    def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Spearman's rank correlation coefficient.

        Measures monotonic dependence (non-parametric).

        Args:
            x: First series
            y: Second series

        Returns:
            Spearman's rho in [-1, 1]
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return np.nan

        rho, _ = spearmanr(x_clean, y_clean)

        return rho

    @staticmethod
    def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Kendall's tau correlation coefficient.

        Measures concordance (non-parametric).

        Args:
            x: First series
            y: Second series

        Returns:
            Kendall's tau in [-1, 1]
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return np.nan

        tau, _ = kendalltau(x_clean, y_clean)

        return tau

    @staticmethod
    def empirical_tail_dependence(
        x: np.ndarray,
        y: np.ndarray,
        quantile: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Compute empirical tail dependence coefficients.

        Args:
            x: First series
            y: Second series
            quantile: Quantile threshold for tail (default 5%)

        Returns:
            Tuple of (lower_tail, upper_tail) dependence coefficients
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        n = len(x_clean)

        if n < 50:
            logger.warning(
                f"Sample size too small for reliable tail dependence: n={n}"
            )
            return np.nan, np.nan

        # Compute quantiles
        x_lower = np.quantile(x_clean, quantile)
        x_upper = np.quantile(x_clean, 1 - quantile)
        y_lower = np.quantile(y_clean, quantile)
        y_upper = np.quantile(y_clean, 1 - quantile)

        # Lower tail dependence
        # P(Y < y_q | X < x_q)
        both_lower = np.sum((x_clean < x_lower) & (y_clean < y_lower))
        x_lower_count = np.sum(x_clean < x_lower)

        if x_lower_count > 0:
            lambda_lower = both_lower / x_lower_count
        else:
            lambda_lower = 0

        # Upper tail dependence
        # P(Y > y_q | X > x_q)
        both_upper = np.sum((x_clean > x_upper) & (y_clean > y_upper))
        x_upper_count = np.sum(x_clean > x_upper)

        if x_upper_count > 0:
            lambda_upper = both_upper / x_upper_count
        else:
            lambda_upper = 0

        return lambda_lower, lambda_upper

    @staticmethod
    def correlation_breakdown(
        x: np.ndarray,
        y: np.ndarray,
        quantiles: list = [0.25, 0.5, 0.75],
    ) -> dict:
        """
        Compute correlations in different regions (lower, middle, upper).

        Args:
            x: First series
            y: Second series
            quantiles: Quantiles to split regions

        Returns:
            Dictionary with correlations by region
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        results = {}

        # Overall
        results["overall"] = DependenceMetrics.pearson_correlation(x_clean, y_clean)

        # By quantile regions
        x_quantiles = np.quantile(x_clean, quantiles)

        # Lower region
        mask_lower = x_clean < x_quantiles[0]
        if np.sum(mask_lower) > 10:
            results["lower"] = DependenceMetrics.pearson_correlation(
                x_clean[mask_lower], y_clean[mask_lower]
            )

        # Middle region
        mask_middle = (x_clean >= x_quantiles[0]) & (x_clean <= x_quantiles[-1])
        if np.sum(mask_middle) > 10:
            results["middle"] = DependenceMetrics.pearson_correlation(
                x_clean[mask_middle], y_clean[mask_middle]
            )

        # Upper region
        mask_upper = x_clean > x_quantiles[-1]
        if np.sum(mask_upper) > 10:
            results["upper"] = DependenceMetrics.pearson_correlation(
                x_clean[mask_upper], y_clean[mask_upper]
            )

        return results

    @staticmethod
    def concordance_discordance(
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[int, int, float]:
        """
        Compute number of concordant and discordant pairs.

        Args:
            x: First series
            y: Second series

        Returns:
            Tuple of (concordant, discordant, ratio)
        """
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        n = len(x_clean)

        concordant = 0
        discordant = 0

        # Pairwise comparison
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_clean[j] - x_clean[i]
                dy = y_clean[j] - y_clean[i]

                if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                    concordant += 1
                elif (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
                    discordant += 1
                # Ties are ignored

        total = concordant + discordant
        ratio = concordant / total if total > 0 else 0

        return concordant, discordant, ratio

    @staticmethod
    def all_metrics(
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Compute all dependence metrics.

        Args:
            x: First series
            y: Second series

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "pearson": DependenceMetrics.pearson_correlation(x, y),
            "spearman": DependenceMetrics.spearman_rho(x, y),
            "kendall": DependenceMetrics.kendall_tau(x, y),
        }

        # Tail dependence
        lambda_lower, lambda_upper = DependenceMetrics.empirical_tail_dependence(x, y)
        metrics["tail_dependence"] = {
            "lower": lambda_lower,
            "upper": lambda_upper,
        }

        # Correlation breakdown
        metrics["correlation_breakdown"] = DependenceMetrics.correlation_breakdown(
            x, y
        )

        return metrics
