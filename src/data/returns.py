"""Return calculation utilities."""

from typing import Optional
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReturnCalculator:
    """Calculate various types of returns from price data."""

    @staticmethod
    def log_returns(prices: np.ndarray) -> np.ndarray:
        """
        Calculate log returns: ln(P_t / P_{t-1}).

        Args:
            prices: Array of prices

        Returns:
            Array of log returns (length = len(prices) - 1)
        """
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(np.log(prices))
        return returns

    @staticmethod
    def simple_returns(prices: np.ndarray) -> np.ndarray:
        """
        Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}.

        Args:
            prices: Array of prices

        Returns:
            Array of simple returns
        """
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    @staticmethod
    def pct_change(prices: np.ndarray) -> np.ndarray:
        """
        Calculate percentage change: 100 * (P_t - P_{t-1}) / P_{t-1}.

        Args:
            prices: Array of prices

        Returns:
            Array of percentage changes
        """
        return ReturnCalculator.simple_returns(prices) * 100

    @staticmethod
    def returns_from_dataframe(
        df: pd.DataFrame,
        price_col: str = "close",
        method: str = "log",
    ) -> pd.Series:
        """
        Calculate returns from DataFrame.

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            method: Return calculation method ('log' or 'simple')

        Returns:
            Series of returns
        """
        prices = df[price_col].values

        if method == "log":
            returns = ReturnCalculator.log_returns(prices)
        elif method == "simple":
            returns = ReturnCalculator.simple_returns(prices)
        elif method == "pct":
            returns = ReturnCalculator.pct_change(prices)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create series with aligned index (skip first row)
        return_series = pd.Series(returns, index=df.index[1:])

        logger.debug(
            f"Calculated {method} returns: {len(return_series)} values, "
            f"mean={return_series.mean():.6f}, std={return_series.std():.6f}"
        )

        return return_series

    @staticmethod
    def realized_volatility(
        returns: np.ndarray,
        window: Optional[int] = None,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate realized volatility.

        Args:
            returns: Array of returns
            window: Rolling window size (None = use all data)
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year for annualization

        Returns:
            Realized volatility
        """
        if window is not None:
            returns = returns[-window:]

        vol = np.nanstd(returns)

        if annualize:
            vol *= np.sqrt(periods_per_year)

        return vol

    @staticmethod
    def rolling_returns(
        prices: np.ndarray,
        window: int,
        method: str = "log",
    ) -> np.ndarray:
        """
        Calculate rolling returns.

        Args:
            prices: Array of prices
            window: Window size for rolling returns
            method: Return calculation method

        Returns:
            Array of rolling returns
        """
        n = len(prices)
        if n < window + 1:
            return np.array([])

        rolling_rets = np.zeros(n - window)

        for i in range(len(rolling_rets)):
            if method == "log":
                rolling_rets[i] = np.log(prices[i + window]) - np.log(prices[i])
            elif method == "simple":
                rolling_rets[i] = (prices[i + window] - prices[i]) / prices[i]

        return rolling_rets

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        mean_return = np.nanmean(returns) * periods_per_year
        std_return = np.nanstd(returns) * np.sqrt(periods_per_year)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    @staticmethod
    def drawdown(prices: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series.

        Args:
            prices: Array of prices

        Returns:
            Array of drawdowns (negative values)
        """
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        return drawdown

    @staticmethod
    def max_drawdown(prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            prices: Array of prices

        Returns:
            Maximum drawdown (negative value)
        """
        dd = ReturnCalculator.drawdown(prices)
        return np.min(dd)
