"""Performance metrics for backtesting."""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Calculate performance metrics for trading strategies."""

    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """
        Calculate total return.

        Args:
            equity_curve: Series of equity values

        Returns:
            Total return as decimal
        """
        if len(equity_curve) < 2:
            return 0.0

        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    @staticmethod
    def annualized_return(
        equity_curve: pd.Series,
        periods_per_year: int = 35040,  # 15-min bars in a year
    ) -> float:
        """
        Calculate annualized return.

        Args:
            equity_curve: Series of equity values
            periods_per_year: Number of periods per year

        Returns:
            Annualized return
        """
        if len(equity_curve) < 2:
            return 0.0

        total_return = PerformanceMetrics.total_return(equity_curve)
        n_periods = len(equity_curve)
        years = n_periods / periods_per_year

        if years == 0:
            return 0.0

        annualized = (1 + total_return) ** (1 / years) - 1

        return annualized

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 35040,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annualized risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

        return sharpe

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 35040,
    ) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility).

        Args:
            returns: Series of returns
            risk_free_rate: Annualized risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year

        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)

        return sortino

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of equity values

        Returns:
            Maximum drawdown as negative decimal
        """
        if len(equity_curve) < 2:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        return drawdown.min()

    @staticmethod
    def calmar_ratio(
        equity_curve: pd.Series,
        periods_per_year: int = 35040,
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            equity_curve: Series of equity values
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        ann_return = PerformanceMetrics.annualized_return(
            equity_curve, periods_per_year
        )
        max_dd = abs(PerformanceMetrics.max_drawdown(equity_curve))

        if max_dd == 0:
            return 0.0

        return ann_return / max_dd

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).

        Args:
            returns: Series of returns

        Returns:
            Win rate as decimal
        """
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        total = len(returns)

        return wins / total

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Series of returns

        Returns:
            Profit factor
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def average_win_loss_ratio(returns: pd.Series) -> float:
        """
        Calculate average win to average loss ratio.

        Args:
            returns: Series of returns

        Returns:
            Win/loss ratio
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0:
            return np.inf if len(wins) > 0 else 0.0

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 0.0

        return avg_win / avg_loss

    @staticmethod
    def recovery_factor(equity_curve: pd.Series) -> float:
        """
        Calculate recovery factor (net profit / max drawdown).

        Args:
            equity_curve: Series of equity values

        Returns:
            Recovery factor
        """
        net_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
        max_dd = abs(
            PerformanceMetrics.max_drawdown(equity_curve) * equity_curve.iloc[0]
        )

        if max_dd == 0:
            return 0.0

        return net_profit / max_dd

    @staticmethod
    def tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio (average top percentile / average bottom percentile).

        Args:
            returns: Series of returns
            percentile: Percentile for tails (e.g., 0.05 for 5%)

        Returns:
            Tail ratio
        """
        top_threshold = returns.quantile(1 - percentile)
        bottom_threshold = returns.quantile(percentile)

        top_avg = returns[returns >= top_threshold].mean()
        bottom_avg = abs(returns[returns <= bottom_threshold].mean())

        if bottom_avg == 0:
            return 0.0

        return top_avg / bottom_avg

    @staticmethod
    def compute_all_metrics(
        equity_curve: pd.Series,
        returns: pd.Series,
        periods_per_year: int = 35040,
        risk_free_rate: float = 0.0,
    ) -> Dict:
        """
        Compute all performance metrics.

        Args:
            equity_curve: Series of equity values
            returns: Series of returns
            periods_per_year: Number of periods per year
            risk_free_rate: Annualized risk-free rate

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "total_return": PerformanceMetrics.total_return(equity_curve),
            "annualized_return": PerformanceMetrics.annualized_return(
                equity_curve, periods_per_year
            ),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            "max_drawdown": PerformanceMetrics.max_drawdown(equity_curve),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(
                equity_curve, periods_per_year
            ),
            "win_rate": PerformanceMetrics.win_rate(returns),
            "profit_factor": PerformanceMetrics.profit_factor(returns),
            "avg_win_loss_ratio": PerformanceMetrics.average_win_loss_ratio(returns),
            "recovery_factor": PerformanceMetrics.recovery_factor(equity_curve),
            "tail_ratio": PerformanceMetrics.tail_ratio(returns),
            "volatility": returns.std() * np.sqrt(periods_per_year),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict, title: str = "Performance Metrics") -> None:
        """
        Pretty print performance metrics.

        Args:
            metrics: Dictionary of metrics
            title: Title for the printout
        """
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        print(f"\nReturns:")
        print(f"  Total Return:       {metrics['total_return']*100:>10.2f}%")
        print(f"  Annualized Return:  {metrics['annualized_return']*100:>10.2f}%")

        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.3f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.3f}")

        print(f"\nRisk:")
        print(f"  Max Drawdown:       {metrics['max_drawdown']*100:>10.2f}%")
        print(f"  Volatility (ann.):  {metrics['volatility']*100:>10.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Win Rate:           {metrics['win_rate']*100:>10.2f}%")
        print(f"  Profit Factor:      {metrics['profit_factor']:>10.3f}")
        print(f"  Avg Win/Loss:       {metrics['avg_win_loss_ratio']:>10.3f}")
        print(f"  Recovery Factor:    {metrics['recovery_factor']:>10.3f}")

        print(f"\nDistribution:")
        print(f"  Tail Ratio:         {metrics['tail_ratio']:>10.3f}")
        print(f"  Skewness:           {metrics['skewness']:>10.3f}")
        print(f"  Kurtosis:           {metrics['kurtosis']:>10.3f}")

        print("\n" + "=" * 80)
