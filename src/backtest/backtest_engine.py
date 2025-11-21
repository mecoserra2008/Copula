"""Backtesting engine for trading strategies."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from .portfolio import Portfolio
from .performance import PerformanceMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Simulates strategy execution on historical data and tracks performance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.portfolio: Optional[Portfolio] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.results: Optional[Dict] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_symbol: Optional[str] = None

    def load_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Load historical data for backtesting.

        Args:
            data: Dictionary of {symbol: DataFrame} with OHLCV data
        """
        self.data = data

        # Validate data
        for symbol, df in data.items():
            required_cols = ["timestamp", "close"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"Data for {symbol} missing required columns: {required_cols}"
                )

        logger.info(f"Loaded data for {len(data)} symbols")

    def load_benchmark(self, benchmark_data: pd.DataFrame, benchmark_symbol: str = "BTCUSDT") -> None:
        """
        Load benchmark data for comparison.

        Args:
            benchmark_data: DataFrame with OHLCV data for benchmark
            benchmark_symbol: Symbol name for benchmark (e.g., 'BTCUSDT')
        """
        required_cols = ["timestamp", "close"]
        if not all(col in benchmark_data.columns for col in required_cols):
            raise ValueError(
                f"Benchmark data missing required columns: {required_cols}"
            )

        self.benchmark_data = benchmark_data
        self.benchmark_symbol = benchmark_symbol
        logger.info(f"Loaded benchmark data for {benchmark_symbol}")

    def run(
        self,
        strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        show_progress: bool = True,
    ) -> Dict:
        """
        Run backtest with a strategy.

        Args:
            strategy: Strategy object with on_data() method
            start_date: Start date for backtest
            end_date: End date for backtest
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with backtest results
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("=" * 80)
        logger.info("Starting Backtest")
        logger.info("=" * 80)

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
        )

        # Get all timestamps (assuming all data is aligned)
        first_symbol = list(self.data.keys())[0]
        timestamps = self.data[first_symbol]["timestamp"].values

        # Filter by date range
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]

        logger.info(f"Backtest period: {len(timestamps)} bars")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Initialize strategy
        strategy.initialize(self.portfolio, self.data)

        # Run backtest
        iterator = tqdm(timestamps, desc="Backtesting") if show_progress else timestamps

        for timestamp in iterator:
            # Get current data for all symbols
            current_data = {}
            prices = {}

            for symbol, df in self.data.items():
                row = df[df["timestamp"] == timestamp]
                if not row.empty:
                    current_data[symbol] = row.iloc[0]
                    prices[symbol] = row.iloc[0]["close"]

            # Update portfolio prices
            self.portfolio.update_prices(prices, timestamp)

            # Call strategy
            try:
                strategy.on_data(timestamp, current_data, self.portfolio)
            except Exception as e:
                logger.error(f"Strategy error at {timestamp}: {e}")
                continue

        # Finalize
        final_timestamp = timestamps[-1] if timestamps else datetime.now()
        strategy.finalize(self.portfolio, final_timestamp)

        # Compute results
        self.results = self._compute_results()

        logger.info("\n" + "=" * 80)
        logger.info("Backtest Complete")
        logger.info("=" * 80)

        return self.results

    def _compute_results(self) -> Dict:
        """Compute backtest results and metrics."""
        if self.portfolio is None:
            return {}

        # Get equity curve and returns
        equity_df = self.portfolio.get_equity_curve()
        returns = self.portfolio.get_returns()

        if len(equity_df) == 0:
            logger.warning("No equity data available")
            return {"error": "No equity data"}

        # Compute performance metrics
        metrics = PerformanceMetrics.compute_all_metrics(
            equity_curve=equity_df["equity"],
            returns=returns,
        )

        # Portfolio summary
        portfolio_summary = self.portfolio.get_summary()

        # Trades
        trades_df = self.portfolio.get_trades_df()

        results = {
            "metrics": metrics,
            "portfolio_summary": portfolio_summary,
            "equity_curve": equity_df,
            "returns": returns,
            "trades": trades_df,
            "num_trades": len(trades_df),
        }

        # Add benchmark comparison if available
        if self.benchmark_data is not None:
            benchmark_metrics = self._compute_benchmark_comparison(equity_df, returns)
            results["benchmark"] = benchmark_metrics

        return results

    def _compute_benchmark_comparison(self, equity_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        Compute benchmark comparison metrics.

        Args:
            equity_df: Strategy equity curve
            returns: Strategy returns

        Returns:
            Dictionary with benchmark comparison metrics
        """
        if self.benchmark_data is None:
            return {}

        # Align benchmark data with equity curve timestamps
        equity_timestamps = equity_df.index
        benchmark_aligned = self.benchmark_data[
            self.benchmark_data["timestamp"].isin(equity_timestamps)
        ].copy()

        if len(benchmark_aligned) == 0:
            logger.warning("No overlapping timestamps with benchmark")
            return {}

        # Calculate benchmark returns
        benchmark_aligned = benchmark_aligned.sort_values("timestamp")
        benchmark_prices = benchmark_aligned["close"].values
        benchmark_returns = pd.Series(
            np.diff(benchmark_prices) / benchmark_prices[:-1],
            index=benchmark_aligned["timestamp"].iloc[1:].values
        )

        # Align returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            logger.warning("No common timestamps for benchmark comparison")
            return {}

        strategy_returns_aligned = returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]

        # Compute benchmark metrics
        benchmark_total_return = (benchmark_prices[-1] / benchmark_prices[0] - 1) * 100
        benchmark_ann_return = PerformanceMetrics.annualized_return(
            pd.Series(benchmark_prices, index=benchmark_aligned["timestamp"]),
            periods_per_year=35040  # 15-minute bars
        )

        # Compute comparison metrics
        excess_returns = strategy_returns_aligned - benchmark_returns_aligned
        information_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(35040)
            if excess_returns.std() != 0 else 0
        )

        # Beta and Alpha
        covariance = np.cov(strategy_returns_aligned, benchmark_returns_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_returns_aligned)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

        strategy_mean_return = strategy_returns_aligned.mean() * 35040
        benchmark_mean_return = benchmark_returns_aligned.mean() * 35040
        alpha = strategy_mean_return - (beta * benchmark_mean_return)

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(35040)

        # Up/down capture
        up_periods = benchmark_returns_aligned > 0
        down_periods = benchmark_returns_aligned < 0

        up_capture = (
            (strategy_returns_aligned[up_periods].mean() / benchmark_returns_aligned[up_periods].mean())
            if up_periods.sum() > 0 and benchmark_returns_aligned[up_periods].mean() != 0 else 0
        )

        down_capture = (
            (strategy_returns_aligned[down_periods].mean() / benchmark_returns_aligned[down_periods].mean())
            if down_periods.sum() > 0 and benchmark_returns_aligned[down_periods].mean() != 0 else 0
        )

        return {
            "symbol": self.benchmark_symbol,
            "total_return": benchmark_total_return,
            "annualized_return": benchmark_ann_return,
            "returns": benchmark_returns_aligned,
            "prices": benchmark_aligned,
            "information_ratio": information_ratio,
            "beta": beta,
            "alpha": alpha,
            "tracking_error": tracking_error,
            "up_capture": up_capture,
            "down_capture": down_capture,
        }

    def print_results(self) -> None:
        """Print backtest results."""
        if self.results is None:
            logger.warning("No results to print. Run backtest first.")
            return

        if "error" in self.results:
            logger.error(f"Backtest error: {self.results['error']}")
            return

        # Print performance metrics
        PerformanceMetrics.print_metrics(
            self.results["metrics"], title="Backtest Performance"
        )

        # Print portfolio summary
        print("\nPortfolio Summary:")
        summary = self.results["portfolio_summary"]
        print(f"  Initial Capital:    ${summary['initial_capital']:>12,.2f}")
        print(f"  Final Equity:       ${summary['final_equity']:>12,.2f}")
        print(f"  Total Return:       {summary['total_return_pct']:>12.2f}%")
        print(f"  Number of Trades:   {summary['num_trades']:>12}")
        print(f"  Final Cash:         ${summary['cash']:>12,.2f}")
        print(f"  Open Positions:     {summary['num_positions']:>12}")

        # Print benchmark comparison if available
        if "benchmark" in self.results and self.results["benchmark"]:
            print("\n" + "=" * 80)
            print("Benchmark Comparison")
            print("=" * 80)

            benchmark = self.results["benchmark"]
            strategy_return = summary['total_return_pct']

            print(f"\nReturns:")
            print(f"  Strategy Return:        {strategy_return:>12.2f}%")
            print(f"  {benchmark['symbol']} Return:   {benchmark['total_return']:>12.2f}%")
            print(f"  Outperformance:         {strategy_return - benchmark['total_return']:>12.2f}%")

            print(f"\nRisk-Adjusted Metrics:")
            print(f"  Information Ratio:      {benchmark['information_ratio']:>12.4f}")
            print(f"  Beta:                   {benchmark['beta']:>12.4f}")
            print(f"  Alpha (annualized):     {benchmark['alpha']:>12.2%}")
            print(f"  Tracking Error:         {benchmark['tracking_error']:>12.2%}")

            print(f"\nCapture Ratios:")
            print(f"  Up Capture:             {benchmark['up_capture']:>12.2%}")
            print(f"  Down Capture:           {benchmark['down_capture']:>12.2%}")

        print("\n" + "=" * 80)

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive backtest results with benchmark comparison.

        Args:
            save_path: Optional path to save the figure

        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.gridspec import GridSpec
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot results.")
            return

        if self.results is None:
            logger.warning("No results to plot. Run backtest first.")
            return

        equity_df = self.results["equity_curve"]
        returns = self.results["returns"]
        has_benchmark = "benchmark" in self.results and self.results["benchmark"]

        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity curve with benchmark
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(equity_df.index, equity_df["equity"], label="Strategy", linewidth=2)

        if has_benchmark:
            benchmark = self.results["benchmark"]
            benchmark_prices = benchmark["prices"]
            # Normalize benchmark to same starting value as strategy
            normalized_benchmark = (
                benchmark_prices["close"].values / benchmark_prices["close"].values[0]
                * self.initial_capital
            )
            ax1.plot(
                benchmark_prices["timestamp"].values,
                normalized_benchmark,
                label=f'{benchmark["symbol"]} Benchmark',
                linewidth=2,
                alpha=0.7,
                linestyle="--"
            )

        ax1.set_title("Equity Curve vs Benchmark", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Date", fontsize=11)
        ax1.set_ylabel("Equity ($)", fontsize=11)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.tick_params(axis='x', rotation=45)

        # 2. Drawdown comparison
        ax2 = fig.add_subplot(gs[0, 2])
        running_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red", label="Strategy")

        if has_benchmark:
            benchmark_prices_values = benchmark_prices["close"].values
            benchmark_running_max = pd.Series(benchmark_prices_values).expanding().max()
            benchmark_drawdown = (
                (benchmark_prices_values - benchmark_running_max.values) / benchmark_running_max.values * 100
            )
            ax2.fill_between(
                benchmark_prices["timestamp"].values,
                benchmark_drawdown,
                0,
                alpha=0.3,
                color="blue",
                label=benchmark["symbol"]
            )

        ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=10)
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. Returns distribution
        ax3 = fig.add_subplot(gs[1, 0])
        returns_clean = returns.dropna()
        ax3.hist(returns_clean * 100, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
        ax3.axvline(returns_clean.mean() * 100, color="red", linestyle="--", linewidth=2, label="Mean")
        ax3.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax3.set_title("Returns Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Return (%)", fontsize=10)
        ax3.set_ylabel("Frequency", fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cumulative returns
        ax4 = fig.add_subplot(gs[1, 1])
        cumulative_returns = (1 + returns).cumprod() - 1
        ax4.plot(cumulative_returns.index, cumulative_returns.values * 100, linewidth=2, color="green")
        ax4.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax4.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Date", fontsize=10)
        ax4.set_ylabel("Cumulative Return (%)", fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Rolling Sharpe ratio
        ax5 = fig.add_subplot(gs[1, 2])
        window = min(100, len(returns) // 4)
        if window > 10:
            rolling_sharpe = (
                returns.rolling(window=window).mean() / returns.rolling(window=window).std()
                * np.sqrt(35040)
            )
            ax5.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color="purple")
            ax5.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            ax5.axhline(1, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Sharpe=1")
            ax5.set_title(f"Rolling Sharpe ({window} bars)", fontsize=12, fontweight="bold")
            ax5.set_xlabel("Date", fontsize=10)
            ax5.set_ylabel("Sharpe Ratio", fontsize=10)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='x', rotation=45)

        # 6. Monthly returns heatmap
        ax6 = fig.add_subplot(gs[2, :2])
        if len(returns) > 30:
            try:
                import seaborn as sns

                # Convert to DataFrame with datetime index
                returns_df = pd.DataFrame({"returns": returns.values}, index=pd.to_datetime(returns.index))
                returns_df["year"] = returns_df.index.year
                returns_df["month"] = returns_df.index.month

                # Calculate monthly returns
                monthly_returns = returns_df.groupby(["year", "month"])["returns"].apply(
                    lambda x: (1 + x).prod() - 1
                )

                # Pivot to matrix
                monthly_matrix = monthly_returns.unstack(fill_value=0) * 100

                if len(monthly_matrix) > 0:
                    sns.heatmap(
                        monthly_matrix,
                        annot=True,
                        fmt=".1f",
                        cmap="RdYlGn",
                        center=0,
                        cbar_kws={'label': 'Return (%)'},
                        ax=ax6,
                        linewidths=0.5
                    )
                    ax6.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight="bold")
                    ax6.set_xlabel("Month", fontsize=10)
                    ax6.set_ylabel("Year", fontsize=10)
            except Exception as e:
                logger.warning(f"Could not create monthly heatmap: {e}")
                ax6.text(0.5, 0.5, "Monthly heatmap not available", ha="center", va="center", transform=ax6.transAxes)

        # 7. Performance metrics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        metrics = self.results["metrics"]
        summary_text = f"""
Strategy Performance
{'='*25}
Total Return:     {metrics['returns']['total_return']:.2f}%
Ann. Return:      {metrics['returns']['annualized_return']:.2f}%
Sharpe Ratio:     {metrics['risk_adjusted']['sharpe_ratio']:.3f}
Sortino Ratio:    {metrics['risk_adjusted']['sortino_ratio']:.3f}
Calmar Ratio:     {metrics['risk_adjusted']['calmar_ratio']:.3f}
Max Drawdown:     {metrics['risk']['max_drawdown']:.2f}%
Win Rate:         {metrics['trade_stats']['win_rate']:.2f}%
Profit Factor:    {metrics['trade_stats']['profit_factor']:.3f}
"""

        if has_benchmark:
            benchmark = self.results["benchmark"]
            summary = self.results["portfolio_summary"]
            summary_text += f"""
Benchmark Comparison
{'='*25}
{benchmark['symbol']} Return: {benchmark['total_return']:.2f}%
Outperformance:   {summary['total_return_pct'] - benchmark['total_return']:.2f}%
Information Ratio:{benchmark['information_ratio']:.3f}
Beta:             {benchmark['beta']:.3f}
Alpha:            {benchmark['alpha']:.2%}
"""

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Overall title
        fig.suptitle("Comprehensive Backtest Report", fontsize=16, fontweight="bold", y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()
