"""Backtesting engine for trading strategies."""

from typing import Dict, List, Optional
import pandas as pd
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

        return results

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

        print("\n" + "=" * 80)

    def plot_results(self) -> None:
        """
        Plot backtest results.

        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot results.")
            return

        if self.results is None:
            logger.warning("No results to plot. Run backtest first.")
            return

        equity_df = self.results["equity_curve"]
        returns = self.results["returns"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        axes[0, 0].plot(equity_df.index, equity_df["equity"])
        axes[0, 0].set_title("Equity Curve")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Equity ($)")
        axes[0, 0].grid(True)

        # Drawdown
        running_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Drawdown")
        axes[0, 1].grid(True)

        # Returns distribution
        axes[1, 0].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Returns Distribution")
        axes[1, 0].set_xlabel("Return")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)

        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values)
        axes[1, 1].set_title("Cumulative Returns")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Cumulative Return")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
