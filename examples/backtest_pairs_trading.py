"""Example of backtesting pairs trading strategy."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bybit_fetcher import BybitDataFetcher
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.pairs_trading import CopulaPairsTradingStrategy
from src.utils.logger import setup_logger

logger = setup_logger("backtest_pairs", level="INFO")


def main():
    """Run pairs trading backtest."""
    logger.info("=" * 80)
    logger.info("Pairs Trading Strategy Backtest")
    logger.info("=" * 80)

    # Configuration
    symbol1 = "BTCUSDT"
    symbol2 = "ETHUSDT"
    benchmark_symbol = "BTCUSDT"  # Using BTC as benchmark
    interval = "15"  # 15-minute bars
    days = 30

    # Fetch data
    logger.info(f"\nFetching data for {symbol1}, {symbol2}, and benchmark...")
    fetcher = BybitDataFetcher()

    df1 = fetcher.fetch_and_cache(symbol1, interval=interval, days=days)
    df2 = fetcher.fetch_and_cache(symbol2, interval=interval, days=days)
    df_benchmark = fetcher.fetch_and_cache(benchmark_symbol, interval=interval, days=days)

    if df1.empty or df2.empty:
        logger.error("Failed to fetch data")
        return

    logger.info(f"Fetched {len(df1)} bars for {symbol1}")
    logger.info(f"Fetched {len(df2)} bars for {symbol2}")
    if not df_benchmark.empty:
        logger.info(f"Fetched {len(df_benchmark)} bars for {benchmark_symbol} (benchmark)")

    # Prepare data for backtesting
    data = {
        symbol1: df1,
        symbol2: df2,
    }

    # Initialize strategy
    strategy = CopulaPairsTradingStrategy(
        symbol1=symbol1,
        symbol2=symbol2,
        lookback_period=500,
        copula_type="gaussian",
        entry_threshold=0.05,
        exit_threshold=0.5,
        refit_frequency=100,
        position_size=0.5,
    )

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000.0,
        transaction_cost=0.001,
        slippage=0.0005,
    )

    # Load data
    engine.load_data(data)

    # Load benchmark if available
    if not df_benchmark.empty:
        engine.load_benchmark(df_benchmark, benchmark_symbol)

    # Run backtest
    logger.info("\nRunning backtest...")
    results = engine.run(strategy, show_progress=True)

    # Print results
    engine.print_results()

    # Try to plot results
    try:
        engine.plot_results()
    except Exception as e:
        logger.info(f"\nPlotting not available: {e}")
        logger.info("Install matplotlib to enable plotting: pip install matplotlib")


if __name__ == "__main__":
    main()
