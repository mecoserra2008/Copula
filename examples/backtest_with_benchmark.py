"""Example of backtesting with Bitcoin benchmark comparison."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bybit_fetcher import BybitDataFetcher
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.pairs_trading import CopulaPairsTradingStrategy
from src.utils.logger import setup_logger

logger = setup_logger("backtest_benchmark", level="INFO")


def main():
    """Run pairs trading backtest with Bitcoin benchmark."""
    logger.info("=" * 80)
    logger.info("Pairs Trading Backtest with Bitcoin Benchmark")
    logger.info("=" * 80)

    # Configuration
    symbol1 = "ETHUSDT"
    symbol2 = "SOLUSDT"
    benchmark_symbol = "BTCUSDT"
    interval = "15"  # 15-minute bars
    days = 30

    # Fetch data
    logger.info(f"\nFetching data for {symbol1}, {symbol2}, and {benchmark_symbol}...")
    fetcher = BybitDataFetcher()

    df1 = fetcher.fetch_and_cache(symbol1, interval=interval, days=days)
    df2 = fetcher.fetch_and_cache(symbol2, interval=interval, days=days)
    df_benchmark = fetcher.fetch_and_cache(benchmark_symbol, interval=interval, days=days)

    if df1.empty or df2.empty or df_benchmark.empty:
        logger.error("Failed to fetch data")
        return

    logger.info(f"  ✓ Fetched {len(df1)} bars for {symbol1}")
    logger.info(f"  ✓ Fetched {len(df2)} bars for {symbol2}")
    logger.info(f"  ✓ Fetched {len(df_benchmark)} bars for {benchmark_symbol}")

    # Prepare data for backtesting
    data = {
        symbol1: df1,
        symbol2: df2,
    }

    # Initialize strategy
    logger.info("\nInitializing copula pairs trading strategy...")
    strategy = CopulaPairsTradingStrategy(
        symbol1=symbol1,
        symbol2=symbol2,
        lookback_period=500,
        copula_type="student_t",  # Student-t copula captures tail dependence
        entry_threshold=0.05,
        exit_threshold=0.5,
        refit_frequency=100,
        position_size=0.5,
    )

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000.0,
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,  # 0.05% slippage
    )

    # Load data and benchmark
    engine.load_data(data)
    engine.load_benchmark(df_benchmark, benchmark_symbol)
    logger.info(f"  ✓ Loaded benchmark: {benchmark_symbol}")

    # Run backtest
    logger.info("\nRunning backtest...")
    results = engine.run(strategy, show_progress=True)

    # Print comprehensive results including benchmark comparison
    logger.info("\n")
    engine.print_results()

    # Plot comprehensive charts with benchmark
    logger.info("\nGenerating comprehensive backtest charts...")
    try:
        engine.plot_results(save_path="plots/backtest_with_benchmark.png")
        logger.info("  ✓ Charts saved to plots/backtest_with_benchmark.png")
    except Exception as e:
        logger.warning(f"  Plotting not available: {e}")
        logger.info("  Install matplotlib and seaborn: pip install matplotlib seaborn")

    # Additional analysis
    if "benchmark" in results and results["benchmark"]:
        logger.info("\n" + "=" * 80)
        logger.info("Key Insights")
        logger.info("=" * 80)

        benchmark = results["benchmark"]
        strategy_return = results["portfolio_summary"]["total_return_pct"]
        metrics = results["metrics"]

        # Performance comparison
        if strategy_return > benchmark["total_return"]:
            logger.info(
                f"\n✓ Strategy OUTPERFORMED {benchmark_symbol} by "
                f"{strategy_return - benchmark['total_return']:.2f}%"
            )
        else:
            logger.info(
                f"\n✗ Strategy UNDERPERFORMED {benchmark_symbol} by "
                f"{benchmark['total_return'] - strategy_return:.2f}%"
            )

        # Risk-adjusted performance
        logger.info(f"\nRisk-Adjusted Performance:")
        logger.info(f"  - Sharpe Ratio: {metrics['risk_adjusted']['sharpe_ratio']:.3f}")
        logger.info(f"  - Information Ratio: {benchmark['information_ratio']:.3f}")

        if benchmark["information_ratio"] > 0:
            logger.info(
                f"  ✓ Positive Information Ratio indicates skill-based outperformance"
            )

        # Beta analysis
        logger.info(f"\nMarket Exposure:")
        logger.info(f"  - Beta: {benchmark['beta']:.3f}")

        if abs(benchmark["beta"]) < 0.5:
            logger.info(
                f"  ✓ Low beta ({abs(benchmark['beta']):.3f}) - Strategy is market-neutral"
            )
        elif abs(benchmark["beta"]) < 1.0:
            logger.info(
                f"  ✓ Moderate beta ({abs(benchmark['beta']):.3f}) - Partial market exposure"
            )
        else:
            logger.info(
                f"  ⚠ High beta ({abs(benchmark['beta']):.3f}) - Strong market correlation"
            )

        # Alpha
        if benchmark["alpha"] > 0:
            logger.info(
                f"  ✓ Positive Alpha ({benchmark['alpha']:.2%}) - Excess return after "
                f"adjusting for market risk"
            )
        else:
            logger.info(
                f"  ✗ Negative Alpha ({benchmark['alpha']:.2%}) - Underperformed "
                f"risk-adjusted benchmark"
            )

        # Capture ratios
        logger.info(f"\nUp/Down Capture:")
        logger.info(f"  - Up Capture: {benchmark['up_capture']:.2%}")
        logger.info(f"  - Down Capture: {benchmark['down_capture']:.2%}")

        if benchmark["up_capture"] > 1.0 and abs(benchmark["down_capture"]) < 1.0:
            logger.info(
                f"  ✓ Excellent! Captures more upside and less downside than benchmark"
            )
        elif benchmark["up_capture"] > 1.0:
            logger.info(f"  ✓ Captures more upside than benchmark")
        elif abs(benchmark["down_capture"]) < 1.0:
            logger.info(f"  ✓ Captures less downside than benchmark")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)

    main()
