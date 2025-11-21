"""Example of backtesting statistical arbitrage strategy."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bybit_fetcher import BybitDataFetcher
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.statistical_arbitrage import CopulaStatisticalArbitrageStrategy
from src.utils.logger import setup_logger

logger = setup_logger("backtest_stat_arb", level="INFO")


def main():
    """Run statistical arbitrage backtest."""
    logger.info("=" * 80)
    logger.info("Statistical Arbitrage Strategy Backtest")
    logger.info("=" * 80)

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("ETHUSDT", "SOLUSDT"),
        ("BTCUSDT", "XRPUSDT"),
    ]
    interval = "15"
    days = 30

    # Fetch data
    logger.info(f"\nFetching data for {len(symbols)} symbols...")
    fetcher = BybitDataFetcher()

    data = {}
    for symbol in symbols:
        df = fetcher.fetch_and_cache(symbol, interval=interval, days=days)
        if not df.empty:
            data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} bars")
        else:
            logger.warning(f"  {symbol}: No data")

    if len(data) < 2:
        logger.error("Insufficient data for strategy")
        return

    # Initialize strategy
    strategy = CopulaStatisticalArbitrageStrategy(
        pairs=pairs,
        lookback_period=500,
        copula_type="gaussian",
        rebalance_frequency=20,
        entry_threshold=0.1,
        num_positions=4,
        position_size=0.8,
    )

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000.0,
        transaction_cost=0.001,
        slippage=0.0005,
    )

    # Load data
    engine.load_data(data)

    # Run backtest
    logger.info("\nRunning backtest...")
    results = engine.run(strategy, show_progress=True)

    # Print results
    engine.print_results()

    # Print trade summary
    if "trades" in results:
        trades_df = results["trades"]
        logger.info(f"\nTrade Summary:")
        logger.info(f"  Total trades: {len(trades_df)}")

        if len(trades_df) > 0:
            by_symbol = trades_df.groupby("symbol").agg(
                {
                    "quantity": "count",
                    "value": "sum",
                }
            )
            logger.info(f"\nTrades by symbol:")
            logger.info(by_symbol)

    # Try to plot
    try:
        engine.plot_results()
    except Exception as e:
        logger.info(f"\nPlotting not available: {e}")


if __name__ == "__main__":
    main()
