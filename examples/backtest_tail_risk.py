"""Example of backtesting tail risk hedging strategy."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bybit_fetcher import BybitDataFetcher
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.tail_risk_hedging import CopulaTailRiskHedgingStrategy
from src.utils.logger import setup_logger

logger = setup_logger("backtest_tail_risk", level="INFO")


def main():
    """Run tail risk hedging backtest."""
    logger.info("=" * 80)
    logger.info("Tail Risk Hedging Strategy Backtest")
    logger.info("=" * 80)

    # Configuration
    base_asset = "BTCUSDT"
    hedge_assets = ["ETHUSDT", "SOLUSDT"]
    interval = "15"
    days = 30

    # Fetch data
    logger.info(f"\nFetching data...")
    fetcher = BybitDataFetcher()

    data = {}
    all_symbols = [base_asset] + hedge_assets

    for symbol in all_symbols:
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
    strategy = CopulaTailRiskHedgingStrategy(
        base_asset=base_asset,
        hedge_assets=hedge_assets,
        lookback_period=500,
        copula_type="student_t",  # Student-t for tail dependence
        rebalance_frequency=20,
        tail_risk_threshold=0.1,
        max_hedge_ratio=0.3,
        base_allocation=0.7,
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

    # Compare with buy-and-hold
    logger.info("\n" + "=" * 80)
    logger.info("Comparison with Buy-and-Hold")
    logger.info("=" * 80)

    # Simple buy-and-hold return
    df_base = data[base_asset]
    bh_return = (df_base["close"].iloc[-1] / df_base["close"].iloc[0]) - 1

    strategy_return = results["metrics"]["total_return"]

    logger.info(f"\nBuy-and-Hold {base_asset}:")
    logger.info(f"  Total Return: {bh_return * 100:.2f}%")

    logger.info(f"\nTail Risk Hedging Strategy:")
    logger.info(f"  Total Return: {strategy_return * 100:.2f}%")
    logger.info(f"  Max Drawdown: {results['metrics']['max_drawdown'] * 100:.2f}%")

    # Try to plot
    try:
        engine.plot_results()
    except Exception as e:
        logger.info(f"\nPlotting not available: {e}")


if __name__ == "__main__":
    main()
