"""Basic example of using the copula framework."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.manager.copula_manager import CopulaManager
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("example", level="INFO")


def main():
    """Run basic copula analysis example."""
    logger.info("=" * 80)
    logger.info("Basic Copula Analysis Example")
    logger.info("=" * 80)

    # Initialize manager
    manager = CopulaManager()

    # Define pair to analyze
    symbol1 = "BTCUSDT"
    symbol2 = "ETHUSDT"

    logger.info(f"\nAnalyzing pair: {symbol1} - {symbol2}")

    # Fit copula models
    results = manager.fit_pair(
        symbol1=symbol1,
        symbol2=symbol2,
        interval="15",  # 15-minute candles
        days=30,  # Last 30 days
        copula_types=["gaussian", "student_t", "clayton", "gumbel", "frank"],
    )

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nPair: {symbol1} - {symbol2}")
    logger.info(f"Observations: {results['n_observations']}")
    logger.info(f"Interval: {results['interval']} minutes")
    logger.info(f"Days: {results['days']}")

    # Best copula
    best = results["best_copula"]
    logger.info(f"\n--- Best Copula ---")
    logger.info(f"Type: {best['type']}")
    logger.info(f"AIC: {best['aic']:.2f}")
    logger.info(f"BIC: {best['bic']:.2f}")
    logger.info(f"Log-Likelihood: {best['log_likelihood']:.2f}")
    logger.info(f"Kendall's Tau: {best['kendall_tau']:.4f}")
    logger.info(
        f"Tail Dependence: "
        f"Lower={best['tail_dependence']['lower']:.4f}, "
        f"Upper={best['tail_dependence']['upper']:.4f}"
    )
    logger.info(f"Parameters: {best['params']}")

    # All copulas comparison
    logger.info(f"\n--- All Copulas Comparison ---")
    logger.info(f"{'Type':<15} {'AIC':>10} {'BIC':>10} {'Kendall τ':>10}")
    logger.info("-" * 50)

    for copula_type, result in results["copulas"].items():
        logger.info(
            f"{copula_type:<15} {result['aic']:>10.2f} {result['bic']:>10.2f} "
            f"{result['kendall_tau']:>10.4f}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
