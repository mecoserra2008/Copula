"""Example of analyzing multiple cryptocurrency pairs."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.manager.copula_manager import CopulaManager
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("multi_pair_example", level="INFO")


def main():
    """Run multi-pair copula analysis."""
    logger.info("=" * 80)
    logger.info("Multi-Pair Copula Analysis Example")
    logger.info("=" * 80)

    # Initialize manager
    manager = CopulaManager()

    # Define pairs to analyze
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("ETHUSDT", "SOLUSDT"),
        ("BTCUSDT", "XRPUSDT"),
        ("ETHUSDT", "XRPUSDT"),
    ]

    logger.info(f"\nAnalyzing {len(pairs)} pairs...")

    # Fit copulas for all pairs
    all_results = manager.fit_multiple_pairs(
        pairs=pairs,
        interval="15",
        days=30,
        copula_types=["gaussian", "student_t", "clayton", "gumbel", "frank"],
    )

    # Create comparison DataFrame
    comparison_df = manager.compare_pairs(all_results)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{comparison_df.to_string()}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    # Most common best copula
    best_copula_counts = comparison_df["best_copula"].value_counts()
    logger.info(f"\nMost common best copula:")
    for copula, count in best_copula_counts.items():
        logger.info(f"  {copula}: {count} pairs")

    # Average Kendall's tau
    avg_tau = comparison_df["kendall_tau"].mean()
    logger.info(f"\nAverage Kendall's tau: {avg_tau:.4f}")

    # Pairs with strongest dependence
    logger.info(f"\nPairs with strongest dependence:")
    top_pairs = comparison_df.nlargest(3, "kendall_tau")
    for _, row in top_pairs.iterrows():
        logger.info(
            f"  {row['symbol1']}-{row['symbol2']}: "
            f"τ={row['kendall_tau']:.4f} ({row['best_copula']})"
        )

    # Tail dependence analysis
    logger.info(f"\nTail dependence summary:")
    avg_lower_tail = comparison_df["lower_tail_dep"].mean()
    avg_upper_tail = comparison_df["upper_tail_dep"].mean()
    logger.info(f"  Average lower tail dependence: {avg_lower_tail:.4f}")
    logger.info(f"  Average upper tail dependence: {avg_upper_tail:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Multi-pair analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
