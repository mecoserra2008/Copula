"""Example of risk analysis using copulas."""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.manager.copula_manager import CopulaManager
from src.analysis.risk import RiskMetrics
from src.analysis.dependence import DependenceMetrics
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("risk_example", level="INFO")


def main():
    """Run risk analysis example."""
    logger.info("=" * 80)
    logger.info("Risk Analysis Example")
    logger.info("=" * 80)

    # Initialize manager
    manager = CopulaManager()

    # Analyze BTC-ETH pair
    symbol1 = "BTCUSDT"
    symbol2 = "ETHUSDT"

    logger.info(f"\nAnalyzing pair: {symbol1} - {symbol2}")

    # Fit copula
    results = manager.fit_pair(
        symbol1=symbol1,
        symbol2=symbol2,
        interval="15",
        days=30,
    )

    # Extract returns
    r1 = results["data"]["returns1"]
    r2 = results["data"]["returns2"]

    # Dependence metrics
    logger.info("\n" + "=" * 80)
    logger.info("DEPENDENCE METRICS")
    logger.info("=" * 80)

    metrics = DependenceMetrics.all_metrics(r1, r2)

    logger.info(f"\nCorrelation measures:")
    logger.info(f"  Pearson:  {metrics['pearson']:.4f}")
    logger.info(f"  Spearman: {metrics['spearman']:.4f}")
    logger.info(f"  Kendall:  {metrics['kendall']:.4f}")

    logger.info(f"\nTail dependence (empirical):")
    logger.info(f"  Lower: {metrics['tail_dependence']['lower']:.4f}")
    logger.info(f"  Upper: {metrics['tail_dependence']['upper']:.4f}")

    logger.info(f"\nCorrelation breakdown:")
    for region, corr in metrics['correlation_breakdown'].items():
        logger.info(f"  {region.capitalize()}: {corr:.4f}")

    # Risk metrics
    logger.info("\n" + "=" * 80)
    logger.info("RISK METRICS")
    logger.info("=" * 80)

    confidence = 0.95

    # Individual asset risk
    var1 = RiskMetrics.value_at_risk(r1, confidence)
    var2 = RiskMetrics.value_at_risk(r2, confidence)
    cvar1 = RiskMetrics.conditional_value_at_risk(r1, confidence)
    cvar2 = RiskMetrics.conditional_value_at_risk(r2, confidence)

    logger.info(f"\nIndividual asset risk (95% confidence):")
    logger.info(f"  {symbol1}:")
    logger.info(f"    VaR:  {var1*100:.3f}%")
    logger.info(f"    CVaR: {cvar1*100:.3f}%")
    logger.info(f"  {symbol2}:")
    logger.info(f"    VaR:  {var2*100:.3f}%")
    logger.info(f"    CVaR: {cvar2*100:.3f}%")

    # Portfolio risk (50/50 allocation)
    logger.info(f"\nPortfolio risk (50/50 allocation):")

    for weight1 in [0.3, 0.5, 0.7]:
        port_var = RiskMetrics.portfolio_var(r1, r2, weight1, confidence)
        port_cvar = RiskMetrics.portfolio_cvar(r1, r2, weight1, confidence)

        logger.info(f"\n  Weight {symbol1}: {weight1*100:.0f}%")
        logger.info(f"    Portfolio VaR:  {port_var*100:.3f}%")
        logger.info(f"    Portfolio CVaR: {port_cvar*100:.3f}%")

    # Diversification benefit
    logger.info(f"\n" + "-" * 80)
    logger.info("DIVERSIFICATION BENEFIT (50/50 portfolio)")
    logger.info("-" * 80)

    div_benefit = RiskMetrics.diversification_benefit(
        r1, r2, weight1=0.5, confidence=confidence
    )

    logger.info(f"\n  Weighted sum of individual VaRs: {div_benefit['weighted_var']*100:.3f}%")
    logger.info(f"  Actual portfolio VaR:            {div_benefit['portfolio_var']*100:.3f}%")
    logger.info(f"  Diversification benefit:         {div_benefit['diversification_benefit']*100:.3f}%")
    logger.info(f"  Diversification benefit:         {div_benefit['diversification_benefit_pct']:.2f}%")

    # Upside/downside correlation
    logger.info(f"\n" + "-" * 80)
    logger.info("UPSIDE/DOWNSIDE CORRELATION")
    logger.info("-" * 80)

    downside_corr = RiskMetrics.downside_correlation(r1, r2)
    upside_corr = RiskMetrics.upside_correlation(r1, r2)

    logger.info(f"\n  Downside correlation: {downside_corr:.4f}")
    logger.info(f"  Upside correlation:   {upside_corr:.4f}")

    if downside_corr > upside_corr:
        logger.info(f"\n  → Higher correlation during downside (increased crash risk)")
    elif upside_corr > downside_corr:
        logger.info(f"\n  → Higher correlation during upside (joint rallies)")
    else:
        logger.info(f"\n  → Similar correlation in both directions")

    logger.info("\n" + "=" * 80)
    logger.info("Risk analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
