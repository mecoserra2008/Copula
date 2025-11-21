"""Demo script showcasing copula visualizations and benchmark comparison."""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bybit_fetcher import BybitDataFetcher
from src.transformations.empirical import EmpiricalTransform
from src.copulas.gaussian import GaussianCopula
from src.copulas.student_t import StudentTCopula
from src.copulas.clayton import ClaytonCopula
from src.copulas.gumbel import GumbelCopula
from src.copulas.frank import FrankCopula
from src.copulas.visualization import CopulaVisualizer
from src.manager import CopulaManager
from src.utils.logger import setup_logger

logger = setup_logger("demo_viz", level="INFO")


def demo_copula_visualizations():
    """Demonstrate copula visualization capabilities."""
    logger.info("=" * 80)
    logger.info("Copula Visualization Demo")
    logger.info("=" * 80)

    # Fetch data
    logger.info("\nFetching data for BTC-ETH pair...")
    fetcher = BybitDataFetcher()

    df1 = fetcher.fetch_and_cache("BTCUSDT", interval="15", days=30)
    df2 = fetcher.fetch_and_cache("ETHUSDT", interval="15", days=30)

    if df1.empty or df2.empty:
        logger.error("Failed to fetch data")
        return

    # Calculate returns
    returns1 = df1["close"].pct_change().dropna().values
    returns2 = df2["close"].pct_change().dropna().values

    # Ensure same length
    min_len = min(len(returns1), len(returns2))
    returns1 = returns1[:min_len]
    returns2 = returns2[:min_len]

    # Transform to uniform
    logger.info("Transforming to uniform marginals...")
    transformer = EmpiricalTransform()
    u1 = transformer.fit_transform(returns1)
    u2 = transformer.fit_transform(returns2)
    u_data = np.column_stack([u1, u2])

    # Fit copulas
    logger.info("\nFitting copulas...")
    copulas = {
        "Gaussian": GaussianCopula(),
        "Student-t": StudentTCopula(),
        "Clayton": ClaytonCopula(),
        "Gumbel": GumbelCopula(),
        "Frank": FrankCopula(),
    }

    fitted_copulas = []
    copula_names = []

    for name, copula in copulas.items():
        try:
            logger.info(f"  Fitting {name} copula...")
            copula.fit(u_data)
            fitted_copulas.append(copula)
            copula_names.append(name)
            logger.info(f"    ✓ AIC: {copula.aic(u_data):.2f}")
        except Exception as e:
            logger.warning(f"    Failed to fit {name}: {e}")

    if not fitted_copulas:
        logger.error("No copulas fitted successfully")
        return

    # Visualizations
    visualizer = CopulaVisualizer()

    # 1. PDF Surface for best copula
    logger.info("\nGenerating 3D PDF surface plot...")
    try:
        visualizer.plot_pdf_surface(
            fitted_copulas[0],
            title=f"{copula_names[0]} Copula PDF - BTC/ETH",
            save_path="plots/copula_pdf_surface.png"
        )
        logger.info("  ✓ Saved to plots/copula_pdf_surface.png")
    except Exception as e:
        logger.warning(f"  Failed to create PDF surface: {e}")

    # 2. CDF Contours
    logger.info("\nGenerating CDF contour plot...")
    try:
        visualizer.plot_cdf_contours(
            fitted_copulas[0],
            title=f"{copula_names[0]} Copula CDF - BTC/ETH",
            save_path="plots/copula_cdf_contours.png"
        )
        logger.info("  ✓ Saved to plots/copula_cdf_contours.png")
    except Exception as e:
        logger.warning(f"  Failed to create CDF contours: {e}")

    # 3. Scatter comparison
    logger.info("\nGenerating scatter comparison...")
    try:
        visualizer.plot_scatter_comparison(
            fitted_copulas[0],
            u_data,
            title=f"Empirical vs {copula_names[0]} Copula - BTC/ETH",
            save_path="plots/copula_scatter_comparison.png"
        )
        logger.info("  ✓ Saved to plots/copula_scatter_comparison.png")
    except Exception as e:
        logger.warning(f"  Failed to create scatter comparison: {e}")

    # 4. Copula comparison (all fitted copulas)
    logger.info("\nGenerating copula comparison grid...")
    try:
        visualizer.plot_copula_comparison(
            fitted_copulas,
            copula_names,
            save_path="plots/copula_comparison.png"
        )
        logger.info("  ✓ Saved to plots/copula_comparison.png")
    except Exception as e:
        logger.warning(f"  Failed to create copula comparison: {e}")

    # 5. Tail dependence illustration
    logger.info("\nGenerating tail dependence illustration...")
    try:
        # Use Student-t copula if available (has tail dependence)
        student_t_copula = next(
            (c for c, n in zip(fitted_copulas, copula_names) if n == "Student-t"),
            fitted_copulas[0]
        )
        visualizer.plot_tail_dependence_illustration(
            student_t_copula,
            u_data,
            save_path="plots/copula_tail_dependence.png"
        )
        logger.info("  ✓ Saved to plots/copula_tail_dependence.png")
    except Exception as e:
        logger.warning(f"  Failed to create tail dependence illustration: {e}")

    # 6. Density heatmap
    logger.info("\nGenerating density heatmap...")
    try:
        visualizer.plot_density_heatmap(
            fitted_copulas[0],
            title=f"{copula_names[0]} Copula Density - BTC/ETH",
            save_path="plots/copula_density_heatmap.png"
        )
        logger.info("  ✓ Saved to plots/copula_density_heatmap.png")
    except Exception as e:
        logger.warning(f"  Failed to create density heatmap: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Visualization demo complete!")
    logger.info("Check the 'plots/' directory for generated visualizations.")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)

    demo_copula_visualizations()
