"""
Advanced Interactive Copula Visualizations

This example demonstrates the advanced Plotly-based visualizations for copulas,
including 3D surfaces, interactive contours, and comparison plots.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.copulas.gaussian import GaussianCopula
from src.copulas.student_t import StudentTCopula
from src.copulas.plotly_viz import InteractiveCopulaVisualizer


def main():
    print("=" * 80)
    print("Advanced Interactive Copula Visualizations")
    print("=" * 80)
    print()

    # Generate some correlated data
    np.random.seed(42)
    n_samples = 1000

    # Create bivariate normal samples with correlation
    rho_true = 0.7
    mean = [0, 0]
    cov = [[1, rho_true], [rho_true, 1]]
    X = np.random.multivariate_normal(mean, cov, n_samples)

    # Transform to uniform using empirical CDF
    from scipy import stats
    U = np.column_stack([
        stats.norm.cdf(X[:, 0]),
        stats.norm.cdf(X[:, 1])
    ])

    print(f"Generated {n_samples} samples with true correlation ρ={rho_true}")
    print()

    # Fit copulas
    print("Fitting Gaussian copula...")
    gauss_copula = GaussianCopula()
    gauss_copula.fit(U)
    print(f"  Fitted parameters: rho={gauss_copula.params_['rho']:.4f}")
    print()

    print("Fitting Student-T copula...")
    t_copula = StudentTCopula()
    t_copula.fit(U)
    print(f"  Fitted parameters: rho={t_copula.params_['rho']:.4f}, df={t_copula.params_['df']:.2f}")
    print()

    # Create visualizer
    viz = InteractiveCopulaVisualizer()

    # 1. 3D Surface Plot of PDF
    print("Creating 3D surface plots...")
    print("  - Gaussian copula PDF...")
    viz.plot_3d_surface(
        gauss_copula,
        mode="pdf",
        title="Gaussian Copula PDF (Interactive 3D)",
        grid_size=40,
        show=False,
        save_html="outputs/gaussian_pdf_3d.html"
    )

    print("  - Student-T copula PDF...")
    viz.plot_3d_surface(
        t_copula,
        mode="pdf",
        title="Student-T Copula PDF (Interactive 3D)",
        grid_size=40,
        show=False,
        save_html="outputs/t_copula_pdf_3d.html"
    )
    print("  ✓ Saved to outputs/gaussian_pdf_3d.html and outputs/t_copula_pdf_3d.html")
    print()

    # 2. Interactive Contour Plots
    print("Creating interactive contour plots...")
    viz.plot_contour_interactive(
        gauss_copula,
        mode="pdf",
        title="Gaussian Copula PDF Contours (Interactive)",
        grid_size=80,
        show=False,
        save_html="outputs/gaussian_contour.html"
    )

    viz.plot_contour_interactive(
        t_copula,
        mode="pdf",
        title="Student-T Copula PDF Contours (Interactive)",
        grid_size=80,
        show=False,
        save_html="outputs/t_copula_contour.html"
    )
    print("  ✓ Saved contour plots")
    print()

    # 3. Scatter Comparison
    print("Creating scatter comparison plots...")
    viz.plot_scatter_comparison(
        gauss_copula,
        U,
        n_samples=1000,
        title="Gaussian Copula: Empirical vs Samples",
        show=False,
        save_html="outputs/gaussian_scatter_comparison.html"
    )

    viz.plot_scatter_comparison(
        t_copula,
        U,
        n_samples=1000,
        title="Student-T Copula: Empirical vs Samples",
        show=False,
        save_html="outputs/t_scatter_comparison.html"
    )
    print("  ✓ Saved scatter comparison plots")
    print()

    # 4. Conditional Distribution
    print("Creating conditional distribution plots...")
    viz.plot_conditional_distribution(
        gauss_copula,
        v_values=[0.1, 0.25, 0.5, 0.75, 0.9],
        title="Gaussian Copula Conditional Distribution C(u|v)",
        show=False,
        save_html="outputs/gaussian_conditional.html"
    )

    viz.plot_conditional_distribution(
        t_copula,
        v_values=[0.1, 0.25, 0.5, 0.75, 0.9],
        title="Student-T Copula Conditional Distribution C(u|v)",
        show=False,
        save_html="outputs/t_conditional.html"
    )
    print("  ✓ Saved conditional distribution plots")
    print()

    # 5. Tail Dependence Analysis
    print("Creating tail dependence analysis...")
    viz.plot_tail_dependence_analysis(
        gauss_copula,
        U,
        title="Gaussian Copula Tail Dependence",
        show=False,
        save_html="outputs/gaussian_tail_dependence.html"
    )

    viz.plot_tail_dependence_analysis(
        t_copula,
        U,
        title="Student-T Copula Tail Dependence",
        show=False,
        save_html="outputs/t_tail_dependence.html"
    )
    print("  ✓ Saved tail dependence analysis")
    print()

    # 6. Copula Comparison Grid
    print("Creating copula comparison grid...")
    viz.plot_copula_comparison_grid(
        copulas=[gauss_copula, t_copula],
        names=["Gaussian", "Student-T"],
        grid_size=40,
        title="Gaussian vs Student-T Copula Comparison",
        show=False,
        save_html="outputs/copula_comparison.html"
    )
    print("  ✓ Saved comparison grid")
    print()

    print("=" * 80)
    print("All visualizations created successfully!")
    print("=" * 80)
    print()
    print("Open the HTML files in outputs/ to view interactive plots:")
    print("  - gaussian_pdf_3d.html - Interactive 3D surface")
    print("  - gaussian_contour.html - Interactive contour plot")
    print("  - gaussian_scatter_comparison.html - Sample comparison")
    print("  - gaussian_conditional.html - Conditional distributions")
    print("  - gaussian_tail_dependence.html - Tail dependence")
    print("  - copula_comparison.html - Side-by-side comparison")
    print()
    print("(Similar files created for Student-T copula)")
    print()


if __name__ == "__main__":
    # Create outputs directory
    import os
    os.makedirs("outputs", exist_ok=True)

    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print()
        print("Please install plotly: pip install plotly")
        print("And scipy for data generation: pip install scipy")
