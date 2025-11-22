"""
Manifold Theory and Information Geometry for Copulas

This example demonstrates:
1. Fisher information metric computation
2. Geodesic distances on the copula manifold
3. Natural gradient optimization
4. Information-theoretic metrics
5. Manifold interpolation between copulas
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.copulas.gaussian import GaussianCopula
from src.copulas.student_t import StudentTCopula
from src.copulas.manifold_theory import (
    CopulaManifold,
    RiemannianOptimizer,
    ManifoldInterpolator,
    compute_kl_divergence,
    compute_mutual_information
)


def print_section(title):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def main():
    print_section("Manifold Theory and Information Geometry for Copulas")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    # Create data with known correlation
    rho_true = 0.6
    mean = [0, 0]
    cov = [[1, rho_true], [rho_true, 1]]
    X = np.random.multivariate_normal(mean, cov, n_samples)

    # Transform to uniform
    from scipy import stats
    U = np.column_stack([
        stats.norm.cdf(X[:, 0]),
        stats.norm.cdf(X[:, 1])
    ])

    print(f"Generated {n_samples} samples with true correlation ρ={rho_true:.4f}")

    # =========================================================================
    # 1. Information Geometry Metrics
    # =========================================================================
    print_section("1. Information Geometry Metrics")

    # Fit Gaussian copula
    gauss_copula = GaussianCopula()
    gauss_copula.fit(U)

    # Get information geometry metrics
    metrics_gauss = gauss_copula.information_geometry_metrics(U)

    print("Gaussian Copula Metrics:")
    print(f"  Correlation (ρ): {metrics_gauss['correlation']:.4f}")
    print(f"  Mutual Information: {metrics_gauss['mutual_information']:.4f}")
    print(f"  Fisher Information: {metrics_gauss['fisher_information'][0,0]:.2f}")
    print(f"  Fisher Determinant: {metrics_gauss['fisher_determinant']:.2f}")
    print(f"  Manifold Volume Element: {metrics_gauss['manifold_volume']:.2f}")
    print(f"  Kendall's τ: {gauss_copula.kendall_tau():.4f}")
    print(f"  Spearman's ρ: {gauss_copula.spearman_rho():.4f}")

    # Fit Student-T copula
    t_copula = StudentTCopula()
    t_copula.fit(U)

    metrics_t = t_copula.information_geometry_metrics(U)

    print()
    print("Student-T Copula Metrics:")
    print(f"  Correlation (ρ): {metrics_t['correlation']:.4f}")
    print(f"  Degrees of Freedom (ν): {metrics_t['degrees_of_freedom']:.2f}")
    print(f"  Mutual Information: {metrics_t['mutual_information']:.4f}")
    print(f"  Tail Dependence (λₗ): {metrics_t['tail_dependence_lower']:.4f}")
    print(f"  Tail Dependence (λᵤ): {metrics_t['tail_dependence_upper']:.4f}")
    print(f"  Kendall's τ: {t_copula.kendall_tau():.4f}")
    print(f"  Spearman's ρ: {t_copula.spearman_rho():.4f}")

    # =========================================================================
    # 2. Fisher Information and Manifold Geometry
    # =========================================================================
    print_section("2. Fisher Information and Manifold Geometry")

    # Create manifold
    manifold = CopulaManifold(gauss_copula)

    # Compute Fisher information at current parameters
    fisher = manifold.fisher_information_matrix(U)
    print(f"Fisher Information Matrix:")
    print(f"  {fisher}")
    print()

    # Distance to independence
    dist_indep = manifold.manifold_distance_to_independence(U)
    print(f"Manifold Distance to Independence: {dist_indep:.4f}")
    print()

    # Scalar curvature
    curvature = manifold.compute_scalar_curvature(U)
    print(f"Scalar Curvature of Manifold: {curvature:.6f}")

    # =========================================================================
    # 3. Geodesic Distances and Paths
    # =========================================================================
    print_section("3. Geodesic Distances and Paths on the Manifold")

    # Define two points on the manifold
    params1 = {"rho": 0.3}
    params2 = {"rho": 0.8}

    # Compute geodesic distance
    geo_dist = manifold.geodesic_distance(params1, params2, U, n_steps=20)
    print(f"Geodesic distance between ρ=0.3 and ρ=0.8: {geo_dist:.4f}")
    print()

    # Compute geodesic path
    geo_path = manifold.geodesic_path(params1, params2, U, n_points=10)
    print("Geodesic path (10 points):")
    for i, params in enumerate(geo_path):
        print(f"  Step {i}: ρ = {params['rho']:.4f}")

    # =========================================================================
    # 4. Natural Gradient Optimization
    # =========================================================================
    print_section("4. Natural Gradient Optimization (Riemannian)")

    print("Optimizing Gaussian copula using natural gradient descent...")
    print()

    # Create optimizer
    gauss_copula_opt = GaussianCopula()
    optimizer = RiemannianOptimizer(
        gauss_copula_opt,
        learning_rate=0.01,
        max_iter=50,
        tol=1e-6
    )

    # Initial guess
    initial_params = {"rho": 0.2}

    # Optimize
    optimal_params = optimizer.fit(U, initial_params)

    print(f"Initial ρ: {initial_params['rho']:.4f}")
    print(f"Optimal ρ (Natural Gradient): {optimal_params['rho']:.4f}")
    print(f"Standard MLE ρ: {gauss_copula.params_['rho']:.4f}")
    print()
    print("Natural gradient finds the same optimum but follows")
    print("a coordinate-independent path on the manifold!")

    # =========================================================================
    # 5. KL Divergence Between Copulas
    # =========================================================================
    print_section("5. Kullback-Leibler Divergence")

    # Compute KL divergence from Gaussian to t-copula
    kl_gauss_to_t = compute_kl_divergence(gauss_copula, t_copula, n_samples=5000)
    kl_t_to_gauss = compute_kl_divergence(t_copula, gauss_copula, n_samples=5000)

    print(f"KL(Gaussian || Student-T): {kl_gauss_to_t:.6f}")
    print(f"KL(Student-T || Gaussian): {kl_t_to_gauss:.6f}")
    print()
    print("Note: KL divergence is asymmetric (not a distance metric)")

    # =========================================================================
    # 6. Manifold Interpolation
    # =========================================================================
    print_section("6. Manifold Interpolation Between Copulas")

    # Create two Gaussian copulas with different correlations
    copula_low = GaussianCopula()
    copula_low.params_ = {"rho": 0.2}
    copula_low.is_fitted_ = True

    copula_high = GaussianCopula()
    copula_high.params_ = {"rho": 0.9}
    copula_high.is_fitted_ = True

    # Interpolate
    print("Interpolating from ρ=0.2 to ρ=0.9:")
    print()

    for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        params_interp = ManifoldInterpolator.interpolate_copulas(
            copula_low, copula_high, alpha
        )
        print(f"  α={alpha:.2f}: ρ={params_interp['rho']:.4f}")

    # =========================================================================
    # 7. Mutual Information Analysis
    # =========================================================================
    print_section("7. Mutual Information Analysis")

    print("Mutual Information as a function of correlation:")
    print()

    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cop = GaussianCopula()
        cop.params_ = {"rho": rho}
        cop.is_fitted_ = True
        cop.n_obs_ = 100

        mi = compute_mutual_information(cop, n_samples=5000)

        # Theoretical MI for Gaussian copula: -0.5 * log(1 - ρ²)
        mi_theory = -0.5 * np.log(1 - rho**2)

        print(f"  ρ={rho:.1f}: MI={mi:.4f} (theory: {mi_theory:.4f})")

    print()
    print("Mutual information increases with correlation strength!")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")

    print("This example demonstrated:")
    print("  ✓ Fisher information metric on the copula manifold")
    print("  ✓ Geodesic distances and paths")
    print("  ✓ Natural gradient optimization (Riemannian)")
    print("  ✓ KL divergence between different copulas")
    print("  ✓ Manifold interpolation")
    print("  ✓ Information-theoretic metrics")
    print()
    print("The copula manifold provides a geometric view of statistical inference,")
    print("where the Fisher information defines a natural Riemannian metric.")
    print()
    print("Key insights:")
    print("  • The manifold distance respects the statistical geometry")
    print("  • Natural gradient follows coordinate-independent optimization paths")
    print("  • Information geometry unifies statistical and geometric perspectives")
    print()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print()
        print("Please install required packages:")
        print("  pip install numpy scipy")
