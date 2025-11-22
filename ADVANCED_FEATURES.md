# Advanced Copula Features

This document describes the advanced features available in this copula library, including interactive visualizations and manifold theory integration.

## Table of Contents

1. [Pure Numpy Implementation](#pure-numpy-implementation)
2. [Interactive Plotly Visualizations](#interactive-plotly-visualizations)
3. [Manifold Theory & Information Geometry](#manifold-theory--information-geometry)
4. [Usage Examples](#usage-examples)

---

## Pure Numpy Implementation

All copula implementations (Gaussian and Student-T) now use **only numpy** with no scipy dependencies.

### Features

- **Gaussian Copula** (`src/copulas/gaussian.py`):
  - Normal PDF, CDF, and inverse CDF (Beasley-Springer-Moro algorithm)
  - Bivariate normal PDF (closed-form)
  - Bivariate normal CDF (Drezner-Wesolowsky approximation)
  - Multivariate normal sampling (Cholesky decomposition)

- **Student-T Copula** (`src/copulas/student_t.py`):
  - Log-gamma function (Lanczos approximation)
  - T-distribution PDF, CDF, and inverse CDF
  - Regularized incomplete beta function (continued fractions)
  - Multivariate-T sampling (chi-squared construction)
  - Coordinate descent optimization

### Benefits

- ✅ Zero scipy dependencies
- ✅ Portable and lightweight
- ✅ Numerically stable
- ✅ Full control over algorithms

---

## Interactive Plotly Visualizations

The `InteractiveCopulaVisualizer` class provides rich, interactive visualizations.

### Available Visualizations

1. **3D Surface Plots** - Rotate and zoom PDF/CDF surfaces
2. **Interactive Contours** - Hover for values, zoom, pan
3. **Scatter Comparisons** - Compare empirical data with copula samples
4. **Conditional Distributions** - C(u|v) for different conditioning values
5. **Tail Dependence Analysis** - Visualize lower and upper tail behavior
6. **Copula Comparison Grids** - Side-by-side comparison of multiple copulas
7. **Parameter Sensitivity** - How density changes with parameters

### Example Usage

```python
from src.copulas.gaussian import GaussianCopula
from src.copulas.plotly_viz import InteractiveCopulaVisualizer

# Fit copula
copula = GaussianCopula()
copula.fit(U)  # U is your uniform data

# Create visualizer
viz = InteractiveCopulaVisualizer()

# Generate 3D surface plot
viz.plot_3d_surface(
    copula,
    mode="pdf",
    save_html="copula_3d.html"
)

# Interactive contour plot
viz.plot_contour_interactive(
    copula,
    mode="pdf",
    save_html="copula_contour.html"
)

# Conditional distribution
viz.plot_conditional_distribution(
    copula,
    v_values=[0.25, 0.5, 0.75],
    save_html="conditional.html"
)
```

### Features

- 🎨 Beautiful color schemes (Viridis, Plasma, etc.)
- 🔍 Hover tooltips with precise values
- 📊 Interactive zoom, pan, rotate
- 💾 Export to HTML for sharing
- 📱 Responsive design

---

## Manifold Theory & Information Geometry

The copula parameter space forms a **statistical manifold** equipped with the Fisher information metric.

### Core Concepts

#### 1. Fisher Information Metric

The Fisher information matrix defines a Riemannian metric on the parameter manifold:

```python
from src.copulas.manifold_theory import CopulaManifold

manifold = CopulaManifold(copula)
fisher = manifold.fisher_information_matrix(U)
```

This metric measures:
- **Local curvature** of the statistical manifold
- **Natural distance** between parameter points
- **Information content** of the distribution

#### 2. Geodesic Distances

The shortest path between parameters on the manifold:

```python
params1 = {"rho": 0.3}
params2 = {"rho": 0.8}

distance = manifold.geodesic_distance(params1, params2, U)
path = manifold.geodesic_path(params1, params2, U, n_points=10)
```

#### 3. Natural Gradient Optimization

Optimization that respects the manifold geometry:

```python
from src.copulas.manifold_theory import RiemannianOptimizer

optimizer = RiemannianOptimizer(copula, learning_rate=0.01)
optimal_params = optimizer.fit(U, initial_params)
```

Natural gradient = Fisher-inverse × gradient:
- **Coordinate independent** - same result regardless of parameterization
- **Faster convergence** - follows natural geometry
- **Theoretically optimal** - steepest descent on manifold

#### 4. Information-Theoretic Metrics

Built-in methods for information theory:

```python
# Kendall's tau
tau = copula.kendall_tau()

# Spearman's rho
rho_s = copula.spearman_rho()

# Mutual information
mi = compute_mutual_information(copula, n_samples=10000)

# KL divergence between copulas
kl = compute_kl_divergence(copula1, copula2, n_samples=10000)

# All metrics at once
metrics = copula.information_geometry_metrics(U)
print(metrics['fisher_information'])
print(metrics['mutual_information'])
print(metrics['manifold_volume'])
```

#### 5. Manifold Interpolation

Smoothly interpolate between different copula models:

```python
from src.copulas.manifold_theory import ManifoldInterpolator

# Interpolate between two copulas
params_mid = ManifoldInterpolator.interpolate_copulas(
    copula1, copula2, alpha=0.5
)

# Create interpolation path
path = ManifoldInterpolator.create_interpolation_path(
    copula1, copula2, n_steps=20
)
```

### Mathematical Background

The statistical manifold view provides several insights:

1. **Fisher-Rao Metric**: The Fisher information is the unique Riemannian metric invariant under sufficient statistics.

2. **Geodesics**: Shortest paths minimize:
   ```
   L = ∫₀¹ √(θ̇ᵀ I(θ(t)) θ̇) dt
   ```
   where I(θ) is the Fisher information matrix.

3. **Natural Gradient**: For θ = θ + η ∇̃L, the natural gradient is:
   ```
   ∇̃L = I(θ)⁻¹ ∇L
   ```

4. **Mutual Information**: For Gaussian copula with correlation ρ:
   ```
   MI = -½ log(1 - ρ²)
   ```

5. **Scalar Curvature**: Measures intrinsic curvature of the manifold.

---

## Usage Examples

### Example 1: Complete Visualization Pipeline

```python
import numpy as np
from src.copulas.gaussian import GaussianCopula
from src.copulas.student_t import StudentTCopula
from src.copulas.plotly_viz import InteractiveCopulaVisualizer

# Generate data
U = np.random.uniform(0, 1, (1000, 2))

# Fit copulas
gauss = GaussianCopula()
gauss.fit(U)

t_cop = StudentTCopula()
t_cop.fit(U)

# Visualize
viz = InteractiveCopulaVisualizer()

# 3D surfaces
viz.plot_3d_surface(gauss, save_html="gauss_3d.html")
viz.plot_3d_surface(t_cop, save_html="t_3d.html")

# Compare
viz.plot_copula_comparison_grid(
    [gauss, t_cop],
    ["Gaussian", "Student-T"],
    save_html="comparison.html"
)
```

### Example 2: Manifold Theory Analysis

```python
from src.copulas.gaussian import GaussianCopula
from src.copulas.manifold_theory import CopulaManifold, compute_mutual_information

# Fit copula
copula = GaussianCopula()
copula.fit(U)

# Analyze manifold geometry
manifold = CopulaManifold(copula)

# Fisher information
fisher = manifold.fisher_information_matrix(U)
print(f"Fisher information: {fisher}")

# Curvature
curvature = manifold.compute_scalar_curvature(U)
print(f"Scalar curvature: {curvature}")

# Distance to independence
dist = manifold.manifold_distance_to_independence(U)
print(f"Distance to independence: {dist}")

# Information metrics
metrics = copula.information_geometry_metrics(U)
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### Example 3: Natural Gradient Optimization

```python
from src.copulas.gaussian import GaussianCopula
from src.copulas.manifold_theory import RiemannianOptimizer

# Create copula
copula = GaussianCopula()

# Optimize using natural gradient
optimizer = RiemannianOptimizer(
    copula,
    learning_rate=0.05,
    max_iter=100,
    tol=1e-6
)

# Fit with initial guess
initial_params = {"rho": 0.1}
optimal_params = optimizer.fit(U, initial_params)

print(f"Optimal parameters: {optimal_params}")
```

---

## Complete Examples

Run the provided example scripts:

### Interactive Visualizations

```bash
python examples/advanced_visualizations.py
```

This generates:
- 3D interactive surface plots
- Interactive contour maps
- Scatter comparisons
- Conditional distributions
- Tail dependence analysis
- Multi-copula comparisons

All saved as HTML files in `outputs/`.

### Manifold Theory Demo

```bash
python examples/manifold_theory_demo.py
```

This demonstrates:
- Fisher information computation
- Geodesic distances and paths
- Natural gradient optimization
- KL divergence between copulas
- Manifold interpolation
- Mutual information analysis

---

## Dependencies

### Core (Required)
- `numpy` - All computations

### Visualizations (Optional)
- `plotly` - Interactive plots

```bash
pip install plotly
```

### Data Generation (Examples only)
- `scipy` - Used only for generating example data

---

## Mathematical Foundations

### 1. Copula Density

For a copula C(u₁, u₂), the density is:

```
c(u₁, u₂) = ∂²C(u₁, u₂) / ∂u₁∂u₂
```

### 2. Gaussian Copula

```
C_ρ(u₁, u₂) = Φ_ρ(Φ⁻¹(u₁), Φ⁻¹(u₂))
```

where Φ is the standard normal CDF and Φ_ρ is the bivariate normal CDF.

### 3. Student-T Copula

```
C_ρ,ν(u₁, u₂) = T_ρ,ν(t_ν⁻¹(u₁), t_ν⁻¹(u₂))
```

where t_ν is the Student-T CDF with ν degrees of freedom.

### 4. Fisher Information (Gaussian Copula)

For a single parameter ρ:

```
I(ρ) = n / (1 - ρ²)²
```

### 5. Mutual Information (Gaussian Copula)

```
MI = -½ log(1 - ρ²)
```

### 6. Tail Dependence (Student-T)

```
λₗ = λᵤ = 2 T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
```

---

## Advanced Topics

### Manifold Curvature

The scalar curvature measures how the manifold curves in space:
- **Positive curvature**: Sphere-like (geodesics converge)
- **Zero curvature**: Flat (Euclidean geometry)
- **Negative curvature**: Hyperbolic (geodesics diverge)

### Natural Coordinates

The Fisher metric provides natural (coordinate-independent) measurements:
- Distances don't depend on parameterization
- Angles between directions are intrinsic
- Optimization follows natural geometry

### Information Monotonicity

The Fisher information satisfies:
- **Monotonicity**: Information can only decrease under transformations
- **Additivity**: Information from independent sources adds
- **Invariance**: Preserved under sufficient statistics

---

## References

1. **Copula Theory**:
   - Nelsen, R. B. (2006). "An Introduction to Copulas"
   - Joe, H. (2014). "Dependence Modeling with Copulas"

2. **Information Geometry**:
   - Amari, S. & Nagaoka, H. (2000). "Methods of Information Geometry"
   - Ay, N., et al. (2017). "Information Geometry"

3. **Natural Gradient**:
   - Amari, S. (1998). "Natural Gradient Works Efficiently in Learning"
   - Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method"

---

## Contributing

We welcome contributions! Areas of interest:
- Additional copula families
- More advanced manifold computations (geodesic integration, parallel transport)
- Interactive 3D manifold visualizations
- Quantum information geometric extensions

---

## License

Same as the main project.
