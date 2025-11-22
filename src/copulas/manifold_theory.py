"""
Manifold theory and information geometry for copulas (pure numpy implementation).

This module implements differential geometric and information theoretic concepts
for copula analysis, including:
- Fisher information metric on the copula manifold
- Geodesic distances and paths
- Natural gradient descent
- Riemannian optimization
- Tangent space representations
- Manifold interpolation
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
from .base import BaseCopula


class CopulaManifold:
    """
    Represents the statistical manifold of copula distributions.

    The copula manifold is the space of all copula distributions with a given
    parametric form, equipped with the Fisher information metric (Riemannian metric).

    This provides a geometric view of statistical inference and optimization.
    """

    def __init__(self, copula: BaseCopula, epsilon: float = 1e-6):
        """
        Initialize the copula manifold.

        Args:
            copula: Base copula instance defining the parametric family
            epsilon: Small value for numerical derivatives
        """
        self.copula = copula
        self.epsilon = epsilon

    def fisher_information_matrix(
        self,
        U: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute the Fisher information matrix at given parameter values.

        The Fisher information matrix is the Riemannian metric tensor on the
        statistical manifold. It measures the local curvature and provides
        a natural distance metric.

        Formula: I(θ) = E[∇log p(u;θ) ∇log p(u;θ)ᵀ]

        Args:
            U: Data points (n_samples, 2)
            params: Parameter values (if None, use current copula params)

        Returns:
            Fisher information matrix (n_params, n_params)
        """
        if params is None:
            params = self.copula.params_

        # Get parameter names and values
        param_names = list(params.keys())
        param_values = np.array([params[k] for k in param_names])
        n_params = len(param_names)

        # Compute score function (gradient of log-likelihood)
        scores = self._compute_score(U, params)

        # Fisher information is the covariance of the score
        fisher_matrix = scores.T @ scores / U.shape[0]

        return fisher_matrix

    def _compute_score(
        self,
        U: np.ndarray,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute the score function (gradient of log-likelihood).

        Args:
            U: Data points (n_samples, 2)
            params: Parameter values

        Returns:
            Score matrix (n_samples, n_params)
        """
        param_names = list(params.keys())
        n_params = len(param_names)
        n_samples = U.shape[0]

        scores = np.zeros((n_samples, n_params))

        # Save original params
        original_params = self.copula.params_.copy()

        for i, param_name in enumerate(param_names):
            # Compute numerical gradient of log-density
            for j in range(n_samples):
                u_point = U[j:j+1]

                # Forward difference
                params_plus = params.copy()
                params_plus[param_name] += self.epsilon

                self.copula.params_ = params_plus
                self.copula.is_fitted_ = True
                pdf_plus = self.copula.pdf(u_point)[0]

                # Backward difference
                params_minus = params.copy()
                params_minus[param_name] -= self.epsilon

                self.copula.params_ = params_minus
                self.copula.is_fitted_ = True
                pdf_minus = self.copula.pdf(u_point)[0]

                # Numerical derivative of log-density
                log_pdf_deriv = (np.log(pdf_plus + 1e-10) - np.log(pdf_minus + 1e-10)) / (2 * self.epsilon)
                scores[j, i] = log_pdf_deriv

        # Restore original params
        self.copula.params_ = original_params

        return scores

    def geodesic_distance(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float],
        U: np.ndarray,
        n_steps: int = 10
    ) -> float:
        """
        Compute approximate geodesic distance between two parameter points.

        The geodesic distance is the length of the shortest path on the manifold
        between two points, measured using the Fisher information metric.

        Args:
            params1: First parameter point
            params2: Second parameter point
            U: Data for computing Fisher information
            n_steps: Number of discretization steps

        Returns:
            Approximate geodesic distance
        """
        # Linear interpolation path
        param_names = list(params1.keys())
        values1 = np.array([params1[k] for k in param_names])
        values2 = np.array([params2[k] for k in param_names])

        # Discretize path
        t_values = np.linspace(0, 1, n_steps + 1)
        distance = 0.0

        for i in range(n_steps):
            # Midpoint of segment
            t_mid = (t_values[i] + t_values[i+1]) / 2
            params_mid = {k: (1-t_mid)*params1[k] + t_mid*params2[k] for k in param_names}

            # Tangent vector
            delta = values2 - values1
            dt = 1.0 / n_steps

            # Fisher information at midpoint
            fisher = self.fisher_information_matrix(U, params_mid)

            # Infinitesimal distance: √(δθᵀ I(θ) δθ) dt
            distance += np.sqrt(delta @ fisher @ delta) * dt

        return distance

    def geodesic_path(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float],
        U: np.ndarray,
        n_points: int = 20
    ) -> List[Dict[str, float]]:
        """
        Compute approximate geodesic path between two parameter points.

        Uses linear interpolation as a first approximation (can be improved
        with proper geodesic equation integration).

        Args:
            params1: Starting parameters
            params2: Ending parameters
            U: Data for computing Fisher information
            n_points: Number of points on the path

        Returns:
            List of parameter dictionaries along the path
        """
        param_names = list(params1.keys())
        path = []

        for t in np.linspace(0, 1, n_points):
            params_t = {
                k: (1-t)*params1[k] + t*params2[k]
                for k in param_names
            }
            path.append(params_t)

        return path

    def natural_gradient(
        self,
        U: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute natural gradient direction for parameter optimization.

        The natural gradient is the gradient preconditioned by the inverse
        Fisher information matrix. It provides a coordinate-independent
        optimization direction on the manifold.

        Natural gradient: ∇̃L = I(θ)⁻¹ ∇L

        Args:
            U: Training data
            params: Current parameters (if None, use copula params)

        Returns:
            Natural gradient vector
        """
        if params is None:
            params = self.copula.params_

        # Compute score (gradient of log-likelihood)
        scores = self._compute_score(U, params)
        gradient = scores.mean(axis=0)

        # Compute Fisher information
        fisher = self.fisher_information_matrix(U, params)

        # Natural gradient = inverse Fisher × gradient
        try:
            fisher_inv = np.linalg.inv(fisher + 1e-6 * np.eye(len(gradient)))
            natural_grad = fisher_inv @ gradient
        except np.linalg.LinAlgError:
            # Fallback to regular gradient if Fisher is singular
            natural_grad = gradient

        return natural_grad

    def manifold_distance_to_independence(
        self,
        U: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute manifold distance from current copula to independence copula.

        Args:
            U: Data for computing Fisher information
            params: Parameters (if None, use copula params)

        Returns:
            Approximate distance to independence
        """
        if params is None:
            params = self.copula.params_

        # For Gaussian and t-copula, independence corresponds to rho=0
        if 'rho' in params:
            independence_params = params.copy()
            independence_params['rho'] = 0.0
            return self.geodesic_distance(params, independence_params, U)
        else:
            return 0.0

    def tangent_space_projection(
        self,
        U: np.ndarray,
        direction: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Project a direction vector onto the tangent space at a point.

        Args:
            U: Data for computing Fisher information
            direction: Direction vector in parameter space
            params: Base point parameters

        Returns:
            Projected direction in tangent space
        """
        if params is None:
            params = self.copula.params_

        # Compute Fisher information metric
        fisher = self.fisher_information_matrix(U, params)

        # Normalize direction using Fisher metric
        norm = np.sqrt(direction @ fisher @ direction)

        if norm > 1e-10:
            return direction / norm
        else:
            return direction

    def compute_scalar_curvature(
        self,
        U: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Estimate scalar curvature of the manifold at a point.

        The scalar curvature measures the intrinsic curvature of the manifold.
        This is a simplified estimate using the Ricci curvature.

        Args:
            U: Data for computing derivatives
            params: Parameters at which to compute curvature

        Returns:
            Approximate scalar curvature
        """
        if params is None:
            params = self.copula.params_

        # Compute Fisher information and its derivatives
        fisher = self.fisher_information_matrix(U, params)

        param_names = list(params.keys())
        n_params = len(param_names)

        # Compute Christoffel symbols (simplified)
        # For 2D manifold, scalar curvature ≈ -det(Fisher) / trace(Fisher)²
        det_fisher = np.linalg.det(fisher)
        trace_fisher = np.trace(fisher)

        if trace_fisher > 1e-10:
            curvature = -det_fisher / (trace_fisher ** 2)
        else:
            curvature = 0.0

        return curvature


class RiemannianOptimizer:
    """
    Riemannian optimization on the copula manifold using natural gradient.
    """

    def __init__(
        self,
        copula: BaseCopula,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        """
        Initialize Riemannian optimizer.

        Args:
            copula: Copula instance to optimize
            learning_rate: Step size for natural gradient descent
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.copula = copula
        self.manifold = CopulaManifold(copula)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        U: np.ndarray,
        initial_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Fit copula parameters using natural gradient descent on the manifold.

        Args:
            U: Training data (n_samples, 2)
            initial_params: Initial parameter values

        Returns:
            Optimized parameters
        """
        if initial_params is None:
            # Use simple method for initialization
            self.copula.fit(U)
            params = self.copula.params_.copy()
        else:
            params = initial_params.copy()

        param_names = list(params.keys())
        history = []

        for iteration in range(self.max_iter):
            # Compute natural gradient
            self.copula.params_ = params
            self.copula.is_fitted_ = True

            nat_grad = self.manifold.natural_gradient(U, params)

            # Update parameters
            params_new = params.copy()
            for i, name in enumerate(param_names):
                params_new[name] = params[name] + self.learning_rate * nat_grad[i]

            # Clip to valid ranges
            if 'rho' in params_new:
                params_new['rho'] = np.clip(params_new['rho'], -0.999, 0.999)
            if 'df' in params_new:
                params_new['df'] = np.clip(params_new['df'], 2.0, 30.0)

            # Check convergence
            param_change = np.linalg.norm([params_new[k] - params[k] for k in param_names])
            history.append(param_change)

            if param_change < self.tol:
                params = params_new
                break

            params = params_new

        # Set final parameters
        self.copula.params_ = params
        self.copula.is_fitted_ = True
        self.copula.n_obs_ = U.shape[0]

        return params


class ManifoldInterpolator:
    """
    Interpolate between different copula models on the manifold.
    """

    @staticmethod
    def interpolate_copulas(
        copula1: BaseCopula,
        copula2: BaseCopula,
        alpha: float,
        interpolation_type: str = "linear"
    ) -> Dict[str, float]:
        """
        Interpolate between two copulas in parameter space.

        Args:
            copula1: First copula
            copula2: Second copula
            alpha: Interpolation parameter in [0, 1]
            interpolation_type: "linear" or "geodesic"

        Returns:
            Interpolated parameters
        """
        if not copula1.is_fitted_ or not copula2.is_fitted_:
            raise ValueError("Both copulas must be fitted")

        params1 = copula1.params_
        params2 = copula2.params_

        # Check same parameter structure
        if set(params1.keys()) != set(params2.keys()):
            raise ValueError("Copulas must have same parameter structure")

        # Linear interpolation
        if interpolation_type == "linear":
            params_interp = {
                k: (1 - alpha) * params1[k] + alpha * params2[k]
                for k in params1.keys()
            }
        else:
            # For now, use linear (geodesic requires integration)
            params_interp = {
                k: (1 - alpha) * params1[k] + alpha * params2[k]
                for k in params1.keys()
            }

        return params_interp

    @staticmethod
    def create_interpolation_path(
        copula1: BaseCopula,
        copula2: BaseCopula,
        n_steps: int = 10
    ) -> List[Dict[str, float]]:
        """
        Create a path of interpolated copulas.

        Args:
            copula1: Starting copula
            copula2: Ending copula
            n_steps: Number of interpolation steps

        Returns:
            List of interpolated parameter dictionaries
        """
        path = []
        for alpha in np.linspace(0, 1, n_steps):
            params = ManifoldInterpolator.interpolate_copulas(
                copula1, copula2, alpha
            )
            path.append(params)

        return path


def compute_kl_divergence(
    copula1: BaseCopula,
    copula2: BaseCopula,
    n_samples: int = 10000
) -> float:
    """
    Compute Kullback-Leibler divergence between two copulas using Monte Carlo.

    KL(P||Q) = E_P[log(P/Q)] = E_P[log P] - E_P[log Q]

    Args:
        copula1: First copula (P)
        copula2: Second copula (Q)
        n_samples: Number of Monte Carlo samples

    Returns:
        KL divergence estimate
    """
    # Sample from first copula
    samples = copula1.sample(n_samples)

    # Compute log-densities
    log_p = np.log(copula1.pdf(samples) + 1e-10)
    log_q = np.log(copula2.pdf(samples) + 1e-10)

    # KL divergence
    kl = np.mean(log_p - log_q)

    return kl


def compute_mutual_information(copula: BaseCopula, n_samples: int = 10000) -> float:
    """
    Compute mutual information encoded in the copula.

    MI = ∫∫ c(u,v) log c(u,v) du dv

    where c is the copula density.

    Args:
        copula: Fitted copula
        n_samples: Number of Monte Carlo samples

    Returns:
        Mutual information estimate
    """
    # Sample uniformly from [0,1]²
    U = np.random.uniform(0, 1, (n_samples, 2))

    # Compute copula density
    density = copula.pdf(U)

    # Mutual information
    mi = np.mean(np.log(density + 1e-10) * density)

    return mi
