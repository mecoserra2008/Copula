"""Frank copula implementation."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau
from scipy.integrate import quad

from .base import BivariateCopula
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FrankCopula(BivariateCopula):
    """
    Frank copula.

    The Frank copula is an Archimedean copula that exhibits symmetric dependence
    with no tail dependence. It is useful for modeling general symmetric
    dependencies.

    Generator: φ(t) = -log[(e^{-θt} - 1)/(e^{-θ} - 1)], θ ∈ ℝ, θ ≠ 0
    C(u,v) = -1/θ * log[1 + (e^{-θu} - 1)(e^{-θv} - 1)/(e^{-θ} - 1)]

    Parameters:
        theta: Dependence parameter (θ ∈ ℝ, θ ≠ 0)
            θ = 0: Independence (limiting case)
            θ > 0: Positive dependence
            θ < 0: Negative dependence
            θ → ∞: Perfect positive dependence
            θ → -∞: Perfect negative dependence
    """

    def __init__(self, theta_bounds: tuple = (-20, 20)):
        """
        Initialize Frank copula.

        Args:
            theta_bounds: Bounds for theta parameter
        """
        super().__init__()
        self.theta_bounds = theta_bounds
        self.n_params_ = 1  # theta parameter

    def _debye(self, theta: float, order: int = 1) -> float:
        """
        Compute Debye function D_k(θ) = k/θ^k ∫_0^θ t^k/(e^t - 1) dt.

        Used for the relationship between Kendall's tau and theta.

        Args:
            theta: Parameter value
            order: Order of Debye function (1 or 2)

        Returns:
            Debye function value
        """
        if abs(theta) < 1e-6:
            return 1.0  # Limit as theta -> 0

        def integrand(t):
            if t < 1e-10:
                return 0
            return t**order / (np.exp(t) - 1)

        integral, _ = quad(integrand, 0, abs(theta))

        return order / (abs(theta) ** order) * integral

    def fit(self, U: np.ndarray) -> "FrankCopula":
        """
        Fit Frank copula to uniform data.

        Uses the relationship between Kendall's tau and theta:
        τ = 1 - 4/θ[1 - D_1(θ)]
        where D_1 is the Debye function.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Self
        """
        # Validate
        self._validate_bivariate(U)

        n = U.shape[0]

        # Compute Kendall's tau
        tau, _ = kendalltau(U[:, 0], U[:, 1])

        # Initial estimate from Kendall's tau
        # Use approximation: θ ≈ 5.7 * τ for small tau
        if abs(tau) < 0.3:
            theta_init = 5.7 * tau
        else:
            # Use iterative search
            def tau_error(theta):
                if abs(theta) < 1e-6:
                    return tau**2
                D1 = self._debye(theta, order=1)
                tau_pred = 1 - 4 / theta * (1 - D1)
                return (tau - tau_pred) ** 2

            result = minimize_scalar(
                tau_error,
                bounds=self.theta_bounds,
                method="bounded",
            )
            theta_init = result.x

        theta_init = np.clip(theta_init, self.theta_bounds[0], self.theta_bounds[1])

        # Refine estimate using MLE
        def neg_log_likelihood(theta):
            if abs(theta) < 1e-6:
                return 1e10  # Independence case

            if not (self.theta_bounds[0] <= theta <= self.theta_bounds[1]):
                return 1e10

            try:
                # Compute log-likelihood
                u, v = U[:, 0], U[:, 1]

                # Clip to avoid numerical issues
                u = np.clip(u, 1e-10, 1 - 1e-10)
                v = np.clip(v, 1e-10, 1 - 1e-10)

                # Compute terms
                exp_theta = np.exp(-theta)
                exp_theta_u = np.exp(-theta * u)
                exp_theta_v = np.exp(-theta * v)

                numerator = (exp_theta_u - 1) * (exp_theta_v - 1)
                denominator = exp_theta - 1

                # log c(u,v) = log(-θ) + log(exp_theta - 1)
                #              - θ(u+v) - 2*log(denominator + numerator)
                log_dens = (
                    np.log(abs(theta))
                    + np.log(exp_theta - 1)
                    - theta * (u + v)
                    - 2 * np.log(denominator + numerator)
                )

                # Handle numerical issues
                log_dens = log_dens[np.isfinite(log_dens)]

                if len(log_dens) == 0:
                    return 1e10

                return -np.sum(log_dens)

            except Exception as e:
                logger.debug(f"Error in likelihood computation: {e}")
                return 1e10

        # Optimize
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=self.theta_bounds,
            method="bounded",
        )

        if result.success and result.fun < 1e9:
            theta_opt = result.x
        else:
            logger.warning(
                f"MLE optimization failed, using tau estimate: {result.message}"
            )
            theta_opt = theta_init

        # Avoid exactly zero
        if abs(theta_opt) < 0.01:
            theta_opt = 0.01 if tau > 0 else -0.01

        self.params_ = {"theta": theta_opt}
        self.n_obs_ = n
        self.is_fitted_ = True

        logger.info(f"Fitted FrankCopula: theta={theta_opt:.4f}")

        return self

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Frank copula density.

        c(u,v) = -θ(e^{-θ} - 1)e^{-θ(u+v)} / [(e^{-θ} - 1) + (e^{-θu} - 1)(e^{-θv} - 1)]^2

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of density values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing PDF")

        self._validate_bivariate(U)

        theta = self.params_["theta"]
        u, v = U[:, 0], U[:, 1]

        # Clip to avoid numerical issues
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        # Compute terms
        exp_theta = np.exp(-theta)
        exp_theta_u = np.exp(-theta * u)
        exp_theta_v = np.exp(-theta * v)

        numerator = -theta * (exp_theta - 1) * np.exp(-theta * (u + v))
        denominator = (exp_theta - 1) + (exp_theta_u - 1) * (exp_theta_v - 1)

        density = numerator / (denominator**2)

        return np.abs(density)  # Ensure positive

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Frank copula CDF.

        C(u,v) = -1/θ * log[1 + (e^{-θu} - 1)(e^{-θv} - 1)/(e^{-θ} - 1)]

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing CDF")

        self._validate_bivariate(U)

        theta = self.params_["theta"]
        u, v = U[:, 0], U[:, 1]

        # Clip to avoid numerical issues
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        # Compute CDF
        exp_theta = np.exp(-theta)
        exp_theta_u = np.exp(-theta * u)
        exp_theta_v = np.exp(-theta * v)

        numerator = (exp_theta_u - 1) * (exp_theta_v - 1)
        denominator = exp_theta - 1

        cdf = -1 / theta * np.log(1 + numerator / denominator)

        return cdf

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from Frank copula.

        Uses conditional sampling method.

        Args:
            n_samples: Number of samples to generate

        Returns:
            2D array of uniform samples (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before sampling")

        theta = self.params_["theta"]

        # Sample u uniformly
        u = np.random.uniform(0, 1, n_samples)

        # Sample t uniformly
        t = np.random.uniform(0, 1, n_samples)

        # Conditional sampling for v
        exp_theta = np.exp(-theta)
        exp_theta_u = np.exp(-theta * u)

        # Solve for v: t = C(v|u)
        # v = -1/θ * log(1 + t(e^{-θ} - 1)/(1 - t + t*e^{-θu}))
        numerator = t * (exp_theta - 1)
        denominator = 1 - t + t * exp_theta_u

        v = -1 / theta * np.log(1 + numerator / denominator)

        # Clip to [0,1]
        v = np.clip(v, 0, 1)

        U = np.column_stack([u, v])

        return U

    def tail_dependence(self) -> tuple:
        """
        Compute theoretical tail dependence coefficients for Frank copula.

        Frank copula has no tail dependence:
        λ_L = λ_U = 0

        Returns:
            Tuple of (lower_tail=0, upper_tail=0)
        """
        return 0.0, 0.0
