"""Gumbel copula implementation."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau

from .base import BivariateCopula
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GumbelCopula(BivariateCopula):
    """
    Gumbel copula.

    The Gumbel copula is an Archimedean copula that exhibits upper tail
    dependence. It is useful for modeling joint upside movements.

    Generator: φ(t) = (-log t)^θ, θ ≥ 1
    C(u,v) = exp{-[(-log u)^θ + (-log v)^θ]^{1/θ}}

    Parameters:
        theta: Dependence parameter (θ ≥ 1)
            θ = 1: Independence
            θ → ∞: Perfect dependence
    """

    def __init__(self, theta_bounds: tuple = (1.01, 20)):
        """
        Initialize Gumbel copula.

        Args:
            theta_bounds: Bounds for theta parameter (must be >= 1)
        """
        super().__init__()
        self.theta_bounds = (max(1.01, theta_bounds[0]), theta_bounds[1])
        self.n_params_ = 1  # theta parameter

    def fit(self, U: np.ndarray) -> "GumbelCopula":
        """
        Fit Gumbel copula to uniform data.

        Uses the relationship between Kendall's tau and theta:
        τ = 1 - 1/θ
        => θ = 1 / (1 - τ)

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
        if tau <= 0:
            # Gumbel requires positive dependence
            logger.warning(
                f"Kendall's tau={tau:.4f} <= 0, setting to small positive value"
            )
            tau = 0.01

        theta_init = 1 / (1 - tau)
        theta_init = np.clip(theta_init, self.theta_bounds[0], self.theta_bounds[1])

        # Refine estimate using MLE
        def neg_log_likelihood(theta):
            if not (self.theta_bounds[0] <= theta <= self.theta_bounds[1]):
                return 1e10

            try:
                # Compute log-likelihood
                u, v = U[:, 0], U[:, 1]

                # Clip to avoid numerical issues
                u = np.clip(u, 1e-10, 1 - 1e-10)
                v = np.clip(v, 1e-10, 1 - 1e-10)

                # Compute terms
                log_u = np.log(u)
                log_v = np.log(v)

                A = (-log_u) ** theta + (-log_v) ** theta
                A_1_theta = A ** (1 / theta)

                # log c(u,v) = log(C(u,v)) + (θ-1)[log(-log u) + log(-log v)]
                #              + (1/θ - 2)log(A) + log(A^{1/θ} + θ - 1) - log(uv)
                log_C = -A_1_theta

                log_dens = (
                    log_C
                    + (theta - 1) * (np.log(-log_u) + np.log(-log_v))
                    + (1 / theta - 2) * np.log(A)
                    + np.log(A_1_theta + theta - 1)
                    - log_u
                    - log_v
                )

                # Handle numerical issues
                log_dens = log_dens[np.isfinite(log_dens)]

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

        if result.success:
            theta_opt = result.x
        else:
            logger.warning(
                f"MLE optimization failed, using tau estimate: {result.message}"
            )
            theta_opt = theta_init

        self.params_ = {"theta": theta_opt}
        self.n_obs_ = n
        self.is_fitted_ = True

        logger.info(f"Fitted GumbelCopula: theta={theta_opt:.4f}")

        return self

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Gumbel copula density.

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
        log_u = np.log(u)
        log_v = np.log(v)

        A = (-log_u) ** theta + (-log_v) ** theta
        A_1_theta = A ** (1 / theta)

        # Compute density
        density = (
            self.cdf(U)
            / (u * v)
            * A_1_theta
            * (A_1_theta + theta - 1)
            * A ** (-2 + 1 / theta)
            * (-log_u) ** (theta - 1)
            * (-log_v) ** (theta - 1)
        )

        return density

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Gumbel copula CDF.

        C(u,v) = exp{-[(-log u)^θ + (-log v)^θ]^{1/θ}}

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
        A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        cdf = np.exp(-(A ** (1 / theta)))

        return cdf

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from Gumbel copula.

        Uses the Marshall-Olkin algorithm:
        1. Sample s from stable distribution
        2. Sample e1, e2 from exponential(1)
        3. Compute u = exp(-e1/s), v = exp(-e2/s)

        Args:
            n_samples: Number of samples to generate

        Returns:
            2D array of uniform samples (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before sampling")

        theta = self.params_["theta"]

        # Sample from stable distribution using Chambers' method
        alpha = 1 / theta

        # Sample uniform and exponential
        w = np.random.uniform(-np.pi / 2, np.pi / 2, n_samples)
        e = np.random.exponential(1, n_samples)

        # Stable distribution sample
        s = (
            np.sin(alpha * w)
            / (np.cos(w) ** (1 / alpha))
            * (np.cos(w - alpha * w) / e) ** ((1 - alpha) / alpha)
        )

        # Sample exponentials
        e1 = np.random.exponential(1, n_samples)
        e2 = np.random.exponential(1, n_samples)

        # Compute uniforms
        u = np.exp(-e1 / s)
        v = np.exp(-e2 / s)

        U = np.column_stack([u, v])

        return U

    def tail_dependence(self) -> tuple:
        """
        Compute theoretical tail dependence coefficients for Gumbel copula.

        Gumbel copula has:
        - Lower tail dependence: λ_L = 0
        - Upper tail dependence: λ_U = 2 - 2^{1/θ}

        Returns:
            Tuple of (lower_tail, upper_tail)
        """
        theta = self.params_["theta"]

        lambda_lower = 0.0
        lambda_upper = 2 - 2 ** (1 / theta)

        return lambda_lower, lambda_upper

    def conditional_cdf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        condition_on: int = 1,
    ) -> np.ndarray:
        """
        Compute conditional CDF for Gumbel copula.

        Args:
            u: First variable
            v: Second variable
            condition_on: Which variable to condition on (0 or 1)

        Returns:
            Conditional CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        theta = self.params_["theta"]

        # Clip
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        # Compute A
        A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        A_1_theta = A ** (1 / theta)

        # Compute C(u,v)
        C = np.exp(-A_1_theta)

        if condition_on == 1:
            # C(u|v) = C(u,v) * A^{1/θ - 1} * (-log v)^{θ-1} / v
            cond_cdf = C * A_1_theta / A * (-np.log(v)) ** (theta - 1) / v
        else:
            # C(v|u) = C(u,v) * A^{1/θ - 1} * (-log u)^{θ-1} / u
            cond_cdf = C * A_1_theta / A * (-np.log(u)) ** (theta - 1) / u

        return np.clip(cond_cdf, 0, 1)
