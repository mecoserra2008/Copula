"""Clayton copula implementation."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau

from .base import BivariateCopula
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClaytonCopula(BivariateCopula):
    """
    Clayton copula.

    The Clayton copula is an Archimedean copula that exhibits lower tail
    dependence. It is useful for modeling joint downside risk.

    Generator: φ(t) = (1/θ)(t^{-θ} - 1), θ > 0
    C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

    Parameters:
        theta: Dependence parameter (θ > 0)
            θ → 0: Independence
            θ → ∞: Perfect dependence
    """

    def __init__(self, theta_bounds: tuple = (0.01, 20)):
        """
        Initialize Clayton copula.

        Args:
            theta_bounds: Bounds for theta parameter
        """
        super().__init__()
        self.theta_bounds = theta_bounds
        self.n_params_ = 1  # theta parameter

    def fit(self, U: np.ndarray) -> "ClaytonCopula":
        """
        Fit Clayton copula to uniform data.

        Uses the relationship between Kendall's tau and theta:
        τ = θ / (θ + 2)
        => θ = 2τ / (1 - τ)

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
            # Clayton requires positive dependence
            logger.warning(
                f"Kendall's tau={tau:.4f} <= 0, setting to small positive value"
            )
            tau = 0.01

        theta_init = 2 * tau / (1 - tau)
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

                # log c(u,v) = log(1+θ) - (1+θ)log(u) - (1+θ)log(v) - (1/θ + 2)log(u^{-θ} + v^{-θ} - 1)
                log_dens = (
                    np.log(1 + theta)
                    - (1 + theta) * np.log(u)
                    - (1 + theta) * np.log(v)
                    - (1 / theta + 2) * np.log(u ** (-theta) + v ** (-theta) - 1)
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

        logger.info(f"Fitted ClaytonCopula: theta={theta_opt:.4f}")

        return self

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Clayton copula density.

        c(u,v) = (1+θ)(uv)^{-(1+θ)}(u^{-θ} + v^{-θ} - 1)^{-(1/θ + 2)}

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

        # Compute density
        density = (
            (1 + theta)
            * (u * v) ** (-(1 + theta))
            * (u ** (-theta) + v ** (-theta) - 1) ** (-(1 / theta + 2))
        )

        return density

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Clayton copula CDF.

        C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

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
        cdf = (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)

        return cdf

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from Clayton copula.

        Uses conditional sampling method:
        1. Sample u ~ Uniform(0,1)
        2. Sample w ~ Uniform(0,1)
        3. Compute v = u(w^{-θ/(1+θ)} - 1 + u^θ)^{-1/θ}

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

        # Sample w uniformly
        w = np.random.uniform(0, 1, n_samples)

        # Conditional sampling
        v = u * (w ** (-theta / (1 + theta)) - 1 + u**theta) ** (-1 / theta)

        U = np.column_stack([u, v])

        return U

    def tail_dependence(self) -> tuple:
        """
        Compute theoretical tail dependence coefficients for Clayton copula.

        Clayton copula has:
        - Lower tail dependence: λ_L = 2^{-1/θ}
        - Upper tail dependence: λ_U = 0

        Returns:
            Tuple of (lower_tail, upper_tail)
        """
        theta = self.params_["theta"]

        lambda_lower = 2 ** (-1 / theta)
        lambda_upper = 0.0

        return lambda_lower, lambda_upper

    def conditional_cdf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        condition_on: int = 1,
    ) -> np.ndarray:
        """
        Compute conditional CDF for Clayton copula.

        C(u|v) = v^{-(1+θ)}(u^{-θ} + v^{-θ} - 1)^{-(1/θ + 1)}
        C(v|u) = u^{-(1+θ)}(u^{-θ} + v^{-θ} - 1)^{-(1/θ + 1)}

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

        if condition_on == 1:
            # C(u|v)
            cond_cdf = v ** (-(1 + theta)) * (
                u ** (-theta) + v ** (-theta) - 1
            ) ** (-(1 / theta + 1))
        else:
            # C(v|u)
            cond_cdf = u ** (-(1 + theta)) * (
                u ** (-theta) + v ** (-theta) - 1
            ) ** (-(1 / theta + 1))

        return np.clip(cond_cdf, 0, 1)
