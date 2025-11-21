"""Gaussian (Normal) copula implementation."""

import numpy as np
from scipy import stats
from scipy.stats import norm, multivariate_normal

from .base import BivariateCopula
from ..utils.logger import get_logger
from ..utils.validators import validate_or_raise, DataValidator

logger = get_logger(__name__)


class GaussianCopula(BivariateCopula):
    """
    Gaussian (Normal) copula.

    The Gaussian copula is derived from the multivariate normal distribution.
    It is characterized by a correlation matrix and exhibits symmetric dependence.

    Parameters:
        rho: Correlation coefficient in [-1, 1]
    """

    def __init__(self):
        """Initialize Gaussian copula."""
        super().__init__()
        self.n_params_ = 1  # rho parameter

    def fit(self, U: np.ndarray) -> "GaussianCopula":
        """
        Fit Gaussian copula to uniform data.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Self
        """
        # Validate
        self._validate_bivariate(U)

        n = U.shape[0]

        # Transform to normal marginals using inverse CDF
        X = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

        # Estimate correlation
        rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]

        # Clip to valid range
        rho = np.clip(rho, -0.999, 0.999)

        self.params_ = {"rho": rho}
        self.n_obs_ = n
        self.is_fitted_ = True

        logger.info(f"Fitted GaussianCopula: rho={rho:.4f}")

        return self

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian copula density.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of density values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing PDF")

        self._validate_bivariate(U)

        rho = self.params_["rho"]

        # Transform to normal marginals
        X = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

        # Compute copula density
        # c(u,v) = φ_ρ(Φ^{-1}(u), Φ^{-1}(v)) / [φ(Φ^{-1}(u)) φ(Φ^{-1}(v))]
        # where φ_ρ is the bivariate normal density and φ is univariate normal density

        # Bivariate normal density
        Sigma = np.array([[1, rho], [rho, 1]])
        mvn = multivariate_normal(mean=[0, 0], cov=Sigma)
        bivariate_density = mvn.pdf(X)

        # Univariate normal densities
        univariate_density = norm.pdf(X[:, 0]) * norm.pdf(X[:, 1])

        # Copula density
        copula_density = bivariate_density / univariate_density

        return copula_density

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian copula CDF.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing CDF")

        self._validate_bivariate(U)

        rho = self.params_["rho"]

        # Transform to normal marginals
        X = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

        # Compute CDF using bivariate normal CDF
        Sigma = np.array([[1, rho], [rho, 1]])
        mvn = multivariate_normal(mean=[0, 0], cov=Sigma)

        # Compute CDF for each row
        cdf_values = np.array([mvn.cdf(x) for x in X])

        return cdf_values

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from Gaussian copula.

        Args:
            n_samples: Number of samples to generate

        Returns:
            2D array of uniform samples (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before sampling")

        rho = self.params_["rho"]

        # Sample from bivariate normal
        Sigma = np.array([[1, rho], [rho, 1]])
        X = multivariate_normal.rvs(mean=[0, 0], cov=Sigma, size=n_samples)

        # Ensure 2D
        if n_samples == 1:
            X = X.reshape(1, -1)

        # Transform to uniform using normal CDF
        U = norm.cdf(X)

        return U

    def tail_dependence(self) -> tuple:
        """
        Compute theoretical tail dependence coefficients.

        For Gaussian copula, both tail dependence coefficients are 0
        (asymptotically independent tails).

        Returns:
            Tuple of (lower_tail=0, upper_tail=0)
        """
        return 0.0, 0.0

    def conditional_cdf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        condition_on: int = 1,
    ) -> np.ndarray:
        """
        Compute conditional CDF for Gaussian copula.

        Args:
            u: First variable
            v: Second variable
            condition_on: Which variable to condition on (0 or 1)

        Returns:
            Conditional CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]

        # Transform to normal
        x = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))

        if condition_on == 1:
            # C(u|v) = Φ((Φ^{-1}(u) - ρ Φ^{-1}(v)) / √(1 - ρ²))
            z = (x - rho * y) / np.sqrt(1 - rho**2)
        else:
            # C(v|u) = Φ((Φ^{-1}(v) - ρ Φ^{-1}(u)) / √(1 - ρ²))
            z = (y - rho * x) / np.sqrt(1 - rho**2)

        cond_cdf = norm.cdf(z)

        return cond_cdf
