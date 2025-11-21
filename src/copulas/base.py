"""Base copula class defining the interface for all copula implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np

from ..utils.logger import get_logger
from ..utils.validators import DataValidator, validate_or_raise

logger = get_logger(__name__)


class BaseCopula(ABC):
    """
    Abstract base class for all copula models.

    All copula implementations must inherit from this class and implement
    the abstract methods.
    """

    def __init__(self):
        """Initialize base copula."""
        self.is_fitted_ = False
        self.params_ = {}
        self.n_obs_ = 0
        self.n_params_ = 0

    @abstractmethod
    def fit(self, U: np.ndarray) -> "BaseCopula":
        """
        Fit copula to uniform data.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, n_dims)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute copula probability density function.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, n_dims)

        Returns:
            Array of density values
        """
        pass

    @abstractmethod
    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute copula cumulative distribution function.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, n_dims)

        Returns:
            Array of CDF values
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from copula.

        Args:
            n_samples: Number of samples to generate

        Returns:
            2D array of uniform samples (n_samples, n_dims)
        """
        pass

    def log_likelihood(self, U: np.ndarray) -> float:
        """
        Compute log-likelihood of data under the copula.

        Args:
            U: 2D array of uniform [0,1] data

        Returns:
            Log-likelihood value
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing log-likelihood")

        densities = self.pdf(U)

        # Handle numerical issues
        densities = np.clip(densities, 1e-10, None)

        log_lik = np.sum(np.log(densities))

        return log_lik

    def aic(self, U: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion.

        Args:
            U: 2D array of uniform [0,1] data

        Returns:
            AIC value (lower is better)
        """
        log_lik = self.log_likelihood(U)
        aic = 2 * self.n_params_ - 2 * log_lik

        return aic

    def bic(self, U: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.

        Args:
            U: 2D array of uniform [0,1] data

        Returns:
            BIC value (lower is better)
        """
        n = U.shape[0]
        log_lik = self.log_likelihood(U)
        bic = np.log(n) * self.n_params_ - 2 * log_lik

        return bic

    def kendall_tau(self, U: np.ndarray) -> float:
        """
        Compute Kendall's tau from data.

        Args:
            U: 2D array of uniform data (n_samples, 2)

        Returns:
            Kendall's tau correlation coefficient

        Raises:
            ValueError: If U is not 2-dimensional
        """
        if U.shape[1] != 2:
            raise ValueError("Kendall's tau only defined for bivariate data")

        from scipy.stats import kendalltau

        tau, _ = kendalltau(U[:, 0], U[:, 1])

        return tau

    def spearman_rho(self, U: np.ndarray) -> float:
        """
        Compute Spearman's rho from data.

        Args:
            U: 2D array of uniform data (n_samples, 2)

        Returns:
            Spearman's rho correlation coefficient

        Raises:
            ValueError: If U is not 2-dimensional
        """
        if U.shape[1] != 2:
            raise ValueError("Spearman's rho only defined for bivariate data")

        from scipy.stats import spearmanr

        rho, _ = spearmanr(U[:, 0], U[:, 1])

        return rho

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Compute theoretical tail dependence coefficients.

        Returns:
            Tuple of (lower_tail, upper_tail) dependence coefficients

        Note:
            Default implementation returns (0, 0). Override in subclasses
            with non-zero tail dependence.
        """
        return 0.0, 0.0

    def get_params(self) -> Dict:
        """
        Get fitted parameters.

        Returns:
            Dictionary of parameter names and values
        """
        return self.params_.copy()

    def summary(self) -> Dict:
        """
        Get summary statistics of fitted copula.

        Returns:
            Dictionary with copula information
        """
        if not self.is_fitted_:
            return {"fitted": False}

        lower_tail, upper_tail = self.tail_dependence()

        return {
            "fitted": True,
            "type": self.__class__.__name__,
            "n_obs": self.n_obs_,
            "n_params": self.n_params_,
            "params": self.params_,
            "tail_dependence": {
                "lower": lower_tail,
                "upper": upper_tail,
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        if self.is_fitted_:
            params_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in self.params_.items()
            )
            return f"{self.__class__.__name__}({params_str})"
        else:
            return f"{self.__class__.__name__}(not fitted)"


class BivariateCopula(BaseCopula):
    """
    Base class for bivariate copulas.

    Provides additional methods specific to 2-dimensional copulas.
    """

    def __init__(self):
        """Initialize bivariate copula."""
        super().__init__()

    def _validate_bivariate(self, U: np.ndarray) -> None:
        """
        Validate that data is bivariate.

        Args:
            U: Data to validate

        Raises:
            ValueError: If data is not 2-dimensional
        """
        if U.ndim != 2:
            raise ValueError("Data must be 2-dimensional")

        if U.shape[1] != 2:
            raise ValueError(
                f"Bivariate copula requires 2 columns, got {U.shape[1]}"
            )

        # Validate uniform data
        validate_or_raise(DataValidator.validate_uniform, U[:, 0], "U[:,0]")
        validate_or_raise(DataValidator.validate_uniform, U[:, 1], "U[:,1]")

    def conditional_cdf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        condition_on: int = 1,
    ) -> np.ndarray:
        """
        Compute conditional CDF: C(u|v) or C(v|u).

        Args:
            u: First variable
            v: Second variable
            condition_on: Which variable to condition on (0 or 1)

        Returns:
            Conditional CDF values

        Note:
            Default implementation uses numerical differentiation.
            Override for analytical solutions.
        """
        h = 1e-6

        if condition_on == 1:
            # C(u|v) = ∂C(u,v)/∂v
            U_plus = np.column_stack([u, v + h])
            U_minus = np.column_stack([u, v - h])
            cond_cdf = (self.cdf(U_plus) - self.cdf(U_minus)) / (2 * h)
        else:
            # C(v|u) = ∂C(u,v)/∂u
            U_plus = np.column_stack([u + h, v])
            U_minus = np.column_stack([u - h, v])
            cond_cdf = (self.cdf(U_plus) - self.cdf(U_minus)) / (2 * h)

        return np.clip(cond_cdf, 0, 1)
