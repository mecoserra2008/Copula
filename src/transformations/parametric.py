"""Parametric transformation to uniform marginals."""

from typing import Literal, Optional
import numpy as np
from scipy import stats

from ..utils.logger import get_logger
from ..utils.validators import DataValidator, validate_or_raise

logger = get_logger(__name__)


class ParametricTransform:
    """
    Transform data to uniform [0,1] by fitting parametric distributions.

    Supports normal, t, and skewed-t distributions.
    """

    def __init__(
        self,
        distribution: Literal["norm", "t", "skewt"] = "norm",
    ):
        """
        Initialize parametric transform.

        Args:
            distribution: Distribution to fit ('norm', 't', or 'skewt')
        """
        self.distribution = distribution
        self.params_ = None
        self.dist_ = None
        self.is_fitted_ = False

    def fit(self, data: np.ndarray) -> "ParametricTransform":
        """
        Fit parametric distribution to data.

        Args:
            data: 1D array of observations

        Returns:
            Self
        """
        # Validate data
        validate_or_raise(DataValidator.validate_returns, data, "data")

        # Remove NaN
        clean_data = data[~np.isnan(data)]

        if len(clean_data) < 10:
            raise ValueError("Need at least 10 observations to fit distribution")

        # Fit distribution
        if self.distribution == "norm":
            # Normal distribution
            mu, sigma = stats.norm.fit(clean_data)
            self.params_ = {"loc": mu, "scale": sigma}
            self.dist_ = stats.norm(**self.params_)

        elif self.distribution == "t":
            # Student's t distribution
            df, loc, scale = stats.t.fit(clean_data)
            self.params_ = {"df": df, "loc": loc, "scale": scale}
            self.dist_ = stats.t(**self.params_)

        elif self.distribution == "skewt":
            # Skewed t distribution (if available)
            try:
                from scipy.stats import skewt
                # Fit skewed t
                params = skewt.fit(clean_data)
                self.params_ = {
                    "a": params[0],
                    "df": params[1],
                    "loc": params[2],
                    "scale": params[3],
                }
                self.dist_ = skewt(**self.params_)
            except (ImportError, AttributeError):
                # Fall back to regular t distribution
                logger.warning(
                    "skewt not available, falling back to t distribution"
                )
                df, loc, scale = stats.t.fit(clean_data)
                self.params_ = {"df": df, "loc": loc, "scale": scale}
                self.dist_ = stats.t(**self.params_)
                self.distribution = "t"

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        self.is_fitted_ = True

        logger.debug(
            f"Fitted {self.distribution} distribution with params: {self.params_}"
        )

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to uniform [0,1] using fitted distribution's CDF.

        Args:
            data: Array to transform

        Returns:
            Transformed data in [0,1]

        Raises:
            ValueError: If transform is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted before transform")

        # Handle NaN
        is_nan = np.isnan(data)

        if np.all(is_nan):
            return data.copy()

        # Apply CDF
        u = self.dist_.cdf(data)

        # Restore NaN
        u[is_nan] = np.nan

        # Clip to avoid exact 0 and 1
        u = np.clip(u, 1e-10, 1 - 1e-10)

        logger.debug(
            f"Transformed data: mean={np.nanmean(u):.4f}, "
            f"min={np.nanmin(u):.4f}, max={np.nanmax(u):.4f}"
        )

        return u

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.

        Args:
            data: Array to fit and transform

        Returns:
            Transformed data in [0,1]
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform [0,1] data back to original scale using PPF.

        Args:
            u: Uniform data in [0,1]

        Returns:
            Data in original scale

        Raises:
            ValueError: If transform is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted before inverse_transform")

        # Validate uniform data
        validate_or_raise(DataValidator.validate_uniform, u, "u")

        # Handle NaN
        is_nan = np.isnan(u)

        if np.all(is_nan):
            return u.copy()

        # Apply PPF (inverse CDF)
        x = self.dist_.ppf(u)

        # Restore NaN
        x[is_nan] = np.nan

        return x

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function.

        Args:
            x: Points to evaluate PDF

        Returns:
            PDF values
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted")

        return self.dist_.pdf(x)

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Compute log-likelihood of data under fitted distribution.

        Args:
            data: Data to evaluate

        Returns:
            Log-likelihood
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted")

        clean_data = data[~np.isnan(data)]
        return np.sum(self.dist_.logpdf(clean_data))


class ParametricCopulaTransform:
    """
    Multivariate parametric transform for copula modeling.

    Fits parametric distributions to each marginal and transforms
    to uniform [0,1] marginals.
    """

    def __init__(
        self,
        n_dims: int,
        distribution: Literal["norm", "t", "skewt"] = "norm",
    ):
        """
        Initialize multivariate parametric transform.

        Args:
            n_dims: Number of dimensions/series
            distribution: Distribution to fit to each marginal
        """
        self.n_dims = n_dims
        self.distribution = distribution
        self.transforms_ = [
            ParametricTransform(distribution=distribution) for _ in range(n_dims)
        ]
        self.is_fitted_ = False

    def fit(self, *data: np.ndarray) -> "ParametricCopulaTransform":
        """
        Fit parametric distributions to each dimension.

        Args:
            *data: Variable number of 1D arrays (one per dimension)

        Returns:
            Self

        Raises:
            ValueError: If number of arrays doesn't match n_dims
        """
        if len(data) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} arrays, got {len(data)}"
            )

        for i, d in enumerate(data):
            self.transforms_[i].fit(d)

        self.is_fitted_ = True

        logger.info(
            f"Fitted parametric copula transform ({self.distribution}) "
            f"for {self.n_dims} dimensions"
        )

        return self

    def transform(self, *data: np.ndarray) -> np.ndarray:
        """
        Transform multiple series to uniform [0,1].

        Args:
            *data: Variable number of 1D arrays

        Returns:
            2D array of shape (n_samples, n_dims) with uniform marginals

        Raises:
            ValueError: If transform is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted before transform")

        if len(data) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} arrays, got {len(data)}"
            )

        # Transform each dimension
        u_list = []
        for i, d in enumerate(data):
            u = self.transforms_[i].transform(d)
            u_list.append(u)

        # Stack into matrix
        U = np.column_stack(u_list)

        return U

    def fit_transform(self, *data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            *data: Variable number of 1D arrays

        Returns:
            2D array with uniform marginals
        """
        return self.fit(*data).transform(*data)

    def inverse_transform(self, U: np.ndarray) -> list:
        """
        Transform uniform data back to original scales.

        Args:
            U: 2D array of uniform data (n_samples, n_dims)

        Returns:
            List of arrays in original scales

        Raises:
            ValueError: If transform is not fitted or shape mismatch
        """
        if not self.is_fitted_:
            raise ValueError("Transform must be fitted before inverse_transform")

        if U.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} columns, got {U.shape[1]}"
            )

        # Inverse transform each dimension
        data_list = []
        for i in range(self.n_dims):
            x = self.transforms_[i].inverse_transform(U[:, i])
            data_list.append(x)

        return data_list
