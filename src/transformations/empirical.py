"""Empirical transformation to uniform marginals."""

from typing import Optional
import numpy as np
from scipy import stats

from ..utils.logger import get_logger
from ..utils.validators import DataValidator, validate_or_raise

logger = get_logger(__name__)


class EmpiricalTransform:
    """
    Transform data to uniform [0,1] using empirical CDF.

    This is a non-parametric approach that uses the empirical cumulative
    distribution function to map data to uniform marginals.
    """

    def __init__(self):
        """Initialize empirical transform."""
        self.data_: Optional[np.ndarray] = None
        self.is_fitted_ = False

    def fit(self, data: np.ndarray) -> "EmpiricalTransform":
        """
        Fit the empirical CDF to data.

        Args:
            data: 1D array of observations

        Returns:
            Self
        """
        # Validate data
        validate_or_raise(DataValidator.validate_returns, data, "data")

        # Remove NaN values
        self.data_ = data[~np.isnan(data)].copy()
        self.data_.sort()

        self.is_fitted_ = True

        logger.debug(f"Fitted empirical transform on {len(self.data_)} observations")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to uniform [0,1] using empirical CDF.

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

        # Compute empirical CDF using percentile ranks
        # Add small adjustment to avoid exact 0 and 1
        n = len(self.data_)
        u = np.zeros_like(data, dtype=float)

        for i, x in enumerate(data):
            if not is_nan[i]:
                # Count observations <= x
                rank = np.sum(self.data_ <= x)
                # Use (rank - 0.5) / n to avoid 0 and 1
                u[i] = (rank - 0.5) / n
            else:
                u[i] = np.nan

        # Clip to (0, 1) exclusive
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
        Transform uniform [0,1] data back to original scale.

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

        n = len(self.data_)
        x = np.zeros_like(u, dtype=float)

        for i, ui in enumerate(u):
            if not is_nan[i]:
                # Find corresponding quantile
                idx = int(np.round(ui * n))
                idx = np.clip(idx, 0, n - 1)
                x[i] = self.data_[idx]
            else:
                x[i] = np.nan

        return x


class EmpiricalCopulaTransform:
    """
    Multivariate empirical transform for copula modeling.

    Transforms multiple series to uniform [0,1] marginals using
    empirical CDFs.
    """

    def __init__(self, n_dims: int):
        """
        Initialize multivariate empirical transform.

        Args:
            n_dims: Number of dimensions/series
        """
        self.n_dims = n_dims
        self.transforms_ = [EmpiricalTransform() for _ in range(n_dims)]
        self.is_fitted_ = False

    def fit(self, *data: np.ndarray) -> "EmpiricalCopulaTransform":
        """
        Fit empirical CDFs to each dimension.

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

        logger.info(f"Fitted empirical copula transform for {self.n_dims} dimensions")

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
