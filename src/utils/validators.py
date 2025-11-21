"""Data validation utilities."""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class DataValidator:
    """Validator for copula data."""

    @staticmethod
    def validate_returns(
        data: np.ndarray, name: str = "data"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate return data.

        Args:
            data: Array of returns
            name: Name of the data for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if empty
        if data.size == 0:
            return False, f"{name} is empty"

        # Check for all NaN
        if np.all(np.isnan(data)):
            return False, f"{name} contains only NaN values"

        # Check for infinite values
        if np.any(np.isinf(data)):
            return False, f"{name} contains infinite values"

        # Check for too many NaN values (more than 10%)
        nan_ratio = np.sum(np.isnan(data)) / data.size
        if nan_ratio > 0.1:
            return (
                False,
                f"{name} contains too many NaN values ({nan_ratio*100:.1f}%)",
            )

        # Check for sufficient variance
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 10:
            return False, f"{name} has insufficient valid data points (< 10)"

        if np.std(valid_data) == 0:
            return False, f"{name} has zero variance"

        return True, None

    @staticmethod
    def validate_pair_data(
        data1: np.ndarray, data2: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate paired data for copula fitting.

        Args:
            data1: First series
            data2: Second series

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check individual series
        valid1, error1 = DataValidator.validate_returns(data1, "data1")
        if not valid1:
            return False, error1

        valid2, error2 = DataValidator.validate_returns(data2, "data2")
        if not valid2:
            return False, error2

        # Check length match
        if len(data1) != len(data2):
            return False, f"Data length mismatch: {len(data1)} vs {len(data2)}"

        # Check for sufficient overlapping valid data
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
        n_valid = np.sum(valid_mask)

        if n_valid < 30:
            return (
                False,
                f"Insufficient overlapping valid data points: {n_valid} (need >= 30)",
            )

        return True, None

    @staticmethod
    def validate_uniform(
        u: np.ndarray, name: str = "u", tolerance: float = 1e-6
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate uniform [0,1] data.

        Args:
            u: Uniform data
            name: Name for error messages
            tolerance: Tolerance for bounds checking

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation
        valid, error = DataValidator.validate_returns(u, name)
        if not valid:
            return False, error

        # Check bounds
        valid_data = u[~np.isnan(u)]
        if np.any(valid_data < -tolerance) or np.any(valid_data > 1 + tolerance):
            return False, f"{name} contains values outside [0,1] range"

        return True, None

    @staticmethod
    def validate_correlation_matrix(
        corr: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate correlation matrix.

        Args:
            corr: Correlation matrix

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shape
        if corr.ndim != 2:
            return False, "Correlation matrix must be 2-dimensional"

        if corr.shape[0] != corr.shape[1]:
            return False, "Correlation matrix must be square"

        # Check symmetry
        if not np.allclose(corr, corr.T):
            return False, "Correlation matrix must be symmetric"

        # Check diagonal
        if not np.allclose(np.diag(corr), 1.0):
            return False, "Correlation matrix diagonal must be 1"

        # Check values in [-1, 1]
        if np.any(corr < -1) or np.any(corr > 1):
            return False, "Correlation values must be in [-1, 1]"

        # Check positive semi-definite
        try:
            eigenvalues = np.linalg.eigvalsh(corr)
            if np.any(eigenvalues < -1e-10):
                return False, "Correlation matrix must be positive semi-definite"
        except np.linalg.LinAlgError:
            return False, "Failed to compute eigenvalues of correlation matrix"

        return True, None

    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not symbol:
            return False, "Symbol cannot be empty"

        if not isinstance(symbol, str):
            return False, "Symbol must be a string"

        if len(symbol) < 4:
            return False, "Symbol too short"

        # Check for valid characters (alphanumeric)
        if not symbol.isalnum():
            return False, "Symbol must be alphanumeric"

        return True, None

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate pandas DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"

        return True, None


def validate_or_raise(
    validator_func: Any, *args: Any, **kwargs: Any
) -> None:
    """
    Run validation and raise ValueError if invalid.

    Args:
        validator_func: Validation function to call
        *args: Positional arguments for validator
        **kwargs: Keyword arguments for validator

    Raises:
        ValueError: If validation fails
    """
    valid, error = validator_func(*args, **kwargs)
    if not valid:
        raise ValueError(error)
