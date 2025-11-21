"""Data preprocessing utilities."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..utils.validators import DataValidator

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and clean financial data."""

    @staticmethod
    def clean_ohlcv(
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        fill_missing: bool = True,
        remove_outliers: bool = False,
        outlier_std: float = 5.0,
    ) -> pd.DataFrame:
        """
        Clean OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            remove_duplicates: Remove duplicate timestamps
            fill_missing: Fill missing values
            remove_outliers: Remove outliers
            outlier_std: Number of standard deviations for outlier detection

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        original_len = len(df)

        # Remove duplicates
        if remove_duplicates and "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            n_duplicates = original_len - len(df)
            if n_duplicates > 0:
                logger.info(f"Removed {n_duplicates} duplicate timestamps")

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Fill missing values
        if fill_missing:
            # Forward fill then backward fill
            df = df.fillna(method="ffill").fillna(method="bfill")

        # Remove outliers
        if remove_outliers:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    lower = mean - outlier_std * std
                    upper = mean + outlier_std * std

                    outliers = (df[col] < lower) | (df[col] > upper)
                    n_outliers = outliers.sum()

                    if n_outliers > 0:
                        logger.warning(
                            f"Found {n_outliers} outliers in {col}, replacing with NaN"
                        )
                        df.loc[outliers, col] = np.nan

        # Final cleanup
        df = df.dropna(subset=["close"])

        logger.info(
            f"Cleaning complete: {original_len} → {len(df)} rows "
            f"({original_len - len(df)} removed)"
        )

        return df

    @staticmethod
    def align_timestamps(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two DataFrames by timestamp (inner join).

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            timestamp_col: Name of timestamp column

        Returns:
            Tuple of aligned DataFrames
        """
        # Merge on timestamp
        merged = pd.merge(
            df1,
            df2,
            on=timestamp_col,
            how="inner",
            suffixes=("_1", "_2"),
        )

        if len(merged) == 0:
            logger.warning("No overlapping timestamps found")
            return pd.DataFrame(), pd.DataFrame()

        # Split back
        cols1 = [c for c in merged.columns if c.endswith("_1") or c == timestamp_col]
        cols2 = [c for c in merged.columns if c.endswith("_2") or c == timestamp_col]

        df1_aligned = merged[cols1].rename(
            columns={c: c.replace("_1", "") for c in cols1}
        )
        df2_aligned = merged[cols2].rename(
            columns={c: c.replace("_2", "") for c in cols2}
        )

        logger.info(
            f"Aligned timestamps: {len(df1)} & {len(df2)} → {len(merged)} rows"
        )

        return df1_aligned.reset_index(drop=True), df2_aligned.reset_index(drop=True)

    @staticmethod
    def resample_ohlcv(
        df: pd.DataFrame,
        freq: str = "1H",
        timestamp_col: str = "datetime",
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to different frequency.

        Args:
            df: DataFrame with OHLCV data
            freq: Resampling frequency (e.g., '1H', '4H', '1D')
            timestamp_col: Name of datetime column

        Returns:
            Resampled DataFrame
        """
        df = df.copy()
        df = df.set_index(timestamp_col)

        # Resample
        resampled = df.resample(freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Remove rows with NaN (from weekends/gaps)
        resampled = resampled.dropna()

        resampled = resampled.reset_index()

        logger.info(f"Resampled from {len(df)} to {len(resampled)} rows at {freq}")

        return resampled

    @staticmethod
    def remove_extreme_returns(
        returns: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Remove extreme returns (e.g., > 50% change).

        Args:
            returns: Array of returns
            threshold: Maximum absolute return (0.5 = 50%)

        Returns:
            Returns with extremes replaced by NaN
        """
        returns = returns.copy()
        mask = np.abs(returns) > threshold
        n_extreme = np.sum(mask)

        if n_extreme > 0:
            logger.warning(
                f"Removing {n_extreme} extreme returns (>{threshold*100}%)"
            )
            returns[mask] = np.nan

        return returns

    @staticmethod
    def winsorize(
        data: np.ndarray,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
    ) -> np.ndarray:
        """
        Winsorize data by capping extreme values.

        Args:
            data: Array of data
            lower_percentile: Lower percentile for capping
            upper_percentile: Upper percentile for capping

        Returns:
            Winsorized data
        """
        data = data.copy()
        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            return data

        lower = np.percentile(data[valid_mask], lower_percentile)
        upper = np.percentile(data[valid_mask], upper_percentile)

        data[data < lower] = lower
        data[data > upper] = upper

        logger.debug(f"Winsorized data: [{lower:.6f}, {upper:.6f}]")

        return data
