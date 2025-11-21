"""Multi-pair copula manager for orchestrating copula modeling."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..data.bybit_fetcher import BybitDataFetcher
from ..data.preprocessor import DataPreprocessor
from ..data.returns import ReturnCalculator
from ..transformations.empirical import EmpiricalCopulaTransform
from ..transformations.parametric import ParametricCopulaTransform
from ..copulas.gaussian import GaussianCopula
from ..copulas.student_t import StudentTCopula
from ..copulas.clayton import ClaytonCopula
from ..copulas.gumbel import GumbelCopula
from ..copulas.frank import FrankCopula
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.validators import DataValidator

logger = get_logger(__name__)


class CopulaManager:
    """
    Manager for multi-pair copula modeling.

    Orchestrates:
    - Data fetching from Bybit
    - Data preprocessing
    - Return calculation
    - Transformation to uniform marginals
    - Copula fitting
    - Model selection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize copula manager.

        Args:
            api_key: Bybit API key (optional)
            api_secret: Bybit API secret (optional)
        """
        self.config = get_config()
        self.fetcher = BybitDataFetcher(api_key=api_key, api_secret=api_secret)
        self.preprocessor = DataPreprocessor()

        # Copula classes
        self.copula_classes = {
            "gaussian": GaussianCopula,
            "student_t": StudentTCopula,
            "clayton": ClaytonCopula,
            "gumbel": GumbelCopula,
            "frank": FrankCopula,
        }

        logger.info("Initialized CopulaManager")

    def fetch_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        interval: str = "15",
        days: int = 30,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and align data for a pair of symbols.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            interval: Kline interval in minutes
            days: Number of days to fetch
            use_cache: Whether to use cached data

        Returns:
            Tuple of (df1, df2) aligned DataFrames
        """
        logger.info(f"Fetching data for pair: {symbol1} - {symbol2}")

        # Fetch data
        if use_cache:
            df1 = self.fetcher.fetch_and_cache(symbol1, interval, days)
            df2 = self.fetcher.fetch_and_cache(symbol2, interval, days)
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            df1 = self.fetcher.fetch_klines(symbol1, interval, start_time, end_time)
            df2 = self.fetcher.fetch_klines(symbol2, interval, start_time, end_time)

        # Clean data
        df1 = self.preprocessor.clean_ohlcv(df1)
        df2 = self.preprocessor.clean_ohlcv(df2)

        # Align timestamps
        df1_aligned, df2_aligned = self.preprocessor.align_timestamps(df1, df2)

        logger.info(f"Fetched and aligned {len(df1_aligned)} data points")

        return df1_aligned, df2_aligned

    def prepare_returns(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        return_method: str = "log",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate returns from price data.

        Args:
            df1: First symbol DataFrame
            df2: Second symbol DataFrame
            return_method: Return calculation method ('log' or 'simple')

        Returns:
            Tuple of (returns1, returns2) arrays
        """
        # Calculate returns
        returns1 = ReturnCalculator.returns_from_dataframe(
            df1, price_col="close", method=return_method
        )
        returns2 = ReturnCalculator.returns_from_dataframe(
            df2, price_col="close", method=return_method
        )

        # Convert to numpy arrays
        r1 = returns1.values
        r2 = returns2.values

        # Validate
        valid, error = DataValidator.validate_pair_data(r1, r2)
        if not valid:
            raise ValueError(f"Invalid return data: {error}")

        logger.info(
            f"Calculated returns: n={len(r1)}, "
            f"r1: μ={np.mean(r1):.6f}, σ={np.std(r1):.6f}, "
            f"r2: μ={np.mean(r2):.6f}, σ={np.std(r2):.6f}"
        )

        return r1, r2

    def transform_to_uniform(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Transform returns to uniform [0,1] marginals.

        Args:
            r1: First series returns
            r2: Second series returns
            method: Transformation method ('empirical' or 'parametric')

        Returns:
            2D array of uniform data (n_samples, 2)
        """
        if method is None:
            method = self.config.transformation_method

        logger.info(f"Transforming to uniform marginals using {method} method")

        if method == "empirical":
            transform = EmpiricalCopulaTransform(n_dims=2)
        elif method == "parametric":
            dist = self.config.get("transformations.parametric.distribution", "norm")
            transform = ParametricCopulaTransform(n_dims=2, distribution=dist)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

        # Fit and transform
        U = transform.fit_transform(r1, r2)

        logger.info(
            f"Transformed to uniform: "
            f"u1: μ={np.mean(U[:,0]):.4f}, "
            f"u2: μ={np.mean(U[:,1]):.4f}"
        )

        return U

    def fit_copulas(
        self,
        U: np.ndarray,
        copula_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fit multiple copula models to uniform data.

        Args:
            U: 2D array of uniform data (n_samples, 2)
            copula_types: List of copula types to fit

        Returns:
            Dictionary with fitted copula models and metrics
        """
        if copula_types is None:
            copula_types = self.config.copula_types

        logger.info(f"Fitting {len(copula_types)} copula types: {copula_types}")

        results = {}

        for copula_type in copula_types:
            try:
                # Get copula class
                if copula_type not in self.copula_classes:
                    logger.warning(f"Unknown copula type: {copula_type}, skipping")
                    continue

                copula_class = self.copula_classes[copula_type]

                # Initialize and fit
                copula = copula_class()
                copula.fit(U)

                # Compute metrics
                aic = copula.aic(U)
                bic = copula.bic(U)
                log_lik = copula.log_likelihood(U)
                kendall_tau = copula.kendall_tau(U)
                lower_tail, upper_tail = copula.tail_dependence()

                results[copula_type] = {
                    "copula": copula,
                    "aic": aic,
                    "bic": bic,
                    "log_likelihood": log_lik,
                    "kendall_tau": kendall_tau,
                    "tail_dependence": {
                        "lower": lower_tail,
                        "upper": upper_tail,
                    },
                    "params": copula.get_params(),
                }

                logger.info(
                    f"  {copula_type}: AIC={aic:.2f}, BIC={bic:.2f}, "
                    f"τ={kendall_tau:.4f}, tail=({lower_tail:.4f}, {upper_tail:.4f})"
                )

            except Exception as e:
                logger.error(f"Error fitting {copula_type} copula: {e}")
                continue

        if not results:
            raise ValueError("No copulas were successfully fitted")

        return results

    def select_best_copula(
        self,
        results: Dict,
        criterion: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Select best copula based on information criterion.

        Args:
            results: Dictionary of copula fitting results
            criterion: Selection criterion ('aic' or 'bic')

        Returns:
            Tuple of (best_type, best_result)
        """
        if criterion is None:
            criterion = self.config.selection_criterion

        if criterion not in ["aic", "bic"]:
            raise ValueError(f"Unknown criterion: {criterion}")

        logger.info(f"Selecting best copula using {criterion.upper()}")

        # Find minimum
        best_type = min(results.keys(), key=lambda k: results[k][criterion])
        best_result = results[best_type]

        logger.info(
            f"Best copula: {best_type} "
            f"({criterion.upper()}={best_result[criterion]:.2f})"
        )

        return best_type, best_result

    def fit_pair(
        self,
        symbol1: str,
        symbol2: str,
        interval: str = "15",
        days: int = 30,
        copula_types: Optional[List[str]] = None,
        return_method: str = "log",
        transformation: Optional[str] = None,
    ) -> Dict:
        """
        Complete copula modeling pipeline for a pair.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            interval: Kline interval in minutes
            days: Number of days to fetch
            copula_types: List of copula types to fit
            return_method: Return calculation method
            transformation: Transformation method

        Returns:
            Dictionary with complete results
        """
        logger.info(f"=" * 80)
        logger.info(f"Fitting copula for pair: {symbol1} - {symbol2}")
        logger.info(f"=" * 80)

        # Fetch data
        df1, df2 = self.fetch_pair_data(symbol1, symbol2, interval, days)

        # Calculate returns
        r1, r2 = self.prepare_returns(df1, df2, return_method)

        # Transform to uniform
        U = self.transform_to_uniform(r1, r2, transformation)

        # Fit copulas
        copula_results = self.fit_copulas(U, copula_types)

        # Select best
        best_type, best_result = self.select_best_copula(copula_results)

        # Compile full results
        results = {
            "pair": (symbol1, symbol2),
            "n_observations": len(U),
            "interval": interval,
            "days": days,
            "returns_method": return_method,
            "transformation_method": transformation or self.config.transformation_method,
            "copulas": copula_results,
            "best_copula": {
                "type": best_type,
                **best_result,
            },
            "data": {
                "returns1": r1,
                "returns2": r2,
                "uniform": U,
            },
        }

        logger.info(f"Completed copula modeling for {symbol1} - {symbol2}")

        return results

    def fit_multiple_pairs(
        self,
        pairs: List[Tuple[str, str]],
        interval: str = "15",
        days: int = 30,
        **kwargs,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Fit copulas for multiple pairs.

        Args:
            pairs: List of (symbol1, symbol2) tuples
            interval: Kline interval
            days: Number of days to fetch
            **kwargs: Additional arguments for fit_pair

        Returns:
            Dictionary mapping pairs to results
        """
        logger.info(f"Fitting copulas for {len(pairs)} pairs")

        all_results = {}

        for i, (symbol1, symbol2) in enumerate(pairs, 1):
            logger.info(f"\nPair {i}/{len(pairs)}: {symbol1} - {symbol2}")

            try:
                results = self.fit_pair(
                    symbol1,
                    symbol2,
                    interval=interval,
                    days=days,
                    **kwargs,
                )
                all_results[(symbol1, symbol2)] = results

            except Exception as e:
                logger.error(f"Error fitting pair {symbol1}-{symbol2}: {e}")
                continue

        logger.info(f"\nCompleted {len(all_results)}/{len(pairs)} pairs successfully")

        return all_results

    def compare_pairs(
        self,
        all_results: Dict[Tuple[str, str], Dict],
    ) -> pd.DataFrame:
        """
        Create comparison DataFrame of all fitted pairs.

        Args:
            all_results: Dictionary of results from fit_multiple_pairs

        Returns:
            DataFrame with comparison metrics
        """
        rows = []

        for (symbol1, symbol2), result in all_results.items():
            best = result["best_copula"]

            row = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "best_copula": best["type"],
                "aic": best["aic"],
                "bic": best["bic"],
                "log_likelihood": best["log_likelihood"],
                "kendall_tau": best["kendall_tau"],
                "lower_tail_dep": best["tail_dependence"]["lower"],
                "upper_tail_dep": best["tail_dependence"]["upper"],
                "n_obs": result["n_observations"],
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        return df
