"""Base strategy class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from datetime import datetime

from ..backtest.portfolio import Portfolio
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All trading strategies should inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.portfolio: Optional[Portfolio] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.params: Dict = {}

    def initialize(
        self,
        portfolio: Portfolio,
        data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Initialize strategy with portfolio and data.

        Called once before backtest starts.

        Args:
            portfolio: Portfolio object
            data: Dictionary of historical data {symbol: DataFrame}
        """
        self.portfolio = portfolio
        self.data = data

        logger.info(f"Initialized strategy: {self.name}")
        self.on_initialize()

    def on_initialize(self) -> None:
        """
        Custom initialization logic.

        Override this method to add custom initialization.
        """
        pass

    @abstractmethod
    def on_data(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> None:
        """
        Called on each bar of data.

        This is the main strategy logic that must be implemented.

        Args:
            timestamp: Current timestamp
            current_data: Dictionary of {symbol: current_bar_data}
            portfolio: Portfolio object for executing trades
        """
        pass

    def finalize(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """
        Finalize strategy.

        Called once after backtest completes.

        Args:
            portfolio: Portfolio object
            timestamp: Final timestamp
        """
        logger.info(f"Finalizing strategy: {self.name}")
        self.on_finalize(portfolio, timestamp)

    def on_finalize(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """
        Custom finalization logic.

        Override this method to add custom finalization (e.g., close positions).

        Args:
            portfolio: Portfolio object
            timestamp: Final timestamp
        """
        # Default: close all positions
        portfolio.close_all_positions(timestamp)

    def set_params(self, **params) -> None:
        """
        Set strategy parameters.

        Args:
            **params: Parameter name-value pairs
        """
        self.params.update(params)
        logger.info(f"Updated parameters: {params}")

    def get_param(self, name: str, default=None):
        """
        Get strategy parameter.

        Args:
            name: Parameter name
            default: Default value if parameter not set

        Returns:
            Parameter value
        """
        return self.params.get(name, default)


class CopulaStrategy(BaseStrategy):
    """
    Base class for copula-based trading strategies.

    Extends BaseStrategy with copula-specific functionality.
    """

    def __init__(self, name: str = "CopulaStrategy"):
        """Initialize copula strategy."""
        super().__init__(name)
        self.copula = None
        self.transform = None

    def fit_copula(
        self,
        data1: pd.Series,
        data2: pd.Series,
        copula_type: str = "gaussian",
    ):
        """
        Fit a copula to pair data.

        Args:
            data1: First series
            data2: Second series
            copula_type: Type of copula to fit

        Returns:
            Fitted copula object
        """
        from ..copulas.gaussian import GaussianCopula
        from ..copulas.student_t import StudentTCopula
        from ..copulas.clayton import ClaytonCopula
        from ..copulas.gumbel import GumbelCopula
        from ..copulas.frank import FrankCopula
        from ..transformations.empirical import EmpiricalCopulaTransform
        import numpy as np

        # Map copula types
        copula_map = {
            "gaussian": GaussianCopula,
            "student_t": StudentTCopula,
            "clayton": ClaytonCopula,
            "gumbel": GumbelCopula,
            "frank": FrankCopula,
        }

        if copula_type not in copula_map:
            raise ValueError(f"Unknown copula type: {copula_type}")

        # Transform to uniform marginals
        transform = EmpiricalCopulaTransform(n_dims=2)
        U = transform.fit_transform(data1.values, data2.values)

        # Fit copula
        copula_class = copula_map[copula_type]
        copula = copula_class()
        copula.fit(U)

        return copula, transform

    def compute_conditional_probability(
        self,
        u: float,
        v: float,
        condition_on: int = 1,
    ) -> float:
        """
        Compute conditional CDF using fitted copula.

        Args:
            u: First variable value (in [0,1])
            v: Second variable value (in [0,1])
            condition_on: Which variable to condition on (0 or 1)

        Returns:
            Conditional probability
        """
        if self.copula is None:
            raise ValueError("Copula not fitted. Call fit_copula() first.")

        import numpy as np

        u_arr = np.array([u])
        v_arr = np.array([v])

        cond_prob = self.copula.conditional_cdf(u_arr, v_arr, condition_on)

        return cond_prob[0]
