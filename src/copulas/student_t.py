"""Student's t copula implementation."""

import numpy as np
from scipy import stats
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize

from .base import BivariateCopula
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StudentTCopula(BivariateCopula):
    """
    Student's t copula.

    The t-copula is derived from the multivariate t distribution.
    It exhibits symmetric dependence and tail dependence (unlike Gaussian).
    Heavy tails make it suitable for modeling extreme events.

    Parameters:
        rho: Correlation coefficient in [-1, 1]
        df: Degrees of freedom (> 2)
    """

    def __init__(self, df_bounds: tuple = (2, 30)):
        """
        Initialize Student's t copula.

        Args:
            df_bounds: Bounds for degrees of freedom parameter
        """
        super().__init__()
        self.df_bounds = df_bounds
        self.n_params_ = 2  # rho and df parameters

    def fit(self, U: np.ndarray) -> "StudentTCopula":
        """
        Fit Student's t copula to uniform data.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Self
        """
        # Validate
        self._validate_bivariate(U)

        n = U.shape[0]

        # Initial estimate: use method of moments for df
        # and empirical correlation for rho

        # Transform to t-distributed marginals (assuming df=5 initially)
        df_init = 5
        X_init = t.ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df_init)
        rho_init = np.corrcoef(X_init[:, 0], X_init[:, 1])[0, 1]
        rho_init = np.clip(rho_init, -0.999, 0.999)

        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            rho, df = params

            # Bounds check
            if not (-0.999 < rho < 0.999):
                return 1e10
            if not (self.df_bounds[0] <= df <= self.df_bounds[1]):
                return 1e10

            try:
                # Transform to t-distributed marginals
                X = t.ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

                # Compute log-likelihood
                # log c(u,v) = log t_ρ,ν - sum(log t_ν)
                # where t_ρ,ν is bivariate t density and t_ν is univariate t density

                # Bivariate t density
                Sigma = np.array([[1, rho], [rho, 1]])

                # For each sample
                log_lik = 0
                for i in range(n):
                    x = X[i]

                    # Bivariate t log density
                    d = 2  # dimension
                    det_Sigma = 1 - rho**2

                    # x^T Σ^{-1} x
                    Sigma_inv = np.array([[1, -rho], [-rho, 1]]) / det_Sigma
                    quad_form = x @ Sigma_inv @ x

                    log_bivariate = (
                        np.log(stats.gamma((df + d) / 2))
                        - np.log(stats.gamma(df / 2))
                        - d / 2 * np.log(df * np.pi)
                        - 0.5 * np.log(det_Sigma)
                        - (df + d) / 2 * np.log(1 + quad_form / df)
                    )

                    # Univariate t log densities
                    log_univariate = t.logpdf(x[0], df=df) + t.logpdf(x[1], df=df)

                    # Copula log density
                    log_lik += log_bivariate - log_univariate

                return -log_lik

            except Exception as e:
                logger.debug(f"Error in likelihood computation: {e}")
                return 1e10

        # Optimize
        x0 = [rho_init, df_init]
        bounds = [(-0.999, 0.999), self.df_bounds]

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if result.success:
            rho_opt, df_opt = result.x
        else:
            logger.warning(
                f"Optimization did not converge, using initial values: {result.message}"
            )
            rho_opt, df_opt = rho_init, df_init

        self.params_ = {"rho": rho_opt, "df": df_opt}
        self.n_obs_ = n
        self.is_fitted_ = True

        logger.info(f"Fitted StudentTCopula: rho={rho_opt:.4f}, df={df_opt:.2f}")

        return self

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Student's t copula density.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of density values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing PDF")

        self._validate_bivariate(U)

        rho = self.params_["rho"]
        df = self.params_["df"]

        # Transform to t-distributed marginals
        X = t.ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

        n = X.shape[0]
        densities = np.zeros(n)

        Sigma = np.array([[1, rho], [rho, 1]])
        det_Sigma = 1 - rho**2
        Sigma_inv = np.array([[1, -rho], [-rho, 1]]) / det_Sigma

        for i in range(n):
            x = X[i]

            # Bivariate t density
            d = 2
            quad_form = x @ Sigma_inv @ x

            log_bivariate = (
                np.log(stats.gamma((df + d) / 2))
                - np.log(stats.gamma(df / 2))
                - d / 2 * np.log(df * np.pi)
                - 0.5 * np.log(det_Sigma)
                - (df + d) / 2 * np.log(1 + quad_form / df)
            )

            bivariate_density = np.exp(log_bivariate)

            # Univariate t densities
            univariate_density = t.pdf(x[0], df=df) * t.pdf(x[1], df=df)

            # Copula density
            densities[i] = bivariate_density / univariate_density

        return densities

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Student's t copula CDF.

        Note: This is computationally expensive and uses numerical integration.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing CDF")

        self._validate_bivariate(U)

        # For t-copula CDF, use numerical integration or approximation
        # This is computationally expensive, so we use a simple Monte Carlo approach

        logger.warning(
            "t-copula CDF computation is approximate (using Monte Carlo)"
        )

        rho = self.params_["rho"]
        df = self.params_["df"]

        # Transform to t-distributed marginals
        X = t.ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

        # Use empirical approximation with many samples
        n_mc = 10000
        Sigma = np.array([[1, rho], [rho, 1]])

        cdf_values = np.zeros(U.shape[0])

        for i, x in enumerate(X):
            # Generate samples from bivariate t
            samples = multivariate_t.rvs(
                loc=[0, 0],
                shape=Sigma,
                df=df,
                size=n_mc,
            )

            # Count samples in lower orthant
            count = np.sum((samples[:, 0] <= x[0]) & (samples[:, 1] <= x[1]))
            cdf_values[i] = count / n_mc

        return cdf_values

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from Student's t copula.

        Args:
            n_samples: Number of samples to generate

        Returns:
            2D array of uniform samples (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before sampling")

        rho = self.params_["rho"]
        df = self.params_["df"]

        # Sample from bivariate t
        Sigma = np.array([[1, rho], [rho, 1]])
        X = multivariate_t.rvs(
            loc=[0, 0],
            shape=Sigma,
            df=df,
            size=n_samples,
        )

        # Ensure 2D
        if n_samples == 1:
            X = X.reshape(1, -1)

        # Transform to uniform using t CDF
        U = t.cdf(X, df=df)

        return U

    def tail_dependence(self) -> tuple:
        """
        Compute theoretical tail dependence coefficients for t-copula.

        For t-copula, both tail dependence coefficients are equal:
        λ_L = λ_U = 2 * T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

        where T_ν is the CDF of Student's t with ν degrees of freedom.

        Returns:
            Tuple of (lower_tail, upper_tail)
        """
        rho = self.params_["rho"]
        df = self.params_["df"]

        # Compute tail dependence
        arg = -np.sqrt((df + 1) * (1 - rho) / (1 + rho))
        lambda_tail = 2 * t.cdf(arg, df=df + 1)

        return lambda_tail, lambda_tail
