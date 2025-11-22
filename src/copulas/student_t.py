"""Student's t copula implementation using only numpy."""

import numpy as np

from .base import BivariateCopula
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Pure numpy implementations of statistical functions
def _log_gamma(x):
    """
    Compute log of gamma function using Lanczos approximation.

    This is numerically stable for x > 0.
    """
    # Lanczos coefficients
    g = 7
    coef = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])

    x = np.atleast_1d(x)
    result = np.zeros_like(x, dtype=float)

    # Use Stirling's approximation for large x
    large_mask = x > 100
    if np.any(large_mask):
        x_large = x[large_mask]
        result[large_mask] = (x_large - 0.5) * np.log(x_large) - x_large + 0.5 * np.log(2 * np.pi)

    # Use Lanczos for smaller x
    small_mask = ~large_mask
    if np.any(small_mask):
        x_small = x[small_mask]
        z = x_small - 1

        # Compute sum
        x_sum = coef[0]
        for i in range(1, g + 2):
            x_sum += coef[i] / (z + i)

        # Final result
        t = z + g + 0.5
        result[small_mask] = (z + 0.5) * np.log(t) - t + np.log(np.sqrt(2 * np.pi) * x_sum)

    if result.size == 1:
        return float(result[0])
    return result


def _gamma(x):
    """Compute gamma function."""
    return np.exp(_log_gamma(x))


def _beta(a, b):
    """Compute beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
    return np.exp(_log_gamma(a) + _log_gamma(b) - _log_gamma(a + b))


def _t_pdf(x, df):
    """
    Student's t distribution PDF.

    Formula: f(x;ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
    """
    log_pdf = (
        _log_gamma((df + 1) / 2)
        - _log_gamma(df / 2)
        - 0.5 * np.log(df * np.pi)
        - (df + 1) / 2 * np.log(1 + x**2 / df)
    )
    return np.exp(log_pdf)


def _t_cdf(x, df):
    """
    Student's t distribution CDF using numerical integration.

    Uses adaptive Simpson's rule for integration.
    """
    x = np.atleast_1d(x)
    result = np.zeros_like(x, dtype=float)

    for i, xi in enumerate(x):
        if xi < -10:
            result[i] = 0.0
        elif xi > 10:
            result[i] = 1.0
        else:
            # Use relationship with beta distribution for better numerical stability
            # For x > 0: CDF(x) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2)
            # where I is the regularized incomplete beta function

            if xi == 0:
                result[i] = 0.5
            elif xi > 0:
                t = df / (df + xi**2)
                result[i] = 1 - 0.5 * _incomplete_beta_reg(t, df/2, 0.5)
            else:
                t = df / (df + xi**2)
                result[i] = 0.5 * _incomplete_beta_reg(t, df/2, 0.5)

    if result.size == 1:
        return float(result[0])
    return result


def _incomplete_beta_reg(x, a, b):
    """
    Regularized incomplete beta function using continued fraction.

    I_x(a,b) = B_x(a,b) / B(a,b)
    """
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use continued fraction expansion
    EPSILON = 1e-10
    MAX_ITER = 200

    # Logarithm of beta function
    log_beta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)

    # Front factor
    front = np.exp(a * np.log(x) + b * np.log(1 - x) - log_beta) / a

    # Continued fraction using modified Lentz's algorithm
    f = 1.0
    c = 1.0
    d = 0.0

    for m in range(1, MAX_ITER):
        # Even step
        numerator = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))

        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))

        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < EPSILON:
            break

    return front * f


def _t_ppf(p, df):
    """
    Student's t inverse CDF (quantile function) using Newton-Raphson method.
    """
    p = np.atleast_1d(p)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    result = np.zeros_like(p, dtype=float)

    for i, pi in enumerate(p):
        # Initial guess using normal approximation
        if pi < 0.5:
            sign = -1
            pi_use = 2 * pi
        else:
            sign = 1
            pi_use = 2 * (1 - pi)

        # Initial guess from normal distribution
        from .gaussian import _norm_ppf
        x = _norm_ppf(0.5 + sign * (1 - pi_use) / 2)

        # Newton-Raphson iterations
        for _ in range(20):
            fx = _t_cdf(x, df) - (pi if sign > 0 else pi)
            if abs(fx) < 1e-10:
                break

            fpx = _t_pdf(x, df)
            if fpx < 1e-30:
                break

            x_new = x - fx / fpx
            if abs(x_new - x) < 1e-10:
                break
            x = x_new

        result[i] = x

    if result.size == 1:
        return float(result[0])
    return result


def _sample_multivariate_t(mean, shape, df, size):
    """
    Sample from multivariate t distribution.

    Uses the fact that if X ~ N(0, Σ) and W ~ χ²(ν)/ν independent,
    then X/√W ~ t_ν(0, Σ).

    Args:
        mean: Mean vector
        shape: Shape matrix (correlation matrix)
        df: Degrees of freedom
        size: Number of samples

    Returns:
        Samples from multivariate t
    """
    d = len(mean)

    # Sample from multivariate normal
    L = np.linalg.cholesky(shape)
    Z = np.random.randn(size, d)
    X = Z @ L.T

    # Sample from chi-squared and scale
    chi2_samples = np.random.chisquare(df, size) / df

    # Construct t samples
    T = X / np.sqrt(chi2_samples)[:, np.newaxis] + mean

    return T


def _simple_minimize(func, x0, bounds, max_iter=100):
    """
    Simple bounded optimization using coordinate descent.

    Args:
        func: Function to minimize
        x0: Initial point
        bounds: List of (min, max) tuples for each parameter
        max_iter: Maximum iterations

    Returns:
        Optimized parameters
    """
    x = np.array(x0, dtype=float)
    n_params = len(x)

    best_x = x.copy()
    best_f = func(x)

    step_size = 0.1

    for iteration in range(max_iter):
        improved = False

        # Try to improve each parameter
        for i in range(n_params):
            # Try increasing
            x_try = x.copy()
            x_try[i] = min(x[i] + step_size, bounds[i][1])
            f_try = func(x_try)

            if f_try < best_f:
                best_x = x_try.copy()
                best_f = f_try
                improved = True
                continue

            # Try decreasing
            x_try = x.copy()
            x_try[i] = max(x[i] - step_size, bounds[i][0])
            f_try = func(x_try)

            if f_try < best_f:
                best_x = x_try.copy()
                best_f = f_try
                improved = True

        x = best_x.copy()

        # Adaptive step size
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
        else:
            step_size = min(step_size * 1.1, 0.5)

    return best_x, best_f


class StudentTCopula(BivariateCopula):
    """
    Student's t copula.

    The t-copula is derived from the multivariate t distribution.
    It exhibits symmetric dependence and tail dependence (unlike Gaussian).
    Heavy tails make it suitable for modeling extreme events.

    Parameters:
        rho: Correlation coefficient in [-1, 1]
        df: Degrees of freedom (> 2)

    This implementation uses only numpy, no scipy dependencies.
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
        df_init = 5.0
        X_init = _t_ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df_init)
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
                X = _t_ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

                # Compute log-likelihood
                # log c(u,v) = log t_ρ,ν - sum(log t_ν)

                # Bivariate t density
                det_Sigma = 1 - rho**2

                # For each sample
                log_lik = 0
                for i in range(n):
                    x = X[i]

                    # x^T Σ^{-1} x
                    Sigma_inv = np.array([[1, -rho], [-rho, 1]]) / det_Sigma
                    quad_form = x @ Sigma_inv @ x

                    # Bivariate t log density
                    d = 2  # dimension
                    log_bivariate = (
                        _log_gamma((df + d) / 2)
                        - _log_gamma(df / 2)
                        - d / 2 * np.log(df * np.pi)
                        - 0.5 * np.log(det_Sigma)
                        - (df + d) / 2 * np.log(1 + quad_form / df)
                    )

                    # Univariate t log densities
                    log_univariate = (
                        np.log(_t_pdf(x[0], df) + 1e-30) +
                        np.log(_t_pdf(x[1], df) + 1e-30)
                    )

                    # Copula log density
                    log_lik += log_bivariate - log_univariate

                return -log_lik

            except Exception as e:
                logger.debug(f"Error in likelihood computation: {e}")
                return 1e10

        # Optimize using simple coordinate descent
        x0 = [rho_init, df_init]
        bounds = [(-0.999, 0.999), self.df_bounds]

        result_x, result_f = _simple_minimize(
            neg_log_likelihood,
            x0,
            bounds,
            max_iter=50
        )

        rho_opt, df_opt = result_x

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
        X = _t_ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

        n = X.shape[0]
        densities = np.zeros(n)

        det_Sigma = 1 - rho**2
        Sigma_inv = np.array([[1, -rho], [-rho, 1]]) / det_Sigma

        for i in range(n):
            x = X[i]

            # Bivariate t density
            d = 2
            quad_form = x @ Sigma_inv @ x

            log_bivariate = (
                _log_gamma((df + d) / 2)
                - _log_gamma(df / 2)
                - d / 2 * np.log(df * np.pi)
                - 0.5 * np.log(det_Sigma)
                - (df + d) / 2 * np.log(1 + quad_form / df)
            )

            bivariate_density = np.exp(log_bivariate)

            # Univariate t densities
            univariate_density = _t_pdf(x[0], df=df) * _t_pdf(x[1], df=df)

            # Copula density
            densities[i] = bivariate_density / (univariate_density + 1e-30)

        return densities

    def cdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Student's t copula CDF.

        Note: This uses Monte Carlo approximation.

        Args:
            U: 2D array of uniform [0,1] data (n_samples, 2)

        Returns:
            Array of CDF values
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before computing CDF")

        self._validate_bivariate(U)

        logger.warning(
            "t-copula CDF computation is approximate (using Monte Carlo)"
        )

        rho = self.params_["rho"]
        df = self.params_["df"]

        # Transform to t-distributed marginals
        X = _t_ppf(np.clip(U, 1e-10, 1 - 1e-10), df=df)

        # Use empirical approximation with many samples
        n_mc = 10000
        shape = np.array([[1.0, rho], [rho, 1.0]])

        cdf_values = np.zeros(U.shape[0])

        for i, x in enumerate(X):
            # Generate samples from bivariate t
            samples = _sample_multivariate_t(
                mean=np.array([0.0, 0.0]),
                shape=shape,
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
        shape = np.array([[1.0, rho], [rho, 1.0]])
        X = _sample_multivariate_t(
            mean=np.array([0.0, 0.0]),
            shape=shape,
            df=df,
            size=n_samples,
        )

        # Transform to uniform using t CDF
        U = _t_cdf(X, df=df)

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
        lambda_tail = 2 * _t_cdf(arg, df=df + 1)

        return lambda_tail, lambda_tail

    def information_geometry_metrics(self, U: np.ndarray) -> dict:
        """
        Compute information geometry metrics for the t-copula.

        Returns geometric and information-theoretic properties.

        Args:
            U: Data for computing metrics (n_samples, 2)

        Returns:
            Dictionary with metrics
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]
        df = self.params_["df"]
        n = U.shape[0]

        # Approximate Fisher information (numerical)
        # For t-copula, no closed form, so we estimate
        fisher_rho = n / (1 - rho**2)**2  # Approximate (similar to Gaussian)

        # Estimate for df parameter is more complex
        fisher_df = n / (2 * df**2)  # Rough approximation

        fisher_matrix = np.array([
            [fisher_rho, 0],
            [0, fisher_df]
        ])

        fisher_det = np.linalg.det(fisher_matrix)
        volume_element = np.sqrt(fisher_det)

        # Mutual information (estimated numerically)
        # Sample and compute empirical MI
        samples = U
        copula_density = self.pdf(samples)
        mutual_info = np.mean(np.log(copula_density + 1e-10))

        # Tail dependence
        lambda_l, lambda_u = self.tail_dependence()

        return {
            "fisher_information": fisher_matrix,
            "fisher_determinant": fisher_det,
            "manifold_volume": volume_element,
            "mutual_information": mutual_info,
            "correlation": rho,
            "degrees_of_freedom": df,
            "tail_dependence_lower": lambda_l,
            "tail_dependence_upper": lambda_u,
        }

    def kendall_tau(self) -> float:
        """
        Compute Kendall's tau for the t-copula.

        For t-copula: τ = (2/π) * arcsin(ρ) (same as Gaussian)

        Returns:
            Kendall's tau coefficient
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]
        tau = (2 / np.pi) * np.arcsin(rho)

        return tau

    def spearman_rho(self) -> float:
        """
        Compute Spearman's rho for the t-copula.

        For t-copula: ρ_S ≈ (6/π) * arcsin(ρ/2) (approximate, like Gaussian)

        Returns:
            Spearman's rho coefficient
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]
        rho_s = (6 / np.pi) * np.arcsin(rho / 2)

        return rho_s
