"""Gaussian (Normal) copula implementation using only numpy."""

import numpy as np

from .base import BivariateCopula
from ..utils.logger import get_logger
from ..utils.validators import validate_or_raise, DataValidator

logger = get_logger(__name__)


# Pure numpy implementations of statistical functions
def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _norm_cdf(x):
    """
    Standard normal CDF using error function approximation.

    Uses the approximation: Φ(x) ≈ 0.5 * (1 + erf(x/√2))
    We implement erf using a rational approximation.
    """
    # Constants for error function approximation (Abramowitz and Stegun)
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x) / np.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def _norm_ppf(p):
    """
    Standard normal inverse CDF (quantile function).

    Uses Beasley-Springer-Moro algorithm for approximation.
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # Coefficients for rational approximation
    a = np.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ])

    b = np.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ])

    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ])

    d = np.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ])

    # Define break-points
    p_low = 0.02425
    p_high = 1 - p_low

    # Rational approximation for central region
    def rational_approx(t):
        return t - ((a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5) /
                   (1 + b[0]*t + b[1]*t**2 + b[2]*t**3 + b[3]*t**4 + b[4]*t**5))

    # Rational approximation for tail region
    def tail_approx(t):
        return (c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5) / \
               (1 + d[0]*t + d[1]*t**2 + d[2]*t**3 + d[3]*t**4)

    # Vectorized implementation
    result = np.zeros_like(p, dtype=float)

    # Central region
    central_mask = (p >= p_low) & (p <= p_high)
    q = p[central_mask] - 0.5
    r = q * q
    result[central_mask] = q * (a[0] + a[1]*r + a[2]*r**2 + a[3]*r**3 + a[4]*r**4 + a[5]*r**5) / \
                          (1 + b[0]*r + b[1]*r**2 + b[2]*r**3 + b[3]*r**4 + b[4]*r**5)

    # Lower tail
    lower_mask = p < p_low
    if np.any(lower_mask):
        q = np.sqrt(-2 * np.log(p[lower_mask]))
        result[lower_mask] = tail_approx(q)

    # Upper tail
    upper_mask = p > p_high
    if np.any(upper_mask):
        q = np.sqrt(-2 * np.log(1 - p[upper_mask]))
        result[upper_mask] = -tail_approx(q)

    return result


def _bivariate_normal_pdf(x, y, rho):
    """
    Bivariate normal PDF with zero mean and correlation rho.

    Formula: f(x,y) = 1/(2π√(1-ρ²)) * exp(-q/(2(1-ρ²)))
    where q = x² - 2ρxy + y²
    """
    det = 1 - rho**2
    q = x**2 - 2*rho*x*y + y**2
    return np.exp(-q / (2*det)) / (2 * np.pi * np.sqrt(det))


def _bivariate_normal_cdf(x, y, rho):
    """
    Bivariate normal CDF using Drezner-Wesolowsky approximation.

    This is an approximation but quite accurate for most practical purposes.
    """
    # Handle edge cases
    if rho == 0:
        return _norm_cdf(x) * _norm_cdf(y)

    # Drezner-Wesolowsky constants
    a = np.array([0.3253030, 0.4211071, 0.1334425, 0.006374323])
    b = np.array([0.1337764, 0.6243247, 1.3425378, 2.2626645])

    h = -x
    k = -y
    hk = h * k

    # If correlation is -1 or 1
    if abs(rho) >= 0.999:
        if rho > 0:
            return min(_norm_cdf(-h), _norm_cdf(-k))
        else:
            lower_bound = max(0, _norm_cdf(-h) - _norm_cdf(k))
            return lower_bound

    hs = (h * h + k * k) / 2
    asr = np.arcsin(rho)
    sn = 0

    for i in range(4):
        for sign in [-1, 1]:
            xs = (np.sin(asr * (sign * b[i] + 1) / 2))**2
            sn += a[i] * np.exp((xs * hk - hs) / (1 - xs))

    bvn = _norm_cdf(-h) * _norm_cdf(-k) + sn * asr / (4 * np.pi)

    return max(0, min(1, bvn))


def _sample_multivariate_normal(mean, cov, size):
    """
    Sample from multivariate normal distribution using Cholesky decomposition.

    Args:
        mean: Mean vector
        cov: Covariance matrix
        size: Number of samples

    Returns:
        Samples from multivariate normal
    """
    # Cholesky decomposition
    L = np.linalg.cholesky(cov)

    # Generate standard normal samples
    Z = np.random.randn(size, len(mean))

    # Transform to desired distribution
    X = Z @ L.T + mean

    return X


class GaussianCopula(BivariateCopula):
    """
    Gaussian (Normal) copula.

    The Gaussian copula is derived from the multivariate normal distribution.
    It is characterized by a correlation matrix and exhibits symmetric dependence.

    Parameters:
        rho: Correlation coefficient in [-1, 1]

    This implementation uses only numpy, no scipy dependencies.
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
        X = _norm_ppf(np.clip(U, 1e-10, 1 - 1e-10))

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
        X = _norm_ppf(np.clip(U, 1e-10, 1 - 1e-10))

        # Compute copula density
        # c(u,v) = φ_ρ(Φ^{-1}(u), Φ^{-1}(v)) / [φ(Φ^{-1}(u)) φ(Φ^{-1}(v))]
        # where φ_ρ is the bivariate normal density and φ is univariate normal density

        # Bivariate normal density
        bivariate_density = _bivariate_normal_pdf(X[:, 0], X[:, 1], rho)

        # Univariate normal densities
        univariate_density = _norm_pdf(X[:, 0]) * _norm_pdf(X[:, 1])

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
        X = _norm_ppf(np.clip(U, 1e-10, 1 - 1e-10))

        # Compute CDF using bivariate normal CDF
        cdf_values = np.array([
            _bivariate_normal_cdf(X[i, 0], X[i, 1], rho)
            for i in range(len(X))
        ])

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
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, rho], [rho, 1.0]])
        X = _sample_multivariate_normal(mean, cov, n_samples)

        # Transform to uniform using normal CDF
        U = _norm_cdf(X)

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
        x = _norm_ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = _norm_ppf(np.clip(v, 1e-10, 1 - 1e-10))

        if condition_on == 1:
            # C(u|v) = Φ((Φ^{-1}(u) - ρ Φ^{-1}(v)) / √(1 - ρ²))
            z = (x - rho * y) / np.sqrt(1 - rho**2)
        else:
            # C(v|u) = Φ((Φ^{-1}(v) - ρ Φ^{-1}(u)) / √(1 - ρ²))
            z = (y - rho * x) / np.sqrt(1 - rho**2)

        cond_cdf = _norm_cdf(z)

        return cond_cdf

    def information_geometry_metrics(self, U: np.ndarray) -> dict:
        """
        Compute information geometry metrics for the copula.

        Returns geometric and information-theoretic properties based on the
        Fisher information metric on the statistical manifold.

        Args:
            U: Data for computing metrics (n_samples, 2)

        Returns:
            Dictionary with metrics:
            - fisher_information: Fisher information matrix
            - fisher_determinant: Determinant of Fisher matrix
            - manifold_volume: Volume element on manifold
            - mutual_information: Estimated mutual information
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]

        # Fisher information for Gaussian copula has closed form
        # For bivariate Gaussian copula, I(ρ) = n / (1 - ρ²)²
        n = U.shape[0]
        fisher_scalar = n / (1 - rho**2)**2
        fisher_matrix = np.array([[fisher_scalar]])

        # Determinant
        fisher_det = fisher_scalar

        # Volume element (sqrt of determinant)
        volume_element = np.sqrt(fisher_det)

        # Mutual information for Gaussian copula: -0.5 * log(1 - ρ²)
        mutual_info = -0.5 * np.log(1 - rho**2)

        return {
            "fisher_information": fisher_matrix,
            "fisher_determinant": fisher_det,
            "manifold_volume": volume_element,
            "mutual_information": mutual_info,
            "correlation": rho,
        }

    def kendall_tau(self) -> float:
        """
        Compute Kendall's tau for the Gaussian copula.

        For Gaussian copula: τ = (2/π) * arcsin(ρ)

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
        Compute Spearman's rho for the Gaussian copula.

        For Gaussian copula: ρ_S = (6/π) * arcsin(ρ/2)

        Returns:
            Spearman's rho coefficient
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted")

        rho = self.params_["rho"]
        rho_s = (6 / np.pi) * np.arcsin(rho / 2)

        return rho_s
