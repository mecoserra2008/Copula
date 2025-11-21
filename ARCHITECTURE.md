# Bybit Copula Framework Architecture

## Overview
A comprehensive framework for modeling dependencies between multiple cryptocurrency pairs using copula theory, leveraging Bybit historical 15-minute data.

## Core Concepts

### What are Copulas?
Copulas are statistical tools that model the dependence structure between random variables independently from their marginal distributions. They are particularly useful in finance for:
- Capturing non-linear dependencies between assets
- Modeling tail dependencies (extreme events)
- Risk management and portfolio optimization
- Pairs trading strategies

### Why Use Copulas for Crypto Trading?
1. **Non-linear dependencies**: Crypto pairs often exhibit complex relationships that linear correlation misses
2. **Tail dependencies**: Copulas can model extreme market movements
3. **Flexible marginal distributions**: Each asset can have its own distribution
4. **Multiple pairs**: Can model dependencies across multiple assets simultaneously

## Framework Architecture

### 1. Data Layer (`src/data/`)
- **BybitDataFetcher**: Retrieves historical OHLCV data at 15-minute intervals
- **DataPreprocessor**: Cleans, validates, and transforms raw data
- **ReturnCalculator**: Computes log returns and other transformations

### 2. Copula Models (`src/copulas/`)

#### Base Classes
- **BaseCopula**: Abstract base class defining the copula interface
  - `fit()`: Estimate copula parameters from data
  - `pdf()`: Probability density function
  - `cdf()`: Cumulative distribution function
  - `sample()`: Generate random samples
  - `aic()`, `bic()`: Model selection criteria

#### Implemented Copulas

**Elliptical Copulas**:
- **GaussianCopula**: Models linear dependencies with normal distribution
  - Parameters: Correlation matrix
  - Best for: Symmetric dependencies

- **StudentTCopula**: Heavy-tailed version of Gaussian
  - Parameters: Correlation matrix, degrees of freedom
  - Best for: Extreme events, tail dependencies

**Archimedean Copulas**:
- **ClaytonCopula**: Lower tail dependence
  - Parameters: theta (θ > 0)
  - Best for: Downside risk, crash modeling

- **GumbelCopula**: Upper tail dependence
  - Parameters: theta (θ ≥ 1)
  - Best for: Joint rallies, boom modeling

- **FrankCopula**: Symmetric, no tail dependence
  - Parameters: theta (θ ∈ ℝ, θ ≠ 0)
  - Best for: General symmetric dependencies

### 3. Transformation Layer (`src/transformations/`)
- **EmpiricalTransform**: Non-parametric transformation to uniform marginals
- **ParametricTransform**: Fit distributions (normal, t, skewed-t) to marginals
- **PIT (Probability Integral Transform)**: Convert data to copula space [0,1]

### 4. Multi-Pair Management (`src/manager/`)
- **CopulaManager**: Orchestrates copula modeling for multiple pairs
  - Manages data fetching for multiple symbols
  - Fits multiple copula types to each pair
  - Selects best copula based on AIC/BIC
  - Provides unified interface for analysis

### 5. Analysis Tools (`src/analysis/`)
- **DependenceMetrics**: Kendall's tau, Spearman's rho, tail dependence
- **BacktestEngine**: Test copula-based trading strategies
- **RiskMetrics**: VaR, CVaR, correlation breakdown analysis

### 6. Utilities (`src/utils/`)
- **Config**: Configuration management
- **Logger**: Logging setup
- **Validators**: Data validation utilities

## Data Flow

```
Bybit API (15-min data)
    ↓
BybitDataFetcher
    ↓
DataPreprocessor (clean, validate)
    ↓
ReturnCalculator (log returns)
    ↓
EmpiricalTransform (→ uniform [0,1])
    ↓
CopulaFitting (Gaussian, t, Clayton, Gumbel, Frank)
    ↓
Model Selection (AIC/BIC)
    ↓
Analysis & Trading Signals
```

## Project Structure

```
Copula/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bybit_fetcher.py      # Bybit API integration
│   │   ├── preprocessor.py        # Data cleaning
│   │   └── returns.py             # Return calculations
│   ├── copulas/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseCopula abstract class
│   │   ├── gaussian.py            # Gaussian copula
│   │   ├── student_t.py           # Student-t copula
│   │   ├── clayton.py             # Clayton copula
│   │   ├── gumbel.py              # Gumbel copula
│   │   └── frank.py               # Frank copula
│   ├── transformations/
│   │   ├── __init__.py
│   │   ├── empirical.py           # Empirical CDF
│   │   └── parametric.py          # Parametric marginals
│   ├── manager/
│   │   ├── __init__.py
│   │   └── copula_manager.py      # Multi-pair orchestration
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dependence.py          # Dependence metrics
│   │   └── risk.py                # Risk metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── validators.py
├── examples/
│   ├── basic_usage.py             # Simple example
│   ├── multi_pair_analysis.py    # Multiple pairs
│   └── trading_strategy.py       # Strategy backtest
├── tests/
│   ├── __init__.py
│   ├── test_copulas.py
│   ├── test_data.py
│   └── test_transformations.py
├── data/                          # Cached data (gitignored)
├── config/
│   └── config.yaml                # Configuration
├── requirements.txt
├── setup.py
├── .gitignore
├── ARCHITECTURE.md                # This file
└── README.md

```

## Key Design Decisions

1. **Modular Design**: Each copula type is a separate class for easy extension
2. **Abstract Base Class**: Ensures consistent interface across all copulas
3. **Separation of Concerns**: Data fetching, transformation, and modeling are independent
4. **Configuration-Driven**: External config for API keys, symbols, parameters
5. **Type Hints**: Full type annotations for better IDE support and error catching
6. **Logging**: Comprehensive logging for debugging and monitoring
7. **Testing**: Unit tests for all core components

## Dependencies

### Core Libraries
- `numpy`: Numerical computations
- `scipy`: Statistical functions and optimization
- `pandas`: Data manipulation
- `pybit`: Bybit API client

### Additional
- `matplotlib`, `seaborn`: Visualization
- `pyyaml`: Configuration
- `pytest`: Testing
- `requests`: HTTP requests

## Usage Patterns

### Basic Usage
```python
from src.data import BybitDataFetcher
from src.manager import CopulaManager

# Fetch data
fetcher = BybitDataFetcher()
data = fetcher.fetch_multiple(['BTCUSDT', 'ETHUSDT'], interval='15', days=30)

# Fit copulas
manager = CopulaManager()
results = manager.fit_pair('BTCUSDT', 'ETHUSDT', data)

# Get best copula
best_copula = results['best_copula']
print(f"Best model: {best_copula['type']} (AIC: {best_copula['aic']})")
```

### Multi-Pair Analysis
```python
pairs = [
    ('BTCUSDT', 'ETHUSDT'),
    ('BTCUSDT', 'SOLUSDT'),
    ('ETHUSDT', 'SOLUSDT')
]

manager = CopulaManager()
all_results = manager.fit_multiple_pairs(pairs, data)

# Compare dependencies
for pair, result in all_results.items():
    print(f"{pair}: τ={result['kendall_tau']:.3f}, tail_dep={result['tail_dep']:.3f}")
```

## Next Steps

1. Implement base copula framework
2. Add specific copula types
3. Integrate Bybit data fetching
4. Create example notebooks
5. Add backtesting capabilities
6. Build trading strategies on top

## References

- Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
- Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and dependence in risk management.
- Patton, A. J. (2012). A review of copula models for economic time series.
