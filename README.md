# Bybit Copula Framework

A comprehensive Python framework for modeling dependencies between cryptocurrency pairs using copula theory, leveraging Bybit historical 15-minute data.

## Overview

This framework provides a complete toolkit for:
- Fetching historical cryptocurrency data from Bybit
- Preprocessing and cleaning OHLCV data
- Calculating returns and transforming to uniform marginals
- Fitting multiple copula models (Gaussian, Student-t, Clayton, Gumbel, Frank)
- Analyzing dependencies and tail risks
- Computing portfolio risk metrics

## Features

### Data Layer
- **Bybit Integration**: Fetch historical 15-minute OHLCV data
- **Data Preprocessing**: Clean, validate, and align time series
- **Return Calculations**: Log returns, simple returns, volatility metrics

### Copula Models

#### Elliptical Copulas
- **Gaussian Copula**: Symmetric dependence, no tail dependence
- **Student-t Copula**: Heavy tails, symmetric tail dependence

#### Archimedean Copulas
- **Clayton Copula**: Lower tail dependence (crash modeling)
- **Gumbel Copula**: Upper tail dependence (boom modeling)
- **Frank Copula**: Symmetric, no tail dependence

### Analysis Tools
- **Dependence Metrics**: Pearson, Spearman, Kendall's tau, tail dependence
- **Risk Metrics**: VaR, CVaR, diversification benefits
- **Model Selection**: Automatic selection using AIC/BIC

## Installation

### Requirements

Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mecoserra2008/Copula.git
cd Copula
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.manager.copula_manager import CopulaManager

# Initialize manager
manager = CopulaManager()

# Analyze a pair
results = manager.fit_pair(
    symbol1="BTCUSDT",
    symbol2="ETHUSDT",
    interval="15",  # 15-minute candles
    days=30,        # Last 30 days
)

# Get best copula
best = results["best_copula"]
print(f"Best copula: {best['type']}")
print(f"Kendall's tau: {best['kendall_tau']:.4f}")
print(f"Tail dependence: {best['tail_dependence']}")
```

### Multi-Pair Analysis

```python
# Define pairs
pairs = [
    ("BTCUSDT", "ETHUSDT"),
    ("BTCUSDT", "SOLUSDT"),
    ("ETHUSDT", "SOLUSDT"),
]

# Fit copulas for all pairs
all_results = manager.fit_multiple_pairs(pairs, interval="15", days=30)

# Create comparison DataFrame
comparison = manager.compare_pairs(all_results)
print(comparison)
```

### Risk Analysis

```python
from src.analysis.risk import RiskMetrics
from src.analysis.dependence import DependenceMetrics

# Get returns from results
r1 = results["data"]["returns1"]
r2 = results["data"]["returns2"]

# Compute dependence metrics
metrics = DependenceMetrics.all_metrics(r1, r2)
print(f"Kendall's tau: {metrics['kendall']:.4f}")
print(f"Tail dependence: {metrics['tail_dependence']}")

# Compute portfolio VaR (50/50 allocation)
var = RiskMetrics.portfolio_var(r1, r2, weight1=0.5, confidence=0.95)
print(f"Portfolio VaR (95%): {var*100:.3f}%")

# Diversification benefit
div_benefit = RiskMetrics.diversification_benefit(r1, r2, weight1=0.5)
print(f"Diversification benefit: {div_benefit['diversification_benefit_pct']:.2f}%")
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Bybit API (optional for public data)
bybit:
  api_key: ""
  api_secret: ""

# Data settings
data:
  interval: "15"
  default_days: 30
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "SOLUSDT"

# Copula settings
copulas:
  types:
    - "gaussian"
    - "student_t"
    - "clayton"
    - "gumbel"
    - "frank"
  selection_criterion: "aic"  # or "bic"

# Transformation method
transformations:
  method: "empirical"  # or "parametric"
```

## Examples

The `examples/` directory contains complete working examples:

### 1. Basic Usage
```bash
python examples/basic_usage.py
```
Demonstrates basic copula fitting for a single pair.

### 2. Multi-Pair Analysis
```bash
python examples/multi_pair_analysis.py
```
Shows how to analyze multiple pairs and compare results.

### 3. Risk Analysis
```bash
python examples/risk_analysis.py
```
Comprehensive risk analysis including VaR, CVaR, and diversification benefits.

## Project Structure

```
Copula/
├── src/
│   ├── data/              # Data fetching and preprocessing
│   │   ├── bybit_fetcher.py
│   │   ├── preprocessor.py
│   │   └── returns.py
│   ├── copulas/           # Copula implementations
│   │   ├── base.py
│   │   ├── gaussian.py
│   │   ├── student_t.py
│   │   ├── clayton.py
│   │   ├── gumbel.py
│   │   └── frank.py
│   ├── transformations/   # Marginal transformations
│   │   ├── empirical.py
│   │   └── parametric.py
│   ├── manager/           # Orchestration layer
│   │   └── copula_manager.py
│   ├── analysis/          # Analysis tools
│   │   ├── dependence.py
│   │   └── risk.py
│   └── utils/             # Utilities
│       ├── config.py
│       ├── logger.py
│       └── validators.py
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── config/                # Configuration files
├── ARCHITECTURE.md        # Detailed architecture
└── README.md             # This file
```

## Copula Theory Background

### What are Copulas?

Copulas are functions that link univariate marginal distributions to form multivariate distributions. They separate the dependence structure from the marginal distributions, allowing flexible modeling of dependencies.

**Key formula**: For a bivariate distribution:
```
F(x,y) = C(F_X(x), F_Y(y))
```

Where:
- `F(x,y)` is the joint distribution
- `F_X(x)`, `F_Y(y)` are marginal CDFs
- `C` is the copula function

### When to Use Which Copula?

| Copula | Best For | Tail Dependence |
|--------|----------|-----------------|
| **Gaussian** | Symmetric dependencies, normal markets | None |
| **Student-t** | Extreme events, crash scenarios | Both tails |
| **Clayton** | Downside risk, crash modeling | Lower tail only |
| **Gumbel** | Upside movements, boom scenarios | Upper tail only |
| **Frank** | General symmetric dependencies | None |

### Interpretation Guide

**Kendall's Tau (τ)**:
- Range: [-1, 1]
- 0: Independence
- Positive: Positive association
- Negative: Negative association

**Tail Dependence (λ)**:
- Range: [0, 1]
- 0: No tail dependence (asymptotically independent)
- Higher values: Stronger tail dependence
- λ_L: Lower tail (joint crashes)
- λ_U: Upper tail (joint rallies)

## Use Cases

### 1. Portfolio Risk Management
- Model joint extreme events
- Compute portfolio VaR/CVaR under realistic dependence
- Quantify diversification benefits

### 2. Pairs Trading
- Identify pairs with strong dependencies
- Model deviation from equilibrium
- Generate trading signals based on conditional distributions

### 3. Hedging Strategies
- Design optimal hedging ratios
- Account for tail risks
- Stress test under different market conditions

### 4. Market Research
- Analyze market structure and interconnections
- Identify regime changes in dependencies
- Study contagion effects

## Advanced Features

### Custom Copula Models

Extend the framework by creating custom copulas:

```python
from src.copulas.base import BivariateCopula

class MyCopula(BivariateCopula):
    def fit(self, U):
        # Your fitting logic
        pass

    def pdf(self, U):
        # Your PDF implementation
        pass

    # Implement other required methods...
```

### Custom Transformations

Create custom marginal transformations:

```python
from src.transformations.empirical import EmpiricalTransform

class MyTransform(EmpiricalTransform):
    def transform(self, data):
        # Your transformation logic
        pass
```

## Performance Considerations

- **Caching**: Enable data caching in config to avoid repeated API calls
- **Parallelization**: For multiple pairs, use multiprocessing
- **Sample Size**: Minimum 30 observations recommended for reliable estimation
- **Student-t CDF**: Computationally expensive; uses Monte Carlo approximation

## Troubleshooting

### Common Issues

1. **No data returned from Bybit**
   - Check symbol names (must be exact, e.g., "BTCUSDT")
   - Verify interval is supported (1, 3, 5, 15, 30, 60, etc.)
   - Check internet connection

2. **Copula fitting fails**
   - Ensure sufficient data (>30 observations)
   - Check for NaN values in returns
   - Try different transformation methods

3. **Numerical errors**
   - Reduce date range if data is too large
   - Check for extreme outliers
   - Try winsorization in preprocessing

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

### Academic Papers
- Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
- Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and dependence in risk management.
- Patton, A. J. (2012). A review of copula models for economic time series.

### Bybit API
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/intro)

## License

MIT License - see LICENSE file for details.

## Disclaimer

This framework is for educational and research purposes only. Cryptocurrency trading carries substantial risk. Always conduct your own research and consult with financial professionals before making investment decisions.

## Support

For issues, questions, or contributions:
- GitHub Issues: [github.com/mecoserra2008/Copula/issues](https://github.com/mecoserra2008/Copula/issues)
- Documentation: See `ARCHITECTURE.md` for detailed design documentation

## Acknowledgments

- Built with Python, NumPy, SciPy, and Pandas
- Bybit API integration via `pybit`
- Inspired by quantitative finance research in copula modeling
