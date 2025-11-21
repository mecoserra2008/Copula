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
- **Backtesting trading strategies** with comprehensive performance metrics
- **Implementing copula-based trading strategies** (pairs trading, stat arb, tail risk hedging)

## Features

### Data Layer
- **Bybit Futures Integration**: Fetch historical 15-minute OHLCV data from futures markets
- **Batched Historical Fetching**: Automatically handles Bybit's API limits with smart batching
- **45+ Trading Pairs**: Pre-configured futures symbols across all categories (majors, altcoins, DeFi, meme tokens)
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

### Visualization Tools
- **3D PDF Surface Plots**: Visualize copula probability density functions
- **CDF Contour Plots**: Show copula cumulative distribution contours
- **Scatter Comparisons**: Compare empirical data vs fitted copula
- **Copula Comparison Grids**: Side-by-side comparison of multiple copulas
- **Tail Dependence Illustrations**: Visualize upper and lower tail dependencies
- **Density Heatmaps**: 2D density visualization

### Backtesting Framework
- **Portfolio Management**: Track positions, cash, and equity
- **Performance Metrics**: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- **Benchmark Comparison**: Compare against Bitcoin or custom benchmarks
  - Information Ratio, Alpha, Beta
  - Tracking Error, Up/Down Capture Ratios
  - Outperformance analysis
- **Transaction Costs**: Configurable commissions and slippage
- **Comprehensive Charts**: 9-panel visualization including equity curves, drawdowns, returns distribution, rolling metrics, and monthly heatmaps
- **Detailed Reporting**: Trade history, drawdown analysis, and benchmark comparison

### Trading Strategies
- **Pairs Trading**: Copula-based mean reversion strategy
- **Statistical Arbitrage**: Multi-pair market-neutral strategy
- **Tail Risk Hedging**: Dynamic hedging using tail dependence
- **Custom Strategies**: Extensible base class for building your own

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

### Trading Strategies

```python
from src.strategies.pairs_trading import CopulaPairsTradingStrategy
from src.backtest.backtest_engine import BacktestEngine

# Create pairs trading strategy
strategy = CopulaPairsTradingStrategy(
    symbol1="BTCUSDT",
    symbol2="ETHUSDT",
    lookback_period=500,
    copula_type="gaussian",
    entry_threshold=0.05,  # Enter when prob < 5% or > 95%
    exit_threshold=0.5,    # Exit when prob reverts to 50%
    position_size=0.5
)

# Prepare data
data = {
    "BTCUSDT": btc_df,
    "ETHUSDT": eth_df
}

# Initialize backtest engine
engine = BacktestEngine(
    initial_capital=100000.0,
    transaction_cost=0.001,
    slippage=0.0005
)

# Run backtest
engine.load_data(data)
results = engine.run(strategy)

# Print performance
engine.print_results()
engine.plot_results()
```

### Statistical Arbitrage

```python
from src.strategies.statistical_arbitrage import CopulaStatisticalArbitrageStrategy

# Multi-pair stat arb
strategy = CopulaStatisticalArbitrageStrategy(
    pairs=[
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("ETHUSDT", "SOLUSDT")
    ],
    lookback_period=500,
    copula_type="gaussian",
    rebalance_frequency=20,
    num_positions=4
)

# Backtest
engine.load_data(data)
results = engine.run(strategy)
```

### Tail Risk Hedging

```python
from src.strategies.tail_risk_hedging import CopulaTailRiskHedgingStrategy

# Tail risk hedging strategy
strategy = CopulaTailRiskHedgingStrategy(
    base_asset="BTCUSDT",
    hedge_assets=["ETHUSDT", "SOLUSDT"],
    copula_type="student_t",  # Uses tail dependence
    max_hedge_ratio=0.3,
    base_allocation=0.7
)

# Backtest
engine.load_data(data)
results = engine.run(strategy)
```

### Copula Visualizations

```python
from src.copulas.visualization import CopulaVisualizer
from src.copulas.gaussian import GaussianCopula

# Fit a copula (assuming you have uniform data U)
copula = GaussianCopula()
copula.fit(U)

# Create visualizations
visualizer = CopulaVisualizer()

# 1. 3D PDF surface plot
visualizer.plot_pdf_surface(
    copula,
    title="Gaussian Copula PDF",
    save_path="plots/copula_pdf.png"
)

# 2. CDF contour plot
visualizer.plot_cdf_contours(
    copula,
    title="Gaussian Copula CDF",
    save_path="plots/copula_cdf.png"
)

# 3. Scatter plot comparison (empirical vs copula)
visualizer.plot_scatter_comparison(
    copula,
    U,  # Your empirical data
    save_path="plots/copula_scatter.png"
)

# 4. Compare multiple copulas side-by-side
copulas = [gaussian_copula, student_t_copula, clayton_copula]
names = ["Gaussian", "Student-t", "Clayton"]
visualizer.plot_copula_comparison(
    copulas,
    names,
    save_path="plots/copula_comparison.png"
)

# 5. Tail dependence illustration
visualizer.plot_tail_dependence_illustration(
    student_t_copula,
    U,
    save_path="plots/tail_dependence.png"
)

# 6. Density heatmap
visualizer.plot_density_heatmap(
    copula,
    save_path="plots/density_heatmap.png"
)
```

### Benchmark Comparison

Compare your strategy against Bitcoin or any other benchmark:

```python
from src.backtest.backtest_engine import BacktestEngine

# Initialize engine
engine = BacktestEngine(initial_capital=100000.0)

# Load strategy data
engine.load_data(data)

# Load benchmark (e.g., Bitcoin)
engine.load_benchmark(btc_df, benchmark_symbol="BTCUSDT")

# Run backtest
results = engine.run(strategy)

# Print results with benchmark comparison
engine.print_results()
# Output includes:
#   - Strategy vs Benchmark returns
#   - Information Ratio
#   - Beta and Alpha
#   - Tracking Error
#   - Up/Down Capture Ratios

# Plot comprehensive charts with benchmark
engine.plot_results(save_path="plots/backtest_report.png")
# Creates 9-panel chart including:
#   1. Equity curve vs benchmark
#   2. Drawdown comparison
#   3. Returns distribution
#   4. Cumulative returns
#   5. Rolling Sharpe ratio
#   6. Monthly returns heatmap
#   7. Performance metrics summary
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

### Python Scripts

#### 1. Basic Usage
```bash
python examples/basic_usage.py
```
Demonstrates basic copula fitting for a single pair.

#### 2. Multi-Pair Analysis
```bash
python examples/multi_pair_analysis.py
```
Shows how to analyze multiple pairs and compare results.

#### 3. Risk Analysis
```bash
python examples/risk_analysis.py
```
Comprehensive risk analysis including VaR, CVaR, and diversification benefits.

#### 4. Pairs Trading Backtest
```bash
python examples/backtest_pairs_trading.py
```
Complete backtest of copula-based pairs trading strategy.

#### 5. Statistical Arbitrage Backtest
```bash
python examples/backtest_stat_arb.py
```
Multi-pair statistical arbitrage strategy with market-neutral portfolio.

#### 6. Tail Risk Hedging Backtest
```bash
python examples/backtest_tail_risk.py
```
Dynamic hedging strategy using tail dependence from Student-t copula.

#### 7. Copula Visualizations Demo
```bash
python examples/demo_visualizations.py
```
Demonstrates all copula visualization capabilities including PDF surfaces, CDF contours, scatter comparisons, and tail dependence illustrations.

#### 8. Backtest with Benchmark
```bash
python examples/backtest_with_benchmark.py
```
Complete example showing benchmark comparison with detailed alpha, beta, information ratio, and capture ratio analysis.

### Jupyter Notebooks

#### Complete Framework Tutorial
```bash
jupyter notebook notebooks/01_copula_framework_tutorial.ipynb
```
Comprehensive tutorial covering:
- Data fetching and preprocessing
- Copula fitting and model selection
- Dependence analysis and visualization
- Strategy development
- Backtesting and performance evaluation

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
│   │   ├── frank.py
│   │   └── visualization.py  # NEW: Copula visualization tools
│   ├── transformations/   # Marginal transformations
│   │   ├── empirical.py
│   │   └── parametric.py
│   ├── manager/           # Orchestration layer
│   │   └── copula_manager.py
│   ├── analysis/          # Analysis tools
│   │   ├── dependence.py
│   │   └── risk.py
│   ├── backtest/          # Backtesting framework
│   │   ├── portfolio.py
│   │   ├── performance.py
│   │   └── backtest_engine.py
│   ├── strategies/        # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── pairs_trading.py
│   │   ├── statistical_arbitrage.py
│   │   └── tail_risk_hedging.py
│   └── utils/             # Utilities
│       ├── config.py
│       ├── logger.py
│       └── validators.py
├── examples/              # Example scripts
│   ├── basic_usage.py
│   ├── multi_pair_analysis.py
│   ├── risk_analysis.py
│   ├── backtest_pairs_trading.py
│   ├── backtest_stat_arb.py
│   ├── backtest_tail_risk.py
│   ├── demo_visualizations.py     # NEW: Copula visualization demo
│   └── backtest_with_benchmark.py # NEW: Benchmark comparison demo
├── notebooks/             # Jupyter notebooks
│   └── 01_copula_framework_tutorial.ipynb
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
