# Getting Started with Bybit Copula Framework

Quick start guide to get you up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Git (for cloning)
- Internet connection (for fetching Bybit data)

## Quick Setup (5 Minutes)

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/mecoserra2008/Copula.git
cd Copula

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install framework
pip install -e .
```

### 2. Run Your First Example

```bash
# Fetch data and fit copulas for BTC-ETH pair
python examples/basic_usage.py
```

This will:
- Fetch 30 days of 15-minute futures data from Bybit
- Fit 5 different copula models
- Select the best model using AIC
- Display dependency metrics

### 3. Run a Backtest

```bash
# Backtest pairs trading strategy
python examples/backtest_pairs_trading.py
```

This will:
- Fetch historical futures data
- Run a copula-based pairs trading strategy
- Display comprehensive performance metrics
- Compare against Bitcoin benchmark

## What's Available?

### ⚙️ 45+ Bybit Futures Pairs

Pre-configured in `config/config.yaml`:
- Major: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, etc.
- Layer 1/2: AVAX, NEAR, ARB, OP, ATOM, DOT, etc.
- DeFi: AAVE, UNI, LINK, MKR, COMP, etc.
- Emerging: PEPE, SHIB, APE, SAND, etc.

### 📊 5 Copula Models

- **Gaussian**: Symmetric, no tail dependence
- **Student-t**: Heavy tails, crash modeling
- **Clayton**: Lower tail dependence (downside risk)
- **Gumbel**: Upper tail dependence (rallies)
- **Frank**: Symmetric, general dependence

### 🎯 3 Trading Strategies

1. **Pairs Trading**: Mean reversion using conditional probabilities
2. **Statistical Arbitrage**: Multi-pair market-neutral
3. **Tail Risk Hedging**: Dynamic hedging using tail dependence

## Common Tasks

### Analyze a Different Pair

Edit any example script:

```python
# Change these lines
symbol1 = "BTCUSDT"   # Change to your symbol
symbol2 = "SOLUSDT"   # Change to your symbol
interval = "15"       # 15-minute bars
days = 30             # Last 30 days
```

### Use Different Copula

```python
strategy = CopulaPairsTradingStrategy(
    symbol1="BTCUSDT",
    symbol2="ETHUSDT",
    copula_type="student_t",  # Change: gaussian, student_t, clayton, gumbel, frank
    entry_threshold=0.05,
    position_size=0.5
)
```

### Fetch More Historical Data

```python
# Fetch 90 days instead of 30
df = fetcher.fetch_and_cache(
    symbol="BTCUSDT",
    interval="15",
    days=90  # Change this
)
```

### Change Time Interval

```python
# Use hourly data instead of 15-minute
interval = "60"  # Options: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
```

## Interactive Tutorial

Launch the Jupyter notebook for an interactive walkthrough:

```bash
jupyter notebook notebooks/01_copula_framework_tutorial.ipynb
```

The notebook covers:
- ✅ Data fetching and visualization
- ✅ Copula fitting and model selection
- ✅ Dependence analysis with charts
- ✅ Strategy development
- ✅ Backtesting with performance metrics

## Configuration

All settings are in `config/config.yaml`:

```yaml
data:
  interval: "15"          # Candle interval
  default_days: 30        # Days to fetch
  cache_enabled: true     # Enable caching
  benchmark_symbol: "BTCUSDT"  # Benchmark for comparison

copulas:
  types:                  # Copulas to fit
    - "gaussian"
    - "student_t"
    - "clayton"
    - "gumbel"
    - "frank"
  selection_criterion: "aic"  # or "bic"
```

## Understanding the Output

### Copula Results

```
Best copula: student_t (AIC=1234.56)
Kendall's tau: 0.6523
Lower tail λ: 0.1234  # Crash dependence
Upper tail λ: 0.1234  # Rally dependence
```

**What it means:**
- **Kendall's τ**: Overall correlation (-1 to 1)
- **Lower tail λ**: Joint crash probability
- **Upper tail λ**: Joint rally probability

### Backtest Results

```
Returns:
  Total Return:       25.34%
  Annualized Return: 112.45%

Risk-Adjusted:
  Sharpe Ratio:        2.145  # > 1 is good, > 2 is excellent
  Sortino Ratio:       3.421  # Similar to Sharpe, downside only
  Calmar Ratio:        5.234  # Return/Max drawdown

Risk:
  Max Drawdown:       -8.23%  # Worst peak-to-trough loss

vs Bitcoin Benchmark:
  Strategy Return:     25.34%
  Bitcoin Return:      15.20%
  Outperformance:     +10.14%
```

## Troubleshooting

### "No data returned for XXXUSDT"

The symbol might not be available on Bybit futures. Check available symbols:

```python
from src.data.bybit_fetcher import BybitDataFetcher

fetcher = BybitDataFetcher()
symbols = fetcher.get_available_symbols()
print(f"Available: {len(symbols)} symbols")
print(symbols[:10])  # Show first 10
```

### "Rate limit exceeded"

The fetcher automatically handles rate limits with delays. If you still see this:
- Reduce the number of days
- Fetch fewer symbols at once
- Wait a minute and try again

### "Copula fitting failed"

Possible causes:
- Insufficient data (need at least 30 observations)
- Too many NaN values
- Very low volatility period

Try:
- Increasing `days` parameter
- Using a different pair
- Checking data quality

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install numpy scipy pandas pybit matplotlib seaborn pyyaml tqdm
```

## Next Steps

### 1. Explore Examples

```bash
ls examples/
# Run each one to see different features
```

### 2. Customize Strategies

Copy an existing strategy and modify parameters:
- Entry/exit thresholds
- Position sizing
- Rebalancing frequency
- Copula type

### 3. Run Multi-Pair Analysis

```python
from src.manager import CopulaManager

manager = CopulaManager()
pairs = [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT")]
results = manager.fit_multiple_pairs(pairs)
```

### 4. Backtest on Different Periods

Test strategy robustness across different market conditions:
- Bull market: 2020-2021
- Bear market: 2022
- Sideways: 2023

## Resources

- **Full Documentation**: See `README.md`
- **Architecture**: See `ARCHITECTURE.md`
- **Examples**: See `examples/` directory
- **Notebooks**: See `notebooks/` directory

## Support

For issues or questions:
- Check documentation in `README.md`
- Review example scripts in `examples/`
- Open an issue on GitHub

## Quick Reference Card

```bash
# Fetch data
python examples/basic_usage.py

# Analyze multiple pairs
python examples/multi_pair_analysis.py

# Risk analysis
python examples/risk_analysis.py

# Backtest pairs trading
python examples/backtest_pairs_trading.py

# Backtest stat arb
python examples/backtest_stat_arb.py

# Backtest tail hedging
python examples/backtest_tail_risk.py

# Interactive notebook
jupyter notebook notebooks/01_copula_framework_tutorial.ipynb
```

---

**You're all set!** 🚀

Start with `python examples/basic_usage.py` and explore from there!
