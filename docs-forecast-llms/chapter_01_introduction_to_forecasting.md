---
title: "Chapter 1: Introduction to Forecasting"
description: "Chapter 1: Introduction to Forecasting in the Forecasting with LLMs course. Learn the core concepts and practical guidance for this chapter."
sidebar_label: "Chapter 1: Introduction to Forecasting"
sidebar_position: 1
slug: "/forecast-llms/chapter-01-introduction-to-forecasting"
keywords:
  - forecasting
  - time series
  - LLMs
  - machine learning
  - prediction
---

# Chapter 1: Introduction to Forecasting

> *"Prediction is very difficult, especially about the future."*
> — Niels Bohr

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand what forecasting is and why it matters in business and AI systems.
2. Distinguish between different types of forecasting problems.
3. Identify the key components of a time series.
4. Apply classical forecasting methods using Python.
5. Understand the forecasting pipeline from raw data to production deployment.
6. Evaluate forecasts using appropriate metrics.
7. Recognize when to use statistical models vs. machine learning vs. foundation models.

---

## Prerequisites

- Python 3.9+
- Basic knowledge of Pandas and NumPy
- Familiarity with Matplotlib
- No prior forecasting experience required

Install dependencies before running code in this chapter:

```bash
pip install pandas numpy matplotlib plotly scikit-learn statsmodels
```

---

## 1.1 What Is Forecasting?

Imagine you run a retail store. Every morning you ask yourself:

- *How many customers will walk in today?*
- *Which products will run out of stock this week?*
- *Should I hire extra staff for the upcoming holiday season?*

These questions all require **forecasting** — the art and science of making predictions about the future based on patterns from the past.

Forecasting is the process of estimating future values of a variable based on historical data, domain knowledge, and mathematical models.

It is not guessing. A good forecast is **systematic**, **measurable**, and **improvable**.

### Why Forecasting Matters

Every major business decision involves some view of the future:

| Business Domain | Forecasting Question |
|----------------|---------------------|
| Retail | How much inventory should I stock next month? |
| Finance | What will the stock price be in 30 days? |
| Energy | How much electricity will be consumed tomorrow? |
| Healthcare | How many patients will visit the ER this weekend? |
| Supply Chain | When will demand spike and how large will it be? |
| HR / Workforce | How many employees do we need next quarter? |
| E-Commerce | What will our website traffic look like during a sale? |

Poor forecasting leads to:
- **Overstocking**: wasted capital, storage costs
- **Understocking**: lost sales, unhappy customers
- **Overstaffing**: inflated payroll
- **Understaffing**: poor customer experience
- **Budget overruns**: missed financial targets

Good forecasting leads to:
- Better resource allocation
- Reduced operational costs
- Improved customer satisfaction
- Smarter strategic planning

### The Business Value of a 1% Improvement

Even a small improvement in forecast accuracy has enormous financial impact.

**Example:** A retail chain with $1 billion in annual revenue and $200 million in inventory:
- A 1% improvement in forecast accuracy → approximately $2 million in inventory savings
- Industry studies show companies with best-in-class forecasting carry **10–15% less inventory** than average competitors

---

## 1.2 Types of Forecasting Problems

Not all forecasting problems are the same. Understanding the type of problem helps you pick the right method.

### 1.2.1 By Time Horizon

| Horizon | Range | Example |
|---------|-------|---------|
| Short-term | Hours to days | Electricity load, call center volume |
| Medium-term | Weeks to months | Retail sales, inventory planning |
| Long-term | Months to years | Economic projections, capital planning |

As the forecast horizon grows longer, uncertainty increases. Short-term forecasts are generally more accurate than long-term ones.

### 1.2.2 By Number of Variables

**Univariate Forecasting**: Only one variable is forecasted, using its own historical values.

```
Sales(t) = f(Sales(t-1), Sales(t-2), ..., Sales(t-n))
```

**Multivariate Forecasting**: Multiple variables are used together.

```
Sales(t) = f(Sales(t-1), Price(t), Weather(t), Promotion(t))
```

### 1.2.3 By Output Type

| Output Type | Description | Example |
|------------|-------------|---------|
| Point Forecast | Single predicted value | "Sales will be 500 units" |
| Interval Forecast | Range of likely values | "Sales will be between 450 and 550 units" |
| Probabilistic Forecast | Full probability distribution | "10th percentile: 400, median: 500, 90th: 620" |

Modern AI systems are increasingly focused on **probabilistic forecasts** because they communicate uncertainty honestly.

### 1.2.4 By Frequency

| Frequency | Periods per Year | Example |
|-----------|-----------------|---------|
| Annual | 1 | GDP growth |
| Quarterly | 4 | Corporate earnings |
| Monthly | 12 | Retail sales |
| Weekly | 52 | Supermarket orders |
| Daily | 365 | Website traffic |
| Hourly | 8,760 | Energy consumption |
| Minute-level | 525,600 | Financial trades |

High-frequency data (hourly, minute) requires specialized handling due to volume and complex seasonal patterns.

---

## 1.3 Time Series: The Foundation of Forecasting

A **time series** is a sequence of data points collected at regular time intervals.

```
Index:  t=1    t=2    t=3    t=4    t=5  ...  t=n
Value:  y1     y2     y3     y4     y5   ...   yn
```

### 1.3.1 Components of a Time Series

Every time series can be decomposed into four key components:

```
┌─────────────────────────────────────────────────────┐
│              TIME SERIES DECOMPOSITION              │
│                                                     │
│  Original Series = Trend + Seasonality +            │
│                    Cyclical + Residual               │
│                                                     │
│  Trend (T):        Long-term upward/downward        │
│                    movement                         │
│                    ─────────────────────────►       │
│                                                     │
│  Seasonality (S):  Repeating patterns at            │
│                    fixed periods (weekly,            │
│                    monthly, yearly)                  │
│                    ╭─╮  ╭─╮  ╭─╮  ╭─╮              │
│                    ╯  ╰╯  ╰╯  ╰╯  ╰╯               │
│                                                     │
│  Cyclical (C):     Long irregular cycles            │
│                    (economic cycles, years)         │
│                    ╭──────╮         ╭────           │
│                    ╯       ╰───────╯                │
│                                                     │
│  Residual (R):     Random noise remaining           │
│                    after removing T, S, C           │
│                    ╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿╿            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Trend
The long-term direction of the data. An e-commerce company growing at 20% annually has an upward trend.

#### Seasonality
Predictable, repeating patterns tied to the calendar:
- **Daily seasonality**: Rush hour traffic peaks at 8am and 6pm
- **Weekly seasonality**: Restaurant sales peak on weekends
- **Annual seasonality**: Toy stores see huge December spikes

#### Cyclical Patterns
Longer, irregular waves not tied to fixed calendar periods. Economic recessions and recoveries are cyclical — they repeat but not at fixed intervals.

#### Residual (Noise)
The unexplained variation that remains after removing trend, seasonality, and cyclical patterns. Good models minimize residual, but some noise is irreducible.

### 1.3.2 Additive vs Multiplicative Decomposition

**Additive Model** — components add together:
```
y(t) = Trend(t) + Seasonal(t) + Residual(t)
```
Use when the magnitude of seasonal swings stays constant over time.

**Multiplicative Model** — components multiply together:
```
y(t) = Trend(t) × Seasonal(t) × Residual(t)
```
Use when seasonal swings grow proportionally with the trend (common in retail sales, which is most real-world cases).

---

## 1.4 The Forecasting Landscape

Before diving into agentic AI methods, it's critical to understand the full spectrum of available approaches. This book will take you from the foundations all the way to multi-agent AI systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                   FORECASTING LANDSCAPE                         │
│                                                                 │
│  CLASSICAL METHODS          MACHINE LEARNING                    │
│  ┌──────────────────┐       ┌──────────────────────┐           │
│  │ Moving Average   │       │ Random Forest        │           │
│  │ Exponential      │       │ XGBoost / LightGBM   │           │
│  │ Smoothing        │       │ LSTM / GRU           │           │
│  │ ARIMA / SARIMA   │       │ N-BEATS / N-HiTS     │           │
│  │ Holt-Winters     │       │ TFT (Temporal Fusion │           │
│  └──────────────────┘       │  Transformer)        │           │
│                             └──────────────────────┘           │
│  FOUNDATION MODELS          AGENTIC AI                         │
│  ┌──────────────────┐       ┌──────────────────────┐           │
│  │ TimesFM          │       │ LLM Forecasting      │           │
│  │ Chronos          │       │ RAG Forecasting      │           │
│  │ Lag-Llama        │       │ Forecasting Agents   │           │
│  │ Moirai           │       │ Multi-Agent Systems  │           │
│  │ TimeGPT          │       │ Forecasting Copilot  │           │
│  └──────────────────┘       └──────────────────────┘           │
│                                                                 │
│  Increasing Capability, Flexibility, and Complexity ──────────► │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple univariate, stationary data | ARIMA, Exponential Smoothing |
| Multiple related series, structured data | ML (XGBoost, LightGBM) |
| Long sequences, non-linear patterns | LSTM, Transformer models |
| Zero-shot forecasting, new datasets | Foundation Models |
| Qualitative + quantitative signals | LLM Forecasting, RAG |
| Autonomous adaptive forecasting | Agentic AI Systems |

---

## 1.5 Classical Forecasting Methods: A Quick Primer

Before we adopt modern AI methods, we must understand the classical foundations. These remain competitive baselines and are used heavily in production today.

### 1.5.1 Naive Forecast

The simplest possible method: the forecast equals the last observed value.

```
ŷ(t+h) = y(t)
```

Despite its simplicity, the naive forecast is surprisingly hard to beat, especially for short horizons. It serves as a critical baseline.

**Seasonal Naive**: Use the value from the same period last season.

```
ŷ(t+h) = y(t - m + h)   where m = season length
```

### 1.5.2 Moving Average

Smooths out noise by averaging the last `k` observations.

```
ŷ(t+1) = (y(t) + y(t-1) + ... + y(t-k+1)) / k
```

### 1.5.3 Exponential Smoothing

Assigns exponentially decreasing weights to past observations. Recent observations get more weight than older ones.

**Simple Exponential Smoothing (SES)**:
```
ŷ(t+1) = α × y(t) + (1 - α) × ŷ(t)
```

Where `α` (alpha) is the smoothing parameter (0 < α < 1):
- High α → model reacts quickly to recent changes (reactive)
- Low α → model smooths more heavily (stable)

**Holt's Method** adds a trend component.
**Holt-Winters** adds both trend and seasonality.

### 1.5.4 ARIMA

**AutoRegressive Integrated Moving Average** — the workhorse of classical forecasting.

ARIMA(p, d, q):
- **p**: Number of autoregressive lags
- **d**: Degree of differencing (makes series stationary)
- **q**: Number of moving average terms

```
ARIMA(1,1,1):

y'(t) = c + φ₁·y'(t-1) + θ₁·ε(t-1) + ε(t)

where y'(t) = differenced series
      φ₁    = AR coefficient
      θ₁    = MA coefficient
      ε(t)  = white noise error
```

**SARIMA** extends ARIMA to handle seasonality: SARIMA(p, d, q)(P, D, Q)[m].

---

## 1.6 Hands-On: Your First Forecasting System

Let's build a complete forecasting pipeline using the classic **Airline Passengers Dataset** — monthly international airline passenger counts from 1949 to 1960.

### Dataset

```python
# airline_passengers.csv is available at:
# https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

### 1.6.1 Basic Version: Explore and Decompose

```python
"""
Chapter 1 - Basic Version: Time Series Exploration and Decomposition
Dataset: Airline Passengers (1949-1960)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_airline_data() -> pd.DataFrame:
    """Load and prepare the Airline Passengers dataset."""
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'  # Month Start frequency
    print(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df


df = load_airline_data()
```

**Expected Output:**
```
Loaded 144 records from 1949-01-01 to 1960-12-01

First 5 rows:
            passengers
Month
1949-01-01         112
1949-02-01         118
1949-03-01         132
1949-04-01         129
1949-05-01         121

Basic statistics:
       passengers
count  144.000000
mean   280.298611
std    119.966317
min    104.000000
25%    180.000000
50%    265.500000
75%    360.500000
max    622.000000
```

```python
# ─────────────────────────────────────────────
# 2. VISUALIZE THE TIME SERIES
# ─────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame) -> None:
    """Plot the raw time series with key statistics."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df['passengers'], color='steelblue', linewidth=1.5,
            label='Monthly Passengers')
    ax.fill_between(df.index, df['passengers'], alpha=0.1, color='steelblue')

    # Rolling mean (trend proxy)
    rolling_mean = df['passengers'].rolling(window=12).mean()
    ax.plot(df.index, rolling_mean, color='crimson', linewidth=2,
            linestyle='--', label='12-Month Rolling Mean (Trend)')

    ax.set_title('International Airline Passengers (1949–1960)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers (thousands)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('airline_passengers.png', dpi=150)
    plt.show()
    print("Plot saved to airline_passengers.png")


plot_time_series(df)
```

```python
# ─────────────────────────────────────────────
# 3. DECOMPOSE THE TIME SERIES
# ─────────────────────────────────────────────

def decompose_series(df: pd.DataFrame) -> None:
    """
    Decompose time series into Trend, Seasonal, and Residual components.
    Uses multiplicative model because seasonal amplitude grows with trend.
    """
    decomposition = seasonal_decompose(
        df['passengers'],
        model='multiplicative',
        period=12  # 12 months = 1 year
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    decomposition.observed.plot(ax=axes[0], color='steelblue')
    axes[0].set_title('Observed', fontweight='bold')
    axes[0].set_ylabel('Passengers')

    decomposition.trend.plot(ax=axes[1], color='darkorange')
    axes[1].set_title('Trend Component', fontweight='bold')
    axes[1].set_ylabel('Trend')

    decomposition.seasonal.plot(ax=axes[2], color='forestgreen')
    axes[2].set_title('Seasonal Component', fontweight='bold')
    axes[2].set_ylabel('Seasonality')

    decomposition.resid.plot(ax=axes[3], color='crimson')
    axes[3].set_title('Residual Component', fontweight='bold')
    axes[3].set_ylabel('Residual')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Time Series Decomposition – Airline Passengers',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('decomposition.png', dpi=150)
    plt.show()

    # Print seasonal indices
    seasonal_indices = (
        decomposition.seasonal
        .groupby(decomposition.seasonal.index.month)
        .first()
    )
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print("\nSeasonal Indices (values > 1.0 = above average):")
    print("-" * 35)
    for i, (month, val) in enumerate(zip(month_names, seasonal_indices.values)):
        bar = '█' * int(val * 10)
        print(f"  {month}: {val:.3f}  {bar}")


decompose_series(df)
```

**Expected Output:**
```
Seasonal Indices (values > 1.0 = above average):
-----------------------------------
  Jan: 0.908  █████████
  Feb: 0.883  ████████
  Mar: 1.002  ██████████
  Apr: 0.989  █████████
  May: 0.972  █████████
  Jun: 1.136  ███████████
  Jul: 1.230  ████████████
  Aug: 1.217  ████████████
  Sep: 1.024  ██████████
  Oct: 0.929  █████████
  Nov: 0.808  ████████
  Dec: 0.922  █████████
```

**Interpretation**: July and August have the highest seasonal indices (~1.22–1.23), meaning passenger counts are about 22–23% above the trend line in those months — reflecting summer vacation travel peaks.

### 1.6.2 Advanced Version: Classical Forecasting Methods Compared

```python
"""
Chapter 1 - Advanced Version: Multiple Forecasting Methods with Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """Compute all forecast evaluation metrics."""
    metrics = {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mape(y_true.values, y_pred),
        'SMAPE': smape(y_true.values, y_pred),
    }
    return metrics


# ─────────────────────────────────────────────
# LOAD AND SPLIT DATA
# ─────────────────────────────────────────────

def load_and_split(test_months: int = 24) -> Tuple[pd.Series, pd.Series]:
    """Load data and create train/test split."""
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'

    train = df['passengers'][:-test_months]
    test = df['passengers'][-test_months:]

    print(f"Train: {len(train)} months ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"Test:  {len(test)} months ({test.index[0].date()} to {test.index[-1].date()})")

    return train, test


train, test = load_and_split(test_months=24)


# ─────────────────────────────────────────────
# METHOD 1: NAIVE FORECAST
# ─────────────────────────────────────────────

def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Repeat the last observed value for h steps."""
    return np.array([train.iloc[-1]] * h)


def seasonal_naive_forecast(train: pd.Series, h: int, m: int = 12) -> np.ndarray:
    """Use last season's values repeated forward."""
    last_season = train.values[-m:]
    cycles = (h // m) + 1
    repeated = np.tile(last_season, cycles)
    return repeated[:h]


naive_pred = naive_forecast(train, len(test))
seasonal_naive_pred = seasonal_naive_forecast(train, len(test), m=12)


# ─────────────────────────────────────────────
# METHOD 2: HOLT-WINTERS (EXPONENTIAL SMOOTHING)
# ─────────────────────────────────────────────

def holt_winters_forecast(train: pd.Series, h: int) -> np.ndarray:
    """
    Triple Exponential Smoothing with trend and seasonality.
    Multiplicative model is appropriate when seasonal amplitude
    increases proportionally with trend.
    """
    model = ExponentialSmoothing(
        train,
        trend='multiplicative',
        seasonal='multiplicative',
        seasonal_periods=12,
        initialization_method='estimated'
    )
    fitted_model = model.fit(optimized=True)
    forecast = fitted_model.forecast(h)
    print(f"\nHolt-Winters Parameters:")
    print(f"  Alpha (level):   {fitted_model.params['smoothing_level']:.4f}")
    print(f"  Beta  (trend):   {fitted_model.params['smoothing_trend']:.4f}")
    print(f"  Gamma (seasonal):{fitted_model.params['smoothing_seasonal']:.4f}")
    return forecast.values


hw_pred = holt_winters_forecast(train, len(test))


# ─────────────────────────────────────────────
# METHOD 3: SARIMA
# ─────────────────────────────────────────────

def sarima_forecast(train: pd.Series, h: int) -> np.ndarray:
    """
    SARIMA(1,1,1)(1,1,1)[12] — standard choice for monthly data
    with trend and annual seasonality.

    p=1, d=1, q=1  : non-seasonal AR, differencing, MA
    P=1, D=1, Q=1  : seasonal AR, differencing, MA
    m=12            : 12-month season
    """
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    forecast = fitted_model.forecast(steps=h)
    print(f"\nSARIMA AIC: {fitted_model.aic:.2f}")
    return forecast.values


sarima_pred = sarima_forecast(train, len(test))


# ─────────────────────────────────────────────
# EVALUATE ALL MODELS
# ─────────────────────────────────────────────

results = []
predictions = {
    'Naive':            naive_pred,
    'Seasonal Naive':   seasonal_naive_pred,
    'Holt-Winters':     hw_pred,
    'SARIMA(1,1,1)(1,1,1)[12]': sarima_pred,
}

for model_name, pred in predictions.items():
    metrics = evaluate_forecast(test, pred, model_name)
    results.append(metrics)

results_df = pd.DataFrame(results).set_index('Model')
results_df = results_df.round(2)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string())
print("=" * 60)
print(f"\nBest MAE:   {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
print(f"Best RMSE:  {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.2f})")
print(f"Best MAPE:  {results_df['MAPE'].idxmin()} ({results_df['MAPE'].min():.2f}%)")
```

**Expected Output:**
```
============================================================
MODEL COMPARISON RESULTS
============================================================
                            MAE    RMSE   MAPE  SMAPE
Model
Naive                     95.21  112.84  22.43  19.62
Seasonal Naive            25.83   33.41   5.87   5.71
Holt-Winters              16.78   22.11   3.89   3.82
SARIMA(1,1,1)(1,1,1)[12]  18.42   24.65   4.21   4.14
============================================================

Best MAE:   Holt-Winters (16.78)
Best RMSE:  Holt-Winters (22.11)
Best MAPE:  Holt-Winters (3.89%)
```

```python
# ─────────────────────────────────────────────
# VISUALIZE FORECAST COMPARISON
# ─────────────────────────────────────────────

def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    predictions: dict
) -> None:
    """Plot all forecasts against actual test data."""
    colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson']
    linestyles = ['--', '-.', ':', '-']

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot train history
    ax.plot(train.index, train.values, color='black', linewidth=1.5,
            label='Training Data')

    # Plot actual test values
    ax.plot(test.index, test.values, color='black', linewidth=2.5,
            linestyle='-', label='Actual (Test)', marker='o', markersize=4)

    # Plot each forecast
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax.plot(test.index, pred, color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2, label=model_name, alpha=0.85)

    # Mark the train/test split
    ax.axvline(x=test.index[0], color='gray', linestyle='--',
               linewidth=1, alpha=0.7, label='Train/Test Split')

    ax.set_title('Forecasting Methods Comparison – Airline Passengers',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers (thousands)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150)
    plt.show()


plot_forecast_comparison(train, test, predictions)
```

### 1.6.3 Production Version: Forecasting Pipeline Class

```python
"""
Chapter 1 - Production Version: Reusable Forecasting Pipeline
Follows software engineering best practices: OOP, error handling,
logging, type hints, and reproducibility.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for a forecast result."""
    model_name: str
    predictions: np.ndarray
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesForecaster:
    """
    Production-grade forecasting pipeline supporting multiple methods.

    Usage:
        forecaster = TimeSeriesForecaster(data, target_col='passengers')
        forecaster.train_test_split(test_size=24)
        results = forecaster.run_all_models()
        forecaster.compare_results(results)
    """

    SUPPORTED_MODELS = ['naive', 'seasonal_naive', 'holt_winters', 'sarima']

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        freq: str = 'MS',
        random_state: int = 42
    ) -> None:
        """
        Initialize the forecaster.

        Args:
            data: DataFrame with a DatetimeIndex
            target_col: Name of the column to forecast
            freq: Pandas frequency string (e.g., 'MS' for month start)
            random_state: Seed for reproducibility
        """
        if target_col not in data.columns:
            raise ValueError(f"Column '{target_col}' not found in data. "
                             f"Available: {list(data.columns)}")

        self.data = data.copy()
        self.target_col = target_col
        self.freq = freq
        self.random_state = random_state
        self.series: pd.Series = data[target_col]
        self.train: Optional[pd.Series] = None
        self.test: Optional[pd.Series] = None

        # Set frequency if missing
        if self.series.index.freq is None:
            self.series.index.freq = pd.tseries.frequencies.to_offset(freq)

        logger.info(f"TimeSeriesForecaster initialized | "
                    f"Series: {len(self.series)} points | "
                    f"Target: {target_col} | Freq: {freq}")

    def train_test_split(self, test_size: int) -> None:
        """
        Split series into train and test sets.

        Args:
            test_size: Number of periods to reserve for testing
        """
        if test_size >= len(self.series):
            raise ValueError(
                f"test_size ({test_size}) must be less than "
                f"series length ({len(self.series)})"
            )
        self.train = self.series[:-test_size]
        self.test = self.series[-test_size:]
        logger.info(f"Split | Train: {len(self.train)} | Test: {len(self.test)}")

    def _require_split(self) -> None:
        """Raise if train/test split hasn't been done."""
        if self.train is None or self.test is None:
            raise RuntimeError("Call train_test_split() before forecasting.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard forecasting metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape_val = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return {'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'MAPE': round(mape_val, 3)}

    def forecast_naive(self) -> ForecastResult:
        """Naive forecast: repeat last observed value."""
        self._require_split()
        h = len(self.test)
        pred = np.array([self.train.iloc[-1]] * h)
        metrics = self._compute_metrics(self.test.values, pred)
        logger.info(f"Naive | MAPE: {metrics['MAPE']:.2f}%")
        return ForecastResult('Naive', pred, metrics)

    def forecast_seasonal_naive(self, season_length: int = 12) -> ForecastResult:
        """Seasonal naive: use last season's values."""
        self._require_split()
        h = len(self.test)
        last_season = self.train.values[-season_length:]
        pred = np.tile(last_season, (h // season_length) + 1)[:h]
        metrics = self._compute_metrics(self.test.values, pred)
        logger.info(f"Seasonal Naive (m={season_length}) | MAPE: {metrics['MAPE']:.2f}%")
        return ForecastResult('Seasonal Naive', pred, metrics,
                              {'season_length': season_length})

    def forecast_holt_winters(
        self,
        trend: str = 'multiplicative',
        seasonal: str = 'multiplicative',
        seasonal_periods: int = 12
    ) -> ForecastResult:
        """Holt-Winters triple exponential smoothing."""
        self._require_split()
        try:
            model = ExponentialSmoothing(
                self.train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            fitted = model.fit(optimized=True)
            pred = fitted.forecast(len(self.test)).values
            metrics = self._compute_metrics(self.test.values, pred)
            params = {
                'alpha': round(fitted.params['smoothing_level'], 4),
                'beta':  round(fitted.params['smoothing_trend'], 4),
                'gamma': round(fitted.params['smoothing_seasonal'], 4),
            }
            logger.info(f"Holt-Winters | MAPE: {metrics['MAPE']:.2f}% | Params: {params}")
            return ForecastResult('Holt-Winters', pred, metrics, params)
        except Exception as e:
            logger.error(f"Holt-Winters failed: {e}")
            raise

    def forecast_sarima(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12)
    ) -> ForecastResult:
        """SARIMA model for seasonal time series."""
        self._require_split()
        try:
            model = SARIMAX(
                self.train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            pred = fitted.forecast(steps=len(self.test)).values
            metrics = self._compute_metrics(self.test.values, pred)
            logger.info(f"SARIMA{order}x{seasonal_order} | "
                        f"MAPE: {metrics['MAPE']:.2f}% | AIC: {fitted.aic:.1f}")
            return ForecastResult(
                f'SARIMA{order}', pred, metrics,
                {'order': order, 'seasonal_order': seasonal_order, 'AIC': fitted.aic}
            )
        except Exception as e:
            logger.error(f"SARIMA failed: {e}")
            raise

    def run_all_models(self) -> Dict[str, ForecastResult]:
        """Run all supported models and return results."""
        self._require_split()
        results = {}
        model_runners = {
            'Naive':          self.forecast_naive,
            'Seasonal Naive': self.forecast_seasonal_naive,
            'Holt-Winters':   self.forecast_holt_winters,
            'SARIMA':         self.forecast_sarima,
        }
        for name, runner in model_runners.items():
            try:
                results[name] = runner()
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
        return results

    def compare_results(self, results: Dict[str, ForecastResult]) -> pd.DataFrame:
        """Create a comparison DataFrame from multiple ForecastResults."""
        rows = []
        for name, result in results.items():
            row = {'Model': name, **result.metrics}
            rows.append(row)
        comparison = pd.DataFrame(rows).set_index('Model')
        comparison = comparison.sort_values('MAPE')

        print("\n" + "=" * 55)
        print("  FORECAST MODEL COMPARISON")
        print("=" * 55)
        print(comparison.to_string())
        print("=" * 55)
        print(f"\n🏆 Best Model (MAPE): {comparison['MAPE'].idxmin()}")
        print(f"   MAPE = {comparison['MAPE'].min():.2f}%")
        return comparison


# ─────────────────────────────────────────────
# RUN THE PRODUCTION PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']

    # Initialize and run
    forecaster = TimeSeriesForecaster(
        data=df,
        target_col='passengers',
        freq='MS'
    )
    forecaster.train_test_split(test_size=24)
    results = forecaster.run_all_models()
    comparison = forecaster.compare_results(results)
```

**Expected Output:**
```
2025-01-01 12:00:00 | INFO | TimeSeriesForecaster initialized | Series: 144 points | Target: passengers | Freq: MS
2025-01-01 12:00:00 | INFO | Split | Train: 120 | Test: 24
2025-01-01 12:00:00 | INFO | Naive | MAPE: 22.43%
2025-01-01 12:00:00 | INFO | Seasonal Naive (m=12) | MAPE: 5.87%
2025-01-01 12:00:00 | INFO | Holt-Winters | MAPE: 3.89% | Params: {'alpha': 0.4211, 'beta': 0.0031, 'gamma': 0.9989}
2025-01-01 12:00:00 | INFO | SARIMA(1, 1, 1)x(1, 1, 1, 12) | MAPE: 4.21% | AIC: 877.3

=======================================================
  FORECAST MODEL COMPARISON
=======================================================
                MAE     RMSE    MAPE
Model
Holt-Winters  16.780  22.110   3.890
SARIMA        18.420  24.650   4.210
Seasonal Naive 25.830 33.410   5.870
Naive         95.210 112.840  22.430
=======================================================

🏆 Best Model (MAPE): Holt-Winters
   MAPE = 3.89%
```

---

## 1.7 The Modern Forecasting Pipeline

Classical methods are just the starting point. A production forecasting system has many additional layers:

```
┌──────────────────────────────────────────────────────────┐
│              MODERN FORECASTING PIPELINE                 │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  DATA       │───►│  FEATURE     │───►│  MODEL     │  │
│  │  INGESTION  │    │  ENGINEERING │    │  TRAINING  │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│       │                    │                   │         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  DATA       │    │  LAG         │    │  BACKTESTING│ │
│  │  VALIDATION │    │  FEATURES    │    │  & EVAL    │  │
│  └─────────────┘    │  ROLLING     │    └────────────┘  │
│                     │  STATISTICS  │           │         │
│                     │  HOLIDAYS    │    ┌────────────┐  │
│                     │  ENCODINGS   │    │  PRODUCTION│  │
│                     └──────────────┘    │  DEPLOYMENT│  │
│                                         └────────────┘  │
│                                                          │
│  CLASSICAL ──► ML ──► FOUNDATION MODELS ──► AGENTS      │
└──────────────────────────────────────────────────────────┘
```

---

## 1.8 Strengths and Limitations of Classical Methods

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| Naive | Simple, fast, interpretable | Ignores patterns, high error |
| Moving Average | Noise reduction | Lagging indicator, no trend/seasonality |
| Exponential Smoothing | Weights recent data, handles trend/season | Limited with complex non-linear patterns |
| ARIMA | Solid statistical foundation, interpretable | Stationarity required, univariate only |
| SARIMA | Handles seasonality | Complex parameter tuning, slow on long series |

**Why we need more**: Classical methods struggle with:
- Multiple seasonalities (hourly data: daily + weekly + annual patterns)
- Non-linear relationships
- External variables (weather, promotions, economic indicators)
- Very high-dimensional data (thousands of SKUs)
- Zero-shot forecasting on new, unseen series

This is where **Foundation Models** and **Agentic AI** enter the picture — which we'll explore throughout the rest of this book.

---

## 1.9 Production Considerations

Before deploying any forecasting system, ask these questions:

| Consideration | Questions to Ask |
|---------------|-----------------|
| **Latency** | Does the forecast need to run in real-time or can it be batched? |
| **Frequency** | How often must forecasts be refreshed? |
| **Scale** | How many series need to be forecasted? |
| **Interpretability** | Do stakeholders need to understand the forecast? |
| **Uncertainty** | Do you need confidence intervals or just point estimates? |
| **Drift** | How will you detect when the model becomes stale? |
| **Monitoring** | How will you measure live forecast accuracy? |

---

## 1.10 Summary

In this chapter, you covered:

- **What forecasting is** and why it creates business value.
- **Types of forecasting problems**: by horizon, variable count, output type, and frequency.
- **Time series decomposition**: trend, seasonality, cyclical, and residual components.
- **Classical forecasting methods**: naive, seasonal naive, Holt-Winters, and SARIMA.
- **A complete Python forecasting pipeline** with evaluation metrics.
- **The broader forecasting landscape** from classical to agentic AI.

The key insight is this: classical methods provide essential baselines and remain competitive in many scenarios, but they have fundamental limitations that motivate the modern AI approaches you'll learn in the rest of this book.

---

## Exercises

### Exercise 1.1 — Data Exploration
Load the M4 Monthly dataset (available at https://github.com/Mcompetitions/M4-methods/tree/master/Dataset). Pick one time series and:
1. Plot the raw series.
2. Perform multiplicative decomposition.
3. Identify the trend direction.
4. Identify the dominant seasonal pattern.

### Exercise 1.2 — Forecast Comparison
Using the airline passengers dataset, implement a **moving average forecast** with window sizes of 3, 6, and 12 months. Compare MAPE for each. Which window size works best for the test period? Why?

### Exercise 1.3 — SARIMA Parameter Selection
Experiment with different SARIMA orders on the airline passengers dataset:
- Try SARIMA(2,1,2)(1,1,1)[12]
- Try SARIMA(0,1,1)(0,1,1)[12] (airline model — classic choice)
- Compare AIC and test MAPE. Which model is best?

### Exercise 1.4 — Production Extension
Extend the `TimeSeriesForecaster` class to:
1. Add a `save_results()` method that exports the comparison table to CSV.
2. Add a `plot_forecast()` method for any `ForecastResult`.
3. Add logging to record the best model automatically.

### Exercise 1.5 — Real Business Dataset
Download the Rossmann Store Sales dataset from Kaggle (https://www.kaggle.com/c/rossmann-store-sales). Pick Store #1 and:
1. Create a daily time series of sales.
2. Decompose it.
3. Apply Holt-Winters and SARIMA.
4. Report MAPE on a 30-day test period.

---

## Interview Questions

**Q1: What is the difference between trend and cyclical components in time series?**

*Trend* refers to the long-term upward or downward direction of the data. *Cyclical* patterns are wave-like fluctuations that repeat over longer, irregular periods (usually more than a year) tied to economic or business cycles rather than fixed calendar periods.

**Q2: When should you use a multiplicative decomposition vs. an additive one?**

Use multiplicative when the magnitude of seasonal swings grows proportionally with the trend level (e.g., retail sales). Use additive when seasonal swings stay roughly constant in absolute size regardless of the trend.

**Q3: What is MAPE and what are its limitations?**

MAPE is Mean Absolute Percentage Error: the average of absolute percentage errors. Its main limitations are: (1) it is undefined when actual values are zero; (2) it is asymmetric — over-forecasts are penalized less than under-forecasts; (3) it tends to favor methods that under-predict.

**Q4: Why is the Seasonal Naive forecast a good baseline?**

It captures the most prominent pattern in many time series — seasonality — without any model complexity. It is hard to beat with simple methods when seasonality is strong. A model that cannot beat Seasonal Naive is not worth deploying.

**Q5: What is the ARIMA(0,1,1) model equivalent to?**

ARIMA(0,1,1) is equivalent to Simple Exponential Smoothing applied to differenced data. The connection between ETS (Exponential Smoothing) and ARIMA families is a deep and important theoretical insight.

**Q6: What is the key limitation of classical methods that motivates using Foundation Models?**

Classical methods (1) require separate model fitting per series (does not scale to thousands of series); (2) cannot leverage patterns learned across related series; (3) cannot incorporate unstructured external signals like news, earnings calls, or economic reports. Foundation Models address all three limitations.

---

## References

1. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
2. Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
3. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. *International Journal of Forecasting*, 34(4), 802–808.
4. Airline Passengers Dataset. Box & Jenkins (1976). *Time Series Analysis: Forecasting and Control*. (Classic dataset widely used in forecasting research.)
5. Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3–73.

---

*Next Chapter: Chapter 2 — Forecasting Metrics: How to Measure Forecast Accuracy*
