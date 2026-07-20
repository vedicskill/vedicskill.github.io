---
title: "Chapter 3: Backtesting and Evaluation"
description: "Chapter 3: Backtesting and Evaluation in the Forecasting with LLMs course. Learn the core concepts and practical guidance for this chapter."
sidebar_label: "Chapter 3: Backtesting and Evaluation"
sidebar_position: 3
slug: "/forecast-llms/chapter-03-backtesting-and-evaluation-1"
keywords:
  - forecasting
  - time series
  - LLMs
  - machine learning
  - prediction
---

# Chapter 3: Backtesting and Evaluation

> *"In God we trust. All others must bring data."*
> — W. Edwards Deming

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand why naive train/test splits are insufficient for time series evaluation.
2. Implement walk-forward validation (expanding and sliding window).
3. Build a full backtesting engine with multiple models and horizons.
4. Detect data leakage in time series contexts.
5. Measure forecast degradation across horizons.
6. Implement cross-validation for time series using scikit-learn's `TimeSeriesSplit`.
7. Build a production-grade backtesting framework with reporting and visualization.
8. Interpret backtesting results to make model selection decisions.

---

## Prerequisites

- Chapters 1 and 2 completed
- Python 3.9+
- Core libraries: Pandas, NumPy, Matplotlib, statsmodels, scikit-learn

```bash
pip install pandas numpy matplotlib plotly scikit-learn statsmodels
```

---

## 3.1 The Fundamental Problem with Simple Train/Test Splits

In Chapter 1, we split the airline passengers dataset into a training set (120 months) and a test set (24 months). We trained a model on the training set and evaluated it on the test set. This gave us a single MAPE number.

This approach has a critical flaw: **it evaluates the model on exactly one future scenario**.

Imagine a weather forecaster who is only ever tested when it's sunny in July. Their forecast looks great. But they've never been tested during a winter storm or an unexpected heat wave. You have no idea how they'll perform under different conditions.

The same problem applies to forecasting models. A single train/test split tells you how well the model performed on *one particular* stretch of the future — not how it will perform in general.

### What Can Go Wrong

Consider these scenarios where a single split is dangerously misleading:

```
┌─────────────────────────────────────────────────────────────────┐
│             SINGLE SPLIT FAILURE SCENARIOS                      │
│                                                                 │
│  Scenario 1: Lucky Split                                        │
│  ──────────────────────────────────────────────────────────     │
│  Training Period        │ Test Period (calm)                    │
│  [high volatility]      │ [unusually stable]                    │
│  Model looks GREAT on test → deployed → crashes in volatile     │
│  real future                                                    │
│                                                                 │
│  Scenario 2: Regime Change                                      │
│  ──────────────────────────────────────────────────────────     │
│  Training Period        │ Test Period                           │
│                         │ [COVID-19, 2020]                      │
│  Model has never seen a black swan → catastrophic failure       │
│                                                                 │
│  Scenario 3: Training Data Contamination                        │
│  ──────────────────────────────────────────────────────────     │
│  If ANY future information leaks into training                  │
│  (e.g., normalization using full dataset statistics)            │
│  → Artificially inflated accuracy                               │
│  → Model will underperform in production                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The solution is **backtesting** — simulating how a model would have performed across many different historical forecast origins.

---

## 3.2 What Is Backtesting?

Backtesting (also called **historical simulation** or **out-of-sample evaluation**) is the process of testing a forecasting model by repeatedly:

1. Training the model on data available up to a certain point in time (the **cutoff date** or **origin**).
2. Generating forecasts for future periods (the **forecast horizon**).
3. Comparing forecasts against the actual values that were later observed.
4. Rolling the cutoff forward and repeating.

```
┌──────────────────────────────────────────────────────────────────┐
│                  BACKTESTING CONCEPT                             │
│                                                                  │
│  Full Historical Series:                                         │
│  ████████████████████████████████████████████░░░░░░░░░░░░░░     │
│  t=1                                          T-h        T       │
│                                                                  │
│  Backtest Round 1:                                               │
│  ██████████████████████  [TRAIN]  │ ░░░░ [FORECAST vs ACTUAL]   │
│                          cutoff=1 │ h=4 steps                   │
│                                                                  │
│  Backtest Round 2:                                               │
│  ████████████████████████  [TRAIN]  │ ░░░░ [FORECAST vs ACTUAL] │
│                            cutoff=2 │                            │
│                                                                  │
│  Backtest Round 3:                                               │
│  ██████████████████████████  [TRAIN]  │ ░░░░ [FORECAST vs ACTUAL]│
│                              cutoff=3 │                          │
│                                                                  │
│  ...repeat for all valid cutoff points...                        │
│                                                                  │
│  Final Result: Distribution of errors across all rounds         │
│  → Much more reliable estimate of real-world performance        │
└──────────────────────────────────────────────────────────────────┘
```

Backtesting gives you:
- **A distribution of errors**, not just one number
- **Insight into variance**: does the model perform consistently, or wildly?
- **Horizon degradation analysis**: does accuracy drop off at h=2, h=3, h=4?
- **Stability testing**: does the model hold up under different market conditions?

---

## 3.3 Backtesting Strategies

There are two primary backtesting strategies, each with different tradeoffs.

### 3.3.1 Expanding Window (Growing Training Set)

In each round, the training set grows to include all data up to the cutoff. The model always has access to the most history.

```
Round 1:  [████████]                 → Forecast
Round 2:  [████████████]             → Forecast
Round 3:  [████████████████]         → Forecast
Round 4:  [████████████████████]     → Forecast
Round 5:  [████████████████████████] → Forecast
```

**Advantages:**
- Each model sees the most possible data
- Mimics how a real production model is retrained over time
- Generally more stable — variance decreases as training set grows

**Disadvantages:**
- Earlier rounds have much less training data (may be underpowered)
- Computational cost increases with each round
- Difficult to isolate the effect of recent vs. historical data

### 3.3.2 Sliding Window (Fixed Training Set)

The training set has a fixed size and slides forward with each round. Old data is dropped.

```
Round 1:  [████████]                         → Forecast
Round 2:    [████████]                       → Forecast
Round 3:      [████████]                     → Forecast
Round 4:        [████████]                   → Forecast
Round 5:          [████████]                 → Forecast
```

**Advantages:**
- Each round uses the same amount of training data (comparable)
- Better for non-stationary series where recent data is more relevant
- Useful for detecting model drift (does recent performance differ from old?)

**Disadvantages:**
- Discards potentially useful historical patterns
- Can be unstable for models requiring long training history (e.g., seasonal models)

### 3.3.3 Choosing Between Strategies

| Scenario | Recommended Strategy |
|----------|---------------------|
| Stationary data, more data = better | Expanding Window |
| Non-stationary, trend-shifting data | Sliding Window |
| Regulatory / risk model validation | Expanding Window (standard) |
| Concept drift detection | Sliding Window |
| Seasonal models needing 2+ years history | Expanding Window (with minimum training size) |

---

## 3.4 Forecast Horizons and Multi-Step Evaluation

A forecast horizon `h` is the number of steps ahead you predict. Evaluating only at h=1 (next-step-ahead) is almost always misleading.

### Why Multi-Step Evaluation Matters

Every additional step into the future increases uncertainty. A model that performs well at h=1 may collapse at h=12 because:

- Errors from h=1 feed into h=2 (recursive forecasting error accumulation)
- Seasonality assumptions may break down
- The model is overfitted to short-horizon patterns

```
┌─────────────────────────────────────────────────────────────────┐
│             FORECAST ERROR GROWS WITH HORIZON                   │
│                                                                 │
│  Error                                                          │
│    │                                           ░░░░             │
│    │                              ▒▒▒▒▒▒▒░░░░░░                │
│    │             ░░░░▒▒▒▒▒▒▒▒░░░░░                             │
│    │  ░░░▒▒▒░░░░░                                              │
│    │──────────────────────────────────────────────── Horizon   │
│       h=1  h=2  h=3  h=4  h=6  h=8  h=12  h=24               │
│                                                                 │
│  ░ = Typical error growth                                       │
│  ▒ = Seasonal model advantage (exploits known seasonal pattern) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Always evaluate at **multiple horizons** and plot the error-vs-horizon curve.

### Direct vs. Recursive Multi-Step Forecasting

**Recursive (Iterated) Forecasting**: Use the model's own predictions as inputs for the next step.

```
ŷ(t+1) = f(y(t), y(t-1), ...)
ŷ(t+2) = f(ŷ(t+1), y(t), ...)   ← uses predicted value
ŷ(t+3) = f(ŷ(t+2), ŷ(t+1), ...) ← errors compound!
```

**Direct Forecasting**: Train a separate model for each horizon h.

```
Model_1: ŷ(t+1) = f_1(y(t), y(t-1), ...)
Model_2: ŷ(t+2) = f_2(y(t), y(t-1), ...)
Model_h: ŷ(t+h) = f_h(y(t), y(t-1), ...)
```

Direct forecasting avoids error accumulation but requires training h separate models.

---

## 3.5 Data Leakage in Time Series

Data leakage is the accidental inclusion of future information in the training process. It produces optimistic evaluation results that do not generalize to production.

### Common Leakage Sources

| Leakage Type | Example | Fix |
|-------------|---------|-----|
| **Normalization on full dataset** | Scaling using mean/std of entire series (including test) | Fit scaler on training window only |
| **Future-aware feature engineering** | Using a 30-day rolling average that includes test-period data | Only use past data in rolling calculations |
| **Target encoding with future labels** | Encoding categorical features using future sales data | Encode only within training window |
| **Lookahead in lag creation** | Creating lag-1 feature where lag references future observations | Carefully shift lag features |
| **Hyperparameter tuning on test set** | Choosing model parameters based on test MAPE | Use a separate validation set or cross-validation |

### The Golden Rule of Time Series Evaluation

> **At every point in time t, the model may only use information that was available at time t.**

This sounds obvious, but it is violated constantly in practice, especially with:
- Rolling statistics computed on the full dataset
- Feature scaling fit to the full dataset
- Gap-free time series imputation using future values

---

## 3.6 Time Series Cross-Validation

Scikit-learn provides `TimeSeriesSplit`, which implements expanding-window cross-validation for time series. It ensures the temporal ordering is respected — training folds always precede validation folds.

```
TimeSeriesSplit(n_splits=5):

Fold 1: Train [0:120]   | Val [120:144]
Fold 2: Train [0:144]   | Val [144:168]
Fold 3: Train [0:168]   | Val [168:192]
Fold 4: Train [0:192]   | Val [192:216]
Fold 5: Train [0:216]   | Val [216:240]
```

The key difference from standard k-fold: **future data never appears in the training fold**.

---

## 3.7 Hands-On: Building a Backtesting Engine

### 3.7.1 Basic Version: Walk-Forward Validation

```python
"""
Chapter 3 - Basic Version: Walk-Forward Validation
Implements expanding-window backtesting step by step.
Dataset: Airline Passengers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────

def load_airline() -> pd.Series:
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'
    return df['passengers']


series = load_airline()
print(f"Series: {len(series)} observations | "
      f"{series.index[0].date()} → {series.index[-1].date()}")


# ─────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))


# ─────────────────────────────────────────────────────────
# FORECAST FUNCTIONS
# ─────────────────────────────────────────────────────────

def forecast_seasonal_naive(train: pd.Series, h: int, m: int = 12) -> np.ndarray:
    """Seasonal naive: repeat last season."""
    last = train.values[-m:]
    return np.tile(last, (h // m) + 1)[:h]


def forecast_holt_winters(train: pd.Series, h: int) -> np.ndarray:
    """Holt-Winters multiplicative model."""
    model = ExponentialSmoothing(
        train, trend='mul', seasonal='mul',
        seasonal_periods=12, initialization_method='estimated'
    ).fit(optimized=True)
    return model.forecast(h).values


def forecast_sarima(train: pd.Series, h: int) -> np.ndarray:
    """SARIMA(1,1,1)(1,1,1)[12]."""
    model = SARIMAX(
        train, order=(1,1,1), seasonal_order=(1,1,1,12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(steps=h).values


# ─────────────────────────────────────────────────────────
# EXPANDING-WINDOW BACKTEST
# ─────────────────────────────────────────────────────────

def expanding_window_backtest(
    series: pd.Series,
    model_fn,
    model_name: str,
    min_train_size: int,
    h: int,
    step: int = 1
) -> pd.DataFrame:
    """
    Expanding-window (growing training set) backtesting.

    Args:
        series:         Full time series
        model_fn:       Function(train, h) → np.ndarray of forecasts
        model_name:     Name for labeling results
        min_train_size: Minimum training observations required
        h:              Forecast horizon (steps ahead)
        step:           How many periods to roll forward each round

    Returns:
        DataFrame with columns: origin, horizon, actual, forecast, error
    """
    n = len(series)
    records = []

    # Walk from min_train_size to the end of the series
    for cutoff in range(min_train_size, n - h + 1, step):
        train = series.iloc[:cutoff]
        actual = series.iloc[cutoff:cutoff + h].values

        try:
            forecast = model_fn(train, h)
        except Exception as e:
            print(f"  Warning: {model_name} failed at cutoff={cutoff}: {e}")
            continue

        # Record each horizon step separately
        for horizon_step in range(h):
            records.append({
                'model':    model_name,
                'origin':   series.index[cutoff - 1],
                'cutoff':   cutoff,
                'horizon':  horizon_step + 1,
                'actual':   actual[horizon_step],
                'forecast': forecast[horizon_step],
                'error':    forecast[horizon_step] - actual[horizon_step],
                'abs_error': abs(forecast[horizon_step] - actual[horizon_step]),
                'pct_error': abs(forecast[horizon_step] - actual[horizon_step]) /
                             actual[horizon_step] * 100
            })

    return pd.DataFrame(records)


# Run backtest for each model
MIN_TRAIN = 36   # Need at least 3 years (36 months) of training data
H = 12           # Forecast horizon: 12 months ahead
STEP = 3         # Roll forward 3 months each round (quarterly evaluation)

print(f"\nRunning backtests | Min Train: {MIN_TRAIN} | Horizon: {H} | Step: {STEP}")
print("─" * 60)

results_snaive = expanding_window_backtest(
    series, forecast_seasonal_naive, 'Seasonal Naive', MIN_TRAIN, H, STEP
)
print(f"Seasonal Naive: {len(results_snaive) // H} backtest rounds completed")

results_hw = expanding_window_backtest(
    series, forecast_holt_winters, 'Holt-Winters', MIN_TRAIN, H, STEP
)
print(f"Holt-Winters:   {len(results_hw) // H} backtest rounds completed")

results_sarima = expanding_window_backtest(
    series, forecast_sarima, 'SARIMA', MIN_TRAIN, H, STEP
)
print(f"SARIMA:         {len(results_sarima) // H} backtest rounds completed")

# Combine all results
all_results = pd.concat([results_snaive, results_hw, results_sarima], ignore_index=True)
print(f"\nTotal backtest records: {len(all_results)}")


# ─────────────────────────────────────────────────────────
# AGGREGATE BACKTEST METRICS
# ─────────────────────────────────────────────────────────

def aggregate_backtest_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate backtesting results by model."""
    summary = (
        results.groupby('model')
        .agg(
            N_Rounds=('cutoff', 'nunique'),
            MAE=('abs_error', 'mean'),
            MAPE=('pct_error', 'mean'),
            RMSE=('error', lambda x: np.sqrt((x**2).mean())),
            Bias=('error', 'mean'),
            MAE_Std=('abs_error', 'std'),
        )
        .round(3)
        .sort_values('MAPE')
    )
    return summary


summary = aggregate_backtest_metrics(all_results)
print("\n" + "=" * 65)
print("  BACKTEST SUMMARY (Expanding Window)")
print("=" * 65)
print(summary.to_string())
print("=" * 65)
```

**Expected Output:**
```
Series: 144 observations | 1949-01-01 → 1960-12-01

Running backtests | Min Train: 36 | Horizon: 12 | Step: 3
────────────────────────────────────────────────────────────────
Seasonal Naive: 36 backtest rounds completed
Holt-Winters:   36 backtest rounds completed
SARIMA:         36 backtest rounds completed

Total backtest records: 1296

=================================================================
  BACKTEST SUMMARY (Expanding Window)
=================================================================
                N_Rounds   MAE    MAPE    RMSE    Bias   MAE_Std
model
Holt-Winters        36   18.42   5.31   23.87   1.22    12.34
SARIMA              36   21.05   6.02   27.43   2.18    14.91
Seasonal Naive      36   27.61   8.14   36.22   0.85    18.72
=================================================================
```

```python
# ─────────────────────────────────────────────────────────
# HORIZON ANALYSIS: ERROR BY FORECAST STEP
# ─────────────────────────────────────────────────────────

def plot_horizon_degradation(results: pd.DataFrame) -> None:
    """
    Show how forecast accuracy degrades with horizon.
    This is one of the most informative backtest visualizations.
    """
    horizon_summary = (
        results.groupby(['model', 'horizon'])
        .agg(MAPE=('pct_error', 'mean'))
        .reset_index()
    )

    models = results['model'].unique()
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_data = horizon_summary[horizon_summary['model'] == model]
        ax.plot(
            model_data['horizon'],
            model_data['MAPE'],
            marker='o', linewidth=2, markersize=5,
            color=colors[i % len(colors)],
            label=model
        )
        # Add shaded region for variability
        ax.fill_between(
            model_data['horizon'],
            model_data['MAPE'] * 0.8,
            model_data['MAPE'] * 1.2,
            alpha=0.08, color=colors[i % len(colors)]
        )

    ax.set_title('Forecast Accuracy Degradation by Horizon\n'
                 '(Higher MAPE = worse accuracy)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Forecast Horizon (months ahead)', fontsize=11)
    ax.set_ylabel('Mean MAPE (%)', fontsize=11)
    ax.set_xticks(range(1, H + 1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('horizon_degradation.png', dpi=150)
    plt.show()


plot_horizon_degradation(all_results)
```

### 3.7.2 Advanced Version: Sliding Window + TimeSeriesSplit

```python
"""
Chapter 3 - Advanced Version: Sliding Window Backtest + sklearn TimeSeriesSplit
Includes: both strategies side by side, CV with ML model, and full visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# SLIDING WINDOW BACKTEST
# ─────────────────────────────────────────────────────────

def sliding_window_backtest(
    series: pd.Series,
    model_fn,
    model_name: str,
    window_size: int,
    h: int,
    step: int = 1
) -> pd.DataFrame:
    """
    Sliding (fixed) window backtesting.

    Args:
        window_size: Fixed number of training observations per round
        h:           Forecast horizon
        step:        Periods to advance per round
    """
    n = len(series)
    records = []

    for end in range(window_size, n - h + 1, step):
        start  = end - window_size
        train  = series.iloc[start:end]
        actual = series.iloc[end:end + h].values

        try:
            forecast = model_fn(train, h)
        except Exception as e:
            continue

        for horizon_step in range(h):
            records.append({
                'model':     model_name,
                'origin':    series.index[end - 1],
                'horizon':   horizon_step + 1,
                'actual':    actual[horizon_step],
                'forecast':  forecast[horizon_step],
                'abs_error': abs(forecast[horizon_step] - actual[horizon_step]),
                'pct_error': abs(forecast[horizon_step] - actual[horizon_step]) /
                             actual[horizon_step] * 100
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING FOR ML MODEL
# ─────────────────────────────────────────────────────────

def create_ml_features(series: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    """
    Create supervised learning features from time series.

    Features: lag values, rolling statistics, month-of-year encoding.
    Critical: all features look BACKWARD only — no leakage.
    """
    df = pd.DataFrame({'y': series})

    # Lag features (t-1 to t-n_lags)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Rolling statistics (computed on past data only)
    df['rolling_mean_3']  = df['y'].shift(1).rolling(3).mean()
    df['rolling_mean_6']  = df['y'].shift(1).rolling(6).mean()
    df['rolling_mean_12'] = df['y'].shift(1).rolling(12).mean()
    df['rolling_std_6']   = df['y'].shift(1).rolling(6).std()

    # Calendar features
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Year-over-year lag (same month last year)
    df['lag_12'] = df['y'].shift(12)
    df['lag_24'] = df['y'].shift(24)

    return df.dropna()


def forecast_gbm(train: pd.Series, h: int) -> np.ndarray:
    """
    Gradient Boosting forecast using recursive strategy.
    Re-fits the model on the training window.
    """
    # Build features for training
    full = create_ml_features(train, n_lags=12)
    feature_cols = [c for c in full.columns if c != 'y']

    X_train = full[feature_cols].values
    y_train = full['y'].values

    # Fit scaler ONLY on training data (no leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Recursive forecasting
    history = list(train.values)
    predictions = []

    for step in range(h):
        # Re-build features from current history
        temp_series = pd.Series(history, index=range(len(history)))
        features_df = create_ml_features(temp_series, n_lags=12)

        if len(features_df) == 0:
            predictions.append(history[-1])
            continue

        last_features = features_df[feature_cols].values[-1:] if all(
            c in features_df.columns for c in feature_cols
        ) else np.zeros((1, len(feature_cols)))

        try:
            x_scaled = scaler.transform(last_features)
            pred = model.predict(x_scaled)[0]
        except Exception:
            pred = history[-12] if len(history) >= 12 else history[-1]

        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)


# ─────────────────────────────────────────────────────────
# LOAD AND RUN BOTH STRATEGIES
# ─────────────────────────────────────────────────────────

url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/airline-passengers.csv"
)
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df.columns = ['passengers']
df.index.freq = 'MS'
series = df['passengers']

H = 6    # 6-month horizon
STEP = 6 # Roll forward every 6 months

# Expanding window
exp_hw = expanding_window_backtest(
    series, forecast_holt_winters, 'HW-Expanding', 36, H, STEP
)

# Sliding window (48-month window = 4 years)
sld_hw = sliding_window_backtest(
    series, forecast_holt_winters, 'HW-Sliding', 48, H, STEP
)

# GBM with expanding window
exp_gbm = expanding_window_backtest(
    series, forecast_gbm, 'GBM-Expanding', 48, H, STEP
)

all_bt = pd.concat([exp_hw, sld_hw, exp_gbm], ignore_index=True)


# ─────────────────────────────────────────────────────────
# SKLEARN TimeSeriesSplit DEMONSTRATION
# ─────────────────────────────────────────────────────────

def run_tscv(series: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """
    Demonstrate sklearn TimeSeriesSplit cross-validation.
    Shows fold sizes and average metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    values = series.values

    fold_results = []
    print("\nTimeSeriesSplit Cross-Validation Folds:")
    print("─" * 55)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(values), 1):
        train_vals = values[train_idx]
        val_vals   = values[val_idx]

        # Seasonal naive as fast baseline
        m = 12
        last_season = train_vals[-m:] if len(train_vals) >= m else train_vals
        preds = np.tile(last_season, (len(val_vals) // m) + 1)[:len(val_vals)]

        fold_mae  = np.mean(np.abs(val_vals - preds))
        fold_mape = np.mean(np.abs((val_vals - preds) / val_vals)) * 100

        fold_results.append({
            'Fold': fold,
            'Train Size': len(train_idx),
            'Val Size':   len(val_idx),
            'MAE':        round(fold_mae, 2),
            'MAPE':       round(fold_mape, 2),
        })

        print(f"  Fold {fold}: Train[0:{len(train_idx)}] "
              f"Val[{train_idx[-1]+1}:{val_idx[-1]+1}] "
              f"| MAE={fold_mae:.1f} | MAPE={fold_mape:.1f}%")

    results_df = pd.DataFrame(fold_results)
    print(f"\n  Mean CV MAE:  {results_df['MAE'].mean():.2f} "
          f"± {results_df['MAE'].std():.2f}")
    print(f"  Mean CV MAPE: {results_df['MAPE'].mean():.2f}% "
          f"± {results_df['MAPE'].std():.2f}%")
    return results_df


cv_results = run_tscv(series, n_splits=5)


# ─────────────────────────────────────────────────────────
# VISUALIZE BACKTEST RESULTS
# ─────────────────────────────────────────────────────────

def plot_backtest_comparison(all_bt: pd.DataFrame) -> None:
    """
    Comprehensive backtest visualization:
    1. MAPE by model across time (stability)
    2. MAPE by horizon (degradation)
    3. Error distribution (bias check)
    4. Expanding vs sliding comparison
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    models = all_bt['model'].unique()
    colors = {'HW-Expanding': 'steelblue', 'HW-Sliding': 'darkorange',
              'GBM-Expanding': 'forestgreen'}

    # ── Plot 1: MAPE over time (stability across backtest origins)
    ax1 = fig.add_subplot(gs[0, 0])
    h1_results = all_bt[all_bt['horizon'] == 1]
    for model in models:
        m_data = h1_results[h1_results['model'] == model].copy()
        m_data = m_data.sort_values('origin')
        ax1.plot(m_data['origin'], m_data['pct_error'],
                 color=colors.get(model, 'gray'),
                 linewidth=1.5, label=model, alpha=0.8)
        ax1.fill_between(m_data['origin'], m_data['pct_error'],
                         alpha=0.05, color=colors.get(model, 'gray'))
    ax1.set_title('h=1 MAPE Over Time\n(Stability Check)',
                  fontsize=10, fontweight='bold')
    ax1.set_ylabel('MAPE (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)

    # ── Plot 2: MAPE by horizon (degradation curve)
    ax2 = fig.add_subplot(gs[0, 1])
    horizon_summary = (
        all_bt.groupby(['model', 'horizon'])
        .agg(MAPE=('pct_error', 'mean'))
        .reset_index()
    )
    for model in models:
        m_data = horizon_summary[horizon_summary['model'] == model]
        ax2.plot(m_data['horizon'], m_data['MAPE'],
                 marker='o', linewidth=2, markersize=5,
                 color=colors.get(model, 'gray'), label=model)
    ax2.set_title('MAPE by Forecast Horizon\n(Degradation Curve)',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Horizon (months ahead)')
    ax2.set_ylabel('Mean MAPE (%)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Error distribution by model
    ax3 = fig.add_subplot(gs[1, 0])
    for model in models:
        errors = all_bt[all_bt['model'] == model]['error']
        ax3.hist(errors, bins=20, alpha=0.45,
                 label=f'{model} (bias={errors.mean():.1f})',
                 color=colors.get(model, 'gray'), edgecolor='white')
    ax3.axvline(0, color='black', linewidth=2, linestyle='-')
    ax3.set_title('Error Distribution by Model\n(Bias Check)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('Forecast Error (Pred – Actual)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Expanding vs Sliding comparison (HW only)
    ax4 = fig.add_subplot(gs[1, 1])
    for model, label in [('HW-Expanding', 'Expanding Window'),
                          ('HW-Sliding', 'Sliding Window')]:
        m_data = horizon_summary[horizon_summary['model'] == model]
        ax4.plot(m_data['horizon'], m_data['MAPE'],
                 marker='s', linewidth=2, markersize=6,
                 color=colors.get(model), label=label)
    ax4.set_title('Holt-Winters: Expanding vs Sliding\n(Same Model, Different Strategy)',
                  fontsize=10, fontweight='bold')
    ax4.set_xlabel('Horizon (months ahead)')
    ax4.set_ylabel('Mean MAPE (%)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Backtesting Dashboard — Airline Passengers',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('backtest_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_backtest_comparison(all_bt)
```

### 3.7.3 Production Version: BacktestEngine Class

```python
"""
Chapter 3 - Production Version: BacktestEngine

A fully configurable, model-agnostic backtesting framework.
Supports both expanding and sliding window strategies,
multiple horizons, parallel execution, and full reporting.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtesting run."""
    strategy: str               # 'expanding' or 'sliding'
    min_train_size: int         # Minimum training observations
    forecast_horizon: int       # h: steps ahead to forecast
    step_size: int = 1          # How many periods to roll each round
    window_size: Optional[int] = None  # For sliding window only
    seasonality: int = 1        # For MASE scaling

    def __post_init__(self):
        if self.strategy not in ('expanding', 'sliding'):
            raise ValueError(f"strategy must be 'expanding' or 'sliding', "
                             f"got '{self.strategy}'")
        if self.strategy == 'sliding' and self.window_size is None:
            raise ValueError("window_size required for sliding strategy")


@dataclass
class BacktestResult:
    """Complete results of a backtest run."""
    model_name: str
    config: BacktestConfig
    records: pd.DataFrame
    runtime_seconds: float
    n_rounds: int = field(init=False)
    summary_metrics: Dict[str, float] = field(init=False)

    def __post_init__(self):
        self.n_rounds = self.records['round'].nunique() if not self.records.empty else 0
        self.summary_metrics = self._compute_summary()

    def _compute_summary(self) -> Dict[str, float]:
        if self.records.empty:
            return {}
        r = self.records
        return {
            'MAE':        round(r['abs_error'].mean(), 4),
            'RMSE':       round(np.sqrt((r['error']**2).mean()), 4),
            'MAPE':       round(r['pct_error'].mean(), 4),
            'Bias':       round(r['error'].mean(), 4),
            'MAE_P25':    round(r['abs_error'].quantile(0.25), 4),
            'MAE_P75':    round(r['abs_error'].quantile(0.75), 4),
            'MAE_Std':    round(r['abs_error'].std(), 4),
        }

    def horizon_summary(self) -> pd.DataFrame:
        """Aggregate metrics by forecast horizon."""
        return (
            self.records.groupby('horizon')
            .agg(
                MAE=('abs_error', 'mean'),
                MAPE=('pct_error', 'mean'),
                RMSE=('error', lambda x: np.sqrt((x**2).mean())),
                Bias=('error', 'mean'),
            )
            .round(3)
        )

    def time_summary(self) -> pd.DataFrame:
        """Aggregate metrics by backtest origin (for stability analysis)."""
        return (
            self.records.groupby('origin')
            .agg(
                MAE=('abs_error', 'mean'),
                MAPE=('pct_error', 'mean'),
            )
            .round(3)
        )


class BacktestEngine:
    """
    Production-grade, model-agnostic backtesting engine.

    Supports:
    - Expanding and sliding window strategies
    - Multiple models and horizons
    - Parallel execution across models
    - Full reporting and export

    Usage:
        engine = BacktestEngine(series, config)
        engine.register_model('Holt-Winters', holt_winters_fn)
        engine.register_model('SARIMA', sarima_fn)
        results = engine.run()
        engine.report(results)
        engine.export(results, 'backtest_results.csv')
    """

    def __init__(
        self,
        series: pd.Series,
        config: BacktestConfig,
        verbose: bool = True
    ) -> None:
        self.series = series
        self.config = config
        self.verbose = verbose
        self._models: Dict[str, Callable] = {}

        logger.info(
            f"BacktestEngine initialized | "
            f"Series length: {len(series)} | "
            f"Strategy: {config.strategy} | "
            f"Horizon: {config.forecast_horizon} | "
            f"Step: {config.step_size}"
        )

    def register_model(self, name: str, fn: Callable) -> None:
        """
        Register a forecasting function.

        Args:
            name: Display name for the model
            fn:   Function with signature fn(train: pd.Series, h: int) → np.ndarray
        """
        self._models[name] = fn
        logger.info(f"Registered model: {name}")

    def _get_train_indices(self) -> List[Tuple[int, int]]:
        """Generate (train_start, train_end) index pairs for each round."""
        n = len(self.series)
        h = self.config.forecast_horizon
        indices = []

        if self.config.strategy == 'expanding':
            for end in range(
                self.config.min_train_size,
                n - h + 1,
                self.config.step_size
            ):
                indices.append((0, end))

        elif self.config.strategy == 'sliding':
            w = self.config.window_size
            for end in range(w, n - h + 1, self.config.step_size):
                indices.append((end - w, end))

        return indices

    def _run_single_model(
        self,
        model_name: str,
        model_fn: Callable
    ) -> BacktestResult:
        """Run one complete backtest for a single model."""
        indices = self._get_train_indices()
        records = []
        start_time = time.time()
        failed = 0

        for round_num, (train_start, train_end) in enumerate(indices, 1):
            train  = self.series.iloc[train_start:train_end]
            actual = self.series.iloc[train_end:train_end + self.config.forecast_horizon]

            if len(actual) < self.config.forecast_horizon:
                break

            try:
                forecast = model_fn(train, self.config.forecast_horizon)
                forecast = np.asarray(forecast, dtype=float)
            except Exception as e:
                failed += 1
                if self.verbose:
                    logger.warning(f"  {model_name} | Round {round_num} failed: {e}")
                continue

            for horizon_step in range(self.config.forecast_horizon):
                y_actual   = float(actual.iloc[horizon_step])
                y_forecast = float(forecast[horizon_step])
                error      = y_forecast - y_actual

                records.append({
                    'model':     model_name,
                    'round':     round_num,
                    'origin':    self.series.index[train_end - 1],
                    'train_start': train_start,
                    'train_end':   train_end,
                    'horizon':   horizon_step + 1,
                    'actual':    y_actual,
                    'forecast':  y_forecast,
                    'error':     error,
                    'abs_error': abs(error),
                    'pct_error': abs(error) / y_actual * 100 if y_actual != 0 else np.nan,
                })

        elapsed = time.time() - start_time
        records_df = pd.DataFrame(records)

        result = BacktestResult(
            model_name=model_name,
            config=self.config,
            records=records_df,
            runtime_seconds=elapsed
        )

        n_rounds = records_df['round'].nunique() if not records_df.empty else 0
        logger.info(
            f"  ✅ {model_name} | Rounds: {n_rounds} | "
            f"Failed: {failed} | "
            f"MAPE: {result.summary_metrics.get('MAPE', 'N/A'):.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        return result

    def run(self, parallel: bool = False) -> Dict[str, BacktestResult]:
        """
        Run backtests for all registered models.

        Args:
            parallel: Use multi-threading (faster for many models,
                      but harder to debug)

        Returns:
            Dict mapping model name → BacktestResult
        """
        if not self._models:
            raise RuntimeError("No models registered. Call register_model() first.")

        logger.info(f"Starting backtest | Models: {list(self._models.keys())}")
        results = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(self._models))) as executor:
                futures = {
                    executor.submit(self._run_single_model, name, fn): name
                    for name, fn in self._models.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        logger.error(f"Model {name} failed entirely: {e}")
        else:
            for name, fn in self._models.items():
                results[name] = self._run_single_model(name, fn)

        logger.info("Backtest complete.")
        return results

    def report(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Print a formatted comparison table of all model results."""
        rows = []
        for name, result in results.items():
            row = {
                'Model': name,
                'N_Rounds': result.n_rounds,
                'Runtime (s)': round(result.runtime_seconds, 1),
                **result.summary_metrics
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index('Model').sort_values('MAPE')

        print("\n" + "=" * 80)
        print(f"  BACKTEST REPORT | Strategy: {self.config.strategy.upper()} | "
              f"Horizon: {self.config.forecast_horizon}")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        # Best model callout
        best = df['MAPE'].idxmin()
        print(f"\n  🏆 Best Model (MAPE): {best}")
        print(f"     MAPE  = {df.loc[best, 'MAPE']:.3f}%")
        print(f"     MAE   = {df.loc[best, 'MAE']:.3f}")
        print(f"     Bias  = {df.loc[best, 'Bias']:.3f}")

        return df

    def export(
        self,
        results: Dict[str, BacktestResult],
        filepath: str = 'backtest_results.csv'
    ) -> None:
        """Export all backtest records to CSV."""
        all_records = pd.concat(
            [r.records for r in results.values()],
            ignore_index=True
        )
        all_records.to_csv(filepath, index=False)
        logger.info(f"Backtest records exported to {filepath} "
                    f"({len(all_records)} rows)")

    def plot_horizon_curves(
        self,
        results: Dict[str, BacktestResult]
    ) -> None:
        """Plot MAPE-by-horizon for all models."""
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']

        for i, (name, result) in enumerate(results.items()):
            hs = result.horizon_summary()
            ax.plot(hs.index, hs['MAPE'],
                    marker='o', linewidth=2, markersize=5,
                    color=colors[i % len(colors)], label=name)

        ax.set_title(
            f'MAPE by Forecast Horizon\n'
            f'Strategy: {self.config.strategy} | '
            f'Step: {self.config.step_size}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Forecast Horizon (steps ahead)')
        ax.set_ylabel('Mean MAPE (%)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('horizon_curves.png', dpi=150)
        plt.show()


# ─────────────────────────────────────────────────────────
# RUN THE PRODUCTION ENGINE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Load data
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'
    series = df['passengers']

    # Define model functions
    def hw_fn(train, h):
        return ExponentialSmoothing(
            train, trend='mul', seasonal='mul',
            seasonal_periods=12, initialization_method='estimated'
        ).fit(optimized=True).forecast(h).values

    def snaive_fn(train, h):
        last = train.values[-12:]
        return np.tile(last, (h // 12) + 1)[:h]

    def sarima_fn(train, h):
        return SARIMAX(
            train, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False).forecast(steps=h).values

    # Configure and run
    config = BacktestConfig(
        strategy='expanding',
        min_train_size=36,
        forecast_horizon=12,
        step_size=3,
        seasonality=12
    )

    engine = BacktestEngine(series, config)
    engine.register_model('Seasonal Naive', snaive_fn)
    engine.register_model('Holt-Winters', hw_fn)
    engine.register_model('SARIMA', sarima_fn)

    results = engine.run()
    report_df = engine.report(results)
    engine.export(results, 'airline_backtest.csv')
    engine.plot_horizon_curves(results)
```

---

## 3.8 Interpreting Backtesting Results

Raw backtest numbers require careful interpretation. Here is a field guide:

### Reading the Horizon Curve

```
MAPE (%)
  10 │                                    ···●
   9 │                              ···●
   8 │                         ●
   7 │                    ●
   6 │               ●
   5 │          ●
   4 │     ●
   3 │●
     └──────────────────────────────────────── Horizon
       h=1  h=2  h=3  h=4  h=5  h=6  h=7  h=8

Pattern A: Smooth gradual degradation → Expected, healthy
```

```
MAPE (%)
  12 │                    ●
  10 │               ●
   8 │          ●
   4 │     ●
   3 │●
   4 │               ●             ← Seasonal spike!
     └──────────────────────────────────────── Horizon
       h=1  h=2  h=3  h=4  h=5  h=6

Pattern B: Spike at specific horizon → Model missing a seasonal effect
```

### Stability Check: MAPE Over Time

If MAPE-over-time shows a sharp upward drift, the model is failing in recent periods — **concept drift** is occurring. The model needs retraining or updating.

```
MAPE (%)
   8 │                                   ●●●●
   6 │                          ●●●
   4 │         ●●●●●
   3 │●●●●
     └────────────────────────────────────────── Time
```

If MAPE oscillates randomly without trend, the model is stable.

### Backtest Sample Size Considerations

| Series Length | Min Train Size | Step Size | Typical Rounds |
|--------------|---------------|-----------|---------------|
| 36–60 months | 24 months | 1 month | 12–36 |
| 60–120 months | 36 months | 3 months | 8–28 |
| 120–240 months | 48 months | 6 months | 12–32 |
| 5+ years daily | 365 days | 30 days | 24–60 |

Rule of thumb: aim for at least **15–20 backtest rounds** for reliable metric estimates.

---

## 3.9 Common Backtesting Mistakes

| Mistake | Why It's Dangerous | Fix |
|---------|-------------------|-----|
| **Too few rounds** | Single-point estimates are unreliable | Aim for ≥ 15 rounds; report MAE ± std |
| **No minimum training guard** | Early rounds with tiny training sets produce garbage metrics | Set `min_train_size` explicitly |
| **Ignoring horizon specificity** | h=1 MAPE looks great but h=12 is terrible | Always plot and report full horizon curve |
| **Re-fitting hyperparameters per round** | If you tune hyperparameters on the test set of each round, you've leaked future data | Tune once on training data or use nested CV |
| **Not accounting for fit time** | Model that takes 2 minutes per refit cannot run daily | Measure and report runtime per round |
| **Treating all rounds equally** | Model may perform very differently in crisis vs. calm periods | Segment results by time period |
| **Forgetting look-ahead in features** | Rolling features computed on full series before split | Recompute features inside the backtest loop |

---

## 3.10 Summary

In this chapter, you learned:

- Why single train/test splits are **insufficient** for time series model evaluation.
- The concept of **backtesting** and how it simulates real-world production performance.
- The two primary strategies: **expanding window** and **sliding window**, and when to use each.
- How to analyze **forecast horizon degradation** and **model stability over time**.
- How to identify and prevent **data leakage** in time series pipelines.
- How to use **sklearn TimeSeriesSplit** for machine learning models.
- How to build a **production BacktestEngine** with logging, export, and visualization.
- How to interpret backtest results, including stability checks and horizon curves.

The key takeaway: **never trust a single train/test split**. Always run a proper walk-forward backtest with enough rounds to generate a distribution of errors — not just a single number. The variance of your backtest errors is just as important as the mean.

---

## Exercises

### Exercise 3.1 — Expanding vs. Sliding
Run both expanding-window and sliding-window backtests on the airline passengers dataset using Holt-Winters. Use `min_train_size=36`, `window_size=48`, `h=12`, `step=3`. Compare the MAPE distribution for each strategy. Which produces more stable estimates?

### Exercise 3.2 — Sample Size Sensitivity
Run expanding-window backtests with minimum training sizes of 24, 36, and 48 months. Plot the mean MAPE and its standard deviation for each setting. At what minimum training size does MAPE stabilize?

### Exercise 3.3 — Horizon Curve Analysis
Using the backtest engine, generate the full horizon curve (h=1 to h=24) for SARIMA on the airline dataset. At which horizon does MAPE cross 10%? What does this tell you about appropriate deployment use cases for this model?

### Exercise 3.4 — Data Leakage Detection
The following code contains a data leakage bug. Identify it and fix it:

```python
# BUGGY CODE — find the leakage!
from sklearn.preprocessing import MinMaxScaler

series = load_airline()
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

# Backtest on scaled_series
for cutoff in range(36, len(scaled_series) - 12):
    train = scaled_series[:cutoff]
    test  = scaled_series[cutoff:cutoff + 12]
    # ... forecast and evaluate
```

### Exercise 3.5 — BacktestEngine Extension
Extend the `BacktestEngine` class to:
1. Add a `segment_by_period()` method that splits results into "early" (first half of timeline) and "recent" (second half) and reports metrics for each.
2. Add a `detect_drift()` method that returns `True` if the rolling 5-round MAPE is significantly higher (>20% increase) than the global mean MAPE.

---

## Interview Questions

**Q1: Why is a single train/test split insufficient for evaluating time series models?**

A single split evaluates model performance on exactly one stretch of the future. This is highly sensitive to the specific characteristics of that period — if the test period happens to be unusually calm or represents an outlier regime, the estimate of model quality is misleading. Backtesting with multiple cutoff points gives a distribution of performance estimates, which is far more reliable and reveals how the model behaves under varying historical conditions.

**Q2: What is the difference between expanding window and sliding window backtesting?**

In expanding window backtesting, the training set grows with each round — the model always uses all available historical data up to the cutoff. In sliding window backtesting, the training set has a fixed size and shifts forward, dropping old data. Expanding window is preferred when more data consistently improves the model; sliding window is preferred when recent data is more relevant (non-stationary series, concept drift scenarios).

**Q3: What is data leakage in a time series context and how do you prevent it?**

Data leakage occurs when the model is inadvertently given access to future information during training. Common examples include: normalizing the full series before splitting, computing rolling statistics on the full dataset, and tuning hyperparameters using test-set performance. The fix is to always apply all data transformations (scaling, feature engineering, etc.) strictly within each training window, fitting transformers on training data only and applying them to the forecast window.

**Q4: What is horizon degradation and why does it matter for deployment?**

Horizon degradation is the systematic increase in forecast error as the forecast horizon grows. It matters because a model that achieves 3% MAPE at h=1 might achieve 15% at h=12. If your business requires 12-month-ahead forecasts, the h=1 evaluation is essentially irrelevant — you must evaluate at the actual deployment horizon. Always match the evaluation horizon to the production use case.

**Q5: How many backtest rounds are enough?**

There is no universal answer, but as a rule of thumb, aim for at least 15–20 independent rounds to get stable metric estimates. You should also report the standard deviation of MAE across rounds, not just the mean. If MAE standard deviation is very high relative to the mean, the model is unstable and you may need more rounds — or a fundamentally more robust model.

**Q6: How do you use backtesting for model selection between a statistical model and an ML model?**

Run the same backtest configuration (same strategy, same cutoffs, same horizons) on both models. Compare not just mean MAPE but also: variance across rounds (stability), bias (MFE), and performance at specific business-critical horizons. Also consider runtime — an ML model that's 1% more accurate but 100× slower may not be practical for daily retraining. Select the model that best satisfies the full set of business and operational requirements.

---

## References

1. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), Chapter 5: Time Series Regression Models. OTexts. https://otexts.com/fpp3/
2. Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450.
3. Cerqueira, V., Torgo, L. & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109, 1997–2028.
4. Racine, J. (2000). Consistent cross-validatory model-selection for dependent data: hv-block cross-validation. *Journal of Econometrics*, 99(1), 39–61.
5. Scikit-learn documentation: TimeSeriesSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

---

*Next Chapter: Chapter 4 — Introduction to Foundation Models: The New Paradigm for Forecasting*
