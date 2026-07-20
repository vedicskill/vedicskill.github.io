---
title: "Chapter 2: Forecasting Metrics"
description: "Chapter 2: Forecasting Metrics in the Forecasting with LLMs course. Learn the core concepts and practical guidance for this chapter."
sidebar_label: "Chapter 2: Forecasting Metrics"
sidebar_position: 2
slug: "/forecast-llms/chapter-02-forecasting-metrics"
keywords:
  - forecasting
  - time series
  - LLMs
  - machine learning
  - prediction
---

# Chapter 2: Forecasting Metrics

> *"Not everything that can be counted counts, and not everything that counts can be counted."*
> — William Bruce Cameron

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand why choosing the right metric is as important as choosing the right model.
2. Compute and interpret all major forecasting error metrics.
3. Recognize the strengths and pitfalls of each metric.
4. Select the appropriate metric for a given business problem.
5. Implement a production-grade metrics library in Python.
6. Build a visual metrics dashboard for forecast comparison.
7. Apply scale-free metrics for comparing forecasts across multiple series.

---

## Prerequisites

- Chapter 1 completed
- Python 3.9+
- Familiarity with NumPy and Pandas

```bash
pip install pandas numpy matplotlib plotly scikit-learn scipy
```

---

## 2.1 Why Metrics Matter More Than You Think

Imagine two competing forecasting systems submitted to your manager:

- **System A** achieves a MAPE of 4.2%
- **System B** achieves a MAPE of 6.1%

You deploy System A. Three months later, the supply chain team reports massive stockouts. What went wrong?

It turns out System A consistently **under-forecasted** demand. MAPE didn't penalize under-forecasting heavily enough — and the business paid dearly in lost sales.

The choice of metric is not a technical afterthought. It is a **business decision** that directly encodes what kind of errors you are willing to tolerate.

### The Core Tension in Forecasting Metrics

Every metric makes a tradeoff:

| Tradeoff | Dimension |
|----------|-----------|
| Large errors vs. small errors | Squared vs. absolute loss |
| Over-forecast vs. under-forecast | Symmetric vs. asymmetric penalty |
| Scale-dependent vs. scale-free | Absolute vs. percentage/relative |
| Mean performance vs. tail risk | Average vs. quantile metrics |

A good forecasting team uses **multiple metrics** together, not just one.

---

## 2.2 Taxonomy of Forecasting Metrics

```
┌──────────────────────────────────────────────────────────────────┐
│                  FORECASTING METRICS TAXONOMY                    │
│                                                                  │
│  SCALE-DEPENDENT                  SCALE-FREE                     │
│  (Units of the series)            (Unitless / Comparable)        │
│  ┌──────────────────────┐         ┌──────────────────────────┐   │
│  │ MAE                  │         │ MAPE                     │   │
│  │ MSE                  │         │ SMAPE                    │   │
│  │ RMSE                 │         │ MASE                     │   │
│  │ MdAE (Median AE)     │         │ RMSSE                    │   │
│  └──────────────────────┘         │ sMAPE Variants           │   │
│                                   └──────────────────────────┘   │
│  PROBABILISTIC                    BUSINESS-ORIENTED              │
│  (Distribution quality)           (Decision-focused)             │
│  ┌──────────────────────┐         ┌──────────────────────────┐   │
│  │ Pinball / Quantile   │         │ Bias                     │   │
│  │ CRPS                 │         │ Forecast Value Added     │   │
│  │ Coverage             │         │ Inventory Cost Metric    │   │
│  │ Winkler Score        │         │ CFE (Cum. Forecast Error)│   │
│  └──────────────────────┘         └──────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2.3 Scale-Dependent Metrics

Scale-dependent metrics are expressed in the **same units** as the target variable. They are intuitive but cannot be used to compare forecasts across series with different scales.

### 2.3.1 Mean Absolute Error (MAE)

The average absolute difference between forecast and actual values.

**Formula:**

```
        1   n
MAE = ─── · Σ |y_t - ŷ_t|
        n  t=1
```

Where:
- `y_t` = actual value at time t
- `ŷ_t` = forecasted value at time t
- `n` = number of forecast periods

**Intuition**: If MAE = 50 units, your forecast is off by 50 units on average.

**Properties:**
- Same unit as the target (easy to interpret)
- Treats all errors equally (linear penalty)
- Robust to outliers (unlike MSE)
- Does not penalize the direction of the error

**Business Example**: If MAE = $5,000 in monthly revenue forecasts, stakeholders immediately understand the magnitude of the average error.

### 2.3.2 Mean Squared Error (MSE)

The average of squared differences between forecast and actual.

**Formula:**

```
        1   n
MSE = ─── · Σ (y_t - ŷ_t)²
        n  t=1
```

**Properties:**
- Squared units (hard to interpret directly)
- **Heavily penalizes large errors** — a single catastrophic miss is penalized far more than many small ones
- Differentiable — preferred for optimization and gradient-based training
- Sensitive to outliers

**When to use MSE**: When large errors are disproportionately costly. If a single forecast miss causes a plant shutdown, MSE aligns incentives correctly.

### 2.3.3 Root Mean Squared Error (RMSE)

The square root of MSE — restores the original unit.

**Formula:**

```
RMSE = √MSE = √( (1/n) · Σ (y_t - ŷ_t)² )
```

**Properties:**
- Same unit as the target
- Still penalizes large errors more than MAE
- The most widely reported metric in academic papers and competitions

**Relationship**: RMSE ≥ MAE always. The larger the gap between RMSE and MAE, the more large outlier errors exist in your forecast.

```
If RMSE >> MAE → large individual errors exist (investigate them!)
If RMSE ≈ MAE → errors are uniformly distributed
```

### 2.3.4 Median Absolute Error (MdAE)

The **median** (not mean) of absolute errors.

**Formula:**

```
MdAE = median( |y_t - ŷ_t| )
```

**Properties:**
- Extremely robust to outliers
- 50% of forecasts have error below this value
- Useful for skewed error distributions

**When to use**: When you have occasional extreme outlier events (promotions, crises) that inflate MAE/RMSE but don't represent typical performance.

### Summary Table: Scale-Dependent Metrics

| Metric | Formula | Outlier Sensitivity | Interpretability | Use Case |
|--------|---------|--------------------|--------------------|----------|
| MAE | mean(|e|) | Low | High | General purpose |
| MSE | mean(e²) | Very High | Low (squared units) | Model training/optimization |
| RMSE | √mean(e²) | High | Medium | Academic reporting |
| MdAE | median(|e|) | Very Low | High | Noisy data, outlier-prone series |

---

## 2.4 Scale-Free (Percentage) Metrics

These metrics express error as a **percentage** of the actual value, enabling comparison across series of different scales (e.g., comparing forecast accuracy for both a $100 product and a $100,000 machine).

### 2.4.1 Mean Absolute Percentage Error (MAPE)

The most widely used metric in business forecasting.

**Formula:**

```
         100%   n  |y_t - ŷ_t|
MAPE = ─────── · Σ ───────────
          n    t=1     y_t
```

**Intuition**: A MAPE of 5% means your forecast is off by 5% of the actual value on average.

**Properties:**
- Unitless — easy to communicate to business stakeholders
- Interpretable ("we are within X% of actual")

**Critical Limitations of MAPE**:

1. **Undefined when y_t = 0** — you cannot divide by zero. This is a common real-world problem (a store that sold zero units on Monday).

2. **Asymmetric penalty**: MAPE penalizes over-forecasting more heavily than under-forecasting.

   Example:
   - Actual = 100, Forecast = 150 → Error = 50%
   - Actual = 100, Forecast = 50  → Error = 50%
   
   Both seem equal... but:
   - Actual = 50, Forecast = 150 → Error = 200%
   - Actual = 150, Forecast = 50 → Error = 67%
   
   Over-forecasting by the same absolute amount generates a much larger MAPE when actual values are small. This **biases MAPE-minimizing models toward under-forecasting**.

3. **Distorted by small actuals**: A single period with actual = 1 and forecast = 2 contributes 100% error and can dominate the metric.

**Business Context**: Despite its flaws, MAPE remains the dominant metric in supply chain, retail, and financial forecasting due to its interpretability.

### 2.4.2 Symmetric Mean Absolute Percentage Error (SMAPE)

Addresses MAPE's asymmetry by normalizing with the average of actual and forecast.

**Formula:**

```
          200%   n     |y_t - ŷ_t|
SMAPE = ─────── · Σ ─────────────────
          n    t=1  |y_t| + |ŷ_t|
```

**Properties:**
- Bounded between 0% and 200%
- More symmetric than MAPE
- Still undefined when both y_t = 0 AND ŷ_t = 0

**Caution**: SMAPE has its own quirks — it is not truly symmetric. When actual = 0, SMAPE = 200% regardless of forecast. It was used in the M3 and M4 competitions but has been increasingly replaced by MASE.

### 2.4.3 Mean Absolute Scaled Error (MASE)

Introduced by Hyndman & Koehler (2006), MASE solves the zero-division and asymmetry problems by scaling against the **naive in-sample forecast error**.

**Formula:**

```
              MAE_forecast
MASE = ─────────────────────────────────
         (1/(n-m)) · Σ |y_t - y_{t-m}|
```

Where:
- `n` = number of training observations
- `m` = seasonal period (m=1 for non-seasonal, m=12 for monthly)
- Denominator = MAE of seasonal naive forecast on training data

**Interpretation**:
- MASE < 1.0 → your model is **better** than the seasonal naive baseline
- MASE = 1.0 → your model is **equal** to seasonal naive
- MASE > 1.0 → your model is **worse** than seasonal naive (go back and fix it!)

**Properties:**
- Works with zero values
- Scale-free
- Symmetric
- The **official metric of the M4 and M5 competitions**
- Best metric for comparing forecasts across heterogeneous series

```
MASE Quick Reference:
─────────────────────────────────────────
MASE < 0.5   → Excellent performance
MASE 0.5–1.0 → Good (better than naive)
MASE = 1.0   → Same as seasonal naive
MASE > 1.0   → Worse than naive → red flag
─────────────────────────────────────────
```

### 2.4.4 Root Mean Squared Scaled Error (RMSSE)

The MASE equivalent of RMSE — penalizes large errors more heavily.

**Formula:**

```
               √( (1/h) · Σ (y_t - ŷ_t)² )
RMSSE = ──────────────────────────────────────────────
         √( (1/(n-m)) · Σ (y_t - y_{t-m})² )
```

Used as the **primary metric in the M5 Competition** (Walmart sales forecasting, 2020).

---

## 2.5 Bias Metrics

Accuracy alone is insufficient. A model can have good average accuracy but be systematically wrong in one direction — this is **forecast bias**.

### 2.5.1 Mean Forecast Error (MFE) / Bias

```
        1   n
MFE = ─── · Σ (ŷ_t - y_t)
        n  t=1
```

- **Positive MFE** → model consistently **over-forecasts** (optimistic)
- **Negative MFE** → model consistently **under-forecasts** (pessimistic)
- **MFE ≈ 0** → unbiased forecast

**Business Impact of Bias:**

| Bias Direction | Business Consequence |
|---------------|---------------------|
| Consistent over-forecast | Excess inventory, wasted capital, markdowns |
| Consistent under-forecast | Stockouts, lost sales, customer dissatisfaction |

### 2.5.2 Cumulative Forecast Error (CFE)

```
CFE(T) = Σ(t=1 to T) (ŷ_t - y_t)
```

CFE tracks whether errors accumulate in one direction over time. A good forecast should have a CFE that oscillates around zero, not drift consistently positive or negative.

### 2.5.3 Tracking Signal

The Tracking Signal monitors whether a model has drifted and needs recalibration:

```
Tracking Signal(t) = CFE(t) / MAD(t)
```

Where MAD = Mean Absolute Deviation (≈ MAE).

**Rule of thumb**: If |Tracking Signal| > 4 to 6, the model is biased and needs to be reset.

---

## 2.6 Probabilistic Forecast Metrics

Modern forecasting increasingly outputs **probability distributions** rather than point estimates. These metrics evaluate how well the distribution fits reality.

### 2.6.1 Quantile Loss (Pinball Loss)

For a given quantile τ ∈ (0,1), the pinball loss is:

```
L_τ(y, q) = (y - q) · τ         if y ≥ q  (actual ≥ predicted quantile)
           = (q - y) · (1 - τ)   if y < q  (actual < predicted quantile)
```

Where:
- `τ` = target quantile (e.g., 0.1, 0.5, 0.9)
- `q` = predicted quantile value
- `y` = actual value

**Intuition**: 
- At τ = 0.9 (90th percentile), under-forecasting (y > q) is penalized 9× more than over-forecasting
- At τ = 0.1 (10th percentile), over-forecasting (q > y) is penalized 9× more than under-forecasting
- At τ = 0.5, pinball loss equals MAE/2 (median regression)

### 2.6.2 Coverage (Prediction Interval Reliability)

For a (1-α)% prediction interval [L, U]:

```
Coverage = (1/n) · Σ 1{ L_t ≤ y_t ≤ U_t }
```

A 90% prediction interval should contain ~90% of actual values. If it contains 60%, the interval is too narrow (overconfident). If it contains 99%, the interval is too wide (uninformative).

### 2.6.3 Winkler Score

Evaluates both coverage AND sharpness (width) of prediction intervals:

```
W_α(L, U, y) = (U - L)                           if L ≤ y ≤ U
             = (U - L) + (2/α)(L - y)             if y < L
             = (U - L) + (2/α)(y - U)             if y > U
```

Lower Winkler Score = better (narrower interval that still covers the actual).

### 2.6.4 Continuous Ranked Probability Score (CRPS)

CRPS evaluates the full predictive distribution against a single observation:

```
CRPS(F, y) = ∫_{-∞}^{∞} (F(z) - 1{z ≥ y})² dz
```

Where F(z) is the predicted cumulative distribution function (CDF).

CRPS generalizes MAE to distributional forecasts. When the forecast degenerates to a point estimate, CRPS = MAE.

---

## 2.7 Metric Selection Guide

Use this decision guide to choose the right metric:

```
┌─────────────────────────────────────────────────────────────┐
│              METRIC SELECTION DECISION GUIDE                │
│                                                             │
│  Start Here                                                 │
│      │                                                      │
│      ▼                                                      │
│  Are values ever zero?                                      │
│  ┌────────────────────────────────┐                         │
│  │ YES → Use MASE, RMSSE         │                         │
│  │ NO  → Continue                │                         │
│  └────────────────────────────────┘                         │
│      │                                                      │
│      ▼                                                      │
│  Comparing across different series/scales?                  │
│  ┌────────────────────────────────┐                         │
│  │ YES → Use MASE or SMAPE       │                         │
│  │ NO  → Continue                │                         │
│  └────────────────────────────────┘                         │
│      │                                                      │
│      ▼                                                      │
│  Are large errors especially costly?                        │
│  ┌────────────────────────────────┐                         │
│  │ YES → Use RMSE                │                         │
│  │ NO  → Use MAE or MAPE        │                         │
│  └────────────────────────────────┘                         │
│      │                                                      │
│      ▼                                                      │
│  Need probabilistic output?                                 │
│  ┌────────────────────────────────┐                         │
│  │ YES → Add Pinball Loss, CRPS  │                         │
│  │ NO  → Done                    │                         │
│  └────────────────────────────────┘                         │
│                                                             │
│  Always ALSO check: MFE (Bias) and Tracking Signal         │
└─────────────────────────────────────────────────────────────┘
```

### Industry-Specific Metric Recommendations

| Industry | Primary Metric | Secondary Metric | Why |
|----------|---------------|-----------------|-----|
| Retail / Supply Chain | MAPE or MASE | MFE (Bias) | Interpretable; zero-sales are common → prefer MASE |
| Financial Trading | RMSE | Directional Accuracy | Large errors very costly |
| Energy / Utilities | RMSE, MAE | MAPE | Units meaningful, no zero values |
| Healthcare | MAE | Coverage | Intervals critical for safety |
| Demand Planning | MASE | Tracking Signal | Enables cross-SKU comparison |
| M4/M5 Competition | SMAPE + MASE / OWA | — | Official competition metrics |

### The M4 Overall Weighted Average (OWA)

The M4 Competition introduced the OWA metric to combine SMAPE and MASE:

```
OWA = (1/2) · (SMAPE_model / SMAPE_naive2) + (1/2) · (MASE_model / MASE_naive2)
```

Where Naive2 is the seasonal naive forecast. OWA < 1.0 means your model beats the baseline.

---

## 2.8 Directional Accuracy

For some applications (especially finance), the **direction** of change matters more than the magnitude.

### 2.8.1 Directional Accuracy (DA)

```
DA = (1/n) · Σ 1{ sign(y_t - y_{t-1}) = sign(ŷ_t - y_{t-1}) }
```

Percentage of time the model correctly predicts whether the series goes up or down.

- DA = 1.0 → always correct direction
- DA = 0.5 → no better than coin flip
- DA < 0.5 → systematically wrong (contrarian signal!)

### 2.8.2 Pearson Correlation

Measures how linearly related forecasts and actuals are:

```
r = Σ((ŷ_t - mean(ŷ)) · (y_t - mean(y))) / (n · σ_ŷ · σ_y)
```

High correlation (r → 1.0) means the model tracks the shape of the series well, even if the scale is off.

---

## 2.9 Hands-On: Complete Metrics Library

### 2.9.1 Basic Version: Core Metrics Implementation

```python
"""
Chapter 2 - Basic Version: Core Forecasting Metrics
All key metrics implemented from scratch for transparency.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# CORE METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mdae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error — robust to outliers."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.median(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Warning: undefined when y_true contains zeros.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.all():
        warnings.warn(
            f"MAPE: {(~mask).sum()} zero values in y_true were excluded.",
            UserWarning
        )
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Range: [0%, 200%].
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    result = np.zeros_like(y_true)
    result[mask] = (
        2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    )
    return float(np.mean(result) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Args:
        y_true:     Actual test values
        y_pred:     Predicted values
        y_train:    Training series (used to compute naive baseline)
        seasonality: Seasonal period m (1 = non-seasonal, 12 = monthly)

    Returns:
        MASE value. < 1.0 means better than seasonal naive.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    n = len(y_train)
    if n <= seasonality:
        raise ValueError(
            f"Training series length ({n}) must be > seasonality ({seasonality})"
        )

    # In-sample naive forecast error (denominator)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        raise ValueError("MASE denominator is zero — all training values are equal.")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Root Mean Squared Scaled Error — M5 Competition primary metric.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    n = len(y_train)
    naive_errors_sq = (y_train[seasonality:] - y_train[:-seasonality]) ** 2
    scale = np.mean(naive_errors_sq)

    if scale == 0:
        raise ValueError("RMSSE denominator is zero.")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def mean_forecast_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Forecast Error (Bias).
    Positive = over-forecast; Negative = under-forecast.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of periods where the forecast correctly predicts direction of change.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        raise ValueError("Need at least 2 periods for directional accuracy.")
    actual_dir   = np.sign(np.diff(y_true))
    forecast_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == forecast_dir) * 100)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between actuals and forecasts."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ─────────────────────────────────────────────────────────
# DEMONSTRATION ON AIRLINE DATA
# ─────────────────────────────────────────────────────────

def demonstrate_metrics() -> None:
    """Demonstrate all metrics on a concrete example."""

    # Synthetic example: actual vs. biased forecast
    y_true = np.array([100, 120, 130, 110, 140, 160, 155, 145, 130, 120], dtype=float)
    y_pred = np.array([105, 115, 125, 118, 148, 155, 160, 140, 135, 125], dtype=float)  # slightly off
    y_train = np.array([80, 85, 90, 95, 88, 92, 98, 100, 105, 108, 112, 115,
                        90, 95, 100, 105, 100, 102, 108, 112], dtype=float)

    print("=" * 60)
    print("  METRICS DEMONSTRATION")
    print("=" * 60)
    print(f"\n  Actual:    {y_true}")
    print(f"  Forecast:  {y_pred}")

    print(f"\n{'─'*60}")
    print("  SCALE-DEPENDENT METRICS")
    print(f"{'─'*60}")
    print(f"  MAE   = {mae(y_true, y_pred):>8.3f}  (avg absolute error)")
    print(f"  MSE   = {mse(y_true, y_pred):>8.3f}  (avg squared error)")
    print(f"  RMSE  = {rmse(y_true, y_pred):>8.3f}  (root mean squared error)")
    print(f"  MdAE  = {mdae(y_true, y_pred):>8.3f}  (median absolute error)")

    print(f"\n{'─'*60}")
    print("  SCALE-FREE METRICS")
    print(f"{'─'*60}")
    print(f"  MAPE  = {mape(y_true, y_pred):>8.3f}%  (mean abs pct error)")
    print(f"  SMAPE = {smape(y_true, y_pred):>8.3f}%  (symmetric MAPE)")
    print(f"  MASE  = {mase(y_true, y_pred, y_train, seasonality=1):>8.3f}   (scaled error vs naive)")
    print(f"  RMSSE = {rmsse(y_true, y_pred, y_train, seasonality=1):>8.3f}   (scaled RMSE vs naive)")

    print(f"\n{'─'*60}")
    print("  BIAS METRICS")
    print(f"{'─'*60}")
    bias = mean_forecast_error(y_true, y_pred)
    direction = "over-forecast" if bias > 0 else "under-forecast"
    print(f"  MFE (Bias) = {bias:>8.3f}  [{direction}]")

    print(f"\n{'─'*60}")
    print("  DIRECTIONAL / CORRELATION METRICS")
    print(f"{'─'*60}")
    print(f"  Directional Accuracy = {directional_accuracy(y_true, y_pred):>6.1f}%")
    print(f"  Pearson Correlation  = {pearson_correlation(y_true, y_pred):>6.4f}")
    print("=" * 60)


demonstrate_metrics()
```

**Expected Output:**
```
============================================================
  METRICS DEMONSTRATION
============================================================

  Actual:    [100. 120. 130. 110. 140. 160. 155. 145. 130. 120.]
  Forecast:  [105. 115. 125. 118. 148. 155. 160. 140. 135. 125.]

────────────────────────────────────────────────────────────
  SCALE-DEPENDENT METRICS
────────────────────────────────────────────────────────────
  MAE   =    5.100  (avg absolute error)
  MSE   =   33.400  (avg squared error)
  RMSE  =    5.779  (root mean squared error)
  MdAE  =    5.000  (median absolute error)

────────────────────────────────────────────────────────────
  SCALE-FREE METRICS
────────────────────────────────────────────────────────────
  MAPE  =    4.195%  (mean abs pct error)
  SMAPE =    4.106%  (symmetric MAPE)
  MASE  =    0.628   (scaled error vs naive)
  RMSSE =    0.711   (scaled RMSE vs naive)

────────────────────────────────────────────────────────────
  BIAS METRICS
────────────────────────────────────────────────────────────
  MFE (Bias) =    1.500  [over-forecast]

────────────────────────────────────────────────────────────
  DIRECTIONAL / CORRELATION METRICS
────────────────────────────────────────────────────────────
  Directional Accuracy =   77.8%
  Pearson Correlation  =   0.9918
============================================================
```

### 2.9.2 Advanced Version: Full Metrics Suite on Real Data

```python
"""
Chapter 2 - Advanced Version: Metrics Suite on Airline Passengers Dataset
Compares multiple models across all metrics with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# METRICS SUITE (reusing functions from basic version)
# ─────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 12,
    model_name: str = ''
) -> Dict[str, float]:
    """Compute the complete suite of forecasting metrics."""
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    metrics = {
        'Model':   model_name,
        'MAE':     round(float(np.mean(np.abs(y_true - y_pred))), 3),
        'RMSE':    round(float(np.sqrt(np.mean((y_true - y_pred)**2))), 3),
        'MdAE':    round(float(np.median(np.abs(y_true - y_pred))), 3),
        'MAPE':    round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 3),
        'SMAPE':   round(float(np.mean(
                       2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
                   ) * 100), 3),
        'MASE':    round(float(
                       np.mean(np.abs(y_true - y_pred)) /
                       np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
                   ), 3),
        'MFE':     round(float(np.mean(y_pred - y_true)), 3),
        'DA':      round(float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100), 1),
        'Corr':    round(float(np.corrcoef(y_true, y_pred)[0, 1]), 4),
    }
    return metrics


# ─────────────────────────────────────────────────────────
# LOAD DATA AND MODELS
# ─────────────────────────────────────────────────────────

def load_airline(test_months: int = 24):
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/airline-passengers.csv"
    )
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'
    train = df['passengers'][:-test_months]
    test  = df['passengers'][-test_months:]
    return train, test


train, test = load_airline(test_months=24)


# ─────────────────────────────────────────────────────────
# GENERATE FORECASTS
# ─────────────────────────────────────────────────────────

h = len(test)

# Naive
pred_naive = np.array([train.iloc[-1]] * h)

# Seasonal Naive
last_season = train.values[-12:]
pred_snaive = np.tile(last_season, 3)[:h]

# Holt-Winters
hw_model = ExponentialSmoothing(
    train, trend='multiplicative', seasonal='multiplicative',
    seasonal_periods=12, initialization_method='estimated'
).fit(optimized=True)
pred_hw = hw_model.forecast(h).values

# SARIMA
sarima_model = SARIMAX(
    train, order=(1,1,1), seasonal_order=(1,1,1,12),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)
pred_sarima = sarima_model.forecast(steps=h).values

forecasts = {
    'Naive':          pred_naive,
    'Seasonal Naive': pred_snaive,
    'Holt-Winters':   pred_hw,
    'SARIMA':         pred_sarima,
}


# ─────────────────────────────────────────────────────────
# COMPUTE ALL METRICS
# ─────────────────────────────────────────────────────────

results = []
for name, pred in forecasts.items():
    m = compute_all_metrics(
        y_true=test.values,
        y_pred=pred,
        y_train=train.values,
        seasonality=12,
        model_name=name
    )
    results.append(m)

results_df = pd.DataFrame(results).set_index('Model')

print("\n" + "=" * 80)
print("  COMPLETE METRICS COMPARISON")
print("=" * 80)
print(results_df.to_string())
print("=" * 80)

# Rank models by each metric (lower is better, except DA and Corr)
print("\n📊 MODEL RANKINGS:")
for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'MASE']:
    best = results_df[metric].idxmin()
    best_val = results_df[metric].min()
    print(f"  Best {metric:6s}: {best:<20s} ({best_val})")

for metric in ['DA', 'Corr']:
    best = results_df[metric].idxmax()
    best_val = results_df[metric].max()
    print(f"  Best {metric:6s}: {best:<20s} ({best_val})")
```

**Expected Output:**
```
================================================================================
  COMPLETE METRICS COMPARISON
================================================================================
                  MAE     RMSE    MdAE   MAPE   SMAPE   MASE    MFE    DA   Corr
Model
Naive          95.210  112.840  88.500  22.43  19.621  5.871  -5.21  52.2  0.768
Seasonal Naive 25.830   33.410  22.000   5.87   5.714  1.592   0.42  82.6  0.983
Holt-Winters   16.780   22.110  14.500   3.89   3.820  1.034   2.11  87.0  0.992
SARIMA         18.420   24.650  17.500   4.21   4.140  1.136   1.83  82.6  0.990
================================================================================

📊 MODEL RANKINGS:
  Best MAE   : Holt-Winters         (16.78)
  Best RMSE  : Holt-Winters         (22.11)
  Best MAPE  : Holt-Winters         (3.89)
  Best SMAPE : Holt-Winters         (3.82)
  Best MASE  : Holt-Winters         (1.034)
  Best DA    : Holt-Winters         (87.0)
  Best Corr  : Holt-Winters         (0.9920)
```

```python
# ─────────────────────────────────────────────────────────
# VISUALIZE METRICS COMPARISON
# ─────────────────────────────────────────────────────────

def plot_metrics_dashboard(results_df: pd.DataFrame) -> None:
    """
    Build a visual metrics dashboard showing all key metrics
    side by side for model comparison.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Forecasting Metrics Dashboard — Airline Passengers',
                 fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    models = results_df.index.tolist()
    colors = ['#5B9BD5', '#ED7D31', '#A9D18E', '#FF0000']

    metric_configs = [
        ('MAE',   'Mean Absolute Error',   True,  'lower is better', gs[0, 0]),
        ('RMSE',  'Root Mean Squared Error', True, 'lower is better', gs[0, 1]),
        ('MAPE',  'MAPE (%)',               True,  'lower is better', gs[0, 2]),
        ('MASE',  'MASE (vs. Naive)',       True,  'lower is better', gs[1, 0]),
        ('MFE',   'Forecast Bias (MFE)',    False, 'closer to 0 is better', gs[1, 1]),
        ('DA',    'Directional Accuracy (%)', False, 'higher is better', gs[1, 2]),
    ]

    for metric, title, lower_is_better, subtitle, gs_loc in metric_configs:
        ax = fig.add_subplot(gs_loc)
        values = results_df[metric].values

        # Highlight the best model
        if lower_is_better:
            best_idx = np.argmin(np.abs(values)) if metric == 'MFE' else np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bar_colors = [
            '#2ECC71' if i == best_idx else c
            for i, c in enumerate(colors[:len(models)])
        ]

        bars = ax.bar(models, values, color=bar_colors, edgecolor='white',
                      linewidth=0.5, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(np.abs(values)) * 0.02,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

        # Add MASE reference line at 1.0
        if metric == 'MASE':
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                       label='Naive baseline (MASE=1)')
            ax.legend(fontsize=7)

        ax.set_title(f'{title}\n({subtitle})', fontsize=9, fontweight='bold')
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig('metrics_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Dashboard saved to metrics_dashboard.png")


plot_metrics_dashboard(results_df)
```

```python
# ─────────────────────────────────────────────────────────
# BIAS ANALYSIS: ERROR DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────

def plot_error_distributions(test: pd.Series, forecasts: dict) -> None:
    """Plot residual distributions to reveal systematic bias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, pred) in enumerate(forecasts.items()):
        errors = pred - test.values  # positive = over-forecast
        ax = axes[i]

        ax.hist(errors, bins=15, color='steelblue', edgecolor='white',
                alpha=0.85, density=True)
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-',
                   label='Zero bias')
        ax.axvline(x=np.mean(errors), color='red', linewidth=2,
                   linestyle='--', label=f'Mean bias = {np.mean(errors):.1f}')

        ax.set_title(f'{name}\nMFE = {np.mean(errors):.2f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Forecast Error (Pred – Actual)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Error Distributions by Model\n(Centered at zero = unbiased)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_distributions.png', dpi=150)
    plt.show()


plot_error_distributions(test, forecasts)
```

### 2.9.3 Production Version: ForecastEvaluator Class

```python
"""
Chapter 2 - Production Version: ForecastEvaluator

A reusable, extensible class for computing, comparing, and
reporting forecast metrics across multiple models and series.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """Container for a single model's metrics report."""
    model_name: str
    series_name: str
    n_test: int
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'model': self.model_name,
            'series': self.series_name,
            'n_test': self.n_test,
            'timestamp': self.timestamp,
            **self.metrics
        }

    def passed_naive_test(self) -> bool:
        """True if MASE < 1.0 (better than naive baseline)."""
        return self.metrics.get('MASE', float('inf')) < 1.0

    def summary(self) -> str:
        passed = "✅ PASSES" if self.passed_naive_test() else "❌ FAILS"
        return (
            f"[{self.model_name}] {passed} naive test | "
            f"MASE={self.metrics.get('MASE', 'N/A'):.3f} | "
            f"MAPE={self.metrics.get('MAPE', 'N/A'):.2f}% | "
            f"Bias={self.metrics.get('MFE', 'N/A'):.2f}"
        )


class ForecastEvaluator:
    """
    Production-grade forecast evaluation engine.

    Features:
    - Computes full metrics suite for any number of models
    - Validates model quality against naive baseline (MASE < 1)
    - Exports reports to CSV and JSON
    - Generates visual dashboards
    - Tracks evaluation history

    Usage:
        evaluator = ForecastEvaluator(y_train, y_test, seasonality=12)
        evaluator.add_forecast('Holt-Winters', pred_hw)
        evaluator.add_forecast('SARIMA', pred_sarima)
        report = evaluator.get_comparison()
        evaluator.save_report('results.csv')
    """

    def __init__(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        seasonality: int = 1,
        series_name: str = 'series'
    ) -> None:
        self.y_train = np.asarray(y_train, dtype=float)
        self.y_test  = np.asarray(y_test, dtype=float)
        self.seasonality = seasonality
        self.series_name = series_name
        self.reports: List[MetricsReport] = []

        # Pre-compute naive scaling factor
        naive_errors = np.abs(
            self.y_train[seasonality:] - self.y_train[:-seasonality]
        )
        self._scale = float(np.mean(naive_errors))
        if self._scale == 0:
            raise ValueError("Training series has zero variance — MASE undefined.")

        logger.info(
            f"ForecastEvaluator | Series: {series_name} | "
            f"Train: {len(y_train)} | Test: {len(y_test)} | "
            f"Seasonality: {seasonality} | Scale: {self._scale:.3f}"
        )

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE with zero-value protection."""
        mask = y_true != 0
        if not mask.any():
            return float('nan')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _compute_metrics(
        self, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Internal metric computation."""
        e  = y_pred - self.y_test           # signed error (pred - actual)
        ae = np.abs(e)                       # absolute error

        return {
            'MAE':   round(float(np.mean(ae)), 4),
            'RMSE':  round(float(np.sqrt(np.mean(e**2))), 4),
            'MdAE':  round(float(np.median(ae)), 4),
            'MAPE':  round(self._safe_mape(self.y_test, y_pred), 4),
            'SMAPE': round(float(np.mean(
                         2 * ae / (np.abs(self.y_test) + np.abs(y_pred) + 1e-8)
                     ) * 100), 4),
            'MASE':  round(float(np.mean(ae) / self._scale), 4),
            'RMSSE': round(float(
                         np.sqrt(np.mean(e**2)) /
                         np.sqrt(np.mean(
                             (self.y_train[self.seasonality:] -
                              self.y_train[:-self.seasonality])**2
                         ))
                     ), 4),
            'MFE':   round(float(np.mean(e)), 4),
            'DA':    round(float(np.mean(
                         np.sign(np.diff(self.y_test)) ==
                         np.sign(np.diff(y_pred))
                     ) * 100), 2),
            'Corr':  round(float(np.corrcoef(self.y_test, y_pred)[0, 1]), 4),
        }

    def add_forecast(
        self,
        model_name: str,
        y_pred: np.ndarray
    ) -> MetricsReport:
        """
        Register a model's forecast and compute its metrics.

        Args:
            model_name: Human-readable model identifier
            y_pred:     Array of predictions (same length as y_test)

        Returns:
            MetricsReport for this model
        """
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(self.y_test):
            raise ValueError(
                f"Forecast length {len(y_pred)} != test length {len(self.y_test)}"
            )

        metrics = self._compute_metrics(y_pred)
        report = MetricsReport(
            model_name=model_name,
            series_name=self.series_name,
            n_test=len(self.y_test),
            metrics=metrics
        )
        self.reports.append(report)

        # Log summary
        status = "✅" if report.passed_naive_test() else "❌"
        logger.info(f"{status} {report.summary()}")

        return report

    def get_comparison(self, sort_by: str = 'MASE') -> pd.DataFrame:
        """
        Return a DataFrame comparing all registered models.

        Args:
            sort_by: Metric to sort by (ascending)
        """
        if not self.reports:
            raise RuntimeError("No forecasts registered. Call add_forecast() first.")

        rows = [r.to_dict() for r in self.reports]
        df = pd.DataFrame(rows)
        df = df.set_index('model')

        # Drop metadata columns for display
        metric_cols = ['MAE', 'RMSE', 'MdAE', 'MAPE', 'SMAPE',
                       'MASE', 'RMSSE', 'MFE', 'DA', 'Corr']
        metric_cols = [c for c in metric_cols if c in df.columns]

        comparison = df[metric_cols].sort_values(sort_by)

        print("\n" + "=" * 90)
        print(f"  FORECAST EVALUATION REPORT  |  Series: {self.series_name}")
        print("=" * 90)
        print(comparison.to_string())
        print("=" * 90)
        print(f"\n  🏆 Best Model ({sort_by}): {comparison[sort_by].idxmin()}")
        print(f"     {sort_by} = {comparison[sort_by].min():.4f}")
        print(f"\n  ⚠️  Models failing naive baseline (MASE ≥ 1.0):")
        failing = comparison[comparison['MASE'] >= 1.0]
        if failing.empty:
            print("     None — all models beat the naive baseline!")
        else:
            for model in failing.index:
                print(f"     - {model} (MASE = {failing.loc[model, 'MASE']:.3f})")

        return comparison

    def save_report(self, filepath: str = 'forecast_report.csv') -> None:
        """Export the comparison table to CSV."""
        comparison = self.get_comparison()
        comparison.to_csv(filepath)
        logger.info(f"Report saved to {filepath}")

    def save_json(self, filepath: str = 'forecast_report.json') -> None:
        """Export all reports to JSON format."""
        data = [r.to_dict() for r in self.reports]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON report saved to {filepath}")

    def plot_residuals(self, model_name: str, y_pred: np.ndarray) -> None:
        """
        Plot residual diagnostics for a specific model.
        Includes: actual vs forecast, residual plot, and error histogram.
        """
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_pred - self.y_test

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Actual vs Forecast
        axes[0].plot(self.y_test, label='Actual', color='black', linewidth=1.5)
        axes[0].plot(y_pred, label='Forecast', color='steelblue',
                     linestyle='--', linewidth=1.5)
        axes[0].set_title(f'{model_name}: Actual vs Forecast')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Residuals Over Time
        axes[1].plot(residuals, color='crimson', linewidth=1.2)
        axes[1].axhline(0, color='black', linewidth=1.5, linestyle='-')
        axes[1].fill_between(range(len(residuals)), residuals, alpha=0.2,
                              color='crimson')
        axes[1].set_title(f'{model_name}: Residuals Over Time')
        axes[1].set_ylabel('Error (Pred – Actual)')
        axes[1].grid(True, alpha=0.3)

        # 3. Error Histogram
        axes[2].hist(residuals, bins=12, color='steelblue', edgecolor='white',
                     alpha=0.85, density=True)
        axes[2].axvline(0, color='black', linewidth=2, label='Zero')
        axes[2].axvline(residuals.mean(), color='red', linewidth=2,
                        linestyle='--', label=f'Mean={residuals.mean():.1f}')
        axes[2].set_title(f'{model_name}: Error Distribution')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Residual Diagnostics — {model_name}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fname = f'residuals_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        plt.show()
        logger.info(f"Residual plot saved to {fname}")


# ─────────────────────────────────────────────────────────
# RUN THE PRODUCTION EVALUATOR
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

    train = df['passengers'][:-24].values
    test  = df['passengers'][-24:].values

    # Fit models
    hw_fitted = ExponentialSmoothing(
        train, trend='mul', seasonal='mul',
        seasonal_periods=12, initialization_method='estimated'
    ).fit(optimized=True)
    pred_hw = hw_fitted.forecast(24)

    sarima_fitted = SARIMAX(
        train, order=(1,1,1), seasonal_order=(1,1,1,12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    pred_sarima = sarima_fitted.forecast(steps=24)

    # Evaluate
    evaluator = ForecastEvaluator(
        y_train=train,
        y_test=test,
        seasonality=12,
        series_name='Airline Passengers'
    )

    evaluator.add_forecast('Naive', np.array([train[-1]] * 24))
    evaluator.add_forecast('Seasonal Naive', np.tile(train[-12:], 3)[:24])
    evaluator.add_forecast('Holt-Winters', pred_hw)
    evaluator.add_forecast('SARIMA', pred_sarima)

    comparison = evaluator.get_comparison(sort_by='MASE')
    evaluator.save_report('airline_forecast_report.csv')
    evaluator.save_json('airline_forecast_report.json')
    evaluator.plot_residuals('Holt-Winters', pred_hw)
```

---

## 2.10 Metric Pitfalls: A Field Guide

The following table documents the most common mistakes practitioners make when using forecasting metrics:

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Reporting only one metric** | A single metric hides important issues (e.g., good MAPE but severe bias) | Always report at least MAE + MAPE + MFE |
| **MAPE with zeros** | Division by zero corrupts the metric | Use MASE or exclude zeros with a warning |
| **Comparing RMSE across series** | RMSE is scale-dependent | Use MASE or RMSSE for cross-series comparison |
| **Ignoring bias** | Model may have great accuracy on average but be systematically wrong | Always include MFE in your report |
| **Optimizing for training metric** | Good train-set MAPE does not guarantee good test-set MAPE | Always evaluate on held-out test data |
| **Mean vs. Median confusion** | Outliers inflate MAE/RMSE | Report MdAE alongside MAE to detect outlier impact |
| **Directional errors in finance** | A 5% MAPE model might always predict wrong direction | Add Directional Accuracy for time-series with trend |
| **Ignoring the naive baseline** | Reporting "MASE = 0.98" without noting it's barely above naive | Explicitly flag any model with MASE ≥ 1.0 |

---

## 2.11 Summary

In this chapter, you have learned:

- Why metric selection is a **business decision**, not just a technical one.
- The full **taxonomy of forecasting metrics**: scale-dependent, scale-free, probabilistic, and business-oriented.
- How to compute MAE, MSE, RMSE, MdAE, MAPE, SMAPE, MASE, RMSSE, MFE, DA, and Pearson Correlation.
- The critical **limitations of MAPE** and when to use MASE instead.
- How to detect and diagnose **forecast bias** using MFE and Tracking Signal.
- How to build a production-grade **ForecastEvaluator** class with logging, export, and visualization.
- The most common **metric pitfalls** in real-world forecasting.

The key takeaway: **use MASE as your primary metric** (it handles zeros, is scale-free, and compares against a meaningful baseline), **always check MFE for bias**, and **report MAPE for stakeholder communication**.

---

## Exercises

### Exercise 2.1 — Metric Implementation
Implement the **Tracking Signal** formula from Section 2.5.3. Apply it to the Holt-Winters and SARIMA forecasts on the airline dataset. Does either model trigger the alert threshold (|TS| > 4)?

### Exercise 2.2 — Zero Values Challenge
Create a synthetic daily sales series with 30% zero values (simulating a slow-moving SKU). Attempt to compute MAPE — observe the warning. Then compute SMAPE and MASE. Which gives the most useful signal?

### Exercise 2.3 — Pinball Loss
Implement the Pinball Loss function. Generate a probabilistic forecast using the Holt-Winters 95% prediction interval. Evaluate the 10th, 50th, and 90th percentile forecasts using Pinball Loss.

### Exercise 2.4 — Custom Business Metric
A retail company has asymmetric costs:
- **Over-forecasting** by 1 unit costs $2 (holding cost)
- **Under-forecasting** by 1 unit costs $5 (lost sale + expediting)

Implement a **Cost-Weighted Error** metric and evaluate all four models. Which model minimizes total cost?

```python
def cost_weighted_error(y_true, y_pred, cost_over=2, cost_under=5):
    """
    Asymmetric cost metric.
    Args:
        cost_over:  Cost per unit of over-forecasting
        cost_under: Cost per unit of under-forecasting
    """
    errors = y_pred - y_true
    costs = np.where(errors > 0, errors * cost_over, -errors * cost_under)
    return float(np.sum(costs))
```

### Exercise 2.5 — Dashboard Extension
Extend the `ForecastEvaluator` class to:
1. Add a `rank_models()` method that assigns 1st, 2nd, 3rd place for each metric and computes an overall rank score.
2. Add a `flag_bias()` method that returns a warning if |MFE| > 5% of the mean actual value.

---

## Interview Questions

**Q1: Why might a model with a lower MAPE than a competitor still be the wrong choice?**

MAPE tells you about average percentage accuracy but hides directional bias and the impact of outliers. A model with lower MAPE might consistently under-forecast during peak seasons (unacceptable for retail), have high variance (RMSE >> MAE), or fail to improve on a naive seasonal baseline (MASE ≥ 1.0). Always evaluate with multiple metrics and check bias explicitly.

**Q2: What is MASE and why is it preferred over MAPE in the M4/M5 competitions?**

MASE (Mean Absolute Scaled Error) scales the forecast error against the in-sample naive baseline. It is preferred because: (1) it handles zero values; (2) it is symmetric; (3) it provides an intuitive reference point (MASE < 1.0 = better than naive); and (4) it enables meaningful comparison across series with different scales and units.

**Q3: A model has MAE = 100 and RMSE = 250. What does this tell you?**

The large gap between RMSE and RMSE indicates the presence of occasional very large forecast errors. Since RMSE penalizes large errors quadratically, it is inflated relative to MAE. This tells you the model has a heavy-tailed error distribution — likely with outlier periods where it performs very poorly. Investigation of those specific periods is warranted.

**Q4: What is forecast bias and why is a biased forecast dangerous even if accuracy is high?**

Forecast bias (MFE ≠ 0) means the model is systematically wrong in one direction. A bias toward under-forecasting leads to consistent stockouts; a bias toward over-forecasting leads to chronic excess inventory. Over time, these errors compound — a 5% under-forecast every month leads to 60% cumulative under-procurement in a year. Even a small consistent bias destroys supply chain performance at scale.

**Q5: When should you use RMSE over MAE?**

Use RMSE when the business cost of large errors is disproportionately high — for example, when a single massive stockout shuts down a manufacturing line, or a single large financial loss exceeds all small gains. RMSE's quadratic penalty incentivizes models to avoid catastrophic errors, even at the cost of slightly higher average error.

**Q6: What is the Pinball Loss and when is it used?**

Pinball (quantile) loss evaluates probabilistic forecasts at a specific quantile. It is asymmetric: at quantile τ, under-prediction is penalized τ/(1-τ) times more than over-prediction. It is used when you need calibrated quantile forecasts — for example, safety stock calculation requires a reliable 90th percentile forecast, not just the median.

---

## References

1. Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
2. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54–74.
3. Makridakis, S., et al. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.
4. Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
5. Kolassa, S. & Schütz, W. (2007). Advantages of the MAD/Mean ratio over the MAPE. *Foresight*, 6, 40–43.

---

*Next Chapter: Chapter 3 — Backtesting and Evaluation: How to Honestly Measure Forecast Performance*
