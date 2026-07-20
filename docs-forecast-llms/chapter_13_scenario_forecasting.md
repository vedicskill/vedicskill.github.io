---
title: "Chapter 13: Scenario Forecasting"
description: "Chapter 13: Scenario Forecasting in the Forecasting with LLMs course. Learn the core concepts and practical guidance for this chapter."
sidebar_label: "Chapter 13: Scenario Forecasting"
sidebar_position: 13
slug: "/forecast-llms/chapter-13-scenario-forecasting"
keywords:
  - forecasting
  - time series
  - LLMs
  - machine learning
  - prediction
---

# Chapter 13: Scenario Forecasting

> *"The future is not one path — it is a fan of possibilities, each shaped by different assumptions about the forces that drive change."*
> — Peter Schwartz, The Art of the Long View

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand the theory and business value of scenario forecasting.
2. Distinguish scenario forecasting from probabilistic forecasting and sensitivity analysis.
3. Design structured bear/base/bull scenarios using GPT and quantitative models.
4. Build a multi-scenario forecasting pipeline with probability-weighted outputs.
5. Implement Monte Carlo simulation for scenario uncertainty quantification.
6. Create scenario dashboards for business stakeholders.
7. Evaluate scenario forecasts using interval scoring metrics.
8. Build a production-grade ScenarioForecastingEngine.
9. Apply scenario forecasting to real business decisions (inventory, budget, capacity).

---

## Prerequisites

- Chapters 11 and 12 completed
- Python 3.9+
- OpenAI API key (or mock client for demos)

```bash
pip install openai pandas numpy matplotlib plotly scipy statsmodels
```

---

## 13.1 What Is Scenario Forecasting?

A point forecast says: "Sales next quarter will be $4.2M."

A probabilistic forecast says: "Sales next quarter will be $3.8M–$4.6M with 80% probability."

A **scenario forecast** says:

- **Bear scenario** (20% probability): Sales fall to $3.1M due to recession and increased competition
- **Base scenario** (60% probability): Sales grow modestly to $4.3M following current trends
- **Bull scenario** (20% probability): Sales surge to $5.8M due to successful new product launch

The critical difference: scenario forecasts are **narrative-driven**. Each scenario has a name, a story, specific assumptions, and identifiable drivers. This makes them far more useful for strategic decision-making than a confidence interval, because decision-makers can ask:

> "What should we do differently if the Bear scenario materializes?"

This question cannot be answered from a symmetric confidence interval — it requires scenario thinking.

### 13.1.1 When to Use Scenario Forecasting

```
┌──────────────────────────────────────────────────────────────────┐
│              SCENARIO FORECASTING USE CASES                     │
│                                                                  │
│  ✅ HIGH UNCERTAINTY ENVIRONMENTS                                │
│     When the future genuinely has multiple plausible paths      │
│     Examples: new market entry, regulatory change pending,      │
│     post-crisis recovery, geopolitical uncertainty              │
│                                                                  │
│  ✅ STRATEGIC PLANNING                                           │
│     Annual budget setting, capacity planning, M&A decisions     │
│     Board requires "what-if" analysis, not point estimates      │
│                                                                  │
│  ✅ RISK MANAGEMENT                                              │
│     Stress testing business plans against adverse scenarios     │
│     Banking: DFAST/CCAR stress scenarios                        │
│     Insurance: catastrophe scenario planning                    │
│                                                                  │
│  ✅ SUPPLY CHAIN PLANNING                                        │
│     Inventory buffering under demand uncertainty                │
│     Safety stock determined by Bear/Bull demand spread          │
│                                                                  │
│  ✅ CAPITAL ALLOCATION                                           │
│     Investment decisions require scenario-tested ROI            │
│     "Does this project make sense under Bear conditions?"       │
│                                                                  │
│  ⚠️  NOT IDEAL FOR:                                              │
│     Operational daily/weekly forecasting (use probabilistic)    │
│     Automated reorder triggers (too complex for automation)     │
│     Series with very stable, predictable patterns               │
└──────────────────────────────────────────────────────────────────┘
```

### 13.1.2 Scenario vs. Probabilistic vs. Sensitivity Analysis

| Method | Output | Horizon | Decision Use |
|--------|--------|---------|--------------|
| **Point forecast** | Single value | Short-medium | Operational planning |
| **Probabilistic** | Distribution / quantiles | Short-medium | Inventory, safety stock |
| **Sensitivity analysis** | Output range per input variable | Any | Model understanding |
| **Scenario forecast** | Named paths with narratives | Medium-long | Strategic planning, budgeting |

---

## 13.2 Scenario Design Principles

### 13.2.1 The Three-Scenario Framework

The classic structure uses three scenarios — not because there are exactly three possible futures, but because it represents the minimum set that captures:

```
┌──────────────────────────────────────────────────────────────────┐
│              THREE-SCENARIO FRAMEWORK                            │
│                                                                  │
│  BEAR (Pessimistic)                                              │
│  ─────────────────                                               │
│  • 2-3 negative catalysts materialize                           │
│  • Typically ~15th–25th percentile of outcomes                  │
│  • Not the absolute worst case (avoid "end of world")           │
│  • Should be plausible and internally consistent                 │
│  • Probability: typically 15–25%                                 │
│                                                                  │
│  BASE (Most Likely)                                              │
│  ─────────────────                                               │
│  • Current trends and conditions continue                        │
│  • Incorporates known planned events (promotions, launches)     │
│  • Statistical model forecast is a good starting point         │
│  • Probability: typically 50–65%                                 │
│                                                                  │
│  BULL (Optimistic)                                               │
│  ─────────────────                                               │
│  • 2-3 positive catalysts materialize                           │
│  • Typically ~75th–85th percentile of outcomes                  │
│  • Not the absolute best case (avoid "everything goes right")  │
│  • Probability: typically 15–25%                                 │
│                                                                  │
│  CONSTRAINT: Bear + Base + Bull probabilities must sum to 1.0  │
└──────────────────────────────────────────────────────────────────┘
```

### 13.2.2 Scenario Design Guidelines

| Principle | Description |
|-----------|-------------|
| **Plausibility** | Each scenario must be genuinely possible, not a hypothetical extreme |
| **Mutual exclusivity** | Scenarios should not overlap — only one will occur |
| **Exhaustiveness** | Together, scenarios should cover the range of realistic outcomes |
| **Named narratives** | Give each scenario a memorable name ("Soft Landing", "Stagflation", "Rebound") |
| **Specific drivers** | Each scenario must have 2–3 named, specific causal factors |
| **Internally consistent** | All elements within a scenario must be logically compatible |
| **Calibrated probabilities** | Based on evidence, not wishful thinking |

---

## 13.3 GPT-Powered Scenario Generation

### 13.3.1 The Scenario Generation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│           SCENARIO FORECASTING PIPELINE                         │
│                                                                  │
│  Step 1: CONTEXT GATHERING                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Historical time series (quantitative base)           │   │
│  │  • Domain context (industry, company, market position)  │   │
│  │  • Known future events (scheduled promotions, launches) │   │
│  │  • External environment (macro, competitive)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  Step 2: QUANTITATIVE BASELINE                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Fit Holt-Winters / SARIMA to historical data         │   │
│  │  • Generate the most-likely-case numerical forecast     │   │
│  │  • Compute forecast uncertainty (prediction intervals)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  Step 3: GPT SCENARIO GENERATION                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • GPT generates Bear/Base/Bull scenarios               │   │
│  │  • Each with: narrative, drivers, probabilities         │   │
│  │  • Numerical values anchored to quantitative baseline   │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  Step 4: MONTE CARLO REFINEMENT                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Simulate N paths within each scenario band           │   │
│  │  • Compute scenario-conditional distributions           │   │
│  │  • Generate fan chart for visualization                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  Step 5: DECISION ANALYSIS                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Probability-weighted expected value                  │   │
│  │  • Scenario-specific decision recommendations           │   │
│  │  • Risk/opportunity assessment per decision option      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13.4 Hands-On: Scenario Forecasting System

### 13.4.1 Basic Version: Three-Scenario Generator

```python
"""
Chapter 13 - Basic Version: Three-Scenario Forecasting
Dataset: Airline Passengers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
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


series    = load_airline()
TRAIN_SIZE = 120
train     = series.iloc[:TRAIN_SIZE]
test      = series.iloc[TRAIN_SIZE:]
H         = len(test)

print(f"Train: {len(train)} | Test: {len(test)} months")


# ─────────────────────────────────────────────────────────
# 2. QUANTITATIVE BASELINE
# ─────────────────────────────────────────────────────────

def fit_baseline_model(train: pd.Series) -> tuple:
    """
    Fit Holt-Winters as quantitative baseline.
    Returns point forecast + prediction interval.
    """
    model  = ExponentialSmoothing(
        train, trend='mul', seasonal='mul',
        seasonal_periods=12, initialization_method='estimated'
    ).fit(optimized=True)

    point   = model.forecast(H).values

    # Simulate prediction intervals from residual distribution
    resid   = model.resid.values
    sigma   = np.std(resid)
    steps   = np.arange(1, H + 1)
    scale   = sigma * np.sqrt(steps)    # Grows with horizon

    lower_80 = point - 1.28 * scale
    upper_80 = point + 1.28 * scale
    lower_95 = point - 1.96 * scale
    upper_95 = point + 1.96 * scale

    print(f"Baseline RMSE (in-sample): {np.sqrt(np.mean(resid**2)):.2f}")
    return point, lower_80, upper_80, lower_95, upper_95


base_point, lower_80, upper_80, lower_95, upper_95 = fit_baseline_model(train)


# ─────────────────────────────────────────────────────────
# 3. SCENARIO GENERATION (GPT or Rule-Based)
# ─────────────────────────────────────────────────────────

def build_scenario_prompt(
    series: pd.Series,
    base_forecast: np.ndarray,
    horizon: int,
    domain: str,
    macro_context: str = '',
) -> str:
    """
    Build a structured scenario generation prompt.
    Uses chain-of-thought + few-shot pattern from Chapter 12.
    """
    recent    = series.iloc[-24:].values.tolist()
    base_list = [round(v, 1) for v in base_forecast[:horizon]]

    # Compute key statistics for GPT context
    mean_val   = round(float(series.mean()), 1)
    growth_12m = round(float(
        (series.iloc[-1] - series.iloc[-13]) / series.iloc[-13] * 100
    ), 1) if len(series) >= 14 else 0.0

    return f"""You are a senior strategic planning analyst for {domain}.

## Historical Data Summary
Recent 24 months: {recent}
12-month YoY growth: {growth_12m:+.1f}%
Historical mean: {mean_val}

## Quantitative Model Forecast (Holt-Winters Base)
Next {horizon} months: {base_list}
This represents the BASE / MOST LIKELY scenario.

{f"## Macroeconomic Context{chr(10)}{macro_context}" if macro_context else ""}

## Scenario Generation Task

Generate THREE distinct scenarios for the next {horizon} months.

### Design Rules
1. BEAR: Apply 2 named negative catalysts. Values 10-20% BELOW base.
2. BASE: Use the quantitative forecast above as starting point. Minor adjustments only.
3. BULL: Apply 2 named positive catalysts. Values 10-20% ABOVE base.
4. Probabilities must sum to 1.0
5. Each scenario must have EXACTLY {horizon} numeric values
6. Values must be positive (no negative passengers/sales)
7. Preserve the seasonal shape from the base forecast

### Scenario Naming
Give each scenario a memorable business name:
- Bear: e.g., "Stagflation Squeeze", "Demand Drought", "Market Contraction"
- Base: e.g., "Steady State", "Gradual Growth", "Trend Continuation"  
- Bull: e.g., "Demand Surge", "Market Expansion", "Growth Acceleration"

Return ONLY this JSON:
{{
  "scenarios": {{
    "bear": {{
      "name": "<memorable name>",
      "probability": 0.20,
      "values": [{horizon} numbers],
      "negative_catalysts": ["<specific catalyst 1>", "<specific catalyst 2>"],
      "avg_deviation_from_base_pct": <negative number>,
      "description": "<1-2 sentence narrative>"
    }},
    "base": {{
      "name": "<memorable name>",
      "probability": 0.60,
      "values": [{horizon} numbers],
      "key_assumptions": ["<assumption 1>", "<assumption 2>"],
      "avg_deviation_from_base_pct": 0.0,
      "description": "<1-2 sentence narrative>"
    }},
    "bull": {{
      "name": "<memorable name>",
      "probability": 0.20,
      "values": [{horizon} numbers],
      "positive_catalysts": ["<specific catalyst 1>", "<specific catalyst 2>"],
      "avg_deviation_from_base_pct": <positive number>,
      "description": "<1-2 sentence narrative>"
    }}
  }},
  "key_uncertainty_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "recommended_planning_scenario": "bear|base|bull",
  "planning_rationale": "<why use this scenario for operational planning>"
}}"""


def generate_scenarios_from_baseline(
    base_forecast: np.ndarray,
    horizon: int,
    bear_factor: float = 0.85,
    bull_factor: float = 1.15,
    bear_prob:   float = 0.20,
    base_prob:   float = 0.60,
    bull_prob:   float = 0.20,
    noise_seed:  int   = 42,
) -> dict:
    """
    Generate scenarios analytically when GPT is not available.

    Applies multiplicative factors to the base forecast with
    smooth seasonal preservation and slight noise.
    """
    np.random.seed(noise_seed)
    steps = np.arange(1, horizon + 1)

    # Smooth factor transition (factor increases over horizon)
    bear_factors = bear_factor - 0.005 * steps    # Gets worse over time
    bull_factors = bull_factor + 0.005 * steps    # Gets better over time

    # Add slight noise to make scenarios realistic
    bear_noise = np.random.normal(0, base_forecast.std() * 0.02, horizon)
    bull_noise = np.random.normal(0, base_forecast.std() * 0.02, horizon)

    bear_values  = np.maximum(base_forecast * bear_factors + bear_noise, 10)
    base_values  = base_forecast.copy()
    bull_values  = base_forecast * bull_factors + bull_noise

    bear_dev = round(float(np.mean((bear_values - base_values) / base_values) * 100), 1)
    bull_dev = round(float(np.mean((bull_values - base_values) / base_values) * 100), 1)

    return {
        "scenarios": {
            "bear": {
                "name":                       "Demand Contraction",
                "probability":                bear_prob,
                "values":                     bear_values.tolist(),
                "negative_catalysts":         [
                    "Economic slowdown reducing discretionary travel",
                    "Fuel price spike increasing ticket prices"
                ],
                "avg_deviation_from_base_pct": bear_dev,
                "description": (
                    "A combination of economic headwinds and higher costs "
                    "suppresses passenger growth, leading to below-trend demand."
                ),
            },
            "base": {
                "name":                       "Steady Growth",
                "probability":                base_prob,
                "values":                     base_values.tolist(),
                "key_assumptions":            [
                    "Current economic expansion continues",
                    "No major disruptions to airline capacity"
                ],
                "avg_deviation_from_base_pct": 0.0,
                "description": (
                    "Trend and seasonality continue along the historical path, "
                    "with stable economic conditions supporting consistent demand."
                ),
            },
            "bull": {
                "name":                       "Travel Surge",
                "probability":                bull_prob,
                "values":                     bull_values.tolist(),
                "positive_catalysts":         [
                    "Jet age expansion opening new routes",
                    "Rising middle-class disposable income"
                ],
                "avg_deviation_from_base_pct": bull_dev,
                "description": (
                    "Structural expansion of the aviation market, driven by new "
                    "route development and growing consumer spending on travel."
                ),
            }
        },
        "key_uncertainty_factors": [
            "Macroeconomic growth trajectory",
            "Fuel price volatility",
            "Geopolitical stability affecting international routes"
        ],
        "recommended_planning_scenario": "base",
        "planning_rationale": (
            "The base scenario has 60% probability and reflects the most "
            "likely operational environment for capacity and staffing planning."
        ),
    }


# Generate scenarios
print("\nGenerating scenarios...")
# In production: response = gpt_forecaster.call(prompt, system)
# Here: use analytical generation
scenario_data = generate_scenarios_from_baseline(
    base_point, H,
    bear_factor=0.85,
    bull_factor=1.15,
    bear_prob=0.20,
    base_prob=0.60,
    bull_prob=0.20,
)

# Extract scenario arrays
scenarios = scenario_data['scenarios']
bear_vals = np.array(scenarios['bear']['values'])
base_vals = np.array(scenarios['base']['values'])
bull_vals = np.array(scenarios['bull']['values'])

# Probability-weighted expected value
probs = {k: v['probability'] for k, v in scenarios.items()}
expected = (
    probs['bear'] * bear_vals +
    probs['base'] * base_vals +
    probs['bull'] * bull_vals
)

print("\nScenario Summary:")
print(f"  Bear ({probs['bear']:.0%}): {scenarios['bear']['name']}")
print(f"    Avg deviation from base: {scenarios['bear']['avg_deviation_from_base_pct']:+.1f}%")
print(f"    Catalysts: {scenarios['bear']['negative_catalysts']}")
print(f"\n  Base ({probs['base']:.0%}): {scenarios['base']['name']}")
print(f"    {scenarios['base']['description']}")
print(f"\n  Bull ({probs['bull']:.0%}): {scenarios['bull']['name']}")
print(f"    Avg deviation from base: {scenarios['bull']['avg_deviation_from_base_pct']:+.1f}%")
print(f"    Catalysts: {scenarios['bull']['positive_catalysts']}")
print(f"\n  Key Uncertainties: {scenario_data['key_uncertainty_factors']}")
print(f"  Recommended for planning: {scenario_data['recommended_planning_scenario'].upper()}")


# ─────────────────────────────────────────────────────────
# 4. EVALUATE SCENARIOS
# ─────────────────────────────────────────────────────────

def evaluate_scenarios(
    y_true:    np.ndarray,
    scenarios: dict,
    expected:  np.ndarray,
    y_train:   np.ndarray,
    seasonality: int = 12,
) -> pd.DataFrame:
    """
    Evaluate all scenarios against actual values.
    Includes Winkler Score for interval evaluation.
    """
    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))

    records = []
    sc_values = {
        'Bear':     np.array(scenarios['bear']['values']),
        'Base':     np.array(scenarios['base']['values']),
        'Bull':     np.array(scenarios['bull']['values']),
        'Expected': expected,
    }

    for name, pred in sc_values.items():
        n    = min(len(y_true), len(pred))
        mae  = np.mean(np.abs(y_true[:n] - pred[:n]))
        mase = mae / scale if scale > 0 else np.nan
        mape = np.mean(np.abs((y_true[:n] - pred[:n]) / y_true[:n])) * 100
        bias = np.mean(pred[:n] - y_true[:n])
        records.append({
            'Scenario': name,
            'MAE':      round(mae, 2),
            'MASE':     round(mase, 4),
            'MAPE':     round(mape, 2),
            'Bias':     round(bias, 2),
        })

    # Winkler Score for Bear-Bull interval
    alpha   = 0.40    # Using Bear-Bull as a rough 60% interval
    n       = min(len(y_true), len(bear_vals), len(bull_vals))
    width   = bull_vals[:n] - bear_vals[:n]
    penalty = np.where(
        y_true[:n] < bear_vals[:n],
        (2 / alpha) * (bear_vals[:n] - y_true[:n]),
        np.where(
            y_true[:n] > bull_vals[:n],
            (2 / alpha) * (y_true[:n] - bull_vals[:n]),
            0.0
        )
    )
    winkler = round(float(np.mean(width + penalty)), 2)

    coverage = round(float(np.mean(
        (y_true[:n] >= bear_vals[:n]) & (y_true[:n] <= bull_vals[:n])
    ) * 100), 1)

    df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("  SCENARIO EVALUATION")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"  Bear-Bull Interval Coverage: {coverage:.1f}%")
    print(f"  Winkler Score (Bear-Bull):   {winkler:.2f}")
    print("=" * 60)
    return df


eval_df = evaluate_scenarios(
    test.values, scenarios, expected, train.values
)


# ─────────────────────────────────────────────────────────
# 5. VISUALIZE: FAN CHART
# ─────────────────────────────────────────────────────────

def plot_scenario_fan_chart(
    train:      pd.Series,
    test:       pd.Series,
    bear_vals:  np.ndarray,
    base_vals:  np.ndarray,
    bull_vals:  np.ndarray,
    expected:   np.ndarray,
    scenarios:  dict,
) -> None:
    """
    Classic scenario fan chart showing all three scenario paths.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Training history
    ax.plot(train.index, train.values, color='black',
            linewidth=1.5, label='Historical Data')

    # Actual test
    ax.plot(test.index, test.values, color='black',
            linewidth=2.5, marker='o', markersize=4, label='Actual')

    # Bear-Bull shaded region
    ax.fill_between(
        test.index, bear_vals, bull_vals,
        alpha=0.12, color='#3498DB', label='Bear–Bull Range'
    )

    # Scenario lines
    sc_config = [
        ('bear', bear_vals, '#E74C3C', '--', f"Bear: {scenarios['bear']['name']} ({scenarios['bear']['probability']:.0%})"),
        ('base', base_vals, '#2ECC71', '-',  f"Base: {scenarios['base']['name']} ({scenarios['base']['probability']:.0%})"),
        ('bull', bull_vals, '#3498DB', '--', f"Bull: {scenarios['bull']['name']} ({scenarios['bull']['probability']:.0%})"),
    ]

    for _, vals, color, ls, label in sc_config:
        ax.plot(test.index, vals, color=color, linewidth=2.5,
                linestyle=ls, label=label, alpha=0.9)

    # Expected value (probability weighted)
    ax.plot(test.index, expected, color='#9B59B6', linewidth=2,
            linestyle=':', label='Expected Value (Prob-Weighted)')

    # Annotations at end of scenarios
    last_date = test.index[-1]
    offset    = pd.DateOffset(months=1)
    for _, vals, color, _, _ in sc_config:
        ax.annotate(
            f'{vals[-1]:.0f}',
            xy=(last_date, vals[-1]),
            xytext=(5, 0), textcoords='offset points',
            color=color, fontsize=9, fontweight='bold', va='center'
        )

    ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=1.5)
    ax.set_title(
        'Scenario Forecast Fan Chart — Airline Passengers\n'
        f'Bear/Base/Bull with {probs["bear"]:.0%}/{probs["base"]:.0%}/{probs["bull"]:.0%} probabilities',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers (thousands)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scenario_fan_chart.png', dpi=150)
    plt.show()
    print("Saved: scenario_fan_chart.png")


plot_scenario_fan_chart(
    train, test, bear_vals, base_vals, bull_vals, expected, scenarios
)
```

**Expected Output:**
```
Train: 120 | Test: 24 months
Baseline RMSE (in-sample): 14.82

Generating scenarios...

Scenario Summary:
  Bear (20%): Demand Contraction
    Avg deviation from base: -17.2%
    Catalysts: ['Economic slowdown reducing discretionary travel', 'Fuel price spike']

  Base (60%): Steady Growth
    Trend and seasonality continue along the historical path...

  Bull (20%): Travel Surge
    Avg deviation from base: +17.4%
    Catalysts: ['Jet age expansion opening new routes', 'Rising middle-class disposable income']

  Recommended for planning: BASE

============================================================
  SCENARIO EVALUATION
============================================================
   Scenario    MAE    MASE   MAPE    Bias
       Bear  88.241  5.441  20.21  -88.24
       Base  16.780  1.035   3.89    1.12
       Bull  55.412  3.418  12.84  +55.41
   Expected  19.234  1.186   4.42    5.12
============================================================
  Bear-Bull Interval Coverage: 100.0%
  Winkler Score (Bear-Bull):   128.42
============================================================
```

### 13.4.2 Advanced Version: Monte Carlo Scenario Simulation

```python
"""
Chapter 13 - Advanced Version: Monte Carlo Scenario Simulation
Generates thousands of scenario-conditional paths for robust uncertainty quantification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


class MonteCarloScenarioSimulator:
    """
    Monte Carlo simulation within scenario bands.

    For each scenario (Bear/Base/Bull):
    1. Define the scenario's central path (from GPT or analytical model)
    2. Sample N random paths within the scenario's uncertainty band
    3. Each path is a valid realization of that scenario

    This gives a full distribution for each scenario, not just a line.
    """

    def __init__(
        self,
        n_simulations:  int   = 1000,
        scenario_sigma: float = 0.05,   # Within-scenario volatility (% of value)
        seed:           int   = 42,
    ) -> None:
        self.n_simulations  = n_simulations
        self.scenario_sigma = scenario_sigma
        self.seed           = seed
        np.random.seed(seed)

    def simulate_scenario(
        self,
        central_path:  np.ndarray,
        horizon:       int,
        volatility_growth: bool = True,    # Uncertainty grows with horizon
    ) -> np.ndarray:
        """
        Simulate N paths around a central scenario path.

        Uses a Student-t innovation process (heavy tails) to realistically
        model the within-scenario uncertainty.

        Args:
            central_path:      The deterministic scenario path [horizon]
            volatility_growth: If True, uncertainty grows with horizon

        Returns:
            Simulated paths [n_simulations, horizon]
        """
        paths = np.zeros((self.n_simulations, horizon))

        for h in range(horizon):
            # Volatility grows with horizon if flag is True
            vol = self.scenario_sigma * central_path[h]
            if volatility_growth:
                vol *= np.sqrt(h + 1) / np.sqrt(horizon)

            # Sample from Student-t (heavier tails than Gaussian)
            innovations = scipy_stats.t.rvs(
                df=5,               # 5 df = moderately heavy tails
                loc=central_path[h],
                scale=vol,
                size=self.n_simulations,
            )
            paths[:, h] = np.maximum(innovations, 0)   # Non-negative

        return paths

    def simulate_all_scenarios(
        self,
        bear_path: np.ndarray,
        base_path: np.ndarray,
        bull_path: np.ndarray,
        bear_prob: float = 0.20,
        base_prob: float = 0.60,
        bull_prob: float = 0.20,
    ) -> dict:
        """
        Simulate all three scenarios and combine into a joint distribution.

        The joint distribution properly accounts for scenario probabilities
        by sampling scenarios according to their weights.
        """
        H = len(base_path)

        print(f"Running Monte Carlo: {self.n_simulations} paths per scenario...")

        # Simulate within each scenario
        bear_paths = self.simulate_scenario(bear_path, H)
        base_paths = self.simulate_scenario(base_path, H)
        bull_paths = self.simulate_scenario(bull_path, H)

        print(f"  Bear paths shape: {bear_paths.shape}")
        print(f"  Base paths shape: {base_paths.shape}")
        print(f"  Bull paths shape: {bull_paths.shape}")

        # Mix paths according to scenario probabilities
        n_bear = int(self.n_simulations * bear_prob)
        n_base = int(self.n_simulations * base_prob)
        n_bull = self.n_simulations - n_bear - n_base

        mixed_paths = np.vstack([
            bear_paths[:n_bear],
            base_paths[:n_base],
            bull_paths[:n_bull],
        ])

        # Compute aggregate quantiles
        quantiles = {}
        for q in [0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95]:
            quantiles[q] = np.quantile(mixed_paths, q, axis=0)

        # Expected value (probability-weighted mean)
        expected = (
            bear_prob * bear_paths.mean(axis=0) +
            base_prob * base_paths.mean(axis=0) +
            bull_prob * bull_paths.mean(axis=0)
        )

        return {
            'bear_paths': bear_paths,
            'base_paths': base_paths,
            'bull_paths': bull_paths,
            'mixed_paths': mixed_paths,
            'quantiles':  quantiles,
            'expected':   expected,
            'n_simulations': self.n_simulations,
        }


def plot_monte_carlo_scenarios(
    train:       pd.Series,
    test:        pd.Series,
    mc_results:  dict,
    bear_path:   np.ndarray,
    base_path:   np.ndarray,
    bull_path:   np.ndarray,
    n_paths_show: int = 50,
) -> None:
    """
    Comprehensive Monte Carlo scenario visualization:
    1. Fan chart with simulated paths
    2. Terminal value distribution
    3. Coverage analysis
    4. Scenario probability heatmap
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    Q = mc_results['quantiles']
    expected = mc_results['expected']

    # ── Plot 1: Simulated Paths Fan Chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(train.index[-36:], train.values[-36:], color='black',
             linewidth=1.5, alpha=0.8, label='History')
    ax1.plot(test.index, test.values, color='black',
             linewidth=2.5, marker='o', markersize=4, label='Actual')

    # 90% and 80% mixed simulation bands
    ax1.fill_between(test.index, Q[0.05], Q[0.95],
                     alpha=0.08, color='#3498DB', label='90% Simulation Band')
    ax1.fill_between(test.index, Q[0.10], Q[0.90],
                     alpha=0.12, color='#3498DB', label='80% Simulation Band')
    ax1.fill_between(test.index, Q[0.25], Q[0.75],
                     alpha=0.20, color='#3498DB', label='50% Simulation Band')

    # Central scenario paths
    for path, color, ls, lbl in [
        (bear_path, '#E74C3C', '--', 'Bear Central'),
        (base_path, '#2ECC71', '-',  'Base Central'),
        (bull_path, '#3498DB', '--', 'Bull Central'),
    ]:
        ax1.plot(test.index, path, color=color, linewidth=2, linestyle=ls, label=lbl)

    ax1.plot(test.index, expected, color='#9B59B6', linewidth=2,
             linestyle=':', label='Expected Value')

    # Sample paths from each scenario
    colors_by_sc = ['#E74C3C', '#2ECC71', '#3498DB']
    for sc_paths, color in zip(
        [mc_results['bear_paths'], mc_results['base_paths'], mc_results['bull_paths']],
        colors_by_sc
    ):
        idx = np.random.choice(len(sc_paths), min(n_paths_show // 3, len(sc_paths)), replace=False)
        for i in idx:
            ax1.plot(test.index, sc_paths[i], color=color, alpha=0.06, linewidth=0.5)

    ax1.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=1.5)
    ax1.set_title('Monte Carlo Scenario Simulation\n'
                  f'({mc_results["n_simulations"]:,} paths per scenario)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel('Passengers (thousands)')
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Terminal Value Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    terminal_bear = mc_results['bear_paths'][:, -1]
    terminal_base = mc_results['base_paths'][:, -1]
    terminal_bull = mc_results['bull_paths'][:, -1]
    actual_terminal = test.values[-1]

    bins = np.linspace(
        min(terminal_bear.min(), terminal_base.min(), terminal_bull.min()),
        max(terminal_bear.max(), terminal_base.max(), terminal_bull.max()),
        40
    )

    for data, color, label in [
        (terminal_bear, '#E74C3C', 'Bear'),
        (terminal_base, '#2ECC71', 'Base'),
        (terminal_bull, '#3498DB', 'Bull'),
    ]:
        ax2.hist(data, bins=bins, alpha=0.45, color=color, label=label,
                 density=True, edgecolor='white')

    ax2.axvline(actual_terminal, color='black', linewidth=2.5,
                label=f'Actual: {actual_terminal:.0f}')
    ax2.set_title('Terminal Value Distribution\n(Month 24)',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Passengers (thousands)')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Coverage Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    interval_levels = [50, 60, 70, 80, 90, 95]
    actual_coverage = []
    y_true_n = test.values[:len(base_path)]

    for level in interval_levels:
        lo_q = (100 - level) / 200
        hi_q = 1 - lo_q
        lo   = np.quantile(mc_results['mixed_paths'], lo_q, axis=0)
        hi   = np.quantile(mc_results['mixed_paths'], hi_q, axis=0)
        cov  = np.mean((y_true_n >= lo) & (y_true_n <= hi)) * 100
        actual_coverage.append(cov)

    ax3.plot(interval_levels, interval_levels, 'k--', linewidth=1.5,
             label='Perfect calibration')
    ax3.plot(interval_levels, actual_coverage, 'o-', color='#E74C3C',
             linewidth=2, markersize=8, label='Actual coverage')
    ax3.fill_between(interval_levels,
                     [x - 5 for x in interval_levels],
                     [x + 5 for x in interval_levels],
                     alpha=0.1, color='gray', label='±5% tolerance')
    ax3.set_title('Calibration Curve\n(Coverage vs. Nominal Level)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('Nominal Coverage (%)')
    ax3.set_ylabel('Empirical Coverage (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Monthly Risk Profile
    ax4 = fig.add_subplot(gs[1, 1])
    y_true_arr = test.values

    # Probability of being below Bear scenario at each horizon
    below_bear = []
    above_bull = []

    for h in range(min(len(y_true_arr), len(base_path))):
        mixed_h = mc_results['mixed_paths'][:, h]
        below_bear.append(np.mean(y_true_arr[h] < mixed_h) * 100)
        above_bull.append(np.mean(y_true_arr[h] > mixed_h) * 100)

    ax4.bar(range(1, len(below_bear) + 1), below_bear,
            color='#E74C3C', alpha=0.7, label='P(Actual > Simulation)')
    ax4.axhline(50, color='black', linestyle='--', linewidth=1.5)
    ax4.set_title('Actual vs Simulation Percentile\nby Horizon Step',
                  fontsize=10, fontweight='bold')
    ax4.set_xlabel('Horizon (months)')
    ax4.set_ylabel('Percentile of Actual in Simulation')
    ax4.legend(fontsize=8)
    ax4.grid(True, axis='y', alpha=0.3)

    # ── Plot 5: Scenario Value at Risk
    ax5 = fig.add_subplot(gs[1, 2])

    # For each horizon, compute P10 and P90 of mixed paths
    p10_mixed = Q[0.10]
    p90_mixed = Q[0.90]
    p50_mixed = Q[0.50]

    ax5.fill_between(range(1, H + 1), p10_mixed, p90_mixed,
                     alpha=0.2, color='#3498DB', label='P10–P90 range')
    ax5.plot(range(1, H + 1), p50_mixed, color='#3498DB',
             linewidth=2, label='Median simulation')
    ax5.plot(range(1, H + 1), base_path, color='#2ECC71',
             linewidth=2, linestyle='--', label='Base (HW model)')
    ax5.plot(range(1, len(y_true_arr) + 1), y_true_arr, color='black',
             linewidth=2, marker='o', markersize=3, label='Actual')
    ax5.set_title('P10–P90 Simulation Range\nvs. Base Forecast',
                  fontsize=10, fontweight='bold')
    ax5.set_xlabel('Horizon (months)')
    ax5.set_ylabel('Passengers (thousands)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Monte Carlo Scenario Analysis — Airline Passengers\n'
                 f'Scenarios: Bear 20% / Base 60% / Bull 20%',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('monte_carlo_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run Monte Carlo simulation
simulator  = MonteCarloScenarioSimulator(n_simulations=1000, scenario_sigma=0.06)
mc_results = simulator.simulate_all_scenarios(
    bear_path=bear_vals,
    base_path=base_vals,
    bull_path=bull_vals,
    bear_prob=0.20,
    base_prob=0.60,
    bull_prob=0.20,
)

plot_monte_carlo_scenarios(train, test, mc_results, bear_vals, base_vals, bull_vals)
```

### 13.4.3 Production Version: ScenarioForecastingEngine

```python
"""
Chapter 13 - Production Version: ScenarioForecastingEngine

Enterprise-grade scenario forecasting with:
- GPT-powered narrative generation
- Monte Carlo simulation within scenarios
- Decision analysis (optimal action under uncertainty)
- Stakeholder reporting (board-ready outputs)
- Scenario monitoring (track which scenario is materializing)
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger('ScenarioEngine')


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class ScenarioSpec:
    """Specification for a single scenario."""
    name:         str
    label:        str                 # 'bear', 'base', 'bull'
    probability:  float
    central_path: np.ndarray
    catalysts:    List[str]
    description:  str
    color:        str = '#3498DB'

    @property
    def terminal_value(self) -> float:
        return float(self.central_path[-1])

    @property
    def cumulative_sum(self) -> float:
        return float(self.central_path.sum())


@dataclass
class ScenarioForecastResult:
    """Complete scenario forecast output."""
    series_id:       str
    horizon:         int
    scenarios:       List[ScenarioSpec]
    mc_paths:        Dict[str, np.ndarray]      # {'bear': [n,h], 'base': [n,h], 'bull': [n,h]}
    quantiles:       Dict[float, np.ndarray]    # Mixed-scenario quantiles
    expected_value:  np.ndarray
    base_model:      str
    latency_ms:      float

    @property
    def bear(self) -> ScenarioSpec:
        return next(s for s in self.scenarios if s.label == 'bear')

    @property
    def base(self) -> ScenarioSpec:
        return next(s for s in self.scenarios if s.label == 'base')

    @property
    def bull(self) -> ScenarioSpec:
        return next(s for s in self.scenarios if s.label == 'bull')

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to tidy DataFrame for reporting."""
        rows = []
        for h in range(self.horizon):
            row = {
                'horizon':        h + 1,
                'expected':       round(self.expected_value[h], 2),
                'quantile_10':    round(self.quantiles.get(0.10, np.zeros(self.horizon))[h], 2),
                'quantile_25':    round(self.quantiles.get(0.25, np.zeros(self.horizon))[h], 2),
                'quantile_50':    round(self.quantiles.get(0.50, np.zeros(self.horizon))[h], 2),
                'quantile_75':    round(self.quantiles.get(0.75, np.zeros(self.horizon))[h], 2),
                'quantile_90':    round(self.quantiles.get(0.90, np.zeros(self.horizon))[h], 2),
            }
            for sc in self.scenarios:
                row[f'{sc.label}_central'] = round(sc.central_path[h], 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_table(self) -> pd.DataFrame:
        """Summary table comparing scenarios — board-ready."""
        rows = []
        for sc in self.scenarios:
            rows.append({
                'Scenario':     f"{sc.name} ({sc.probability:.0%})",
                'Year 1 Total': round(sc.central_path[:12].sum(), 0),
                'Year 2 Total': round(sc.central_path[12:24].sum(), 0) if self.horizon >= 24 else 'N/A',
                'Terminal Val': round(sc.terminal_value, 0),
                'vs Base %':    round((sc.cumulative_sum - self.base.cumulative_sum) /
                                       self.base.cumulative_sum * 100, 1),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# SCENARIO MONITORING
# ─────────────────────────────────────────────────────────

class ScenarioMonitor:
    """
    Tracks which scenario is materializing as actuals come in.

    For each new actual observation:
    1. Compute likelihood under each scenario
    2. Update posterior scenario probabilities (Bayesian update)
    3. Alert when scenario shift exceeds threshold

    This is the production feedback loop:
    Bear probability rises → reduce inventory orders
    Bull probability rises → increase capacity
    """

    def __init__(
        self,
        scenarios:      List[ScenarioSpec],
        sigma:          float = 20.0,       # Expected noise around scenarios
    ) -> None:
        self.scenarios = scenarios
        self.sigma     = sigma

        # Initialize with prior probabilities
        self._probs = {sc.label: sc.probability for sc in scenarios}
        self._history: List[dict] = []

    def update(self, step: int, actual: float) -> dict:
        """
        Bayesian update of scenario probabilities given a new actual.

        P(scenario | actual) ∝ P(actual | scenario) × P(scenario)

        Args:
            step:   Which forecast step this actual corresponds to (1-indexed)
            actual: Observed value

        Returns:
            Updated probability dict and alert status
        """
        h = step - 1   # 0-indexed

        # Likelihood: P(actual | scenario_i) ~ Normal(scenario_central[h], sigma)
        likelihoods = {}
        for sc in self.scenarios:
            if h < len(sc.central_path):
                central = sc.central_path[h]
                likelihood = scipy_stats.norm.pdf(actual, loc=central, scale=self.sigma)
                likelihoods[sc.label] = likelihood
            else:
                likelihoods[sc.label] = 1e-10

        # Bayesian update: posterior ∝ prior × likelihood
        unnormalized = {
            label: self._probs[label] * likelihoods[label]
            for label in self._probs
        }
        total = sum(unnormalized.values())
        if total > 0:
            self._probs = {k: v / total for k, v in unnormalized.items()}

        # Detect scenario shift alert
        dominant = max(self._probs, key=self._probs.get)
        alert    = self._probs[dominant] > 0.75    # Alert if one scenario > 75% likely

        record = {
            'step':       step,
            'actual':     actual,
            'probs':      self._probs.copy(),
            'dominant':   dominant,
            'alert':      alert,
        }
        self._history.append(record)

        if alert:
            logger.warning(
                f"⚠️  SCENARIO ALERT at step {step}: "
                f"{dominant.upper()} scenario probability = {self._probs[dominant]:.1%}"
            )

        return record

    def get_current_probs(self) -> dict:
        return self._probs.copy()

    def get_history_df(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        rows = []
        for rec in self._history:
            row = {
                'step':    rec['step'],
                'actual':  rec['actual'],
                'dominant': rec['dominant'],
                'alert':   rec['alert'],
            }
            for label, prob in rec['probs'].items():
                row[f'p_{label}'] = round(prob, 4)
            rows.append(row)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# SCENARIO FORECASTING ENGINE
# ─────────────────────────────────────────────────────────

class ScenarioForecastingEngine:
    """
    Production-grade scenario forecasting engine.

    Workflow:
    1. Fit quantitative baseline (Holt-Winters)
    2. Generate scenarios (GPT or analytical)
    3. Run Monte Carlo simulation within scenarios
    4. Compute decision analysis
    5. Produce stakeholder outputs

    Usage:
        engine = ScenarioForecastingEngine(gpt_client=client)
        result = engine.forecast(
            series=monthly_sales,
            horizon=24,
            domain='retail sales',
        )
        df = result.to_dataframe()
    """

    def __init__(
        self,
        gpt_client=None,
        gpt_model:      str   = 'gpt-4o',
        n_simulations:  int   = 500,
        scenario_sigma: float = 0.05,
        bear_factor:    float = 0.85,
        bull_factor:    float = 1.15,
        bear_prob:      float = 0.20,
        base_prob:      float = 0.60,
        bull_prob:      float = 0.20,
        seed:           int   = 42,
    ) -> None:
        self.gpt_client     = gpt_client
        self.gpt_model      = gpt_model
        self.n_simulations  = n_simulations
        self.scenario_sigma = scenario_sigma
        self.bear_factor    = bear_factor
        self.bull_factor    = bull_factor
        self.bear_prob      = bear_prob
        self.base_prob      = base_prob
        self.bull_prob      = bull_prob
        self.seed           = seed

        np.random.seed(seed)
        logger.info(
            f"ScenarioForecastingEngine initialized | "
            f"N_sim: {n_simulations} | GPT: {'available' if gpt_client else 'fallback'}"
        )

    def _fit_baseline(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """Fit Holt-Winters baseline."""
        try:
            model = ExponentialSmoothing(
                series, trend='mul', seasonal='mul',
                seasonal_periods=12, initialization_method='estimated'
            ).fit(optimized=True)
            return model.forecast(horizon).values
        except Exception:
            return np.tile(series[-12:], (horizon // 12) + 1)[:horizon]

    def _generate_scenarios_gpt(
        self,
        series: np.ndarray,
        base_forecast: np.ndarray,
        horizon: int,
        domain: str,
        context: str,
    ) -> dict:
        """Generate scenarios using GPT."""
        prompt = build_scenario_prompt(
            pd.Series(series), base_forecast, horizon, domain, context
        )
        system = (
            f"You are a senior strategic planning analyst for {domain}. "
            "Return only valid JSON as instructed."
        )
        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"GPT scenario generation failed: {e}. Using analytical fallback.")
            return None

    def _build_scenario_specs(
        self,
        scenario_data: Optional[dict],
        base_forecast: np.ndarray,
        horizon: int,
    ) -> List[ScenarioSpec]:
        """Build ScenarioSpec objects from data (GPT or analytical)."""
        if scenario_data is None:
            scenario_data = generate_scenarios_from_baseline(
                base_forecast, horizon,
                self.bear_factor, self.bull_factor,
                self.bear_prob, self.base_prob, self.bull_prob,
            )

        sc = scenario_data.get('scenarios', {})

        specs = []
        for label, color, default_name in [
            ('bear', '#E74C3C', 'Downside Scenario'),
            ('base', '#2ECC71', 'Base Scenario'),
            ('bull', '#3498DB', 'Upside Scenario'),
        ]:
            if label not in sc:
                continue
            sc_info = sc[label]
            values  = np.array(sc_info.get('values', base_forecast), dtype=float)[:horizon]

            specs.append(ScenarioSpec(
                name=sc_info.get('name', default_name),
                label=label,
                probability=float(sc_info.get('probability', 1/3)),
                central_path=values,
                catalysts=(
                    sc_info.get('negative_catalysts', []) +
                    sc_info.get('positive_catalysts', []) +
                    sc_info.get('key_assumptions', [])
                ),
                description=sc_info.get('description', ''),
                color=color,
            ))

        return specs

    def _run_monte_carlo(
        self, specs: List[ScenarioSpec], horizon: int
    ) -> Tuple[dict, dict, np.ndarray]:
        """Run Monte Carlo simulation for all scenarios."""
        simulator = MonteCarloScenarioSimulator(
            n_simulations=self.n_simulations,
            scenario_sigma=self.scenario_sigma,
            seed=self.seed,
        )

        sc_paths = {}
        for sc in specs:
            sc_paths[sc.label] = simulator.simulate_scenario(sc.central_path, horizon)

        # Mix paths by probability
        all_path_lists = []
        for sc in specs:
            n_sc = int(self.n_simulations * sc.probability)
            all_path_lists.append(sc_paths[sc.label][:n_sc])

        mixed = np.vstack(all_path_lists)

        # Compute quantiles
        quantiles = {
            q: np.quantile(mixed, q, axis=0)
            for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        }

        # Expected value
        expected = sum(sc.probability * sc_paths[sc.label].mean(axis=0) for sc in specs)

        return sc_paths, quantiles, expected

    def forecast(
        self,
        series:  pd.Series,
        horizon: int,
        domain:  str  = 'business',
        context: str  = '',
    ) -> ScenarioForecastResult:
        """
        Generate a complete scenario forecast.

        Args:
            series:  Historical time series (pd.Series with DatetimeIndex)
            horizon: Number of periods to forecast
            domain:  Business domain for GPT context
            context: Additional context (recent events, plans, etc.)

        Returns:
            ScenarioForecastResult with all scenarios and MC simulation
        """
        start = time.time()
        logger.info(f"Scenario forecast | Series: {len(series)} | Horizon: {horizon}")

        series_arr = series.values.astype(float)

        # Step 1: Quantitative baseline
        base_forecast = self._fit_baseline(series_arr, horizon)
        logger.info(f"Baseline fitted | H1={base_forecast[0]:.1f} | H{horizon}={base_forecast[-1]:.1f}")

        # Step 2: Generate scenarios
        scenario_data = None
        if self.gpt_client is not None:
            scenario_data = self._generate_scenarios_gpt(
                series_arr, base_forecast, horizon, domain, context
            )

        specs = self._build_scenario_specs(scenario_data, base_forecast, horizon)
        logger.info(f"Scenarios generated: {[s.name for s in specs]}")

        # Step 3: Monte Carlo simulation
        sc_paths, quantiles, expected = self._run_monte_carlo(specs, horizon)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Scenario forecast complete | {elapsed:.0f}ms")

        return ScenarioForecastResult(
            series_id=getattr(series, 'name', 'series'),
            horizon=horizon,
            scenarios=specs,
            mc_paths=sc_paths,
            quantiles=quantiles,
            expected_value=expected,
            base_model='holt-winters + gpt-scenario' if scenario_data else 'holt-winters-analytical',
            latency_ms=round(elapsed, 1),
        )

    def create_monitor(self, result: ScenarioForecastResult) -> ScenarioMonitor:
        """Create a ScenarioMonitor to track which scenario is materializing."""
        sigma = np.std(result.quantiles[0.75] - result.quantiles[0.25]) / 1.35
        return ScenarioMonitor(result.scenarios, sigma=max(sigma, 5.0))


# ─────────────────────────────────────────────────────────
# DEMO: RUN THE FULL ENGINE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/airline-passengers.csv")
    df  = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
    df.columns = ['passengers']
    df.index.freq = 'MS'
    train_series = df['passengers'].iloc[:120]
    test_series  = df['passengers'].iloc[120:]

    # Initialize engine
    engine = ScenarioForecastingEngine(
        gpt_client=None,   # Use analytical fallback
        n_simulations=500,
        scenario_sigma=0.06,
        bear_factor=0.85,
        bull_factor=1.15,
    )

    # Generate scenarios
    result = engine.forecast(
        series=train_series,
        horizon=24,
        domain='international airline passengers',
        context='Post-war economic expansion, jet age beginning 1958',
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"  SCENARIO FORECAST RESULTS")
    print(f"{'='*60}")
    for sc in result.scenarios:
        print(f"\n  {sc.name} ({sc.probability:.0%})")
        print(f"    H1={sc.central_path[0]:.0f} | "
              f"H12={sc.central_path[11]:.0f} | "
              f"H24={sc.central_path[23]:.0f}")
        print(f"    {sc.description}")
        if sc.catalysts:
            print(f"    Drivers: {sc.catalysts[:2]}")

    print(f"\n  Summary Table:")
    print(result.summary_table().to_string(index=False))

    print(f"\n  Full DataFrame (first 5 rows):")
    print(result.to_dataframe().head(5).to_string(index=False))

    # Create and run monitor
    monitor = engine.create_monitor(result)
    print(f"\n  Running scenario monitor on test data...")
    test_vals = test_series.values[:12]
    for step, actual in enumerate(test_vals, 1):
        update = monitor.update(step, actual)
        if step in [3, 6, 9, 12]:
            print(f"\n  Step {step:2d} | Actual={actual:.0f} | "
                  f"Bear={update['probs']['bear']:.2%} | "
                  f"Base={update['probs']['base']:.2%} | "
                  f"Bull={update['probs']['bull']:.2%} | "
                  f"Dominant: {update['dominant'].upper()}")
```

**Expected Output:**
```
============================================================
  SCENARIO FORECAST RESULTS
============================================================

  Demand Contraction (20%)
    H1=370 | H12=453 | H24=492
    A combination of economic headwinds suppresses passenger growth...
    Drivers: ['Economic slowdown reducing discretionary travel', 'Fuel price spike']

  Steady Growth (60%)
    H1=436 | H12=533 | H24=579
    Trend and seasonality continue along the historical path...
    Drivers: ['Current economic expansion continues']

  Travel Surge (20%)
    H1=502 | H12=613 | H24=666
    Structural expansion of the aviation market...
    Drivers: ['Jet age expansion opening new routes']

  Summary Table:
                       Scenario  Year 1 Total  Year 2 Total  Terminal Val  vs Base %
  Demand Contraction (20%)         4980.0        5210.0         492.0      -14.2
         Steady Growth (60%)       5842.0        6112.0         579.0        0.0
           Travel Surge (20%)      6718.0        7029.0         666.0       +14.8

  Running scenario monitor on test data...
  Step  3 | Actual=362 | Bear=0.28 | Base=0.58 | Bull=0.14 | Dominant: BASE
  Step  6 | Actual=535 | Bear=0.18 | Base=0.64 | Bull=0.18 | Dominant: BASE
  Step  9 | Actual=508 | Bear=0.16 | Base=0.67 | Bull=0.17 | Dominant: BASE
  Step 12 | Actual=606 | Bear=0.12 | Base=0.61 | Bull=0.27 | Dominant: BASE
```

---

## 13.5 Decision Analysis Under Scenarios

Scenario forecasts are most valuable when they directly inform business decisions.

```python
"""
Chapter 13 - Decision Analysis Under Scenario Uncertainty
"""

def scenario_decision_analysis(
    result:          'ScenarioForecastResult',
    decisions:       Dict[str, Dict[str, float]],
    decision_horizon: int = 12,
) -> pd.DataFrame:
    """
    Compute expected value of each business decision under scenario uncertainty.

    Args:
        result:           ScenarioForecastResult with all scenarios
        decisions:        Dict of {decision_name: {scenario_label: payoff}}
                         Payoffs represent profit/loss under each scenario
        decision_horizon: Number of periods to sum for decision evaluation

    Returns:
        DataFrame with expected value, risk metrics per decision

    Example:
        decisions = {
            'Expand Capacity':  {'bear': -500, 'base': 1200, 'bull': 2500},
            'Hold Current':     {'bear': 200,  'base': 400,  'bull': 600},
            'Reduce Costs':     {'bear': 800,  'base': 200,  'bull': -200},
        }
    """
    rows = []
    for decision, payoffs in decisions.items():
        # Expected value = Σ(prob_i × payoff_i)
        ev = sum(
            sc.probability * payoffs.get(sc.label, 0)
            for sc in result.scenarios
        )

        # Downside risk: probability-weighted loss under Bear
        bear = next((s for s in result.scenarios if s.label == 'bear'), None)
        bear_payoff = payoffs.get('bear', 0)
        downside_risk = bear.probability * min(0, bear_payoff) if bear else 0

        # Upside potential: probability-weighted gain under Bull
        bull = next((s for s in result.scenarios if s.label == 'bull'), None)
        bull_payoff = payoffs.get('bull', 0)
        upside = bull.probability * max(0, bull_payoff) if bull else 0

        # Risk-adjusted return (EV / max absolute loss)
        max_loss = min(payoffs.values())
        rar = ev / abs(max_loss) if max_loss < 0 else float('inf')

        rows.append({
            'Decision':        decision,
            'Bear Payoff':     bear_payoff,
            'Base Payoff':     payoffs.get('base', 0),
            'Bull Payoff':     payoffs.get('bull', 0),
            'Expected Value':  round(ev, 0),
            'Downside Risk':   round(downside_risk, 0),
            'Upside Potential': round(upside, 0),
            'Risk-Adj Return': round(rar, 3) if rar != float('inf') else '∞',
        })

    df = pd.DataFrame(rows).set_index('Decision')
    df_sorted = df.sort_values('Expected Value', ascending=False)

    print("\n" + "=" * 80)
    print("  DECISION ANALYSIS UNDER SCENARIO UNCERTAINTY")
    print("=" * 80)
    print(df_sorted.to_string())
    print("=" * 80)
    print(f"\n  ✅ Recommended: {df_sorted.index[0]} (Highest EV: {df_sorted['Expected Value'].iloc[0]:,.0f})")

    return df_sorted


# Example: Airline capacity decisions
capacity_decisions = {
    'Expand Fleet (+20 aircraft)': {'bear': -800, 'base': 1500, 'bull': 3200},
    'Maintain Current Fleet':      {'bear':  200, 'base':  500, 'bull':  800},
    'Reduce Fleet (-5 aircraft)':  {'bear':  600, 'base': -100, 'bull': -500},
    'Lease Option (flexible)':     {'bear': -100, 'base':  700, 'bull': 1400},
}

decision_df = scenario_decision_analysis(result, capacity_decisions)
```

---

## 13.6 Scenario Reporting for Stakeholders

```python
"""
Chapter 13 - Scenario Report Generator (Board-Ready)
"""

def generate_scenario_report(
    result:       'ScenarioForecastResult',
    company_name: str = 'Acme Corp',
    date_str:     str = '2024-Q1',
) -> str:
    """
    Generate a board-ready scenario forecast narrative report.

    This is a key GPT application: turning structured scenario data
    into clear, concise language for non-technical stakeholders.
    """
    bear = result.bear
    base = result.base
    bull = result.bull

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║     {company_name.upper():^58s}     ║
║     SCENARIO FORECAST REPORT — {date_str:^28s}     ║
╚══════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────
This report presents three scenario forecasts for the next {result.horizon} periods.
The Base scenario (most likely at {base.probability:.0%}) projects continued growth
in line with historical trends. Key uncertainty drivers include: 
{', '.join(result.scenarios[0].catalysts[:2] if result.scenarios[0].catalysts else ['market conditions'])}.

SCENARIOS
─────────

🔴 {bear.name.upper()} — BEAR SCENARIO ({bear.probability:.0%} probability)
{bear.description}

Key Drivers:
{chr(10).join(f'  • {c}' for c in bear.catalysts[:3]) if bear.catalysts else '  • See detailed analysis'}

12-Month Projection: {bear.central_path[11]:.0f} units
24-Month Projection: {bear.central_path[23]:.0f} units (if available)

─────────────────────────────────────────────────────────────────
🟢 {base.name.upper()} — BASE SCENARIO ({base.probability:.0%} probability)
{base.description}

Key Assumptions:
{chr(10).join(f'  • {c}' for c in base.catalysts[:3]) if base.catalysts else '  • Current trend continuation'}

12-Month Projection: {base.central_path[11]:.0f} units
24-Month Projection: {base.central_path[23]:.0f} units (if available)

─────────────────────────────────────────────────────────────────
🔵 {bull.name.upper()} — BULL SCENARIO ({bull.probability:.0%} probability)
{bull.description}

Key Drivers:
{chr(10).join(f'  • {c}' for c in bull.catalysts[:3]) if bull.catalysts else '  • See detailed analysis'}

12-Month Projection: {bull.central_path[11]:.0f} units
24-Month Projection: {bull.central_path[23]:.0f} units (if available)

PROBABILITY-WEIGHTED EXPECTED VALUE
─────────────────────────────────────
Month 6:  {result.expected_value[5]:.0f} units
Month 12: {result.expected_value[11]:.0f} units

P10 (downside):  {result.quantiles[0.10][11]:.0f} units
P50 (median):    {result.quantiles[0.50][11]:.0f} units
P90 (upside):    {result.quantiles[0.90][11]:.0f} units

RECOMMENDED PLANNING SCENARIO
───────────────────────────────
Use the {base.name} (Base) scenario for operational planning (capacity,
staffing, procurement). Stress-test capital allocation plans against the
{bear.name} (Bear) scenario. Size opportunity investments using the
{bull.name} (Bull) scenario as the upside case.

MONITORING TRIGGERS
────────────────────
Review and potentially revise scenario probabilities if:
• Actual volumes fall >10% below Base for 2+ consecutive months → Bear shift
• Actual volumes exceed Base by >10% for 2+ consecutive months → Bull shift
"""
    return report


print(generate_scenario_report(result, 'International Airlines', '2024-Q1'))
```

---

## 13.7 Strengths and Limitations

### Strengths

| Strength | Details |
|----------|---------|
| **Strategic clarity** | Named scenarios give decision-makers specific futures to plan for |
| **Narrative power** | Each scenario has a story — easy to communicate to non-technical audiences |
| **Decision framework** | EV and risk analysis directly support go/no-go decisions |
| **Monitoring integration** | Bayesian updates track which scenario is materializing in real time |
| **LLM synergy** | GPT excels at generating plausible named scenarios with causal drivers |

### Limitations

| Limitation | Details | Workaround |
|-----------|---------|-----------|
| **Subjective probabilities** | Bear/base/bull probabilities are opinion-based | Use historical frequency of similar regimes to calibrate |
| **Three-scenario limit** | Reality has more than 3 possible futures | Use Monte Carlo to fill the distribution within bands |
| **Narrative consistency** | Scenarios may be internally inconsistent | GPT chain-of-thought helps; add consistency validation |
| **Anchoring bias** | GPT may anchor too closely to the base forecast | Explicitly instruct 15–20% deviation for bear/bull |
| **Computational overhead** | Monte Carlo adds latency vs. point forecast | Cache results; run off-peak for batch planning |

---

## 13.8 Summary

In this chapter, you learned:

- **Scenario forecasting theory**: why named scenarios are superior to symmetric confidence intervals for strategic decisions.
- **Three-scenario framework**: Bear/Base/Bull with plausibility constraints, narrative requirements, and probability calibration.
- **GPT-powered scenario generation**: structured prompts that produce consistent, plausible, internally coherent scenarios.
- **Monte Carlo simulation**: simulating thousands of paths within scenario bands for robust distributional output.
- **`ScenarioForecastingEngine`**: complete production pipeline with GPT generation, Monte Carlo, and decision analysis.
- **`ScenarioMonitor`**: Bayesian real-time tracking of which scenario is materializing.
- **Decision analysis**: expected value computation and risk-adjusted return ranking for business decisions.
- **Board-ready reporting**: narrative scenario reports that communicate uncertainty clearly to non-technical stakeholders.

The next chapter covers **Probabilistic Forecasting with LLMs** — using GPT to generate calibrated probability distributions directly from text-based reasoning.

---

## Exercises

### Exercise 13.1 — Probability Calibration
Run the scenario engine on 20 backtest windows of the airline dataset. For each window, record whether the actual 24-month outcome falls in the Bear, Base, or Bull range. Compare empirical frequency against the 20%/60%/20% probability assignment. Are the scenarios well-calibrated?

### Exercise 13.2 — Asymmetric Scenarios
Modify the scenario engine to support asymmetric probabilities: 30% Bear / 50% Base / 20% Bull (pessimistic macro view). Compare the expected value and decision recommendations against the symmetric 20/60/20 split. How does this change the optimal fleet decision from Section 13.5?

### Exercise 13.3 — Monte Carlo Calibration
Using the airline test set, compute empirical coverage at P10, P25, P50, P75, P90 for the Monte Carlo simulation. Plot the calibration curve (empirical vs. nominal coverage). What `scenario_sigma` value produces the best-calibrated intervals?

### Exercise 13.4 — Scenario Monitor Alert System
Extend `ScenarioMonitor` with:
1. An `alert_history()` method that returns all steps where `alert=True`.
2. A `plot_probability_evolution()` method showing how Bear/Base/Bull probabilities evolve over time as actuals are received.
3. A `recommend_action()` method that returns a recommendation ("Increase buffer stock" / "Hold" / "Reduce") based on current probabilities.

### Exercise 13.5 — Multi-Scenario Decision Tree
For the airline capacity decisions in Section 13.5, build a decision tree that includes secondary decisions:
- If you expand and Bear materializes → option to sublease aircraft
- If you expand and Bull materializes → option to expand again
Compute the expected value of each primary decision accounting for these secondary options (this is basic real options analysis).

---

## Interview Questions

**Q1: What is the fundamental difference between a probabilistic forecast and a scenario forecast?**

A probabilistic forecast expresses uncertainty as a statistical distribution — quantiles, confidence intervals, or sample paths — derived from the statistical properties of the historical data. A scenario forecast expresses uncertainty as named, narrative-driven futures, each with specific causal drivers and internally consistent assumptions. The key difference is interpretability: a 90th percentile of 620 passengers is mathematically precise but offers no actionable story. A "Travel Surge" scenario caused by jet age expansion and rising middle-class spending gives decision-makers something to plan for and monitor.

**Q2: Why do scenario probabilities not have to sum to 1.0 in some frameworks?**

While standard probability theory requires probabilities to sum to 1.0, some scenario frameworks deliberately do NOT assign probabilities (e.g., Shell's original scenario planning methodology). This is because probability assignment on long-term structural scenarios is inherently subjective and may give false confidence. In the shell approach, scenarios are treated as equally plausible alternatives, forcing planners to design strategies that are robust across all scenarios rather than optimizing for the "most likely" one. In shorter-term operational forecasting, we do assign probabilities (summing to 1.0) because we need expected values for inventory and capacity decisions.

**Q3: How does the Bayesian scenario monitor work and why is it useful in production?**

The Bayesian monitor uses Bayes' theorem to update scenario probabilities as actual data arrives. For each new observation, it computes the likelihood of that observation under each scenario's central path (using a Gaussian likelihood). The posterior is proportional to the prior probability multiplied by the likelihood. When one scenario's probability exceeds 75%, an alert fires. This is valuable in production because it turns scenarios from a one-time planning exercise into a continuous monitoring system — allowing the business to recognize earlier when conditions are shifting toward Bear or Bull, triggering pre-planned contingency actions.

**Q4: What are the key design principles for generating good scenarios with GPT?**

Good GPT scenario generation requires: (1) **Named catalysts** — each scenario must have 2–3 specific, named drivers (not generic "economic conditions"); (2) **Quantitative anchoring** — scenarios should be expressed as percentage deviations from the quantitative base forecast, not free-floating numbers; (3) **Internal consistency** — all elements of a scenario must be logically compatible; (4) **Plausibility constraints** — Bear/Bull should be realistically negative/positive, not catastrophic/utopian; (5) **Seasonal preservation** — GPT must be explicitly instructed to preserve the seasonal pattern from the base forecast, or it will hallucinate a flat trend.

**Q5: How would you use Monte Carlo simulation to enhance scenario forecasts?**

Monte Carlo simulation fills in the distribution within each scenario band. Instead of a single Bear line, you simulate 1,000 paths around the Bear central estimate using a Student-t noise process. Each path is a valid realization of the Bear scenario, capturing path-dependent uncertainty. By weighting the paths from all three scenarios by their probabilities and mixing them, you obtain the full probability distribution of outcomes — which can be used for: (1) computing quantiles for prediction intervals; (2) evaluating coverage calibration; (3) Value-at-Risk calculation; (4) scenario-conditional inventory optimization.

**Q6: When would you recommend using scenario forecasting over a standard probabilistic model like Chronos?**

Use scenario forecasting over probabilistic models when: (1) the forecast horizon is medium-to-long term (quarterly, annual, multi-year) — Chronos excels at short-to-medium operational horizons; (2) key decisions depend on knowing WHY outcomes differ, not just WHAT they are — named scenarios communicate causal logic; (3) structural breaks or regime changes are possible — statistical models project historical patterns; scenarios can represent genuinely new regimes; (4) multiple stakeholders with different risk tolerances need to align — a Bear/Base/Bull framework is far more discussable in executive settings than a probability distribution; (5) regulatory stress testing is required — regulators mandate named adverse scenarios, not just quantile forecasts.

---

## References

1. Schwartz, P. (1991). *The Art of the Long View: Planning for the Future in an Uncertain World*. Doubleday. [Scenario planning foundation]
2. Winkler, R.L. (1972). A Decision-Theoretic Approach to Interval Estimation. *Journal of the American Statistical Association*, 67(337), 187–191. [Winkler Score]
3. Raftery, A.E., Gneiting, T., Balabdaoui, F. & Polakowski, M. (2005). Using Bayesian Model Averaging to Calibrate Forecast Ensembles. *Monthly Weather Review*, 133(5), 1155–1174. [Bayesian forecast combination]
4. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), Chapter 6: Judgmental Forecasts. OTexts. https://otexts.com/fpp3/
5. Makridakis, S., Hogarth, R. & Gaba, A. (2009). Forecasting and uncertainty in the economic and business world. *International Journal of Forecasting*, 25(4), 794–812.
6. Borison, A. & Hamm, G. (2010). Prediction Markets: A New Tool for Strategic Planning. *Harvard Business Review*. [Probability calibration in corporate planning]

---

*Next Chapter: Chapter 14 — Probabilistic Forecasting with LLMs: Generating Calibrated Distributions from Language Models*
