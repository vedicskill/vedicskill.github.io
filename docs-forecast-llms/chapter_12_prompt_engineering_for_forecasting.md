---
title: "Chapter 12: Prompt Engineering for Forecasting"
description: "Chapter 12: Prompt Engineering for Forecasting in the Forecasting with LLMs course. Learn the core concepts and practical guidance for this chapter."
sidebar_label: "Chapter 12: Prompt Engineering for Forecasting"
sidebar_position: 12
slug: "/forecast-llms/chapter-12-prompt-engineering-for-forecasting"
keywords:
  - forecasting
  - time series
  - LLMs
  - machine learning
  - prediction
---

# Chapter 12: Prompt Engineering for Forecasting

> *"The difference between a good prompt and a bad prompt is the difference between a useful forecast and an expensive hallucination."*
> — Applied AI Engineering Wisdom

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Apply systematic prompt engineering principles specifically for forecasting tasks.
2. Use chain-of-thought prompting to improve numerical reasoning.
3. Design few-shot prompts with relevant forecasting examples.
4. Build self-consistency techniques to reduce hallucination risk.
5. Implement role-based prompting for domain-specific forecasting.
6. Create prompt templates for different forecasting scenarios.
7. Test and evaluate prompt quality systematically.
8. Build a production prompt library with version control.
9. Apply meta-prompting to automatically improve forecasting prompts.

---

## Prerequisites

- Chapter 11 completed
- Python 3.9+
- OpenAI API key

```bash
pip install openai pandas numpy matplotlib jinja2 pydantic
```

---

## 12.1 Why Prompt Engineering Matters for Forecasting

In Chapter 11 we saw that GPT can produce reasonable qualitative forecasts and serve as a powerful hybrid assistant. But we also saw that raw GPT outputs can be inconsistent, imprecise, or outright wrong when prompts are poorly designed.

Prompt engineering is the systematic practice of designing, testing, and refining instructions to language models. For forecasting specifically, the stakes are high — a miscalibrated prompt can produce forecasts that look plausible but carry systematic errors that propagate through downstream business decisions.

### The Forecasting Prompt Engineering Stack

```
┌──────────────────────────────────────────────────────────────────┐
│            FORECASTING PROMPT ENGINEERING STACK                  │
│                                                                  │
│  Layer 5: META-PROMPTING                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Using GPT to improve GPT prompts automatically            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Layer 4: SELF-CONSISTENCY                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Run same prompt N times, aggregate results                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Layer 3: FEW-SHOT LEARNING                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Provide examples of good forecasts in the prompt          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Layer 2: CHAIN-OF-THOUGHT                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Force step-by-step reasoning before the answer           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Layer 1: ROLE + CONTEXT + FORMAT (Foundation)                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  System role, domain context, exact output specification  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12.2 Layer 1: Role, Context, and Format (The Foundation)

Every forecasting prompt must nail three fundamentals before applying advanced techniques.

### 12.2.1 Role Design

The system prompt role shapes GPT's entire reasoning process. For forecasting, the role should be:

1. **Specific** — not "helpful assistant" but "senior demand planning analyst with 15 years of retail experience"
2. **Domain-anchored** — specify the exact industry and function
3. **Behavior-constraining** — tell GPT what it must and must not do

```python
"""
Chapter 12 - Role Design for Forecasting Prompts
"""

# ── WEAK ROLE (avoid)
WEAK_SYSTEM = "You are a helpful assistant that can do forecasting."

# ── STRONG ROLES (by domain)
ROLES = {
    'retail_demand': """You are a senior demand planning analyst with 15 years 
of experience in retail and consumer packaged goods. You specialize in 
seasonal demand forecasting, promotional lift modeling, and inventory 
optimization. You always:
- Acknowledge uncertainty explicitly
- Separate trend from seasonal effects before forecasting  
- Flag any anomalies in the data before proceeding
- Output only valid JSON when asked for structured data
- Never fabricate numbers — use "null" if unsure""",

    'financial': """You are a quantitative analyst at a top-tier investment bank
specializing in macroeconomic forecasting. You have deep expertise in:
- Economic cycle analysis and leading indicators
- Monetary policy impact on financial metrics
- Cross-asset correlations and regime shifts
You always express uncertainty as ranges, not point estimates.
Output only valid JSON when a structured format is requested.""",

    'energy': """You are a senior energy market analyst specializing in 
electricity load forecasting and renewable energy production planning.
You understand:
- Weather-driven demand elasticity
- Grid stability and peak demand events
- Seasonal heating/cooling degree day calculations
Provide data-driven analysis. Never extrapolate beyond the visible trend.
Output only valid JSON when requested.""",

    'supply_chain': """You are a supply chain forecasting expert with experience
in S&OP (Sales and Operations Planning) at global manufacturing companies.
You understand:
- Bullwhip effect and demand amplification
- Lead time variability and safety stock optimization
- SKU-level vs. aggregate forecast reconciliation
Always output valid JSON when structured data is required.""",
}

print("Role Design Examples:")
for domain, role in ROLES.items():
    n_words = len(role.split())
    print(f"  {domain:20s}: {n_words} words")
```

### 12.2.2 Context Injection Framework

Context is what transforms a generic prompt into a domain-specific, accurate forecasting instruction. Use the PADIS framework:

```
P — Period: What time period does the data cover?
A — Audience: Who will use this forecast?
D — Data: What series is being forecast (units, currency, frequency)?
I — Influences: What external factors are known to affect this series?
S — Scope: What is the forecast horizon and granularity?
```

```python
"""
Chapter 12 - Context Injection with PADIS Framework
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ForecastContext:
    """
    Structured context object for forecasting prompts.
    Enforces complete context before prompt generation.
    """
    # P — Period
    start_date:        str
    end_date:          str
    frequency:         str        # 'daily', 'weekly', 'monthly', 'quarterly'

    # A — Audience
    audience:          str        # 'board', 'operations', 'analyst', 'technical'

    # D — Data
    series_name:       str
    units:             str
    domain:            str

    # I — Influences (known exogenous factors)
    known_influences:  List[str] = field(default_factory=list)
    recent_events:     List[str] = field(default_factory=list)

    # S — Scope
    horizon:           int = 12
    horizon_unit:      str = 'months'
    output_granularity: str = 'monthly'

    # Additional metadata
    seasonality_period: Optional[int] = None
    historical_growth:  Optional[float] = None    # % annual growth rate
    volatility:         Optional[str] = None      # 'low', 'medium', 'high'

    def to_context_block(self) -> str:
        """Render context as a structured prompt block."""
        lines = [
            "## Forecast Context (PADIS Framework)",
            "",
            f"**Period**: {self.start_date} to {self.end_date} ({self.frequency})",
            f"**Audience**: {self.audience} — calibrate technical depth accordingly",
            f"**Data**: {self.series_name} in {self.units} ({self.domain})",
            f"**Forecast Scope**: {self.horizon} {self.horizon_unit} ahead, "
            f"at {self.output_granularity} granularity",
        ]

        if self.seasonality_period:
            lines.append(f"**Seasonality**: Period = {self.seasonality_period} {self.frequency}")
        if self.historical_growth:
            lines.append(f"**Historical Growth Rate**: {self.historical_growth:+.1f}% per year")
        if self.volatility:
            lines.append(f"**Volatility**: {self.volatility}")

        if self.known_influences:
            lines.append("\n**Known Influencing Factors:**")
            for inf in self.known_influences:
                lines.append(f"  - {inf}")

        if self.recent_events:
            lines.append("\n**Recent Events to Consider:**")
            for evt in self.recent_events:
                lines.append(f"  - {evt}")

        return '\n'.join(lines)


# Example usage
airline_context = ForecastContext(
    start_date='1949-01',
    end_date='1959-12',
    frequency='monthly',
    audience='operations',
    series_name='International Airline Passengers',
    units='thousands of passengers',
    domain='commercial aviation',
    known_influences=[
        'Summer vacation travel peaks in July-August',
        'Post-WWII economic expansion increasing disposable income',
        'Introduction of jet aircraft expanding capacity',
        'Declining real airfare prices over the period',
    ],
    recent_events=[
        'Boeing 707 entered commercial service (1958)',
        'Increased transatlantic route competition',
    ],
    horizon=12,
    horizon_unit='months',
    seasonality_period=12,
    historical_growth=12.5,
    volatility='low',
)

print(airline_context.to_context_block())
```

### 12.2.3 Output Format Specification

GPT's output must be parseable by Python. Always specify:

1. The exact JSON schema
2. Field types and constraints
3. What to do with missing information (use null, never invent)
4. Whether to include reasoning fields

```python
"""
Chapter 12 - Output Format Templates
"""

from typing import Literal
from pydantic import BaseModel, Field


class ForecastStep(BaseModel):
    """Schema for a single forecast step."""
    step:          int    = Field(..., ge=1, description="Step number (1-indexed)")
    period:        str    = Field(..., description="Period label e.g. '2024-01'")
    point:         float  = Field(..., description="Point forecast value")
    lower_80:      Optional[float] = Field(None, description="80th percentile lower bound")
    upper_80:      Optional[float] = Field(None, description="80th percentile upper bound")
    confidence:    Literal['high', 'medium', 'low'] = Field('medium')
    key_driver:    Optional[str]  = Field(None, description="Primary factor driving this step")


class AnalysisBlock(BaseModel):
    """Schema for qualitative analysis accompanying a forecast."""
    trend:         Literal['up', 'down', 'flat'] = Field(...)
    trend_strength: Literal['strong', 'moderate', 'weak'] = Field(...)
    seasonal_peak_months:  List[str] = Field(default_factory=list)
    seasonal_trough_months: List[str] = Field(default_factory=list)
    primary_risk:  str = Field(..., description="Biggest uncertainty factor")
    yoy_growth_forecast: float = Field(..., description="Expected % YoY change")


class ForecastResponse(BaseModel):
    """Complete structured forecast response."""
    analysis:  AnalysisBlock
    forecast:  List[ForecastStep]
    notes:     Optional[str] = None


def schema_to_prompt_block(schema_class: type) -> str:
    """
    Convert a Pydantic model to a JSON schema description for the prompt.
    This tells GPT exactly what structure to produce.
    """
    schema = schema_class.model_json_schema()
    return f"""## Required Output Format
Return ONLY valid JSON matching this schema:
{schema}

Constraints:
- All numeric fields must be actual numbers (no strings like "N/A")
- Use null for unknown/uncertain values (never invent data)
- The 'forecast' array must have exactly {{horizon}} elements
- Do NOT include any text outside the JSON object"""


def build_format_block(horizon: int, include_intervals: bool = True) -> str:
    """Build a concrete output format example."""
    example_steps = []
    for i in range(1, min(3, horizon + 1)):
        step = {
            "step":       i,
            "period":     f"2024-{i:02d}",
            "point":      "<number>",
        }
        if include_intervals:
            step["lower_80"] = "<number>"
            step["upper_80"] = "<number>"
        step["confidence"]  = "high/medium/low"
        step["key_driver"]  = "<brief description or null>"
        example_steps.append(step)

    if horizon > 2:
        example_steps.append({"note": f"... continue for all {horizon} steps ..."})

    return f"""## Required Output Format
Return ONLY this JSON structure — no other text:
{{
  "analysis": {{
    "trend": "up|down|flat",
    "trend_strength": "strong|moderate|weak",
    "seasonal_peak_months": ["<month>", "<month>"],
    "seasonal_trough_months": ["<month>"],
    "primary_risk": "<biggest uncertainty factor>",
    "yoy_growth_forecast": <number as decimal, e.g. 5.2 means 5.2%>
  }},
  "forecast": {example_steps},
  "notes": "<any important caveats or null>"
}}"""


print(build_format_block(horizon=12, include_intervals=True))
```

---

## 12.3 Layer 2: Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting forces GPT to reason step-by-step before producing the final answer. Research shows CoT dramatically improves numerical reasoning accuracy.

### 12.3.1 Why CoT Helps Forecasting

Without CoT, GPT jumps directly from "here is the data" to "here is the forecast" — skipping the analytical steps that a good forecaster would take. With CoT, it must:

1. Compute the trend explicitly
2. Identify seasonal patterns by month
3. Calculate year-over-year growth rates
4. Consider external influences
5. THEN generate the forecast

Each step catches potential errors before they propagate to the final number.

```python
"""
Chapter 12 - Chain-of-Thought Prompting for Forecasting
"""

import pandas as pd
import numpy as np
import json


def build_cot_forecast_prompt(
    series: pd.Series,
    context: 'ForecastContext',
    horizon: int,
    few_shot_examples: list = None,
) -> str:
    """
    Build a chain-of-thought prompt for time series forecasting.

    The CoT structure forces GPT to:
    Step 1: Analyze trend direction and magnitude
    Step 2: Decompose seasonality
    Step 3: Compute growth rates
    Step 4: Assess uncertainty
    Step 5: Generate forecast
    Step 6: Validate (sanity check the forecast)
    """
    context_vals  = series.iloc[-36:].values.tolist()
    context_dates = [str(d.date()) for d in series.iloc[-36:].index]

    # Compute helper statistics for GPT
    n   = len(series)
    yoy = (series.iloc[-1] - series.iloc[-13]) / series.iloc[-13] * 100 \
          if n >= 14 else None

    monthly_avgs = {}
    for month in range(1, 13):
        month_vals = series[series.index.month == month].values
        if len(month_vals) > 0:
            monthly_avgs[month] = round(float(month_vals.mean()), 1)

    overall_mean = round(float(series.mean()), 1)
    seasonal_indices = {
        m: round(v / overall_mean, 3)
        for m, v in monthly_avgs.items()
    }

    few_shot_block = ""
    if few_shot_examples:
        few_shot_block = "\n## Reference Examples (similar forecasting tasks)\n"
        for i, ex in enumerate(few_shot_examples, 1):
            few_shot_block += f"\n### Example {i}\n{ex}\n"

    prompt = f"""You are a {context.domain} forecasting expert.

{context.to_context_block()}

## Historical Data (Last 36 Months)
Dates:  {context_dates}
Values: {context_vals}

## Pre-Computed Statistics
- Overall mean:         {overall_mean} {context.units}
- Recent YoY growth:   {f"{yoy:+.1f}%" if yoy else "N/A"}
- Seasonal indices by month (>1.0 = above average):
  {json.dumps(seasonal_indices, indent=2)}

{few_shot_block}

## Forecasting Task
Generate a {horizon}-{context.horizon_unit} forecast.

## REQUIRED REASONING STEPS (follow in order)

### Step 1: Trend Analysis
Analyze the trend over the last 36 months.
- Direction (up/down/flat)?
- Linear or accelerating/decelerating?
- Any trend breaks or inflection points?
- Estimated monthly trend increment (units/month)?

### Step 2: Seasonality Decomposition
Using the seasonal indices above:
- Which months are strongest (index > 1.1)?
- Which months are weakest (index < 0.9)?
- Is the seasonal pattern stable or shifting?

### Step 3: Growth Rate Analysis
- What is the recent YoY growth rate?
- Is it consistent with the longer-term historical average?
- Any signs of acceleration or deceleration?

### Step 4: Uncertainty Assessment
- What is the primary source of forecast uncertainty?
- What events could cause a significant deviation (>10%)?
- Should confidence vary across the horizon? Why?

### Step 5: Forecast Generation
Using your analysis from Steps 1-4, generate the {horizon}-step forecast.
For each step: apply trend + seasonal index + growth adjustment.

### Step 6: Sanity Check
Before finalizing:
- Do the forecasts stay within a reasonable range?
- Are seasonal patterns preserved correctly?
- Is YoY growth consistent with your analysis?

## Required Output
Return ONLY this JSON (include your reasoning in the analysis fields):
{{
  "reasoning": {{
    "trend_analysis":        "<summary of Step 1>",
    "trend_monthly_increment": <number>,
    "peak_months":           ["<month>", ...],
    "trough_months":         ["<month>", ...],
    "growth_rate_used":      <number as % e.g. 10.5>,
    "primary_uncertainty":   "<summary of Step 4>",
    "sanity_check_passed":   true/false,
    "sanity_check_notes":    "<any issues found>"
  }},
  "forecast": [
    {{"step": 1, "period": "<YYYY-MM>", "point": <number>, 
      "lower_80": <number>, "upper_80": <number>, "confidence": "high/medium/low"}},
    ... (continue for all {horizon} steps)
  ],
  "summary": "<one sentence forecast summary for non-technical stakeholders>"
}}"""

    return prompt


# Example: Build a CoT prompt for airline passengers
url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/airline-passengers.csv")
df  = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df.columns = ['passengers']
df.index.freq = 'MS'
train_series   = df['passengers'].iloc[:120]

cot_prompt = build_cot_forecast_prompt(
    series=train_series,
    context=airline_context,
    horizon=12,
)

print(f"CoT Prompt length: {len(cot_prompt.split())} words / "
      f"~{len(cot_prompt)//4} tokens")
print("\nFirst 500 chars of prompt:")
print(cot_prompt[:500])
print("...")
```

---

## 12.4 Layer 3: Few-Shot Prompting

Few-shot prompting provides GPT with worked examples — completed forecasting tasks that demonstrate exactly the reasoning quality and output format you expect.

### 12.4.1 Why Few-Shot Works

GPT learns from its context window during inference (not training). By seeing 2–5 high-quality examples of:
- Input time series data
- The reasoning process
- The correct output format

GPT dramatically improves the quality of its forecast for a new series.

### 12.4.2 Building a Few-Shot Example Library

```python
"""
Chapter 12 - Few-Shot Example Library for Forecasting
"""

# ── FEW-SHOT EXAMPLE 1: Monthly retail with strong seasonality
FEW_SHOT_RETAIL = """
### Example: Monthly Retail Sales Forecasting

**Input Data** (monthly sales, $thousands):
Jan: 820, Feb: 780, Mar: 910, Apr: 950, May: 1020,
Jun: 1080, Jul: 1150, Aug: 1120, Sep: 980, Oct: 1050,
Nov: 1380, Dec: 1920  (Year 1)
Jan: 880, Feb: 840, Mar: 975, Apr: 1020, May: 1095

**Expert Reasoning:**

Step 1 (Trend): Consistent upward trend, approximately +8% YoY.
Monthly increment ≈ +6 units.

Step 2 (Seasonality): Clear December peak (index ≈ 1.75),
November secondary peak (index ≈ 1.26), Feb trough (index ≈ 0.71).
Seasonal pattern is stable.

Step 3 (Growth): YoY growth rate +7.3%, consistent with trend.

Step 4 (Uncertainty): Primary risk is holiday promotional intensity.
Confidence: HIGH for Jun-Sep, MEDIUM for Oct-Dec (holiday uncertainty).

Step 5 (Forecast):
Jun: 1095 × 1.00 × 1.073 ≈ 1175 | Jul: 1175 × 1.05 ≈ 1234
Aug: 1205 | Sep: 1052 | Oct: 1127 | Nov: 1481 | Dec: 2062

**Output:**
{
  "reasoning": {
    "trend_monthly_increment": 6.2,
    "growth_rate_used": 7.3,
    "peak_months": ["November", "December"],
    "trough_months": ["February"],
    "primary_uncertainty": "holiday promotional competition",
    "sanity_check_passed": true
  },
  "forecast": [
    {"step": 1, "period": "Jun", "point": 1175, "lower_80": 1110, "upper_80": 1240, "confidence": "high"},
    {"step": 2, "period": "Jul", "point": 1234, "lower_80": 1160, "upper_80": 1308, "confidence": "high"},
    {"step": 7, "period": "Dec", "point": 2062, "lower_80": 1880, "upper_80": 2244, "confidence": "medium"}
  ],
  "summary": "Continued 7% YoY growth with typical holiday peak, December forecast ≈ $2.1M."
}
"""

# ── FEW-SHOT EXAMPLE 2: Weekly energy consumption
FEW_SHOT_ENERGY = """
### Example: Weekly Energy Consumption Forecasting

**Input Data** (weekly MWh, last 8 weeks):
Week 1: 4210, Week 2: 4180, Week 3: 3890, Week 4: 3720,
Week 5: 3610, Week 6: 3580, Week 7: 3790, Week 8: 3850

**Expert Reasoning:**

Step 1 (Trend): Initial decline weeks 1-6 followed by recovery.
Pattern suggests seasonal cooling-to-warming transition.

Step 2 (Seasonality): Spring transition (Mar-Apr): 10-15% below winter peak.
Summer cooling will increase load from late June.

Step 3 (Growth): Flat YoY for utilities. No structural growth expected.

Step 4 (Uncertainty): Weather deviation is primary risk. ±5% per °F temperature deviation.

Step 5 (Forecast): Apply spring trough → summer ramp pattern.
Weeks 9-12: 3920, 4050, 4180, 4350 (summer AC load building).

**Output:**
{
  "reasoning": {
    "trend_monthly_increment": -12.5,
    "growth_rate_used": 0.5,
    "peak_months": ["July", "August", "January"],
    "trough_months": ["April", "May"],
    "primary_uncertainty": "temperature deviation from seasonal norm",
    "sanity_check_passed": true
  },
  "forecast": [
    {"step": 1, "period": "Week 9",  "point": 3920, "lower_80": 3720, "upper_80": 4120, "confidence": "medium"},
    {"step": 4, "period": "Week 12", "point": 4350, "lower_80": 4100, "upper_80": 4600, "confidence": "medium"}
  ],
  "summary": "Spring trough recovery toward summer cooling season peak; high weather sensitivity."
}
"""


class FewShotLibrary:
    """
    Manages a library of few-shot examples for forecasting prompts.
    Selects the most relevant examples based on series characteristics.
    """

    def __init__(self):
        self._examples: dict = {}

    def add_example(
        self,
        name:       str,
        domain:     str,
        frequency:  str,
        pattern:    str,    # 'seasonal', 'trend', 'flat', 'volatile'
        example:    str,
    ) -> None:
        """Register a few-shot example."""
        self._examples[name] = {
            'domain':    domain,
            'frequency': frequency,
            'pattern':   pattern,
            'text':      example,
        }

    def select_examples(
        self,
        domain:     str,
        frequency:  str,
        pattern:    str = None,
        n:          int = 2,
    ) -> list:
        """
        Select the most relevant few-shot examples for a given context.
        Uses domain + frequency matching with pattern as tiebreaker.
        """
        scored = []
        for name, ex in self._examples.items():
            score = 0
            if ex['domain'] == domain:
                score += 3
            elif ex['domain'].split('_')[0] == domain.split('_')[0]:
                score += 1
            if ex['frequency'] == frequency:
                score += 2
            if pattern and ex['pattern'] == pattern:
                score += 1
            scored.append((score, name, ex['text']))

        scored.sort(key=lambda x: -x[0])
        return [text for _, _, text in scored[:n]]


# Build the library
few_shot_lib = FewShotLibrary()
few_shot_lib.add_example(
    'retail_monthly_seasonal', 'retail', 'monthly', 'seasonal', FEW_SHOT_RETAIL
)
few_shot_lib.add_example(
    'energy_weekly_transition', 'energy', 'weekly', 'seasonal', FEW_SHOT_ENERGY
)

# Select relevant examples
examples = few_shot_lib.select_examples(
    domain='commercial_aviation',
    frequency='monthly',
    pattern='seasonal',
    n=1
)
print(f"Selected {len(examples)} few-shot examples")
print(f"Example preview (first 200 chars):\n{examples[0][:200]}..." if examples else "No examples selected")
```

---

## 12.5 Layer 4: Self-Consistency

Self-consistency is a powerful technique for reducing hallucination variance. Instead of running the prompt once and trusting the result, you run it N times (with slight temperature > 0) and aggregate the outputs.

```python
"""
Chapter 12 - Self-Consistency for Forecasting
"""

import numpy as np
import json
from typing import List, Tuple


def run_self_consistency_forecast(
    prompt_fn,          # Function that returns a prompt string
    call_fn,            # Function that calls GPT and returns response string
    parse_fn,           # Function that parses response to np.ndarray
    n_samples: int = 5,
    temperature: float = 0.3,
    aggregation: str = 'median',    # 'mean', 'median', 'trimmed_mean'
    trim_fraction: float = 0.2,     # For trimmed_mean: fraction to trim each side
    horizon: int = 12,
) -> dict:
    """
    Run self-consistency forecasting:
    1. Call GPT N times with the same prompt (temperature > 0)
    2. Parse all N forecasts
    3. Aggregate using median (robust to outliers)
    4. Compute uncertainty from spread across samples

    Args:
        prompt_fn:      Function() → prompt string
        call_fn:        Function(prompt, temp) → response string
        parse_fn:       Function(response) → np.ndarray of forecasts
        n_samples:      Number of independent samples
        temperature:    Temperature for sampling (0.2-0.5 recommended)
        aggregation:    How to combine samples
        horizon:        Expected forecast horizon

    Returns:
        Dict with aggregated forecast and uncertainty estimates
    """
    all_samples = []
    failed      = 0

    prompt = prompt_fn()

    for i in range(n_samples):
        try:
            response = call_fn(prompt, temperature)
            values   = parse_fn(response)

            if len(values) >= horizon * 0.8:   # Accept if at least 80% complete
                # Pad or trim to exactly horizon length
                if len(values) < horizon:
                    values = np.pad(values, (0, horizon - len(values)),
                                    mode='edge')
                all_samples.append(values[:horizon])
            else:
                print(f"  Sample {i+1}: insufficient values ({len(values)}), skipping")
                failed += 1

        except Exception as e:
            print(f"  Sample {i+1} failed: {e}")
            failed += 1

    if not all_samples:
        raise RuntimeError("All self-consistency samples failed")

    samples_array = np.array(all_samples)   # [n_valid, horizon]
    n_valid = len(all_samples)

    print(f"\nSelf-consistency: {n_valid}/{n_samples} samples valid "
          f"({failed} failed)")

    # Aggregate
    if aggregation == 'median':
        forecast = np.median(samples_array, axis=0)
    elif aggregation == 'mean':
        forecast = samples_array.mean(axis=0)
    elif aggregation == 'trimmed_mean':
        from scipy import stats
        forecast = stats.trim_mean(samples_array, proportiontocut=trim_fraction, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Uncertainty from spread across samples
    std_across = samples_array.std(axis=0)
    percentile_spread = {
        '10th': np.percentile(samples_array, 10, axis=0),
        '25th': np.percentile(samples_array, 25, axis=0),
        '75th': np.percentile(samples_array, 75, axis=0),
        '90th': np.percentile(samples_array, 90, axis=0),
    }

    # Agreement score: how consistent are the samples?
    cv = std_across / (np.abs(forecast) + 1e-8)    # Coefficient of variation
    agreement = np.mean(cv < 0.10) * 100    # % of steps where CV < 10%

    print(f"  Forecast agreement: {agreement:.0f}% of steps have CV < 10%")
    print(f"  Avg spread (1σ): ±{std_across.mean():.1f} units")

    return {
        'forecast':            forecast,
        'std':                 std_across,
        'lower_80':            percentile_spread['10th'],
        'upper_80':            percentile_spread['90th'],
        'lower_50':            percentile_spread['25th'],
        'upper_50':            percentile_spread['75th'],
        'samples':             samples_array,
        'n_valid':             n_valid,
        'agreement_score':     round(agreement, 1),
        'aggregation_method':  aggregation,
    }


# ── DEMO: Simulate self-consistency on airline data
def simulate_gpt_forecast(series: np.ndarray, horizon: int, noise_factor: float = 0.05) -> str:
    """Simulate GPT forecast with slight variation (for demo)."""
    m    = 12
    base = np.tile(series[-m:], (horizon // m) + 1)[:horizon]
    trend_monthly = (series[-1] - series[-13]) / 12
    trend_arr     = base + trend_monthly * np.arange(1, horizon + 1)
    noise         = np.random.normal(0, series.std() * noise_factor, horizon)
    values        = trend_arr + noise

    forecast = [
        {"step": i+1, "period": f"1960-{i+1:02d}", "point": round(float(v), 1)}
        for i, v in enumerate(values)
    ]
    return json.dumps({"forecast": forecast})


def parse_demo_response(response: str) -> np.ndarray:
    """Parse simulated GPT response."""
    data   = json.loads(response)
    return np.array([f['point'] for f in data['forecast']])


# Load data
url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/airline-passengers.csv")
df  = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df.columns = ['passengers']
train_arr  = df['passengers'].values[:120].astype(float)
test_arr   = df['passengers'].values[120:].astype(float)
H          = len(test_arr)

np.random.seed(42)

sc_result = run_self_consistency_forecast(
    prompt_fn=lambda: "demo_prompt",
    call_fn=lambda p, t: simulate_gpt_forecast(train_arr, H, noise_factor=0.04),
    parse_fn=parse_demo_response,
    n_samples=10,
    temperature=0.3,
    aggregation='median',
    horizon=H,
)

# Evaluate self-consistency vs. single sample
single_sample = parse_demo_response(simulate_gpt_forecast(train_arr, H, 0.04))
scale = np.mean(np.abs(train_arr[12:] - train_arr[:-12]))

single_mase = np.mean(np.abs(test_arr - single_sample)) / scale
sc_mase     = np.mean(np.abs(test_arr - sc_result['forecast'])) / scale

print(f"\nSingle sample MASE:       {single_mase:.4f}")
print(f"Self-consistency MASE:    {sc_mase:.4f}")
print(f"Improvement:              {(single_mase - sc_mase)/single_mase*100:.1f}%")
```

---

## 12.6 Layer 5: Meta-Prompting

Meta-prompting uses GPT to evaluate and improve its own forecasting prompts automatically.

```python
"""
Chapter 12 - Meta-Prompting: GPT Improves Its Own Forecasting Prompts
"""

def build_meta_prompt_evaluator(
    original_prompt: str,
    forecast_result: np.ndarray,
    actual_result:   np.ndarray,
    metrics:         dict,
) -> str:
    """
    Build a meta-prompt that asks GPT to critique and improve a forecasting prompt.

    Meta-prompting workflow:
    1. Run original prompt → get forecast
    2. Evaluate forecast quality (MASE, MAPE, etc.)
    3. Feed prompt + results + metrics back to GPT
    4. Ask GPT to identify what the prompt is missing
    5. Get an improved prompt
    6. Repeat until quality stabilizes
    """
    mase = metrics.get('MASE', 'N/A')
    mape = metrics.get('MAPE', 'N/A')
    bias = metrics.get('Bias', 'N/A')

    # Show where the forecast went wrong
    errors = forecast_result - actual_result
    max_over_step  = int(np.argmax(errors)) + 1
    max_under_step = int(np.argmin(errors)) + 1

    return f"""You are a prompt engineering expert specializing in time series forecasting.

## Task
Analyze the quality of the following forecasting prompt and improve it.

## Original Prompt
```
{original_prompt[:2000]}
{"... [truncated]" if len(original_prompt) > 2000 else ""}
```

## Forecast Performance Metrics
- MASE: {mase} (target: < 0.8; < 1.0 means better than seasonal naive)
- MAPE: {mape}%  
- Bias: {bias} (positive = over-forecast, negative = under-forecast)

## Error Analysis
- Largest over-forecast at step: {max_over_step} (error = +{errors[max_over_step-1]:.1f})
- Largest under-forecast at step: {max_under_step} (error = {errors[max_under_step-1]:.1f})
- Systematic bias direction: {"OVER-forecasting" if float(bias) > 0 else "UNDER-forecasting"}

## Diagnosis Request
Please analyze:
1. What specific information is MISSING from the original prompt that would reduce the bias?
2. What reasoning steps are NOT being enforced that would catch the systematic error?
3. What output format improvements would make parsing more reliable?
4. What few-shot examples (described briefly, not full examples) would help most?

## Required Output
Return ONLY this JSON:
{{
  "diagnosis": {{
    "primary_failure_mode": "<what is mainly going wrong>",
    "missing_context": ["<item 1>", "<item 2>"],
    "missing_reasoning_steps": ["<step 1>", "<step 2>"],
    "bias_root_cause": "<why is it systematically biased>"
  }},
  "improvements": [
    {{
      "category": "context|reasoning|format|few_shot",
      "description": "<what to add/change>",
      "expected_impact": "high|medium|low",
      "implementation": "<specific text to add to the prompt>"
    }}
  ],
  "improved_system_prompt_addition": "<text to prepend to system prompt>",
  "improved_user_prompt_addition": "<text to add before the forecast request>"
}}"""


def meta_prompt_iteration(
    initial_prompt_fn,
    call_fn,
    parse_fn,
    y_true: np.ndarray,
    y_train: np.ndarray,
    n_iterations: int = 3,
    horizon: int = 12,
) -> dict:
    """
    Iteratively improve a forecasting prompt using meta-prompting.

    Each iteration:
    1. Run current prompt → forecast
    2. Evaluate accuracy
    3. Run meta-evaluator → get improvements
    4. Apply improvements → update prompt
    5. Repeat

    Returns history of prompt versions and their performance.
    """
    scale    = np.mean(np.abs(y_train[12:] - y_train[:-12]))
    history  = []
    current_additions = {'system': '', 'user': ''}

    for iteration in range(1, n_iterations + 1):
        print(f"\n[Meta-Prompt Iteration {iteration}/{n_iterations}]")

        # Run forecast with current prompt
        try:
            response = call_fn(initial_prompt_fn(), temperature=0.0)
            forecast = parse_fn(response)
            forecast = np.array(forecast[:horizon], dtype=float)
        except Exception as e:
            print(f"  Forecast failed: {e}")
            continue

        # Evaluate
        n      = min(len(y_true), len(forecast))
        errors = forecast[:n] - y_true[:n]
        mase   = np.mean(np.abs(errors)) / scale
        mape   = np.mean(np.abs(errors / y_true[:n])) * 100
        bias   = np.mean(errors)

        metrics = {'MASE': round(mase, 4), 'MAPE': round(mape, 2), 'Bias': round(bias, 2)}
        print(f"  MASE={mase:.4f} | MAPE={mape:.2f}% | Bias={bias:.2f}")

        history.append({
            'iteration': iteration,
            'metrics':   metrics,
            'additions': current_additions.copy(),
        })

        if iteration < n_iterations:
            # Meta-evaluate
            meta_prompt = build_meta_prompt_evaluator(
                initial_prompt_fn(), forecast, y_true[:n], metrics
            )
            try:
                meta_response = call_fn(meta_prompt, temperature=0.1)
                meta_data     = json.loads(meta_response.strip().lstrip('```json').rstrip('```'))
                improvements  = meta_data.get('improvements', [])

                print(f"  Meta-diagnosis: {meta_data.get('diagnosis', {}).get('primary_failure_mode', 'N/A')}")
                print(f"  {len(improvements)} improvements suggested")

                # Apply top-2 high-impact improvements
                for imp in sorted(improvements,
                                   key=lambda x: {'high': 3, 'medium': 2, 'low': 1}
                                   .get(x.get('expected_impact', 'low'), 0),
                                   reverse=True)[:2]:
                    implementation = imp.get('implementation', '')
                    category       = imp.get('category', 'user')
                    print(f"  Applying: [{category}] {imp.get('description', '')[:60]}")

                    if category == 'context':
                        current_additions['user'] += f"\n\n{implementation}"
                    else:
                        current_additions['system'] += f"\n{implementation}"

            except Exception as e:
                print(f"  Meta-evaluation failed: {e}")

    # Summary
    if len(history) >= 2:
        initial_mase = history[0]['metrics']['MASE']
        final_mase   = history[-1]['metrics']['MASE']
        improvement  = (initial_mase - final_mase) / initial_mase * 100
        print(f"\n📈 Meta-prompting improvement: {improvement:+.1f}% MASE reduction")
        print(f"   Initial MASE: {initial_mase:.4f} → Final MASE: {final_mase:.4f}")

    return {'history': history, 'final_additions': current_additions}
```

---

## 12.7 The Complete Prompt Template Library

```python
"""
Chapter 12 - Production Prompt Template Library
Version-controlled, parameterized templates for all forecasting scenarios.
"""

from string import Template
from datetime import datetime


class ForecastingPromptLibrary:
    """
    Production prompt library with version control and parameterization.

    Templates are stored with metadata for tracking which version
    produced which forecasts — essential for audit trails.

    Usage:
        lib = ForecastingPromptLibrary()
        prompt = lib.get('monthly_seasonal_cot', version='v2.1',
                         series_name='Monthly Sales',
                         horizon=12, ...)
    """

    VERSION = '2.1.0'
    LAST_UPDATED = '2024-01-15'

    # Template: Monthly Seasonal — Chain of Thought
    MONTHLY_SEASONAL_COT = Template("""
You are a $role specializing in $domain.

## Data
Series: $series_name ($units)
Frequency: Monthly | Seasonality period: 12 months
Recent 24 months: $recent_values

## Statistics
Mean: $mean | Std: $std | Recent YoY: $yoy_pct%

## Reasoning Steps (REQUIRED — follow in order)
1. TREND: Describe direction, strength, and monthly increment
2. SEASONALITY: Identify peak months (idx > 1.1) and trough months (idx < 0.9)
3. GROWTH RATE: Compute recent vs. historical average; note any change
4. UNCERTAINTY: Identify the #1 risk factor; assign confidence
5. FORECAST: Apply trend + seasonal adjustment; show your math for Step 1
6. SANITY CHECK: Confirm seasonal pattern preserved; flag any outliers

## Output (ONLY this JSON)
{
  "reasoning": {
    "trend": "$empty",
    "seasonal_peaks": [],
    "seasonal_troughs": [],
    "growth_rate_pct": 0,
    "primary_risk": "$empty",
    "sanity_check": "$empty"
  },
  "forecast": [
    {"step": 1, "period": "$empty", "point": 0, "lower_80": 0, "upper_80": 0, "confidence": "high"},
    ... $horizon steps total
  ]
}
""")

    # Template: Hybrid Adjustment
    HYBRID_ADJUSTMENT = Template("""
You are a $role.

## Base Model Forecast ($base_model_name)
Values: $base_values

## Qualitative Intelligence
Trend: $trend_direction ($trend_strength)
Momentum: $momentum
Risk factors: $risks

## Adjustment Task
Review the base model forecast. Apply adjustments (0.75 to 1.25 range only).
State the reason for any non-trivial adjustment (factor != 1.00).

## Output
{
  "adjustments": [
    {"step": 1, "factor": 1.00, "adjusted": 0, "reason": "no change needed"},
    ... $horizon total
  ],
  "net_adjustment_pct": 0,
  "confidence_in_base_model": "high/medium/low"
}
""")

    # Template: Scenario Generation
    SCENARIO_GENERATION = Template("""
You are a $role scenario planning expert for $domain.

## Context
$context_block

## Base Forecast
$base_values

## Scenario Requirements
Generate three scenarios for $horizon $horizon_unit ahead:
- BEAR (~20th percentile): name the 2 specific negative catalysts
- BASE (~50th percentile): current trend continuation
- BULL (~80th percentile): name the 2 specific positive catalysts

Probabilities must sum to 1.0.
Each scenario must have $horizon numeric values.

## Output
{
  "scenarios": {
    "bear":  {"values": [], "probability": 0.20, "catalysts": [], "label": ""},
    "base":  {"values": [], "probability": 0.60, "catalysts": [], "label": ""},
    "bull":  {"values": [], "probability": 0.20, "catalysts": [], "label": ""}
  },
  "key_uncertainty": "$empty"
}
""")

    # Template: Anomaly Review
    ANOMALY_REVIEW = Template("""
You are a $role data quality analyst.

## Time Series (last 36 months)
$series_data

## Task
Identify any data quality issues or genuine anomalies:
1. Values that appear to be data errors (wrong units, typos, missing)
2. Genuine anomalies (real demand spikes/drops with identifiable causes)
3. Structural breaks (permanent level shifts)

## Output
{
  "anomalies": [
    {"step": 0, "value": 0, "type": "data_error|genuine_anomaly|structural_break",
     "likely_cause": "$empty", "should_exclude_from_training": true}
  ],
  "recommendation": "$empty",
  "clean_series_ready": true
}
""")

    @classmethod
    def get_template(cls, name: str) -> Template:
        """Get a prompt template by name."""
        templates = {
            'monthly_seasonal_cot':  cls.MONTHLY_SEASONAL_COT,
            'hybrid_adjustment':     cls.HYBRID_ADJUSTMENT,
            'scenario_generation':   cls.SCENARIO_GENERATION,
            'anomaly_review':        cls.ANOMALY_REVIEW,
        }
        if name not in templates:
            raise ValueError(
                f"Template '{name}' not found. "
                f"Available: {list(templates.keys())}"
            )
        return templates[name]

    @classmethod
    def build(
        cls,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Build a prompt from a template with substitution.

        Args:
            template_name: Name of template to use
            **kwargs:       Template variables

        Returns:
            Filled prompt string
        """
        # Add library metadata
        kwargs.setdefault('empty', '')
        template = cls.get_template(template_name)

        try:
            return template.safe_substitute(**kwargs).strip()
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    @classmethod
    def list_templates(cls) -> dict:
        """Return all available templates with descriptions."""
        return {
            'monthly_seasonal_cot': 'Monthly seasonal series with chain-of-thought reasoning',
            'hybrid_adjustment':    'Quantitative base model + GPT qualitative adjustment',
            'scenario_generation':  'Bear/base/bull scenario generation',
            'anomaly_review':       'Data quality check and anomaly identification',
        }


# Example usage
lib_prompt = ForecastingPromptLibrary.build(
    'monthly_seasonal_cot',
    role='demand planning analyst',
    domain='commercial aviation',
    series_name='Monthly Airline Passengers',
    units='thousands',
    recent_values=[str(v) for v in train_series.values[-24:].tolist()],
    mean=round(float(train_series.mean()), 1),
    std=round(float(train_series.std()), 1),
    yoy_pct=round(float((train_series.iloc[-1] - train_series.iloc[-13]) /
                         train_series.iloc[-13] * 100), 1),
    horizon=12,
)

print("Library prompt preview (first 400 chars):")
print(lib_prompt[:400])
print("...")
print(f"\nTemplate library version: {ForecastingPromptLibrary.VERSION}")
print(f"Available templates: {list(ForecastingPromptLibrary.list_templates().keys())}")
```

---

## 12.8 Prompt Evaluation Framework

```python
"""
Chapter 12 - Systematic Prompt Evaluation Framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Tuple


class PromptEvaluator:
    """
    Systematically evaluate and compare forecasting prompt variants.

    Workflow:
    1. Define baseline and variant prompts
    2. Run each on N backtest windows
    3. Compute MASE, MAPE, calibration for each
    4. Statistical test for significance
    5. Select the best-performing prompt

    Usage:
        evaluator = PromptEvaluator(call_fn, parse_fn)
        evaluator.add_prompt('baseline_v1', baseline_prompt_fn)
        evaluator.add_prompt('cot_v2',      cot_prompt_fn)
        results = evaluator.run(series, n_rounds=10)
        evaluator.report(results)
    """

    def __init__(
        self,
        call_fn:   Callable,   # fn(prompt, temp) → response string
        parse_fn:  Callable,   # fn(response) → np.ndarray
        temperature: float = 0.0,
    ) -> None:
        self.call_fn     = call_fn
        self.parse_fn    = parse_fn
        self.temperature = temperature
        self._prompts: Dict[str, Callable] = {}

    def add_prompt(self, name: str, prompt_fn: Callable) -> None:
        """Register a prompt variant for evaluation."""
        self._prompts[name] = prompt_fn

    def _run_one_round(
        self,
        train: np.ndarray,
        test:  np.ndarray,
        h:     int,
        scale: float,
    ) -> Dict[str, Dict[str, float]]:
        """Run all prompt variants on one backtest window."""
        results = {}
        for name, prompt_fn in self._prompts.items():
            try:
                prompt   = prompt_fn(train, h)
                response = self.call_fn(prompt, self.temperature)
                forecast = self.parse_fn(response)

                n    = min(len(test), len(forecast), h)
                mae  = np.mean(np.abs(test[:n] - forecast[:n]))
                mase = mae / scale if scale > 0 else np.nan
                mape = np.mean(np.abs((test[:n] - forecast[:n]) /
                                       test[:n])) * 100
                bias = np.mean(forecast[:n] - test[:n])

                results[name] = {
                    'MASE': round(mase, 4),
                    'MAPE': round(mape, 3),
                    'Bias': round(bias, 3),
                    'n':    n,
                }
            except Exception as e:
                print(f"  [{name}] failed: {e}")
                results[name] = {'MASE': np.nan, 'MAPE': np.nan,
                                  'Bias': np.nan, 'n': 0}
        return results

    def run(
        self,
        series:        np.ndarray,
        h:             int,
        n_rounds:      int = 10,
        min_train:     int = 48,
        seasonality:   int = 12,
    ) -> pd.DataFrame:
        """
        Run evaluation across n_rounds backtest windows.

        Returns:
            DataFrame with all round-level results
        """
        n      = len(series)
        scale  = np.mean(np.abs(series[seasonality:] - series[:-seasonality]))
        step   = max(1, (n - min_train - h) // n_rounds)
        rows   = []

        print(f"Evaluating {len(self._prompts)} prompt variants "
              f"× {n_rounds} rounds...")

        for round_num, cutoff in enumerate(
            range(min_train, n - h + 1, step), 1
        ):
            if round_num > n_rounds:
                break
            train = series[:cutoff]
            test  = series[cutoff:cutoff + h]
            if len(test) < h:
                continue

            round_results = self._run_one_round(train, test, h, scale)
            for name, metrics in round_results.items():
                rows.append({
                    'prompt':   name,
                    'round':    round_num,
                    'cutoff':   cutoff,
                    **metrics,
                })

        return pd.DataFrame(rows)

    def report(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a summary comparison report."""
        from scipy import stats as scipy_stats

        summary = (
            results_df.groupby('prompt')
            .agg(
                MASE_mean=('MASE', 'mean'),
                MASE_std=('MASE', 'std'),
                MAPE_mean=('MAPE', 'mean'),
                Bias_mean=('Bias', 'mean'),
                N_rounds=('MASE', 'count'),
            )
            .round(4)
            .sort_values('MASE_mean')
        )

        print("\n" + "=" * 70)
        print("  PROMPT EVALUATION RESULTS")
        print("=" * 70)
        print(summary.to_string())
        print("=" * 70)

        # Pairwise significance testing
        prompts = list(self._prompts.keys())
        if len(prompts) >= 2:
            best = summary.index[0]
            for other in prompts:
                if other == best:
                    continue
                a = results_df[results_df['prompt'] == best]['MASE'].dropna()
                b = results_df[results_df['prompt'] == other]['MASE'].dropna()
                n = min(len(a), len(b))
                if n >= 5:
                    stat, p = scipy_stats.wilcoxon(a.values[:n], b.values[:n])
                    sig     = "✅ SIGNIFICANT" if p < 0.05 else "❌ not significant"
                    print(f"  {best} vs {other}: p={p:.4f} {sig}")

        return summary

    def plot_comparison(self, results_df: pd.DataFrame) -> None:
        """Visualize prompt performance comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        prompts = results_df['prompt'].unique()
        colors  = plt.cm.Set2(np.linspace(0, 1, len(prompts)))

        # MASE distribution
        for i, prompt in enumerate(prompts):
            mases = results_df[results_df['prompt'] == prompt]['MASE'].dropna()
            axes[0].hist(mases, alpha=0.6, label=prompt,
                         color=colors[i], bins=10, edgecolor='white')
        axes[0].axvline(1.0, color='red', linestyle='--',
                        linewidth=2, label='Naive baseline')
        axes[0].set_title('MASE Distribution by Prompt', fontweight='bold')
        axes[0].set_xlabel('MASE')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # MASE by round (stability)
        for i, prompt in enumerate(prompts):
            p_data = results_df[results_df['prompt'] == prompt]
            axes[1].plot(p_data['round'], p_data['MASE'],
                         marker='o', linewidth=1.5, markersize=4,
                         color=colors[i], alpha=0.8, label=prompt)
        axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.5)
        axes[1].set_title('MASE Over Backtest Rounds\n(Stability Check)',
                           fontweight='bold')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('MASE')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Mean MASE bar chart
        summary = results_df.groupby('prompt')['MASE'].mean().sort_values()
        bar_colors = [colors[list(prompts).index(p)] for p in summary.index]
        bars = axes[2].bar(range(len(summary)), summary.values,
                           color=bar_colors, edgecolor='white', width=0.6)
        for bar, val in zip(bars, summary.values):
            axes[2].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
        axes[2].set_xticks(range(len(summary)))
        axes[2].set_xticklabels(summary.index, rotation=15, ha='right', fontsize=9)
        axes[2].axhline(1.0, color='red', linestyle='--', linewidth=1.5)
        axes[2].set_title('Mean MASE\n(Lower = Better)', fontweight='bold')
        axes[2].grid(True, axis='y', alpha=0.3)

        plt.suptitle('Prompt Engineering Evaluation Dashboard',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prompt_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()


# Demo: Evaluate two prompt variants
def basic_prompt_fn(train: np.ndarray, h: int) -> str:
    return f"Forecast next {h} values. Data: {train[-12:].tolist()}. Return JSON: {{\"forecast\": [...]}}"

def cot_prompt_fn(train: np.ndarray, h: int) -> str:
    m     = 12
    yoy   = (train[-1] - train[-13]) / train[-13] * 100 if len(train) > 13 else 0
    base  = np.tile(train[-m:], (h // m) + 1)[:h]
    vals  = [round(v + np.random.normal(0, train.std() * 0.03), 1) for v in base]
    return json.dumps({"forecast": [{"step": i+1, "point": v} for i, v in enumerate(vals)]})

def demo_call_fn(prompt: str, temp: float) -> str:
    """Simulate GPT responses with different quality for each prompt type."""
    if "Return JSON:" in prompt:
        # Basic prompt: noisier output
        train_sim = np.array([500 + i * 2 for i in range(20)])
        vals = train_sim[-1:] * np.random.uniform(0.9, 1.1, 12)
        return json.dumps({"forecast": [{"step": i+1, "point": round(float(v), 1)}
                                         for i, v in enumerate(vals)]})
    else:
        # CoT prompt: better output (actual return from prompt_fn)
        return prompt

def demo_parse_fn(response: str) -> np.ndarray:
    try:
        data = json.loads(response)
        fc   = data.get('forecast', [])
        return np.array([f.get('point', f) if isinstance(f, dict) else float(f)
                          for f in fc], dtype=float)
    except Exception:
        return np.array([])

# Run evaluation
series_arr = df['passengers'].values.astype(float)
evaluator  = PromptEvaluator(demo_call_fn, demo_parse_fn)
evaluator.add_prompt('Basic Prompt',    basic_prompt_fn)
evaluator.add_prompt('CoT Prompt',      cot_prompt_fn)

eval_results = evaluator.run(series_arr, h=12, n_rounds=10, min_train=48)
summary      = evaluator.report(eval_results)
evaluator.plot_comparison(eval_results)
```

---

## 12.9 Production Prompt Best Practices

```python
"""
Chapter 12 - Production Prompt Engineering Checklist
"""

PRODUCTION_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════╗
║       PRODUCTION PROMPT ENGINEERING CHECKLIST                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  DESIGN                                                          ║
║  □ System role is domain-specific (not generic "assistant")     ║
║  □ PADIS context is complete and accurate                       ║
║  □ Output format is explicit JSON schema                        ║
║  □ Temperature = 0.0 for numerical forecasts                    ║
║  □ Chain-of-thought steps specified in order                    ║
║  □ Null policy defined ("use null, never invent")               ║
║                                                                  ║
║  RELIABILITY                                                     ║
║  □ JSON parsing handles malformed responses gracefully           ║
║  □ Retry logic with exponential backoff                          ║
║  □ Classical model fallback when GPT fails                      ║
║  □ Adjustment caps for hybrid mode (±25% max)                   ║
║  □ Self-consistency runs ≥ 5 samples for critical forecasts     ║
║                                                                  ║
║  TESTING                                                         ║
║  □ Evaluate on ≥ 10 backtest rounds before production          ║
║  □ Compare against Holt-Winters baseline (MASE benchmark)       ║
║  □ Test prompt on series from different domains                  ║
║  □ Verify calibration of prediction intervals                    ║
║  □ Run statistical significance test vs. current best prompt    ║
║                                                                  ║
║  OPERATIONS                                                      ║
║  □ Prompt versions tracked in version control (git)             ║
║  □ Each API call logged with: prompt_version, tokens, cost, MASE║
║  □ Cost per series per month monitored with alerts              ║
║  □ Monthly prompt performance review scheduled                  ║
║  □ Meta-prompting improvement run quarterly                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(PRODUCTION_CHECKLIST)
```

---

## 12.10 Summary

In this chapter, you learned:

- **The Forecasting Prompt Engineering Stack**: five layers from Role+Context+Format up to Meta-Prompting.
- **Role design**: domain-specific expert roles that shape GPT's entire reasoning process.
- **PADIS context framework**: Period, Audience, Data, Influences, Scope — ensuring complete context injection.
- **Chain-of-thought prompting**: forcing explicit reasoning steps (trend → seasonality → growth → uncertainty → forecast → sanity check) before the final numbers.
- **Few-shot learning**: a `FewShotLibrary` class that selects the most relevant worked examples by domain and frequency.
- **Self-consistency**: running N independent samples and aggregating with median to reduce hallucination variance.
- **Meta-prompting**: using GPT to automatically diagnose and improve its own forecasting prompts.
- **Production Prompt Template Library**: version-controlled Jinja-style templates for all forecasting scenarios.
- **Systematic prompt evaluation**: the `PromptEvaluator` class with MASE tracking, Wilcoxon significance testing, and visual comparison.

The next chapter moves to **Scenario Forecasting** — using advanced prompting and structured GPT pipelines to generate multi-scenario forecasts for strategic planning.

---

## Exercises

### Exercise 12.1 — PADIS Context Builder
Implement a `ForecastContext.from_series()` class method that automatically populates the PADIS fields by analyzing a time series: (1) detect frequency from DatetimeIndex, (2) compute historical growth rate, (3) identify peak/trough months from seasonal decomposition, (4) classify volatility as low/medium/high from CV.

### Exercise 12.2 — Chain-of-Thought Ablation
Compare the CoT prompt against a "no-reasoning" version on 15 backtest rounds of the airline dataset. Specifically: (a) does the CoT prompt have lower MASE? (b) Is the bias (MFE) smaller? (c) Are the seasonal peaks correctly placed in the CoT version?

### Exercise 12.3 — Few-Shot Example Quality
Create two versions of a few-shot example for monthly retail sales: (a) a high-quality example with correct reasoning and accurate seasonality; (b) a deliberately incorrect example with wrong seasonal indices. Measure how much each version affects GPT forecast quality on a test series.

### Exercise 12.4 — Self-Consistency Sample Count
Run self-consistency with N = [1, 3, 5, 10, 20] samples on the airline dataset. Plot MASE and forecast standard deviation vs. N. At what sample count does the MASE stabilize? What is the cost-accuracy tradeoff curve?

### Exercise 12.5 — ForecastingPromptLibrary Extension
Extend `ForecastingPromptLibrary` to:
1. Add a `DAILY_PATTERN` template for daily series with weekly + annual seasonality.
2. Add a `get_version_history()` class method that returns a dict of all templates with their creation date and last modified date.
3. Add an `A_B_test()` method that runs two templates on the same series and returns the winner with its MASE improvement.

---

## Interview Questions

**Q1: What is chain-of-thought prompting and why does it improve forecast accuracy?**

Chain-of-thought (CoT) prompting explicitly instructs GPT to reason step-by-step before producing the final output. For forecasting, a CoT prompt requires GPT to: (1) identify the trend direction and magnitude; (2) decompose seasonal patterns; (3) compute growth rates; (4) assess uncertainty. Each step provides a checkpoint where reasoning errors can be caught before they propagate to the final number. Without CoT, GPT "shortcuts" from data to answer, often missing structural patterns. Research shows CoT typically improves numerical reasoning accuracy by 10–30% on complex analytical tasks.

**Q2: What is self-consistency and how does it reduce hallucination in forecasts?**

Self-consistency runs the same prompt N times with slight randomness (temperature > 0) and aggregates the results, typically using median. Since GPT's hallucinations are random — different runs produce different wrong answers — they tend to cancel out when aggregated, while the correct reasoning pattern (which is consistent across runs) reinforces. The median is preferred over mean because it is robust to outlier samples where GPT produces a wildly wrong forecast. Using 5–10 samples typically reduces forecast variance by 30–50% compared to a single call.

**Q3: How would you design a few-shot example library for a multi-domain forecasting system?**

A good few-shot library needs: (1) examples organized by domain (retail, energy, finance), frequency (daily, weekly, monthly), and pattern type (seasonal, trending, volatile); (2) each example showing the full reasoning chain — data analysis → seasonality identification → growth rate computation → forecast generation; (3) a similarity-based retrieval system that selects the 2–3 most relevant examples for each new series based on domain and pattern matching; (4) quality-controlled examples validated against known ground truth; (5) version control to track which examples produced the best results over time.

**Q4: What is meta-prompting and when is it useful in production forecasting systems?**

Meta-prompting uses GPT to evaluate and improve its own forecasting prompts. The workflow is: (1) run original prompt → get forecast; (2) evaluate against ground truth; (3) feed original prompt + performance metrics back to GPT asking it to diagnose what's missing; (4) apply suggested improvements to create a new prompt version; (5) repeat. It is most useful when: you have systematic bias (consistent over- or under-forecasting); your prompt produces poor calibration for prediction intervals; you have domain expertise embedded in ground truth data but struggle to articulate it in a prompt. Meta-prompting is a quarterly maintenance activity, not a real-time technique.

**Q5: What should every production forecasting prompt include that most practitioners miss?**

Most practitioners miss three critical elements: (1) **Null policy** — explicitly telling GPT "use null when uncertain, never invent data" prevents fabricated values that look plausible; (2) **Sanity check step** — asking GPT to verify that its forecast preserves the observed seasonal pattern and stays within a plausible range catches many errors before they reach the output; (3) **Audience-calibrated confidence** — telling GPT who the audience is (board vs. operations vs. technical) adjusts whether GPT flags uncertainty appropriately or overconfidently gives a precise point estimate that misleads non-technical stakeholders.

**Q6: How do you evaluate whether a new prompt version is genuinely better or just luckier on your test data?**

Use the Wilcoxon signed-rank test on paired MASE values from N backtest rounds. For each backtest window, you have one MASE for prompt A and one for prompt B. The Wilcoxon test determines whether the differences are systematically positive (A consistently better) or could arise by chance. A p-value < 0.05 means the improvement is statistically significant at 5% confidence. You should also: (1) use ≥ 15 rounds for sufficient statistical power; (2) evaluate on multiple datasets, not just one; (3) compute the effect size (% MASE improvement), not just significance — a 0.5% improvement may be significant but not practically meaningful.

---

## References

1. Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. arXiv:2201.11903.
2. Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171.
3. Brown, T. et al. (2020). Language Models are Few-Shot Learners (GPT-3). *NeurIPS 2020*. arXiv:2005.14165.
4. Zhou, Y. et al. (2023). Large Language Models Are Human-Level Prompt Engineers. *ICLR 2023*. arXiv:2211.01910. [Meta-prompting inspiration]
5. Kojima, T. et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS 2022*. arXiv:2205.11916. [Zero-shot CoT]
6. Liu, P. et al. (2023). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP. *ACM Computing Surveys*, 55(9). [Comprehensive prompting survey]
7. OpenAI Prompt Engineering Guide. https://platform.openai.com/docs/guides/prompt-engineering

---

*Next Chapter: Chapter 13 — Scenario Forecasting: Building Multi-Path Future Visions with LLMs*
