# Regime Detection Engine — V2 Spec

**Status:** roadmap. Do not implement until V1 passes walk-forward validation.
**Builds on:** `regime_engine_v1_final_spec.md`
**Engine version:** `regime-engine-v2.0.0` (when shipped)

---

## 0. Prerequisites

V2 work begins **only after** all of the following hold:

- V1 ships all 9 vertical slices.
- All 10 V1 golden test dates pass.
- V1 has been live in shadow/paper mode for at least one year of out-of-sample data.
- V1 demonstrates measurable strategy improvement vs no-regime baseline (lower drawdown, fewer wrong-environment trades, or improved Sharpe).

V2 inherits every V1 contract:

- `classify(as_of_date)` stateless replay
- NaN cold-start handling
- Asymmetric hysteresis (escalation immediate, de-escalation debounced)
- `raw_label` / `stable_label` / `active_label` triple
- `evidence` and `data_quality` blocks on every output
- Pydantic types
- NYSE trading calendar (US v2)
- No-hallucination rule for the coding agent

V2 does not modify V1 outputs. V2 adds new fields and new classifiers.

V2 also owns the items intentionally descoped from V1:

- PIT constituent breadth.
- Monetary pressure / liquidity pressure.
- Sideways stress warnings.

These were excluded from V1 to avoid shipping unverified data contracts or silently biased classifications.

---

## 1. Layer 1 V2 — Observable State Extensions

### 1A. Trend Direction & Character V2

#### Efficiency Ratio (20d)

```python
directional_move = abs(close[t] - close[t - 20])
path_length = sum(abs(close[i] - close[i-1]) for i in range(t-19, t+1))
efficiency_ratio_20d = directional_move / path_length
```

Interpretation: ~1.0 = clean trend, ~0.0 = noisy chop. Use as evidence layered on ADX. Never sole basis.

#### Hurst Exponent (250d)

- H > 0.55 → trending tendency
- H < 0.45 → mean-reverting tendency
- 0.45–0.55 → random / unclear

Lookback: 250d minimum. Shorter windows are too noisy. Use as evidence only.

#### Breakout / Range Classifier

Two new V2 character labels: `breakout_expansion`, `range_bound`.

`breakout_expansion`:
```text
close breaks 20d or 50d range
AND Bollinger band width expanding
AND volume > 20d average
AND followthrough_rate >= configured threshold
```

`followthrough_rate` definition (must be specified before implementation): of the last N=20 breakouts, the fraction where close held above breakout level for 5+ trading days.

`range_bound`:
```text
abs(return_63d) < 0.05
AND price oscillates inside 20d range
AND ADX_14 < 20
```

#### Trend Slope Strength

Continuous feature, not a label:
```python
slope_sma_50 = (sma_50[t] - sma_50[t-20]) / sma_50[t-20]
slope_sma_200 = (sma_200[t] - sma_200[t-20]) / sma_200[t-20]
```

Feeds into `evidence` for trend_direction.

#### Recovery and Euphoria Labels

`recovery` (new V2 trend_direction label):
```text
prior 252d drawdown <= -0.15
AND return_63d > 0.10
AND close > SMA_50
```

`euphoria` (new V2 trend_direction label):
```text
close > SMA_200
AND return_126d > 0.20
AND realized_vol_21d rising
AND sentiment_score >= configured threshold
```

`sentiment_score` definition required before implementation. Candidate sources: AAII bull-bear, put-call ratio percentile, Investors Intelligence sentiment.

V2 trend_direction precedence (updated):
```text
euphoria > bull > recovery > bear > sideways > transition > unknown
```

---

### 1C. Volatility V2

#### ATR Ratio
```python
atr_ratio = ATR_14 / ATR_50
```
Feeds into volatility_state evidence. New rule trigger:
```text
rising_vol (V2 label):
  ATR_ratio > 1.15
  OR realized_vol_10d > realized_vol_63d * 1.25
```

#### IV vs RV Spread
```python
iv_rv_spread = implied_vol_30d - realized_vol_21d
```
Requires options data feed. Used as evidence for euphoria, vol_crush, and event_window classifiers.

#### Vol Crush

Required prerequisites with definitions locked:
```text
event_window_just_passed:
  as_of_date within 3 NYSE trading days AFTER configured event end

implied_vol_falling_sharply:
  implied_vol_5d_change <= -0.20
```

Rule:
```text
vol_crush:
  realized_vol_10d < realized_vol_21d * 0.75
  AND implied_vol_falling_sharply
  AND event_window_just_passed
```

#### Gap Frequency
```python
gap = abs(open[t] - close[t-1]) / close[t-1]
gap_frequency_20d = count(gap > 0.005) / 20
```
Threshold 0.5% configurable per market. Feeds into liquidity_gap_behavior label (Section 1E).

#### Intraday Range Percentile
```python
intraday_range = (high - low) / close
intraday_range_percentile_252d = percentile_rank(intraday_range, lookback=252)
```

V2 volatility_state precedence (updated):
```text
crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
```

---

### 1D. Breadth V2

PIT constituent breadth begins here, not in V1. V2 must define and validate the historical constituent-membership data pipeline before enabling PIT breadth. Required properties:

- point-in-time SPX membership with effective dates;
- delisted and removed symbols included when they were members on `as_of_date`;
- row-level validation of `date`, `index`, `symbol`, `is_member_as_of_date`, `close`, and moving-average fields;
- explicit rejection of survivorship-biased universes unless a separate biased research mode is approved.

ETF proxy breadth from V1 remains available as fallback evidence, but V2 PIT breadth must not silently fall back to biased current constituents.

#### Stocks Above 200DMA
```python
pct_above_200dma = mean(member.close > member.sma_200)
```

#### Advance-Decline Line
```python
ad_line[t] = ad_line[t-1] + (advances[t] - declines[t])
ad_line_slope_20d = (ad_line[t] - ad_line[t-20]) / 20
```

#### New Highs / New Lows Ratio (52-week)
```python
nh_nl_ratio = new_52w_highs / max(new_52w_highs + new_52w_lows, 1)
```

#### Up-Volume / Down-Volume Ratio
```python
upvol_downvol_ratio = sum(volume[advances]) / max(sum(volume[declines]), 1)
```

#### Sector Breadth
% of GICS sectors with positive 21d returns. For US: count of XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY with `return_21d > 0` divided by 11.

#### Breadth Thrust (Zweig-style)
```text
breadth_thrust:
  10d moving average of pct_advancing
  moves from < 0.40 to > 0.615
  within 10 trading days
```

V2 adds new breadth labels:
- `breadth_thrust` (bullish initiation)
- `broadening_breadth` (recovery confirmation: NH/NL ratio rising AND ad_line_slope > 0)
- `narrowing_breadth` (deterioration: pct_above_50dma falling AND pct_above_200dma falling AND nh_nl_ratio < 0.4)

V2 breadth precedence:
```text
breadth_thrust > divergent_fragile > narrowing_breadth > recovery_breadth > broadening_breadth > weak_breadth > healthy_breadth > neutral_breadth > unknown
```

---

### 1E. Volume / Liquidity Internals V2

New sub-axis (does not exist in V1).

#### Features
- `volume_zscore_20d`
- `gap_frequency_20d` (from 1C)
- `intraday_range_percentile_252d` (from 1C)

#### Labels
```text
normal_volume
panic_volume
liquidity_gap_behavior
unknown
```

#### Rules

`panic_volume`:
```text
volume_zscore_20d > 2.0
AND return_1d < -0.02
```

`liquidity_gap_behavior`:
```text
gap_frequency_20d percentile_252d > 0.75
AND intraday_range_percentile_252d > 0.75
```

`normal_volume`:
```text
otherwise
```

Risk rank:
```yaml
volume_liquidity_risk_rank:
  normal_volume: 0
  liquidity_gap_behavior: 2
  panic_volume: 3
  unknown: 1
```

---

## 2. Layer 2 V2 — Full Structural-Causal State

### 2A. Monetary / Liquidity V2

Monetary pressure was explicitly not implemented in V1. V2 is the first release allowed to implement it, and must lock a clean data contract for 2y yield, 10y yield, and DXY before coding begins.

V1's draft absolute bps thresholds were deferred because they are rate-era dependent. V2 must adapt to rate era.

#### Rate-Era Recalibration

```python
yield_change_zscore = (yield_change_63d - mean_5y) / std_5y
```

Updated rules:
```text
tightening_pressure:
  yield_change_zscore_2y > +1.5
  OR yield_change_zscore_10y > +1.5
  OR DXY_zscore_63d > +1.5

easing_pressure:
  yield_change_zscore_2y < -1.5
  OR yield_change_zscore_10y < -1.5

rate_shock:
  yield_change_zscore_21d_2y > +2.0
  OR yield_change_zscore_21d_10y > +2.0
```

#### Central Bank Text / Sentiment

Source: FOMC minutes, Powell speech transcripts (US); RBI policy statements (India when extended).

Pipeline:
1. Ingest text on release.
2. LLM classifier outputs `{hawkish, dovish, neutral}` with confidence.
3. Output as structured score, fed into `monetary_pressure.evidence` — never as standalone label.

#### Release Timestamp Handling

- Macro releases have a release time (CPI 8:30am ET, FOMC 2:00pm ET, etc.).
- For EOD classification, releases occurring before market close on `as_of_date` are usable.
- Revisions to prior releases must be stored separately. Original release values are point-in-time-correct; revised values are not. The engine must use original values for historical replay.
- Implementation: data store has both `value_first_release` and `value_latest_revision` per data point.

---

### 2B. Inflation / Growth State (new in V2)

#### Labels
```text
goldilocks
inflation_shock
disinflation
recession_scare
recovery_growth
earnings_expansion
earnings_contraction
unknown
```

#### Precedence
```text
inflation_shock > recession_scare > disinflation > goldilocks > recovery_growth > earnings_contraction > earnings_expansion > unknown
```

#### Features
- CPI trend (3m and 6m rate of change)
- Inflation surprise (actual vs consensus, from BLS calendar)
- PMI trend (ISM Manufacturing + Services)
- Earnings revision breadth (% of S&P 500 with positive analyst revisions, last 90 days)
- Commodity returns (Bloomberg Commodity Index 63d return)
- Bond yield trend (10y yield 63d direction)
- Cyclical vs defensive relative strength: `(XLY + XLI) / (XLP + XLU)` ratio, 63d slope

#### Rules

`goldilocks`:
```text
CPI 6m trend stable or falling
AND PMI > 50
AND equities rising
AND credit calm (links Layer 2C)
```

`inflation_shock`:
```text
inflation_surprise positive AND large
OR commodity_return_63d > 0.15
AND yields rising
AND equities AND bonds both weak
```

`disinflation`:
```text
CPI 6m trend falling
AND yields falling
AND PMI > 45
```

`recession_scare`:
```text
yields falling
AND cyclical_vs_defensive_slope < 0
AND credit spreads widening (Layer 2C)
AND equities weak
```

`recovery_growth`:
```text
PMI rising AND > 50
AND cyclical_vs_defensive_slope > 0
AND credit spreads narrowing
```

`earnings_expansion`:
```text
earnings_revision_breadth > 0.55
```

`earnings_contraction`:
```text
earnings_revision_breadth < 0.40
```

---

### 2C. Credit / Funding State (new in V2)

#### Labels
```text
credit_calm
spread_widening
credit_stress
funding_squeeze
deleveraging
unknown
```

#### Precedence
```text
deleveraging > funding_squeeze > credit_stress > spread_widening > credit_calm > unknown
```

#### Features
- Investment grade credit spread (LQD vs Treasury proxy if no direct ICE BAML feed)
- High yield credit spread (HYG vs Treasury proxy if no direct OAS feed)
- Bank index relative strength: `KRE / SPY` 63d slope
- Financial Conditions Index (Chicago Fed NFCI weekly release)
- Dollar index (DXY)
- Short-rate funding stress: SOFR-IORB spread

#### Rules

`credit_calm`:
```text
HY_spread_percentile_504d < 0.50
AND HY_spread 21d trend non-rising
```

`spread_widening`:
```text
HY_spread rising over 21d
AND IG_spread rising over 21d
```

`credit_stress`:
```text
HY_spread_percentile_504d > 0.80
AND equities falling
```

`funding_squeeze`:
```text
DXY_zscore_21d > +1.5
AND SOFR-IORB spread widening
AND risk assets falling
```

`deleveraging`:
```text
equities down
AND bonds weak or unstable
AND DXY rising
AND volatility up
AND avg_pairwise_corr rising (Layer 3 V2)
```

---

### 2D. Event Calendar V2

Add labels to V1's calendar:
- `budget_week` (relevant for India when extended)
- `election_window` (configurable window around major election/budget result)
- `geopolitical_event` (manual flag in YAML for war, sanctions, terrorism)
- `global_rate_decision` (BOE, ECB, BOJ — relevant for US cross-asset moves)

YAML schema extension:
```yaml
events:
  - date: "2026-11-03"
    market: "US"
    type: "election"
    importance: "high"
    window_days: [-5, +10]
  - date: "2026-12-10"
    market: "GLOBAL"
    type: "ECB_decision"
    importance: "medium"
```

---

## 3. Layer 3 V2 — Network Fragility (Full Implementation)

### 3.1 Universe (US)

```yaml
fragility_universe_us:
  sector_etfs:
    - XLB    # Materials
    - XLC    # Communications
    - XLE    # Energy
    - XLF    # Financials
    - XLI    # Industrials
    - XLK    # Technology
    - XLP    # Consumer Staples
    - XLRE   # Real Estate
    - XLU    # Utilities
    - XLV    # Healthcare
    - XLY    # Consumer Discretionary
  cross_asset_etfs:
    - SPY    # US large cap
    - QQQ    # Tech-heavy
    - IWM    # Small cap
    - EFA    # Developed ex-US
    - EEM    # Emerging markets
    - TLT    # Long Treasuries
    - GLD    # Gold
    - HYG    # High yield bonds
    - LQD    # Investment grade bonds
    - USO    # Oil
    - UUP    # Dollar
```

22 assets total. Above the 20-asset preferred floor.

### 3.2 Features

#### Average Pairwise Correlation (63d)
```python
returns_matrix = returns[universe].tail(63)
corr_matrix = returns_matrix.corr()
avg_pairwise_corr = mean(off_diagonal(corr_matrix))
```

#### Correlation Percentile (504d lookback)
```python
avg_pairwise_corr_percentile_504d = percentile_rank(avg_pairwise_corr_history, lookback=504)
```

Default percentile window: **504 trading days** (~2 years). Tunable in V2 calibration phase.

#### Largest Eigenvalue Share
```python
eigenvalues = sorted(eigvals(corr_matrix), reverse=True)
largest_eigenvalue_share = eigenvalues[0] / sum(eigenvalues)
```

#### Effective Rank
```python
p = eigenvalues / sum(eigenvalues)
shannon_entropy = -sum(p_i * log(p_i) for p_i in p if p_i > 0)
effective_rank = exp(shannon_entropy)
```

Low effective rank = diversification collapsing.

#### Absorption Ratio (Top-3 Eigenvalues)
```python
absorption_ratio_top3 = sum(eigenvalues[:3]) / sum(eigenvalues)
```

#### Dispersion Ratio
```python
single_stock_vols = [realized_vol_21d(asset) for asset in universe]
mean_single_vol = mean(single_stock_vols)
index_vol = realized_vol_21d(SPY)
dispersion_ratio = mean_single_vol / index_vol
```

High dispersion = stock-picker market. Low dispersion = correlation-driven.

### 3.3 Labels
```text
diversified_normal
stock_picker_dispersion
rising_fragility
correlation_concentration
correlation_to_one
systemic_stress
unknown
```

### 3.4 Precedence
```text
systemic_stress > correlation_to_one > correlation_concentration > rising_fragility > stock_picker_dispersion > diversified_normal > unknown
```

### 3.5 Rules

`diversified_normal`:
```text
0.25 <= avg_pairwise_corr_percentile_504d <= 0.75
AND effective_rank stable (21d std < 5% of mean)
```

`stock_picker_dispersion`:
```text
avg_pairwise_corr_percentile_504d < 0.30
AND dispersion_ratio percentile_252d > 0.70
AND volatility_state.active_label != crisis_vol
```

`rising_fragility`:
```text
avg_pairwise_corr rising over 21d (positive slope)
AND largest_eigenvalue_share rising over 21d
AND breadth_state.active_label in [weak_breadth, narrowing_breadth, divergent_fragile]
```

`correlation_concentration`:
```text
avg_pairwise_corr_percentile_504d > 0.75
OR largest_eigenvalue_share_percentile_504d > 0.75
OR effective_rank_percentile_504d < 0.25
```

`correlation_to_one`:
```text
avg_pairwise_corr_percentile_504d > 0.90
AND realized_vol_percentile_252d > 0.80
AND drawdown_21d < 0
```

`systemic_stress`:
```text
correlation_to_one
AND credit_funding.active_label in [credit_stress, deleveraging]
AND VIX_percentile_252d > 0.80
AND breadth_state.active_label in [weak_breadth, narrowing_breadth]
```

### 3.6 Risk Rank
```yaml
network_fragility_risk_rank:
  diversified_normal: 0
  stock_picker_dispersion: 1
  rising_fragility: 2
  correlation_concentration: 2
  correlation_to_one: 3
  systemic_stress: 3
  unknown: 2
```

### 3.7 Hysteresis

Asymmetric per V1 rule. De-escalation defaults:
```yaml
network_fragility_deescalation_days:
  rising_fragility: 3
  correlation_concentration: 3
  correlation_to_one: 5
  systemic_stress: 5
```

---

## 4. Layer 4 V2 — Transition Score

### 4.0 Named Warning Extensions

V2 adds named warnings only when they capture a failure mode V1 cannot represent with its deterministic V1 labels.

`sideways_stress_warning`:
```text
trend_direction.active_label = sideways
AND volatility_state.active_label = high_vol
AND breadth_state.active_label in [weak_breadth, divergent_fragile]
```

This captures banking-crisis, election-uncertainty, and macro-shock environments that are stressed but have not committed to V1 `bear`. V1 intentionally emits `stable` for this pattern unless another V1 warning fires; do not backport this warning to V1.

### 4.1 Composition

V2 adds a continuous transition score that **augments** V1's named warnings, not replaces them.

```python
transition_score = weighted_sum([
    volatility_acceleration_score,
    breadth_deterioration_score,
    correlation_concentration_score,
    trend_break_score,
    macro_event_score,
    hmm_probability_shift_score
])
```

### 4.2 Component Score Definitions

Each component produces a 0.0–1.0 score.

`volatility_acceleration_score`:
```python
ratio = realized_vol_10d / realized_vol_63d
score = clip((ratio - 1.0) / 0.5, 0, 1)  # 0 at ratio=1.0, 1 at ratio=1.5
```

`breadth_deterioration_score`:
```python
score = clip((0.50 - pct_above_50dma) / 0.30, 0, 1)  # 0 at 50% breadth, 1 at 20%
```

`correlation_concentration_score`:
```python
score = avg_pairwise_corr_percentile_504d  # already 0-1
```

`trend_break_score`:
```python
distance_from_high = drawdown_from_252d_high  # negative
score = clip(-distance_from_high / 0.15, 0, 1)  # 0 at top, 1 at -15%
```

`macro_event_score`:
```python
score = 1.0 if event_calendar.label in ["fed_week", "cpi_week", "nfp_week"] else 0.0
```

`hmm_probability_shift_score`:
```python
score = abs(hmm.top_state_prob[t] - hmm.top_state_prob[t-5])
```

### 4.3 Weights

**With HMM (full V2):**
```yaml
volatility_acceleration: 0.20
breadth_deterioration: 0.20
correlation_concentration: 0.20
trend_break: 0.20
macro_event: 0.10
hmm_probability_shift: 0.10
```

**Without HMM (V2 partial — if HMM not yet shipped):**
```yaml
volatility_acceleration: 0.225
breadth_deterioration: 0.225
correlation_concentration: 0.225
trend_break: 0.225
macro_event: 0.10
```

### 4.4 Score Interpretation

```text
0.00 - 0.35  →  stable
0.35 - 0.55  →  weakening
0.55 - 0.75  →  transition_warning
0.75 - 1.00  →  high transition risk
```

### 4.5 Integration with V1 Warnings

Transition score is **evidence**, not the regime. Output structure:
```json
{
  "transition_risk": {
    "label": "bull_fragile_warning",
    "transition_score": 0.62,
    "score_interpretation": "transition_warning",
    "score_components": {
      "volatility_acceleration": 0.45,
      "breadth_deterioration": 0.71,
      "correlation_concentration": 0.68,
      "trend_break": 0.20,
      "macro_event": 1.0,
      "hmm_probability_shift": 0.30
    },
    "evidence": {}
  }
}
```

V1 warning labels (`bull_fragile_warning`, `bear_stress_warning`, `crisis_override`, etc.) remain authoritative. The score adds gradation.

### 4.6 Change-Point Detection

Implements one of: BOCPD, PELT, or CUSUM. Pick one for V2 ship; document choice.

Output:
```json
{
  "change_point": {
    "score": 0.78,
    "days_since_last_break": 4
  }
}
```

Feeds `transition_score` as additional evidence (V2.1 — not in initial V2 ship).

---

## 5. Layer 5 V2 — Strategy Response Extensions

### 5.1 Agent Cohort Routing

V2 adds explicit agent routing on top of V1's permission modifiers.

```json
{
  "agent_routing": {
    "active_cohort": "tightening_specialist",
    "fallback_cohort": "default_neutral",
    "blocked_cohorts": ["short_vol", "leveraged_long"]
  }
}
```

Specialist cohorts:
- `crisis_specialist`
- `euphoria_specialist`
- `tightening_specialist`
- `easing_specialist`
- `recovery_specialist`
- `chop_mean_reversion_specialist`
- `bull_low_vol_specialist`
- `bear_stress_specialist`

Routing rules use the V1 scenario precedence extended with V2 macro/fragility states.

### 5.2 Strategy-Family Constraints

V1 has flat allow/block. V2 adds per-family granular constraints.

```json
{
  "strategy_family_constraints": {
    "trend_following": {
      "allowed": true,
      "max_lookback_days": 50,
      "require_breadth_confirmation": false,
      "min_adx": 20
    },
    "mean_reversion": {
      "allowed": true,
      "max_holding_days": 5,
      "require_volume_confirmation": true
    },
    "breakout": {
      "allowed": false,
      "reason": "false_breakout_rate_high_in_chop"
    },
    "short_vol": {
      "allowed": false,
      "reason": "rising_fragility_or_crisis"
    },
    "long_vol": {
      "allowed": true,
      "event_window_only": true
    }
  }
}
```

### 5.3 Vol-Crush Exit Rules

When `volatility_state.active_label = vol_crush`:
- Exit all event-vol longs immediately
- Reduce long-vol exposure
- Normalize risk after `cooldown_days = 5`

### 5.4 No-Flip-Flop Windows

Beyond V1's `post_switch_cooldown`:
- `tightening_pressure` + `fed_week` + `rising_vol` → `no_flip_flop_window = 15 trading days`
- Minimum holding period for reversal trades = 15 trading days during this window

```json
{
  "timing_controls": {
    "no_flip_flop_window_days": 15,
    "post_switch_cooldown_days": 5,
    "minimum_holding_period_days": 15
  }
}
```

### 5.5 Learned PRISM Rules

V2 may incorporate PRISM-derived rules (the user's signal-engine rule-discovery framework) as configurable overrides on top of base V2 modifiers.

PRISM rule contract:
- Walk-forward validated on at least 3 years of data
- Versioned (`prism_rule_id`, `prism_rule_version`)
- Logged for review
- Reversible via single config flag
- Each rule includes: condition, modifier, expected effect, validation metrics

Output:
```json
{
  "prism_overrides_applied": [
    {
      "rule_id": "PRISM_042",
      "version": "1.2.0",
      "modifier": {"position_size_multiplier": 0.6},
      "reason": "tightening_breadth_divergence_pattern"
    }
  ]
}
```

---

## 6. Probabilistic Models

These are **evidence layers**, not final regime labels. Outputs feed into transition_score and other classifiers as additional input. **Never used as standalone label.**

### 6.1 HMM (Hidden Markov Model)

#### Purpose
Infer latent market states from returns and volatility.

#### Inputs
- `return_1d`
- `realized_vol_21d`
- `drawdown_63d`
- `volume_zscore`
- `avg_pairwise_corr` (Layer 3 V2)

#### Model
- Gaussian HMM
- 3 states (recommended): `calm_bull`, `choppy_normal`, `stress_crash`
- Optionally 4 states (split bull into trending vs euphoric) once 3-state version validates

#### Output
```json
{
  "hmm": {
    "top_state": "calm_bull",
    "state_probabilities": {
      "calm_bull": 0.61,
      "choppy_normal": 0.29,
      "stress_crash": 0.10
    },
    "state_persistence_days": 12,
    "model_version": "hmm_3state_v1.0"
  }
}
```

#### Constraint
HMM state is **never** the final regime label. Evidence flows into:
- `transition_score` (via `hmm_probability_shift_score`)
- `volatility_state.evidence`
- `trend_direction.evidence`

#### Training
- Fit on at least 5 years of data
- Refit quarterly on rolling 5-year window
- Compare new model parameters to prior version; alert on >20% parameter drift

---

### 6.2 K-Means / GMM Clustering

#### Purpose
Discover empirical clusters of market days for diagnostic purposes.

#### Inputs
- `return_21d`
- `return_63d`
- `realized_vol_21d`
- `drawdown_63d`
- `ADX_14`
- `avg_pairwise_corr`
- `pct_above_50dma`

#### Output
```json
{
  "cluster": {
    "cluster_id": 2,
    "mapped_label": "high_vol_chop",
    "distance_to_centroid": 0.41,
    "model_version": "gmm_8cluster_v1.0"
  }
}
```

#### Constraint
Clusters must be **manually mapped** to economic labels after inspection. Never auto-label. Mapping is config-versioned.

Mapping artifact (`cluster_label_map.yaml`):
```yaml
cluster_label_map:
  version: "1.0"
  fitted_on: "2026-01-15"
  fitted_window: "2020-01-01..2025-12-31"
  mappings:
    0: "calm_low_vol_bull"
    1: "trending_bull"
    2: "high_vol_chop"
    3: "crisis"
    4: "recovery"
    5: "narrow_leadership"
    6: "broad_decline"
    7: "transition"
```

---

### 6.3 Change-Point Detection

#### Purpose
Detect statistical break points in returns or volatility series.

#### Method Options (pick one for V2 ship)
- Bayesian online change point detection (BOCPD)
- PELT algorithm
- CUSUM

#### Output
```json
{
  "change_point": {
    "score": 0.78,
    "days_since_last_break": 4,
    "method": "BOCPD"
  }
}
```

Feeds `transition_score` as additional evidence.

---

## 7. V3 Research Frontier — Do Not Build in V2

Defer until V2 walk-forward validation complete:

- ORCA-style spectral graph feature model
- SRR-style graph-network model
- Autoencoder anomaly detection
- Transformer / sequence model for regime sequence
- Cross-market regime contagion model (US → NSE → MCX)

Each requires its own justification: walk-forward evidence that the simpler V2 components are insufficient.

---

## 8. V2 Implementation Order

V2 slices, in priority order. Each slice ships end-to-end before the next begins (same vertical-slicing rule as V1).

1. **Layer 3 Network Fragility** — highest immediate value, uses existing data infrastructure (sector ETFs are already in your screeners)
2. **Layer 1 V2 incremental features** — efficiency ratio, ATR ratio, gap frequency, breadth thrust, % above 200DMA. Adds to existing classifiers without changing V1 contracts.
3. **Layer 4 V2 transition score** — composes Layer 1 V2 + Layer 3 V2; can ship without HMM using the renormalized weights
4. **Layer 2C Credit/Funding** — depends on credit data sourcing (HYG, LQD, NFCI)
5. **Layer 2B Inflation/Growth** — depends on macro data sourcing (PMI, CPI, earnings revisions)
6. **HMM module** — runs in parallel to deterministic classifiers, slots into transition_score
7. **K-Means/GMM clustering** — needs manual cluster mapping work; lowest priority
8. **Change-point detection** — feeds transition_score
9. **Layer 5 V2 cohort routing + strategy-family constraints** — composes all prior V2 outputs
10. **Layer 5 V2 PRISM rule integration** — last; requires PRISM framework already producing validated rules

Each slice must pass its own golden test set before commit.

---

## 9. V2 Validation

### 9.1 V2-vs-V1 Performance Gate

Every V2 component must demonstrate, in walk-forward backtest, **at least one** of:
- Lower max drawdown than V1
- Higher Sharpe than V1
- Earlier crisis detection (lower lag in days from event to crisis_override)
- Lower false-switch rate than V1

V2 components that don't beat V1 on at least one metric **must not ship**. Log the failure, document why, move on.

### 9.2 Statistical Validation

- Walk-forward window: same as V1 (rolling train + validation + holdout)
- Out-of-sample period: at least 1 year beyond V1's holdout
- HMM, GMM, change-point: also validated against synthetic regime data with known answers (generate synthetic series with known state changes; verify model recovers them)

### 9.3 Production Validation

- A/B comparison: V1 vs V2 on shadow runs for 60 trading days before V2 takes over routing
- Disagreement log: every day where V1 and V2 differ, log evidence and label, weekly review
- V2 graduates to live routing only after disagreement log has been reviewed and explained

### 9.4 New Golden Test Set Additions for V2

Add dates that exercise V2-specific behavior:

| as_of_date | Test reason |
|---|---|
| 2010-05-06 (Flash Crash) | tests systemic_stress, correlation_to_one |
| 2011-08-08 (US downgrade) | tests credit_stress, funding_squeeze |
| 2015-08-24 (China devaluation) | tests rising_fragility, bull→correlation_to_one transition |
| 2018-10-10 | tests bull→narrowing_breadth→bear_stress sequence |
| 2020-08-15 | tests stock_picker_dispersion (post-COVID rally narrowing) |
| 2021-01-27 (GameStop) | tests dispersion + volume anomalies |
| 2022-09-26 (UK gilt crisis) | tests cross-asset deleveraging |
| 2023-03-13 (SVB) | tests V1 false-negative gap: sideways + high_vol + weak_breadth should fire V2 sideways_stress_warning and Layer 2C credit_stress |
| 2024-08-05 (Yen carry unwind) | tests correlation_to_one + funding_squeeze |

These build on the V1 golden test set; do not replace it.

---

## 10. V2 Coding Agent Prompt

When V2 work begins, give the coding agent this in addition to all V1 rules:

```text
ABSOLUTE RULE: When the spec is ambiguous or silent, stop and ask. Do not invent.
The same V1 forbidden-invention rules apply.

V2 implementation contract:

1. V2 does not modify V1 outputs. V2 adds new fields and new classifiers.
   Existing V1 consumers must continue to work unchanged.

2. Do not invent component score formulas. Use the exact formulas in
   Section 4.2 of the V2 spec.

3. Do not invent transition score weights. Use Section 4.3. If HMM is
   not yet shipped, use the renormalized weights without HMM.

4. Do not auto-label clusters. K-Means/GMM mappings require manual
   review per Section 6.2. Ship the model; do not ship auto-mapping.

5. HMM, GMM, and change-point are evidence layers. Never the final
   regime label. They feed transition_score and evidence dicts only.

6. Network fragility universe is the 22 ETFs in Section 3.1. Do not
   substitute, add, or remove without an explicit config update.

7. Macro release timestamp handling is mandatory. Use point-in-time
   release values. Store revisions separately. Use original values
   for historical replay.

8. PRISM rule integration is optional and last. Do not implement
   PRISM hooks until PRISM itself is producing validated rules.

9. Every V2 component must demonstrate the performance gate
   (Section 9.1) before promotion to live routing.

10. Build vertical slices in the order defined in Section 8.
    Each slice ships end-to-end before the next begins.

Forbidden V2 inventions in addition to V1 list:
- ORCA, SRR, autoencoder, transformer features (these are V3)
- Auto-cluster labeling
- HMM as final label
- New regime labels not in this spec
- Macro thresholds not derived from rate-era recalibration
```

---

## 11. Cross-Reference Index

| Topic | V1 location | V2 location |
|---|---|---|
| Trend Direction | V1 §3 | V2 §1A (additions) |
| Trend Character | V1 §4 | V2 §1A (additions) |
| Volatility State | V1 §5 | V2 §1C (additions) |
| Breadth State | V1 §6 | V2 §1D (additions) |
| Volume / Liquidity | (not in V1) | V2 §1E |
| Event Calendar | V1 §7.2 | V2 §2D (additions) |
| Monetary Pressure | V1 §7.3 (basic) | V2 §2A (full) |
| Inflation / Growth | (not in V1) | V2 §2B |
| Credit / Funding | (not in V1) | V2 §2C |
| Network Fragility | V1 §8 (stub) | V2 §3 (full) |
| Transition Risk | V1 §9 (named warnings) | V2 §4 (score + warnings) |
| Strategy Response | V1 §10 (modifiers) | V2 §5 (cohorts + family constraints) |
| HMM | (not in V1) | V2 §6.1 |
| K-Means / GMM | (not in V1) | V2 §6.2 |
| Change-Point | (not in V1) | V2 §6.3 |
| Golden Test Set | V1 §12.2 | V2 §9.4 (additions) |

---

## 12. Final Principle (carried from V1)

> Do not optimize for beautiful regime labels.
> Optimize for capital protection, strategy routing, replayability, and fast debugging.

Every V2 addition that does not measurably improve those four does not ship.
