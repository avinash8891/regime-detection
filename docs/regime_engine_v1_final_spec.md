# Regime Detection Engine — V1 Final Spec

**Status:** locked, ready for coding-agent handoff
**Market:** US equities (V1)
**Cadence:** EOD
**Engine version:** `regime-engine-v1.0.0`

---

## 0. First Principle

> Do not optimize for beautiful regime labels.
> Optimize for capital protection, strategy routing, replayability, and fast debugging.

Every output must be explainable from its `evidence` fields in under 30 seconds.

---

## 1. Architecture

```text
Observable Market State
+ Structural-Causal State
+ Network Fragility
+ Transition Risk
→ Strategy Response
```

Implementation is split:

```text
V1 = Core Shield (this spec)
V2 = Advanced Regime Engine (see regime_engine_v2_spec.md)
```

V1 ships only:

- Trend Direction
- Trend Character
- Volatility
- Breadth in ETF proxy mode (`RSP/SPY`) only
- Event Calendar
- Simple Transition Warnings
- Strategy Response Modifiers

V1 does **not** ship: PIT constituent breadth, monetary pressure, macro inference, credit/inflation models, HMM, GMM, eigenvalues, correlation network fragility, ORCA/SRR, weighted transition score, Hurst, efficiency ratio, sideways stress warnings.

---

## 2. Global Engine Contract

### 2.1 Entry Points

```python
RegimeEngine.classify(
    as_of_date: date,
    market_data: pd.DataFrame,
    breadth_data: pd.DataFrame | None = None,
    vix_data: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
    config: RegimeConfig | None = None,
) -> RegimeOutput
```

Rules:

- Use only data with date `<= as_of_date`.
- Never use future constituent membership.
- Never use future event outcomes or realized values.
- For scheduled event-calendar rows, lookahead is bounded by `publication_date`, not `event_date`. A future scheduled event may be used only when `publication_date <= as_of_date`.
- For V1 hand-maintained event YAML/CSV, including a scheduled event in the file is treated as the publication act. Coding agents may still populate `publication_date` explicitly when desired.
- `as_of_date` must be an NYSE trading day. If it is not, raise `ValueError` with the nearest prior and next NYSE trading days in the message.
- Do not roll non-trading `as_of_date` values backward or forward.
- Live mode = `classify(as_of_date=today)`.

V1 input contract:

- `market_data` is a long/wide-enough OHLCV DataFrame with at least `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
- US V1 requires `SPY` rows for the market index.
- ETF proxy breadth requires `RSP` rows in the same contract.
- VIX support may be provided either as `vix_data` (preferred) or as a symbol in `market_data`.
- In this repository's deterministic test fixtures, `VIXY` is used as the VIX proxy series (see `tests/fixtures/raw/PROVENANCE.md`). If you provide `VIX`/`^VIX` externally, you must normalize it into the `vix_data` input contract (close series aligned to NYSE sessions) before classification.

V1 helper:

```python
RegimeEngine.classify_window(
    end_date: date,
    market_data: pd.DataFrame,
    lookback_days: int,
    ...
) -> RegimeTimeline
```

`RegimeTimeline` is a Pydantic model:

```python
class RegimeTimeline(BaseModel):
    engine_version: str
    config_version: str
    market: str
    start_date: date
    end_date: date
    trading_calendar: str
    outputs: list[RegimeOutput]
```

`outputs` is sorted ascending by `as_of_date` and contains exactly one `RegimeOutput` per NYSE trading day in the inclusive window. Unknown outputs are emitted, not skipped.

V1.1 helper (deferred):

```python
RegimeEngine.classify_series(
    start_date: date,
    end_date: date,
    market_data: pd.DataFrame,
    ...
) -> pd.DataFrame
```

### 2.2 Stateless Replay Rule

`classify(as_of_date)` must internally recompute raw labels from:

```text
as_of_date - max_required_lookback - max_hysteresis_days
through as_of_date
```

No hidden state. No state files. Replay is deterministic.

### 2.3 Type Contract

`RegimeOutput` is a Pydantic model. All sub-objects are Pydantic models. The JSON in Section 11 is the canonical output shape; Section 10 defines the exhaustive conditional strategy-response fields. Coding agent must not invent dataclasses, TypedDicts, or dicts as substitutes.

### 2.4 Output Versioning

Every top-level output includes:

```json
{
  "engine_version": "regime-engine-v1.0.0",
  "config_version": "core3-v1.0.0",
  "as_of_date": "YYYY-MM-DD",
  "market": "SPY"
}
```

The package version in `pyproject.toml` and emitted `engine_version` must be coupled by test. A mismatch fails CI.

### 2.4.1 Config Loading

V1 ships with the packaged config resource `regime_detection/configs/core3-v1.0.0.yaml`.

Rules:

- `RegimeEngine` loads config once at construction time.
- Default config is loaded from `regime_detection/configs/core3-v1.0.0.yaml`.
- `classify(..., config=...)` may override the engine default only with a validated `RegimeConfig` object.
- `RegimeConfig` is a Pydantic model with `extra="forbid"`; unknown config keys raise.
- `config_version` in output reflects the loaded config.
- Precedence orderings and risk-rank tables are hardcoded in code, not config.

### 2.5 Trading Calendar

V1 uses the **NYSE** trading calendar for all trading-day arithmetic (event windows, lookback counts, hysteresis day counts).

```yaml
trading_calendar: "NYSE"
```

Coding agent must use `pandas_market_calendars` (or equivalent) — not `bdate_range`, which ignores holidays. When the engine is extended to NSE/MCX in later releases, this becomes per-market config.

Non-trading `as_of_date` values are deterministic errors, not data-quality degradation. This prevents calendar-day backtests from silently producing duplicate Friday classifications for weekends or holidays.

### 2.6 Minimum History Requirement

Each classifier declares its minimum required history. If history below the minimum, classifier emits `label="unknown"`, `reason="insufficient_history"`. Boolean rules are **never** evaluated against insufficient data.

| Classifier | Minimum trading days of history before `as_of_date` |
|---|---|
| trend_direction | 200 (for SMA_200) |
| trend_character | 63 (for prior_63d_drawdown) |
| volatility_state | 252 (for vol_percentile_252d) |
| breadth_state | 50 (for SMA_50 of constituents/proxy) |
| transition_risk | 60 (for "prior bear in last 60d" rule) |
| **Engine-wide minimum for full classification** | **320** (covers the deepest chain) |

### 2.7 NaN Handling Rule (Cold Start)

This is a hard rule. Coding agent must not violate it.

> If any required feature for a classifier is `NaN` (e.g., due to insufficient history at the start of a backtest), the classifier must return:
>
> ```json
> {
>   "raw_label": "unknown",
>   "stable_label": "unknown",
>   "active_label": "unknown",
>   "evidence": { "reason": "insufficient_history" },
>   "data_quality": {
>     "status": "insufficient_history",
>     "freshness_days": null,
>     "completeness": null,
>     "reason": "required_feature_is_nan"
>   }
> }
> ```
>
> Boolean rules must **never** be evaluated against `NaN`. Specifically: `close > SMA_200` is forbidden when `SMA_200` is `NaN`. Check for `NaN` first, return `unknown`, then evaluate rules only on non-`NaN` data.

Implementation pattern:

```python
required_features = [sma_50, sma_200, return_63d]
if any(pd.isna(f) for f in required_features):
    return _unknown_output(reason="insufficient_history")
# only now evaluate rules
```

### 2.8 Data Quality Rules

Every classifier output includes:

```json
"data_quality": {
  "status": "ok",
  "freshness_days": 0,
  "completeness": 1.0,
  "reason": null
}
```

Thresholds:

```text
completeness >= 0.90 AND all required features non-NaN  → status=ok, label emitted
0.70 <= completeness < 0.90                             → status=degraded, label emitted with warning
completeness < 0.70                                     → label=unknown, reason=insufficient_data
any required feature is NaN                             → label=unknown, reason=insufficient_history
freshness_days > 3                                      → label=unknown, reason=stale_data
```

### 2.9 Label Contract

Every classifier outputs three labels:

```json
{
  "raw_label": "...",     // today's direct rule output
  "stable_label": "...",  // debounced after hysteresis
  "active_label": "..."   // label used by strategy response
}
```

### 2.10 Asymmetric Hysteresis

Risk escalation immediate, de-escalation debounced.

```python
if risk_rank(raw_label) > risk_rank(stable_label):
    active_label = raw_label  # escalate immediately
    escalation_fast_path = True
else:
    active_label = stable_label  # debounced de-escalation
    escalation_fast_path = False
```

Default V1 hysteresis:

```yaml
hysteresis:
  trend_direction_deescalation_days: 3
  trend_character_deescalation_days: 3
  volatility_deescalation_days: 2
  breadth_deescalation_days: 2
  composite_deescalation_days: 3
  event_calendar_days: 1
```

### 2.11 No Confidence Field in V1

Do not output `"confidence": 0.74`. V1 evidence is rule-based and binary. Probabilistic confidence is a V2 concern (see V2 spec §6).

### 2.12 Precedence and Risk Rank Tables

V1 precedence orderings and `risk_rank` tables are **hardcoded in code**, not config. Coding agent must not invent a config schema for orderings.

### 2.13 No Hallucination Rule

When the spec is ambiguous or silent, the coding agent must stop and ask. Specifically:

- Do not invent feature formulas.
- Do not invent thresholds.
- Do not invent scenario precedence.
- Do not invent state machines.
- Do not invent fallback behavior.
- Do not invent confidence scores.
- Do not invent fields not specified in the schema.

---

## 3. Layer 1A — Trend Direction

### 3.1 Inputs

```text
close
SMA_50
SMA_200
return_63d
```

### 3.2 Labels

```text
bull
bear
sideways
transition
unknown
```

### 3.3 Precedence (highest first)

```text
bull > bear > sideways > transition > unknown
```

### 3.4 Formulas

```python
sma_50 = close.rolling(50).mean()
sma_200 = close.rolling(200).mean()
return_63d = close / close.shift(63) - 1
```

### 3.5 Rules

`bull`:
```text
close > SMA_50 AND close > SMA_200 AND SMA_50 > SMA_200
```

`bear`:
```text
close < SMA_50 AND close < SMA_200 AND SMA_50 < SMA_200
```

`sideways`:
```text
abs(return_63d) < 0.05 AND close within ±5% of SMA_200
```

`transition`:
```text
none of the above match (and all features are non-NaN)
```

`unknown`:
```text
any required feature is NaN
```

### 3.6 Risk Rank

```yaml
trend_direction_risk_rank:
  bull: 0
  sideways: 1
  transition: 2
  bear: 3
  unknown: 2
```

---

## 4. Layer 1B — Trend Character

### 4.1 Inputs

```text
ADX_14
return_10d
return_21d
prior_63d_drawdown
SMA_50  (for recovery_attempt rule)
```

### 4.2 Labels

```text
trending
chop
recovery_attempt
transition
unknown
```

### 4.3 Precedence

```text
recovery_attempt > trending > chop > transition > unknown
```

### 4.4 Formulas

```python
return_10d = close / close.shift(10) - 1
return_21d = close / close.shift(21) - 1
prior_63d_drawdown = close / close.rolling(63).max() - 1
# ADX_14 standard Wilder formula
```

### 4.5 Rules

`recovery_attempt`:
```text
prior_63d_drawdown <= -0.10
AND close > SMA_50
AND return_10d >= 0.05
```

`trending`:
```text
ADX_14 >= 20 AND abs(return_21d) >= 0.05
```

`chop`:
```text
ADX_14 < 20 AND abs(return_10d) < 0.03 AND abs(return_21d) < 0.05
```

`transition`:
```text
none of the above match
```

> Important: Low ADX alone is **not** chop. Low ADX + non-trivial velocity = transition or recovery_attempt.

### 4.6 Risk Rank

```yaml
trend_character_risk_rank:
  trending: 0
  chop: 1
  recovery_attempt: 1
  transition: 2
  unknown: 2
```

---

## 5. Layer 1C — Volatility State

### 5.1 Inputs

```text
21d realized volatility
252d realized volatility percentile
return_1d, return_5d, return_21d
optional VIX percentile
```

### 5.2 Labels

```text
low_vol
normal_vol
high_vol
crisis_vol
unknown
```

### 5.3 Precedence

```text
crisis_vol > high_vol > low_vol > normal_vol > unknown
```

### 5.4 Formulas

```python
daily_returns = close.pct_change()
realized_vol_21d = daily_returns.rolling(21).std() * sqrt(252)
realized_vol_percentile_252d = percentile_rank(realized_vol_21d, lookback=252)
return_1d = close / close.shift(1) - 1
return_5d = close / close.shift(5) - 1
return_21d = close / close.shift(21) - 1
```

### 5.5 Rules

`crisis_vol`:
```text
return_1d <= -0.05
OR return_5d <= -0.08
OR (realized_vol_percentile_252d >= 0.90 AND return_21d <= -0.05)
OR vix_percentile_252d >= 0.95
```

`high_vol`:
```text
realized_vol_percentile_252d >= 0.80
OR vix_percentile_252d >= 0.80
```

`low_vol`:
```text
realized_vol_percentile_252d <= 0.30
```

`normal_vol`:
```text
otherwise (and all features non-NaN)
```

> **Crisis is reactive, not predictive.** `crisis_vol` shifts strategy posture *after* the trigger event, not before. Downstream agents must not assume the regime engine prevents the trigger event itself.

### 5.6 Risk Rank

```yaml
volatility_risk_rank:
  low_vol: 0
  normal_vol: 1
  high_vol: 2
  crisis_vol: 3
  unknown: 2
```

### 5.7 Hysteresis

Escalation to `high_vol` or `crisis_vol`: immediate (per global asymmetric rule). De-escalation: 2 stable days required.

---

## 6. Layer 1D — Breadth State

### 6.1 Modes

```text
etf_proxy
```

US V1 ships `etf_proxy` mode only. Engine output declares `"mode": "etf_proxy"`.

PIT constituent breadth is deferred to v1.1/V2 because it requires a point-in-time historical membership pipeline, delisted symbols, and separate data-quality validation. V1 source code must not implement a PIT breadth path or biased-survivorship fallback.

### 6.2 PIT Constituents Mode (Deferred)

Required input contract:

```json
{
  "date": "YYYY-MM-DD",
  "index": "SPX",
  "symbol": "AAPL",
  "is_member_as_of_date": true,
  "close": 0.0,
  "sma_50": 0.0
}
```

Rules:

- Use only symbols with `is_member_as_of_date=true` for the given `as_of_date`.
- Include delisted/removed symbols if they were members on `as_of_date`.
- Reject non-PIT data unless config explicitly allows biased mode:

```yaml
allow_survivorship_biased_breadth: false
```

If `true`, output must include:

```json
"bias_warning": "survivorship_biased_constituent_universe"
```

This section is retained only to document the future PIT contract. It is out of scope for V1 implementation.

### 6.3 ETF Proxy Mode (US V1)

```yaml
etf_proxy:
  cap_weight_index: SPY
  equal_weight_proxy: RSP
```

### 6.4 Labels

```text
healthy_breadth
neutral_breadth
weak_breadth
divergent_fragile
recovery_breadth
unknown
```

### 6.5 Precedence

```text
divergent_fragile > recovery_breadth > weak_breadth > healthy_breadth > neutral_breadth > unknown
```

### 6.6 PIT Formulas

```python
valid_members = constituents[as_of_date].dropna(subset=["close", "sma_50"])
pct_above_50dma = (valid_members.close > valid_members.sma_50).mean()
breadth_change_20d = pct_above_50dma - pct_above_50dma_20d_ago
index_distance_from_63d_high = close / close.rolling(63).max() - 1
```

### 6.7 PIT Rules

`divergent_fragile`:
```text
index_distance_from_63d_high >= -0.05 AND pct_above_50dma < 0.45
```

`recovery_breadth`:
```text
breadth_change_20d >= 0.15 AND close > SMA_50
```

`weak_breadth`:
```text
pct_above_50dma < 0.45
```

`healthy_breadth`:
```text
pct_above_50dma >= 0.55
```

`neutral_breadth`:
```text
0.45 <= pct_above_50dma < 0.55
```

`unknown`:
```text
valid constituent coverage < 70% OR required features NaN
```

### 6.8 ETF Proxy Formulas

```python
relative_breadth_ratio = RSP_close / SPY_close
relative_breadth_sma50 = relative_breadth_ratio.rolling(50).mean()
relative_breadth_return_20d = relative_breadth_ratio / relative_breadth_ratio.shift(20) - 1
index_distance_from_63d_high = SPY_close / SPY_close.rolling(63).max() - 1
```

### 6.9 ETF Proxy Rules

`divergent_fragile`:
```text
index_distance_from_63d_high >= -0.05
AND relative_breadth_ratio < relative_breadth_sma50
AND relative_breadth_return_20d <= -0.03
```

`weak_breadth`:
```text
relative_breadth_ratio < relative_breadth_sma50
AND relative_breadth_return_20d < 0
```

`healthy_breadth`:
```text
relative_breadth_ratio > relative_breadth_sma50
AND relative_breadth_return_20d >= 0
```

`neutral_breadth`:
```text
otherwise
```

### 6.10 Risk Rank

```yaml
breadth_risk_rank:
  healthy_breadth: 0
  recovery_breadth: 1
  neutral_breadth: 1
  weak_breadth: 2
  divergent_fragile: 3
  unknown: 2
```

---

## 7. Layer 2 — Structural-Causal State

### 7.1 V1 Scope

- Event calendar: required.
- Monetary pressure: not implemented in V1.
- Inflation/growth, credit/funding: **not implemented in V1**. Output `"label": "unknown", "reason": "not_implemented_v1"`.

### 7.2 Event Calendar

Source: manually maintained YAML/CSV loaded and normalized by a helper outside the engine. `RegimeEngine.classify(...)` accepts a normalized DataFrame only; it does not perform file I/O.

```yaml
events:
  - date: "2025-12-10"
    market: "US"
    type: "FOMC"
    importance: "high"
  - date: "2025-12-11"
    market: "US"
    type: "CPI"
    importance: "high"
```

Normalized DataFrame contract:

```text
market
type
importance
date              # required for FOMC, CPI, NFP
publication_date  # required by the normalized contract; loader may default it
```

Rows may leave non-applicable columns null only when a field is not relevant to that event type. Validation is event-type specific.

Labels:
```text
fed_week
cpi_week
nfp_week
expiry_week
earnings_season
normal_calendar
unknown
```

(`rbi_week` is reserved for India market extension; not in US V1.)

Rules:
```text
fed_week:    as_of_date within [-2, +2] NYSE trading days of FOMC, provided publication_date <= as_of_date
cpi_week:    as_of_date within [-1, +1] NYSE trading days of CPI release, provided publication_date <= as_of_date
nfp_week:    as_of_date within [-1, +1] NYSE trading days of NFP release, provided publication_date <= as_of_date
expiry_week: as_of_date within config expiry_rules.monthly_options.window_trading_days around the monthly expiry date derived by rule=third_friday_of_month; V1 default is [-2, 0]
earnings_season: derived from config earnings_seasons, not from event rows
```

Precedence:

```text
fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown
```

If multiple event windows match, `raw_label`, `stable_label`, and `active_label` use this precedence. Evidence must preserve all matches:

```json
{
  "all_matching_events": ["fed_week", "earnings_season"],
  "selected_via_precedence": "fed_week"
}
```

Additional event-specific evidence such as `days_to_fomc` may be included when computable from the event calendar. Importance does not override the hardcoded precedence in V1.

Publication-date defaults for the loader:

- For FOMC, CPI, and NFP rows, if `publication_date` is absent, default it to `event.date - 90 calendar days`.
- For ad-hoc events, if `publication_date` is absent, default it to `event.date`.
- If the normalized event calendar contains an event dated more than 90 calendar days after `as_of_date`, the engine or loader may emit a warning for operator review. This does not fail classification.

V1 event-calendar decisions:

- Scheduled event dates may be used even when `event_date > as_of_date`, but only when `publication_date <= as_of_date`. Outcomes/results are never available early.
- `expiry_week` and `earnings_season` are config-defined rules in `core3-v1.0.0.yaml`, not event-calendar rows.
- `RegimeEngine.classify(...)` stays DataFrame-only for `event_calendar`; file loading belongs outside the engine through a separate loader such as `load_event_calendar()`.

### 7.3 Monetary Pressure (Deferred)

Inputs: 2y yield, 10y yield, DXY.

Labels: `tightening_pressure`, `easing_pressure`, `neutral`, `unknown`.

Rules:
```text
tightening_pressure:
  2y_yield_change_63d >= +50 bps
  OR 10y_yield_change_63d >= +50 bps
  OR DXY_return_63d >= +5%

easing_pressure:
  2y_yield_change_63d <= -50 bps
  OR 10y_yield_change_63d <= -50 bps

neutral: otherwise
```

If any required input unavailable: `label="unknown", reason="feature_not_available"`.

> Note: absolute bps thresholds will need rate-era recalibration. 50bps means different things at 0.5% policy rate vs 5% policy rate. Recalibration approach is in V2 spec §2A.

V1 implementation must not compute monetary pressure. It always emits:

```json
{
  "label": "unknown",
  "reason": "not_implemented_v1"
}
```

The rules above are retained as historical context for the V2 monetary/liquidity design and must not be wired into V1 transition risk.

---

## 8. Layer 3 — Network Fragility

### 8.1 V1 Scope

Not implemented. Breadth state serves as V1 fragility proxy.

```json
{
  "network_fragility": {
    "label": "not_implemented_v1",
    "reason": "breadth_state_used_as_v1_fragility_proxy"
  }
}
```

### 8.2 V2

Network fragility is fully specified in `regime_engine_v2_spec.md` (Section 3).

---

## 9. Layer 4 — Transition Risk

### 9.1 V1 Scope

No weighted score. Named warnings only.

### 9.2 Labels

```text
stable
bull_fragile_warning
bear_stress_warning
recovery_attempt
crisis_override
post_switch_cooldown
unknown
```

### 9.3 Precedence

```text
crisis_override > bear_stress_warning > bull_fragile_warning > recovery_attempt > post_switch_cooldown > stable > unknown
```

### 9.4 Rules

`crisis_override`:
```text
volatility_state.active_label = crisis_vol
```
> V1 emergency override is `crisis_vol` only. Do not reference `crash_condition` — that label does not exist in V1.

`bear_stress_warning`:
```text
trend_direction.active_label = bear
AND volatility_state.active_label in [high_vol, crisis_vol]
AND breadth_state.active_label in [weak_breadth, divergent_fragile, unknown]
```

`bull_fragile_warning`:
```text
trend_direction.active_label = bull
AND breadth_state.active_label = divergent_fragile
```
(No `severity` field. Severity is implicit in the strategy response's `position_size_multiplier`.)

`recovery_attempt`:
```text
trend_character.active_label = recovery_attempt
OR (
  trend_direction.stable_label was bear at any point in last 60 NYSE trading days
  AND close > SMA_50
  AND breadth_state.active_label in [recovery_breadth, healthy_breadth]
)
```

`post_switch_cooldown`:
```text
any axis stable_label changed today
AND days_since_axis_switch <= 5
```
Emergency override (`crisis_vol`) breaks cooldown.

`stable`:
```text
no warning condition active AND no post_switch_cooldown
```

`confirmed_switch` is **not** a separate label. Any axis `stable_label` changing today is a confirmed switch and triggers `post_switch_cooldown`.

---

## 10. Layer 5 — Strategy Response

Strategy response fields fall into two categories:

- **Base fields** are always present. These are the fields in Section 10.1 plus mandatory `modifiers_applied`.
- **Modifier fields** are conditionally present. They are emitted only when their scenario fires and omitted otherwise.

V1 modifier fields are exhaustive:

```text
hard_max_loss_required
block_weak_signals
prefer_cash_or_hedges
take_profit_faster
allow_leverage_expansion
require_breadth_confirmation
reason
```

No other strategy response fields are permitted in V1. Pydantic uses `extra="forbid"` and `model_dump(exclude_none=True)`.

### 10.1 Base Response (default neutral)

```json
{
  "position_size_multiplier": 1.0,
  "leverage_allowed": true,
  "allow_trend_following": true,
  "allow_buy_dip": true,
  "allow_mean_reversion": true,
  "allow_breakout": true,
  "allow_shorts": true,
  "require_confirmation_for_new_longs": false,
  "require_confirmation_for_shorts": false,
  "log_for_review": false,
  "modifiers_applied": []
}
```

`modifiers_applied` contains the scenario names that fired in increasing priority order. If multiple scenarios fire, the highest-priority scenario is last and wins conflicts. For `default_neutral`, it is `[]`.

### 10.2 Unknown Fallback

```json
{
  "position_size_multiplier": 0.75,
  "leverage_allowed": false,
  "allow_trend_following": true,
  "allow_buy_dip": true,
  "allow_mean_reversion": true,
  "allow_breakout": true,
  "allow_shorts": true,
  "require_confirmation_for_new_longs": true,
  "require_confirmation_for_shorts": true,
  "log_for_review": true,
  "reason": "unknown_or_unmapped_regime",
  "modifiers_applied": []
}
```

### 10.3 Scenario Precedence (highest first)

```text
crisis > bear_stress > bull_fragile > sideways_chop > recovery_attempt > bull_healthy_low_vol > default_neutral
```

Apply modifiers in increasing priority order, layered on `default_neutral`. Highest-priority match wins on conflicting fields.

### 10.4 V1 Scenario Modifiers

`crisis` — when `transition_risk.label = crisis_override`:
```json
{
  "position_size_multiplier": 0.25,
  "leverage_allowed": false,
  "hard_max_loss_required": true,
  "block_weak_signals": true,
  "prefer_cash_or_hedges": true,
  "allow_buy_dip": false
}
```

`bear_stress` — when `transition_risk.label = bear_stress_warning`:
```json
{
  "allow_buy_dip": false,
  "position_size_multiplier": 0.5,
  "leverage_allowed": false,
  "require_confirmation_for_shorts": true
}
```

`bull_fragile` — when `transition_risk.label = bull_fragile_warning`:
```json
{
  "position_size_multiplier": 0.5,
  "allow_buy_dip": false,
  "allow_leverage_expansion": false,
  "require_confirmation_for_new_longs": true
}
```

`sideways_chop` — when `trend_character.active_label = chop AND volatility_state.active_label != crisis_vol`:
```json
{
  "allow_trend_following": false,
  "allow_mean_reversion": true,
  "position_size_multiplier": 0.75,
  "take_profit_faster": true
}
```
> Note: this does not erase direction. A bull + chop market is still `direction=bull, character=chop`.

`recovery_attempt` — when `transition_risk.label = recovery_attempt`:
```json
{
  "position_size_multiplier": 0.5,
  "allow_trend_following": true,
  "allow_buy_dip": true,
  "require_breadth_confirmation": true,
  "allow_leverage_expansion": false
}
```

`bull_healthy_low_vol` — when:
```text
trend_direction.active_label = bull
AND trend_character.active_label in [trending, transition]
AND volatility_state.active_label in [low_vol, normal_vol]
AND breadth_state.active_label = healthy_breadth
```
Modifier:
```json
{
  "position_size_multiplier": 1.0,
  "allow_trend_following": true,
  "allow_buy_dip": true,
  "allow_leverage_expansion": true
}
```

---

## 11. Final V1 Output Schema (Canonical)

```json
{
  "engine_version": "regime-engine-v1.0.0",
  "config_version": "core3-v1.0.0",
  "as_of_date": "2025-06-13",
  "market": "SPY",
  "trend_direction": {
    "raw_label": "bull",
    "stable_label": "bull",
    "active_label": "bull",
    "evidence": {
      "close_gt_sma50": true,
      "close_gt_sma200": true,
      "sma50_gt_sma200": true,
      "return_63d": 0.072
    },
    "data_quality": {
      "status": "ok",
      "freshness_days": 0,
      "completeness": 1.0,
      "reason": null
    }
  },
  "trend_character": {
    "raw_label": "chop",
    "stable_label": "chop",
    "active_label": "chop",
    "evidence": {
      "adx_14": 17.8,
      "return_10d": 0.014,
      "return_21d": 0.028
    },
    "data_quality": {
      "status": "ok",
      "freshness_days": 0,
      "completeness": 1.0,
      "reason": null
    }
  },
  "volatility_state": {
    "raw_label": "normal_vol",
    "stable_label": "normal_vol",
    "active_label": "normal_vol",
    "evidence": {
      "realized_vol_21d": 0.142,
      "realized_vol_percentile_252d": 0.52,
      "vix_percentile_252d": 0.48
    },
    "data_quality": {
      "status": "ok",
      "freshness_days": 0,
      "completeness": 1.0,
      "reason": null
    }
  },
  "breadth_state": {
    "mode": "etf_proxy",
    "raw_label": "neutral_breadth",
    "stable_label": "neutral_breadth",
    "active_label": "neutral_breadth",
    "evidence": {
      "proxy": "RSP/SPY",
      "relative_breadth_return_20d": -0.006,
      "relative_breadth_ratio_vs_sma50": -0.003
    },
    "data_quality": {
      "status": "ok",
      "freshness_days": 0,
      "completeness": 1.0,
      "reason": null
    }
  },
  "structural_causal_state": {
    "event_calendar": {
      "raw_label": "normal_calendar",
      "stable_label": "normal_calendar",
      "active_label": "normal_calendar",
      "evidence": {
        "all_matching_events": [],
        "selected_via_precedence": "normal_calendar"
      }
    },
    "monetary_pressure": {
      "label": "unknown",
      "reason": "not_implemented_v1"
    }
  },
  "network_fragility": {
    "label": "not_implemented_v1",
    "reason": "breadth_state_used_as_v1_fragility_proxy"
  },
  "transition_risk": {
    "label": "stable",
    "evidence": {
      "warnings_active": []
    }
  },
  "strategy_response": {
    "position_size_multiplier": 0.75,
    "allow_trend_following": false,
    "allow_mean_reversion": true,
    "leverage_allowed": true,
    "allow_buy_dip": true,
    "allow_breakout": true,
    "allow_shorts": true,
    "require_confirmation_for_new_longs": false,
    "require_confirmation_for_shorts": false,
    "log_for_review": false,
    "modifiers_applied": ["sideways_chop"]
  }
}
```

---

## 12. Implementation Plan

### 12.1 Vertical Slicing (mandatory)

Build in vertical slices, not horizontal layers. Each slice ships end-to-end (feature → classifier → hysteresis → replay → tests → at least one golden date passing) and is committed before the next slice starts.

Slice order:

1. **Foundation** — data ingestion, NYSE trading calendar, NaN cold-start logic, `RegimeOutput` Pydantic model, `classify(as_of_date)` skeleton, data quality contract.
2. **Fixture Verification** — raw SPY/RSP/VIX fixtures, derived golden labels, and a verification report with computed features and predicate evaluations.
3. **Trend Direction** — feature + classifier + hysteresis + replay + tests.
4. **Trend Character** — same pattern.
5. **Volatility State** — same pattern.
6. **Breadth State** — ETF proxy mode only (RSP/SPY). PIT breadth is out of scope for V1.
7. **Event Calendar** — YAML/CSV parser + NYSE trading-day window rules + overlap evidence.
8. **Transition Risk** — composes prior slices, no own features.
9. **Strategy Response** — composes prior slices, no own features.

Each slice must pass its assigned golden date before the next slice begins.

Fixture verification is a hard gate before classifier implementation. If fixture verification produces a label that contradicts the hand-labeled table, the table is wrong by definition because V1 rules are deterministic. Replace the fixture with a date that mechanically produces the intended regime before any classifier slice begins. Do not relax rule predicates to make a fixture pass.

### 12.2 Golden Test Set (build and verify before slice 3)

Build the test fixtures first. Each test passes a DataFrame with **at least 320 trading days** of history ending on the test's `as_of_date`. Single-row DataFrames are forbidden — they will return `unknown` and the test will be meaningless.

Golden fixture artifacts:

```text
tests/fixtures/raw/               # vendor OHLCV/VIX CSVs, committed read-only
tests/fixtures/raw/PROVENANCE.md  # source URL/vendor/fetch date/license
tests/fixtures/derived/golden_dates.yaml
tests/fixtures/verification/golden_dates_report.yaml
```

`scripts/verify_fixtures.py` reads only repo-local raw files and performs no network calls. The verification report records raw values, computed features, predicate evaluations, selected labels, generator commit, and timestamp for every fixture.

Raw fixture CSVs should be marked generated and excluded from normal diffs with `.gitattributes`:

```gitattributes
tests/fixtures/raw/*.csv linguist-generated=true
tests/fixtures/raw/*.csv -diff
```

| as_of_date | Expected trend_direction | Expected character | Expected volatility | Expected breadth | Expected transition_risk |
|---|---|---|---|---|---|
| 2017-06-01 | bull | trending | low_vol | healthy_breadth | stable |
| 2018-02-05 (Volmageddon) | bull | transition | crisis_vol | **pin during fixture verification** | crisis_override |
| 2018-12-24 | bear | trending | high_vol | weak_breadth | bear_stress_warning |
| 2019-09-13 | bull | trending | normal_vol | healthy_breadth | stable |
| 2020-03-16 (COVID crash) | bear | transition | crisis_vol | weak_breadth | crisis_override |
| 2020-04-10 | bear | recovery_attempt | high_vol | recovery_breadth | recovery_attempt |
| 2021-11-15 | bull | trending | low_vol | healthy_breadth | stable |
| 2022-06-13 | bear | trending | high_vol | weak_breadth | bear_stress_warning |
| 2022-10-12 | bear | trending | high_vol | weak_breadth | bear_stress_warning |
| 2024-01-16 | bull | trending | low_vol | healthy_breadth | stable |

These are hand-labeled expectations pending Slice 2 verification. The deterministic rule predicates win over intuition. If a slice can't pass its assigned date after verification, the bug is either the spec or the implementation — investigate before moving on. Do not relax expectations to make tests pass.

After all slices ship, all 10 dates run as a regression suite on every commit.

### 12.3 Validation Beyond Unit Tests

A regime label is useful only if it changes downstream strategy behavior. Track per regime (in walk-forward backtests):

- strategy return, max drawdown, Sharpe
- hit rate, average trade duration
- false switch rate, average detection lag
- time spent in regime
- strategy PnL improvement from regime gating

Do not tune thresholds on the holdout period.

### 12.4 Package and Dependency Contract

V1 uses a Python `src/` package layout:

```text
pyproject.toml
src/regime_detection/
tests/
src/regime_detection/configs/core3-v1.0.0.yaml
scripts/verify_fixtures.py
```

Runtime dependencies:

```text
pandas
numpy
pydantic>=2,<3
pandas_market_calendars
pyyaml
```

Development dependencies:

```text
pytest
pytest-cov
hypothesis
ruff
pyright
```

The package version and emitted `engine_version` are the same versioning concept and must be checked by unit test.

V1 must include a CI or pre-commit check that fails on V2 scaffolding/imports, including HMM libraries, macro fetchers, correlation/eigenvalue modules beyond schema needs, ORCA/SRR, Hurst, efficiency ratio, weighted transition score, and `crash_condition`.

---

## 13. Coding Agent Prompt (drop in at top of agent run)

```text
You are implementing the V1 regime detection engine.

ABSOLUTE RULE: When the spec is ambiguous or silent, stop and ask. Do not invent.

Specifically forbidden inventions:
- feature formulas
- thresholds
- scenario precedence
- state machines
- fallback behavior
- confidence scores
- fields not specified in the schema
- config schemas not specified in the spec

Architecture:
Observable Market State + Structural-Causal State + Network Fragility
+ Transition Risk → Strategy Response

V1 implementation contract:

1. Build vertical slices in this order: Foundation, Fixture Verification,
   Trend Direction, Trend Character, Volatility, Breadth (ETF proxy),
   Event Calendar, Transition Risk, Strategy Response.
   Each slice ships end-to-end before the next begins.

2. Fixture Verification is a hard gate. If verified rule predicates
   contradict a hand-labeled fixture, the fixture table is wrong. Replace
   the fixture with a date that mechanically produces the intended regime.
   Do not relax rule predicates to make fixtures pass.

3. Integration tests must pass DataFrames with at least 320 trading days
   of history ending on the as_of_date. Single-row DataFrames are
   forbidden.

4. NaN handling is strict: if any required feature is NaN, the classifier
   returns label="unknown" with reason="insufficient_history". Boolean
   rules are NEVER evaluated against NaN. Check NaN first, return unknown,
   then evaluate rules only on non-NaN data.

5. Use the NYSE trading calendar for all trading-day arithmetic. Use
   pandas_market_calendars or equivalent. Never use bdate_range.
   classify(as_of_date) must raise ValueError for non-NYSE trading days.

6. classify(as_of_date) is stateless replay-safe. Recompute history
   internally from as_of_date - max_lookback - max_hysteresis_days.

7. Use Pydantic models for RegimeOutput and all sub-objects. Match the
   Section 11 output shape and the Section 10 conditional strategy-response
   field whitelist exactly.

8. Hysteresis is asymmetric: escalation immediate, de-escalation debounced.

9. trend_direction and trend_character are SEPARATE axes. Never collapse
   them into a single label.

10. Breadth has a neutral_breadth label covering 0.45–0.55 to close the
    gap between weak and healthy.

11. transition_risk has no confirmed_switch label. Any axis stable_label
    change today triggers post_switch_cooldown.

12. V1 emergency override is crisis_vol only. Do not reference
    crash_condition — that label does not exist in V1.

13. Strategy response uses scenario precedence (crisis > bear_stress >
    bull_fragile > sideways_chop > recovery_attempt > bull_healthy_low_vol
    > default_neutral). Apply modifiers in increasing priority order
    layered on default_neutral.

14. Every output includes evidence, data_quality, engine_version,
    config_version, as_of_date.

15. No confidence field in V1.

16. Precedence orderings and risk_rank tables are hardcoded in code,
    not config.

17. Do NOT implement: HMM, GMM, eigenvalues, graph models, macro
    inference, credit/inflation models, weighted transition score,
    Hurst, efficiency ratio, ORCA/SRR, severity fields, crash_condition,
    PIT breadth, monetary_pressure, or sideways_stress_warning.

18. V1 source code must not scaffold V2. Add a CI/pre-commit grep check
    blocking HMM libraries, macro fetchers, correlation/eigenvalue modules
    beyond schema needs, ORCA/SRR, Hurst, efficiency ratio, weighted
    transition score, and crash_condition.
```

---

## 14. V2

V2 is specified in a separate document: `regime_engine_v2_spec.md`.

V2 work begins only after V1 ships all 9 vertical slices, all 10 V1 golden test dates pass, V1 has run in shadow mode for at least one year of out-of-sample data, and V1 demonstrates measurable strategy improvement vs no-regime baseline.

The coding agent building V1 must not reference, prepare for, or scaffold V2 components. V1 is its own deliverable.
