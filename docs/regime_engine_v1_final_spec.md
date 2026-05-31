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
    vix_data: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
    config: RegimeConfig | None = None,
    sector_etf_closes: dict[str, pd.Series] | None = None,
    cross_asset_closes: dict[str, pd.Series] | None = None,
    macro_series: dict[str, pd.Series] | None = None,
    pit_constituent_intervals: pd.DataFrame | None = None,
    constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
    aaii_sentiment: pd.DataFrame | None = None,
    implied_vol_30d: pd.Series | None = None,
    central_bank_text_releases: pd.DataFrame | None = None,
    cpi_first_release: pd.Series | None = None,
    news_sentiment: pd.Series | None = None,
    request_source: Literal["direct", "profile_manifest"] = "direct",
    manifest_resolved_inputs: frozenset[str] | None = None,
    manifest_cli_overrides: frozenset[str] | None = None,
) -> RegimeOutput
```

Rules:

- Use only data with date `<= as_of_date`.
- Never use future constituent membership.
- Never use future event data.
- `as_of_date` must be an NYSE trading day. If it is not, raise `ValueError` with the nearest prior and next NYSE trading days in the message.
- Do not roll non-trading `as_of_date` values backward or forward.
- Live mode = `classify(as_of_date=today)`.

V1 input contract:

- `market_data` is a long/wide-enough OHLCV DataFrame with at least `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
- US V1 requires `SPY` rows for the market index.
- ETF proxy breadth requires `RSP` rows in the same contract.
- VIX support may be provided either as `vix_data` or as a `VIX` symbol in market data; when Alpaca does not provide true VIX, `VIXY` is the documented operational proxy. Tests must use deterministic repo fixtures.
- `event_calendar` is required at the engine boundary. Missing event input is a caller error, not a V2 optional seam.
- All V2 inputs are explicit optional seams on `ClassifyRequest`; absent seams remain `None` and are handled by the axis/boundary policy that owns them.
- `breadth_data` is not an engine input. Passing it to `classify`, `classify_window`, or `ClassifyRequest` must fail loudly instead of being ignored.
- Profile-runner calls must set `request_source="profile_manifest"` and pass manifest provenance through `manifest_resolved_inputs` / `manifest_cli_overrides`. Direct calls must not carry manifest metadata.
- Feature-store optional seams must emit `FeatureStore.availability` with the declared absence policy, required inputs, and missing inputs.
- Runtime outputs and operator artifacts must expose per-date classification coverage and rule provenance without changing the archived V1 wire projection.

Canonical request object:

```python
ClassifyRequest(
    end_date: date,
    market_data: pd.DataFrame,
    lookback_days: int = 1,
    event_calendar: pd.DataFrame,
    ...
) -> RegimeTimeline via RegimeEngine.classify_request(...)
```

`classify` and `classify_window` are wrappers over `classify_request`; all boundary validation belongs on the request path.

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

V1 ships with `configs/core3-v1.0.0.yaml`.

Rules:

- `RegimeEngine` loads config once at construction time.
- Default config path is `configs/core3-v1.0.0.yaml`.
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

Thresholds (config-driven; values shown are V1 defaults):

```yaml
data_quality:
  max_freshness_days: 3
  min_completeness: 0.90
```

```text
completeness >= min_completeness AND all required features non-NaN  → status=ok, label emitted
0.70 <= completeness < min_completeness                             → status=degraded, label emitted with warning
completeness < 0.70                                                 → label=unknown, reason=insufficient_data
any required feature is NaN                                         → label=unknown, reason=insufficient_history
freshness_days > max_freshness_days                                 → label=unknown, reason=stale_data
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

Risk escalation defaults to immediate via `default_escalation_days: 1` on each
axis-level hysteresis config section. Raising `default_escalation_days` or
adding an `escalation_days_by_label` override delays stable-label entry for the
configured label; `active_label` still surfaces the riskier raw label on the
first session. De-escalation remains debounced by `deescalation_days_by_label`
and `default_deescalation_days`.

```python
if risk_rank(raw_label) > risk_rank(stable_label):
    active_label = raw_label  # escalate immediately
    escalation_fast_path = True
else:
    active_label = stable_label  # debounced de-escalation
    escalation_fast_path = False
```

Default V1 hysteresis ships under the neutral axis sections, not a separate
flat `hysteresis` block:

```yaml
trend_direction:
  default_escalation_days: 1
  deescalation_days_by_label:
    bear: 3
    transition: 3
    sideways: 3
    bull: 3
    unknown: 3
  default_deescalation_days: 3
trend_character:
  default_escalation_days: 1
  deescalation_days_by_label:
    recovery_attempt: 3
    trending: 3
    chop: 3
    transition: 3
    unknown: 3
  default_deescalation_days: 3
volatility_state:
  default_escalation_days: 1
  deescalation_days_by_label:
    crisis_vol: 2
    high_vol: 2
    rising_vol: 2
    low_vol: 2
    normal_vol: 2
    vol_crush: 2
    unknown: 2
  default_deescalation_days: 2
breadth_state:
  default_escalation_days: 1
  deescalation_days_by_label:
    weak_breadth: 2
    healthy_breadth: 2
    unknown: 2
  default_deescalation_days: 2
```

> The event_calendar output intentionally has **no hysteresis**. Calendar windows are themselves deterministic (`as_of_date` is inside an event window or it is not), so a debounce knob is meaningless. The current wire shape exposes `primary_label` for compact display/precedence and `matching_labels` for all overlapping event windows. It does not construct the usual hysteresis label triple for the calendar output.

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
# Implementation pin: ADX_14 uses pandas ewm(alpha=1/14, adjust=False, min_periods=14)
# for ATR, +DM, -DM, and DX smoothing.
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
daily_returns = close.pct_change(fill_method=None)
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
  rising_vol: 2
  crisis_vol: 3
  vol_crush: 3
  unknown: 2
```

> `rising_vol` and `vol_crush` are V2 §1C labels. They share risk-rank values with `high_vol` and `crisis_vol` respectively because V2 precedence (`crisis_vol > vol_crush > high_vol > rising_vol`) is enforced by rule evaluation order in `evaluate_v2_volatility_label`, not by distinct risk-rank integers. Hysteresis de-escalation uses the rank values; the V2 intra-tier ordering is resolved before hysteresis.

### 5.7 Hysteresis

Escalation to `high_vol` or `crisis_vol`: immediate (per global asymmetric rule). De-escalation: 2 stable days required.

---

## 6. Layer 1D — Breadth State

### 6.1 Modes

```text
etf_proxy
```

US V1 ships `etf_proxy` mode only. Engine output declares `"mode": "etf_proxy"`.

PIT constituent breadth was deferred in V1. It is now implemented in V2 via `breadth_state_v2.py` using the `pit_constituent_biased_research` mode with survivorship-bias warnings. PIT features include `pct_above_50dma`, `pct_above_200dma`, `ad_line`, `nh_nl_ratio`, `upvol_downvol_ratio`, and `breadth_thrust`. The PIT data source is `sp500_ticker_intervals.parquet` + constituent OHLCV.

### 6.2 PIT Constituents Mode (Implemented in V2)

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
recovery_breadth     # PIT mode only — unreachable in V1 ETF proxy mode
unknown
```

> **Mode-availability note:** `recovery_breadth` can only be produced by the §6.7 PIT formulas. The §6.9 ETF-proxy rules (the only V1 mode shipped) cannot fire it. Consequently, every V1 cross-reference to `recovery_breadth` (e.g., the §9.4 `recovery_attempt` clause "breadth in [recovery_breadth, healthy_breadth]") effectively reduces to `healthy_breadth` in V1. The label slot is preserved in the precedence and risk-rank tables so V2's PIT-mode emission can land additively without renumbering.

### 6.5 Precedence

```text
divergent_fragile > recovery_breadth > weak_breadth > healthy_breadth > neutral_breadth > unknown
```

### 6.6 PIT Formulas

```python
valid_members = constituents[as_of_date].dropna(subset=["close", "sma_50"])
pct_above_50dma = (valid_members.close > valid_members.sma_50).mean()
breadth_change_20d = pct_above_50dma - pct_above_50dma_20d_ago
index_distance_from_63d_high = close / close.rolling(63, min_periods=50).max() - 1
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
index_distance_from_63d_high = SPY_close / SPY_close.rolling(63, min_periods=50).max() - 1
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
- Inflation/growth, credit/funding: **not implemented in V1**. Output `"state": "unknown", "reason": "not_implemented_v1"`.

### 7.2 Event Calendar

Source: manually maintained YAML/CSV. Coding agent must accept either format.

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

Config-side filter (selects which market's events the engine consumes):

```yaml
event_calendar:
  market: "US"
```

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
fed_week:    as_of_date within [-2, +2] NYSE trading days of FOMC
cpi_week:    as_of_date within [-1, +1] NYSE trading days of CPI release
nfp_week:    as_of_date within [-1, +1] NYSE trading days of NFP release
expiry_week: as_of_date inside the configured monthly options expiry window (see expiry_rules below)
earnings_season: as_of_date inside one of the configured earnings_seasons windows (see earnings_seasons below)
```

Monthly options expiry config (US V1 default shown):

```yaml
expiry_rules:
  monthly_options:
    rule: third_friday_of_month
    window_trading_days: [-2, 0]
    label: expiry_week
```

`rule` is a `Literal["third_friday_of_month"]`. `window_trading_days: [start, end]` is the inclusive offset (in NYSE trading days) around the third Friday during which the `expiry_week` label fires.

Earnings season config (US V1 defaults shown):

```yaml
earnings_seasons:
  - quarter: Q1
    start_rule: second_monday_of_january
    end_offset_days: 35
  - quarter: Q2
    start_rule: second_monday_of_april
    end_offset_days: 35
  - quarter: Q3
    start_rule: second_monday_of_july
    end_offset_days: 35
  - quarter: Q4
    start_rule: second_monday_of_october
    end_offset_days: 35
```

`quarter` is `Literal["Q1","Q2","Q3","Q4"]`. `start_rule` is one of `Literal["second_monday_of_january","second_monday_of_april","second_monday_of_july","second_monday_of_october"]`. `end_offset_days` is the inclusive calendar-day length of the window beginning at `start_rule`.

Precedence:

```text
fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown
```

If multiple event windows match, `primary_label` uses this precedence for
compact display and `matching_labels` preserves all matches for downstream
logic that cares about event membership:

```json
{
  "primary_label": "fed_week",
  "matching_labels": ["fed_week", "earnings_season"],
  "evidence": {
    "selection_method": "precedence"
  }
}
```

Additional event-specific evidence such as `days_to_fomc` may be included when computable from the event calendar. Importance does not override the hardcoded precedence in V1.

#### Per-event `window_days` Override (Optional)

Individual event rows may override the per-type window via a `window_days` column. The value must be a 2-element `[start_offset, end_offset]` pair in NYSE trading days; absence falls back to the type's default window from §7.2 above. Use sparingly — the override is intended for one-off events whose impact window differs from the standard pattern (e.g., an FOMC meeting paired with mid-meeting Powell remarks that extend the natural window). Any override should be documented in the event row's `notes`/`reason` column for replay clarity.

```yaml
events:
  - date: "2025-12-10"
    market: "US"
    type: "FOMC"
    importance: "high"
    window_days: [-3, 3]   # override: extend default (-2, +2) to (-3, +3)
    notes: "Mid-meeting press leak window"
```

The override is consumed by `compute_event_calendar_outputs` per the row; the type's spec window remains the default for rows without the column. PIT publication-date gating still applies — a row's window cannot fire before its `publication_date`.

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
  "state": "unknown",
  "reason": "not_implemented_v1"
}
```

The rules above are retained as historical context for the V2 monetary/liquidity design and must not be wired into V1.

---

## 8. Layer 3 — Network Fragility

### 8.1 V1 Scope

Not implemented. Breadth state serves as V1 fragility proxy.

```json
{
  "network_fragility": {
    "state": "not_implemented_v1",
    "reason": "breadth_state_used_as_v1_fragility_proxy"
  }
}
```

### 8.2 V2

Network fragility is fully specified in `regime_engine_v2_spec.md` (Section 3).

---

## 9. Layer 4 — Transition Risk

V1 no longer defines an active transition-risk classifier.

The earlier V1 named-warning design has been superseded by the V2 score-first
transition-risk model in `regime_engine_v2_spec.md` §4. Current engine outputs
must use the V2 shape:

```text
transition_risk.state
transition_risk.score
transition_risk.score_components
transition_risk.primary_drivers
transition_risk.triggered_rules
transition_risk.data_quality
```

V1 still defines the base axes and event-calendar fields that V2 transition
risk consumes. It does not define transition-risk labels, precedence, warning
rules, score weights, history windows, or final-state debouncing.

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

### 10.4 Scenario Modifiers

Strategy response consumes the V2-owned `transition_risk.state` when transition
risk is present. V1 strategy behavior remains unchanged unless V2 config is
active. With V2 config active, event-window strategy adjustments consume
`event_calendar.matching_labels` through `strategy_event_modifiers`; they do
not branch on the compact display label.

`crisis` — when `transition_risk.state = crisis`:
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

`bear_stress` — when `transition_risk.state = bear_stress`:
```json
{
  "allow_buy_dip": false,
  "position_size_multiplier": 0.5,
  "leverage_allowed": false,
  "require_confirmation_for_shorts": true
}
```

`bull_fragile` — when `transition_risk.state = fragile_bull`:
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

`recovery_attempt` — when `transition_risk.state = recovery_attempt`:
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

> **V1 wire contract.** This canonical shape is what the engine emits when `config_version == "core3-v1.0.0"`. Byte-identity is enforced by `tests/test_v1_frozen_replay.py` against archived JSON fixtures under `tests/fixtures/v1_frozen_outputs/`. The Python `RegimeOutput` Pydantic model holds the V2-extended internal shape (see §11.1 below) but coerces to the V1 wire shape via `_rewrite_legacy_v1_wire_shapes()` in `models.py` whenever `config_version` matches the V1 string. The two extensions where coercion is load-bearing:
>
> - `structural_causal_state.monetary_pressure` is rewritten to `{"state": "unknown", "reason": "not_implemented_v1"}`.
> - `network_fragility` is rewritten to `{"state": "not_implemented_v1", "reason": "breadth_state_used_as_v1_fragility_proxy"}`.
>
> When `config_version != "core3-v1.0.0"` (V2 mode), the live richer shapes are emitted instead and additional V2-only top-level fields appear — see §11.1.

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
      "sma_50": 482.3,
      "sma_200": 461.1,
      "return_63d": 0.072,
      "close_gt_sma50": true,
      "close_gt_sma200": true,
      "sma50_gt_sma200": true,
      "within_5pct_sma200": true
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
      "return_21d": 0.028,
      "prior_63d_drawdown": -0.032,
      "recovery_attempt": false,
      "trending": false,
      "chop": true,
      "range_bound": false,
      "breakout_expansion": false
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
      "vix_percentile_252d": 0.48,
      "crisis_vol": false,
      "high_vol": false,
      "low_vol": false
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
    "raw_label": "weak_breadth",
    "stable_label": "weak_breadth",
    "active_label": "weak_breadth",
    "evidence": {
      "proxy": "RSP/SPY",
      "relative_breadth_ratio": 0.312,
      "relative_breadth_sma50": 0.315,
      "relative_breadth_return_20d": -0.006,
      "index_distance_from_63d_high": -0.021,
      "divergent_fragile": false,
      "weak_breadth": true,
      "healthy_breadth": false
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
      "primary_label": "normal_calendar",
      "matching_labels": ["normal_calendar"],
      "evidence": {
        "selection_method": "precedence"
      }
    },
    "monetary_pressure": {
      "state": "unknown",
      "reason": "not_implemented_v1"
    }
  },
  "network_fragility": {
    "state": "not_implemented_v1",
    "reason": "breadth_state_used_as_v1_fragility_proxy"
  },
  "transition_risk": {
    "state": "stable",
    "evidence": {
      "triggered_rules": [],
      "stable_changed_today": false,
      "days_since_axis_switch": 2,
      "axis_switch_count": 0,
      "recent_axis_switch_count": 0
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

### 11.1 V2-Extended Output Shape (when `config_version != "core3-v1.0.0"`)

When the engine runs under a V2 config, the wire shape extends in two ways:

**Reshaped fields** (formerly flat `{label, reason}` in V1, now full axis triples in V2):

```json
"structural_causal_state": {
  "event_calendar": { /* unchanged from V1 */ },
  "monetary_pressure": {
    "state": "unknown",
    "evidence": {},
    "data_quality": { "status": "insufficient_history", ... }
  }
},
"network_fragility": {
  "raw_label": "diversified_normal",
  "stable_label": "diversified_normal",
  "active_label": "diversified_normal",
  "evidence": { /* §3.5 rule inputs */ },
  "data_quality": { ... },
  "mode": "sector_cross_asset_24"
}
```

**New optional top-level fields** (each lands when its V2 slice ships and the config + inputs are wired; omitted from the wire via `exclude_none=True` when absent):

```text
inflation_growth_state         v2 §2B
credit_funding_state           v2 §2C (real ICE BofA OAS)
credit_funding_state_proxy     v2 §2C proxy (TLT vs HYG/LQD; Ambiguity Log #71)
credit_funding_effective_state v2 §2C downstream OAS/proxy resolver
volume_liquidity_state         v2 §1E
monetary_pressure_state        v2 §2A (replaces structural_causal_state.monetary_pressure semantically; both coexist for V1 compatibility)
change_point                   v2 §4.6 / §6.3 (BOCPD)
cluster                        v2 §6.2 (GMM)
agent_routing                  v2 §5.1
strategy_family_constraints    v2 §5.2
```

`transition_risk` is specified entirely by V2 §4. V1 does not define an active
transition-risk algorithm or legacy transition-risk label set.

The V1 wire shape in §11 remains the canonical V1 contract; V2 fields strictly extend, never mutate, the V1 base. V1 byte-identity is mechanically enforced by `tests/test_v1_frozen_replay.py`.

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
8. **Transition Risk** — removed from the V1 active contract; implemented by V2 §4.
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

| as_of_date | Expected trend_direction | Expected character | Expected volatility | Expected breadth |
|---|---|---|---|---|
| 2017-06-01 | bull | trending | low_vol | healthy_breadth |
| 2018-02-05 (Volmageddon) | bull | transition | crisis_vol | **pin during fixture verification** |
| 2018-12-24 | bear | trending | high_vol | weak_breadth |
| 2019-09-13 | bull | trending | normal_vol | healthy_breadth |
| 2020-03-16 (COVID crash) | bear | transition | crisis_vol | weak_breadth |
| 2020-04-10 | bear | recovery_attempt | high_vol | recovery_breadth |
| 2021-11-15 | bull | trending | low_vol | healthy_breadth |
| 2022-06-13 | bear | trending | high_vol | weak_breadth |
| 2022-10-12 | bear | trending | high_vol | weak_breadth |
| 2024-01-16 | bull | trending | low_vol | healthy_breadth |

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
configs/core3-v1.0.0.yaml
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

8. Hysteresis is asymmetric: escalation defaults to immediate via each axis
   section's `default_escalation_days: 1`; raising that default or adding
   `escalation_days_by_label` delays stable-label entry while `active_label`
   still surfaces the riskier raw label. De-escalation remains debounced.

9. trend_direction and trend_character are SEPARATE axes. Never collapse
   them into a single label.

10. Breadth has a neutral_breadth label covering 0.45–0.55 to close the
    gap between weak and healthy.

11. transition_risk is V2-owned. Do not add V1 transition-risk labels,
    precedence rules, score weights, history windows, or debounce rules.

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

17. Do NOT implement in V1: HMM, GMM, eigenvalues, graph models, macro
    inference, credit/inflation models, transition risk, Hurst, efficiency
    ratio, ORCA/SRR, severity fields, crash_condition, PIT breadth, or
    monetary_pressure.

18. V1 source code must not scaffold V2. Add a CI/pre-commit grep check
    blocking HMM libraries, macro fetchers, correlation/eigenvalue modules
    beyond schema needs, ORCA/SRR, Hurst, efficiency ratio, weighted
    transition score, and crash_condition.
```

---

## 14. V2

V2 is specified in a separate document: `regime_engine_v2_spec.md`.

### 14.1 Current Status

**V1 qualification: complete (operator-asserted per implementation plan).**
**V2 implementation: in progress.** V2 work has commenced under the unified V1+V2 design described in the implementation plan; V2 slices 1.x, 2.x, and 4.1 (features-only) have shipped on top of V1, with strict V1 byte-identity preservation under the V1 frozen-replay test (`tests/test_v1_frozen_replay.py`).

V2 progress is tracked in:
- `docs/regime_engine_v2_spec.md` §8 (slice priority order)
- The repo's commit log under the `feat(slice-*)` and `docs(spec)` prefixes
- The V2 Implementation Ambiguity Log within `regime_engine_v2_spec.md` (entries #1 through #53 at time of writing) which records every spec ambiguity surfaced during implementation and its resolution

### 14.2 V1 Qualification Gate Definition (Retained for Future V1 Branches)

The qualification criteria below were authored as the gate that any future V1-only revision branch must clear before authorizing parallel V2 work. They are retained here for historical context and as the definition any V1 successor must satisfy:

V2 work begins only after V1 ships all 9 vertical slices, all 10 V1 golden test dates pass, V1 passes historical walk-forward validation over at least one full out-of-sample year, V1 completes 252 consecutive successful NYSE trading sessions of forward shadow mode with frozen classification logic and immutable archived inputs/outputs, and V1 demonstrates measurable strategy improvement vs no-regime baseline.

Historical walk-forward and forward shadow serve different purposes and both are required:

- Historical walk-forward validates the engine logic on unseen historical data using only as-of inputs.
- Forward shadow validates operational stability, data-feed handling, calendar discipline, reproducibility, and incident response under real daily execution.

The forward-shadow counter starts on the next NYSE trading day after the historical walk-forward gate passes. Only successful sessions with archived inputs/outputs count toward the 252-session requirement. Missed sessions extend the window, and any qualification-breaking classification change during shadow restarts the count from session 1 under the new frozen version.

Operational qualification rules for shadow mode are specified separately in `docs/shadow_runner_spec.md`.

### 14.3 V1 Authoring Discipline (Still In Force on V1 Code Paths)

The coding agent working on V1 code paths must not reference, prepare for, or scaffold V2 components. V1 source files remain V1-only deliverables.

This rule applies to V1 code paths and the V1 spec **even now that V2 is in progress**: V2 work occurs in V2 modules (e.g., `trend_direction_v2.py`, `volatility_state_v2.py`, `network_fragility.py`, `monetary_pressure.py`, `volume_liquidity_rules.py`, the v2 sub-blocks of `axis_series.py` and `config.py`), with V1 modules either untouched or extended additively (new optional kwargs defaulting to `None`, new shared helpers like `volatility_state.realized_vol` / `volatility_state.wilders_atr` that v1 was refactored to consume without changing output). V1 byte-identity is enforced by `tests/test_v1_frozen_replay.py` and the `test_v1_contract_byte_identity_when_v2_features_absent` family of tests across each v2 slice.

Concretely: if you are reading this spec to maintain V1 itself (e.g., to fix a V1-only bug, ship a v1-frozen-replay regression, or revise V1 §1–§13), the V1-only discipline above applies in full. If you are reading this spec to understand V1 surfaces that V2 builds on, see the V2 spec §1A / §1C / §1D / §2A / §3 for how V2 extends the shared Literal types (e.g., `TrendDirectionLabel`, `VolatilityLabel`) without altering V1 emission semantics.
