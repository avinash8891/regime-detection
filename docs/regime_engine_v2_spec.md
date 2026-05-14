# Regime Detection Engine — V2 Spec

**Status:** roadmap. Do not implement until V1 passes historical walk-forward validation and forward shadow qualification.
**Builds on:** `regime_engine_v1_final_spec.md`
**Engine version:** `regime-engine-v2.0.0` (when shipped)

---

## 0. Prerequisites

V2 work begins **only after** all of the following hold:

- V1 ships all 9 vertical slices.
- All 10 V1 golden test dates pass.
- V1 passes historical walk-forward validation over at least one full out-of-sample year using frozen V1 code/config and as-of historical inputs only.
- V1 has been live in forward shadow/paper mode for at least 252 consecutive successful NYSE trading sessions with frozen classification logic and immutable archived inputs/outputs.
- V1 demonstrates measurable strategy improvement vs no-regime baseline (lower drawdown, fewer wrong-environment trades, or improved Sharpe).

Historical walk-forward and forward shadow are **not interchangeable**.

- Historical walk-forward validates engine logic on unseen historical data.
- Forward shadow validates operational stability: data ingestion, calendar handling, daily execution discipline, reproducibility, and incident response.

V2 activation requires **both**, in sequence:

1. Historical walk-forward passes first.
2. Forward shadow starts on the next NYSE trading day after the freeze tag.
3. The forward shadow counter increments only on sessions where the runner executes successfully and archived inputs/outputs are written per `docs/shadow_runner_spec.md`.
4. Missed sessions extend the window; they do not count toward the 252-session requirement.
5. Any qualification-breaking classification change during shadow restarts the count from session 1 under the new frozen version.
6. V2 work begins only after the forward shadow window completes under those rules.

Qualification details for the forward shadow runner, storage, freeze policy, and replay verification live in `docs/shadow_runner_spec.md`.
Historical walk-forward qualification details, required artifacts, and pass/fail criteria live in `docs/historical_walkforward_spec.md`.

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

`breakout_expansion` (upside only — see direction note below):
```text
close_breaks_20d_or_50d_range
AND bollinger_band_width_expanding
AND volume_above_20d_average
AND followthrough_rate >= 0.60
```

Operational definitions:

```python
# close_breaks_20d_or_50d_range — strict upside break of the prior-window close-high
breakout_20d  = close[t] > max(close[t-20..t-1])
breakout_50d  = close[t] > max(close[t-50..t-1])
close_breaks_20d_or_50d_range = breakout_20d OR breakout_50d

# bollinger_band_width_expanding — textbook BB (period=20, multiplier=2)
#   bb_width = upper - lower = 4 * std(close[t-19..t], ddof=0)
# "Expanding" compared to 5 sessions ago, matching the 5-day post-break hold
# in followthrough_rate (single coherent timeframe).
bollinger_band_width_expanding = bb_width_20[t] > bb_width_20[t-5]

# volume_above_20d_average — strict, t excluded from the baseline
volume_above_20d_average = volume[t] > mean(volume[t-20..t-1])

# followthrough_rate — fraction of recent upside breakouts that held
#   Walk backwards through history (cap lookback at 504 sessions) and
#   collect the 20 most-recent past sessions where breakout_20d OR
#   breakout_50d fired. For each such session b, "held" iff
#   close[b+i] > breakout_level for every i in 1..5 (continuous 5-day hold).
#   `breakout_level` = the max-of-prior-window that close[b] crossed at b.
followthrough_rate = held_count / 20
```

Direction: `breakout_expansion` fires on **upside** breakouts only — `followthrough_rate`'s definition explicitly requires close to stay **above** the breakout level. Downside breakouts are out of scope for this label.

Cold-start: the rule cannot fire reliably until at least 20 prior upside breakouts have occurred within the trailing 504-session window. This is the strictest warm-up in any V2 label; new universes / early backtest dates will see this label silent.

`0.60` threshold rationale: matches the historical bull-market followthrough baseline (~55-65% per breakout-quality literature; Zweig / O'Neil neighborhood), is symmetric with §1D `nh_nl_ratio < 0.4` (1 − 0.6), and skews the rule modestly toward false-negative bias — the deliberately conservative side, since false positives route through `breakout_specialist` cohort (§5.1) and produce active PnL damage in chop, whereas false negatives only cost opportunity. The value is a **V2 walk-forward calibration placeholder** per §9.1: post-walk-forward evidence may tighten it to 0.65 (if false-positive rate exceeds target) or loosen it to 0.55 (if false-negative rate dominates).

`range_bound`:
```text
abs(return_63d) < 0.05
AND max_midpoint_excursion_20d <= 0.05
AND ADX_14 < 20
```

`max_midpoint_excursion_20d` definition:
```python
midpoint_20d = (max(close[t-19..t]) + min(close[t-19..t])) / 2
max_midpoint_excursion_20d = max(abs(close[i] - midpoint_20d) / midpoint_20d for i in range(t-19, t+1))
```

Semantics: every close in the 20d window must sit within ±5% of the rolling midpoint (the average of the window's high and low close). This pins the literal "oscillates inside" meaning of the rule — closes that orbit a center, rather than total-span containment. The other two conjunctions (`abs(return_63d) < 0.05`, `ADX_14 < 20`) already filter for low directional intensity, so this third clause encodes the structural around-a-center property that the first two do not.

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
AND sentiment_score >= euphoria_sentiment_threshold
```

Operational definitions:

```python
# realized_vol_21d rising — strict 5-session change (Log #68 §1D analogue:
# same memory horizon as `pct_above_50dma rising` / `nh_nl_ratio rising`).
realized_vol_21d_rising = realized_vol_21d[t] > realized_vol_21d[t - 5]

# sentiment_score — AAII bull-bear spread 8-week moving average.
# Source columns: AAII weekly survey (`bullish`, `bearish` percentages).
# Derived:
#   bull_bear_spread       = bullish - bearish              (per weekly row)
#   bull_bear_spread_8w_ma = rolling mean over 8 weekly rows
#   sentiment_score        = bull_bear_spread_8w_ma         (points, not %)
#
# Weekly-to-daily alignment (V1 §2.2 stateless replay):
#   sentiment_score[as_of_date] = the value carried by the latest AAII
#   row with publication_date <= as_of_date (forward-fill from publication
#   date; NEVER consult a future-dated reading).
#
# Cold-start (V1 §2.7 inheritance): until at least 4 weekly readings
# exist on or before as_of_date, sentiment_score is NaN and the euphoria
# rule falsifies. The 8-week MA's `min_periods=1` in the fetcher exposes
# values from week 1, but predicate consumption requires a fuller window.
```

Default: `euphoria_sentiment_threshold = +20` (points of bull-bear-spread 8w-MA). This is a V2 §9.1 walk-forward calibration placeholder, not a fixed spec constant — historical AAII bull-bear 8w-MA distribution (1987–present) has top-10% in the +18 to +22 range; +20 sits near the Yardeni / Stovall conventional "high optimism" anchor. Operators may retune via the `trend_direction_v2.euphoria_sentiment_threshold` yaml key.

Picked source notes: `bull_bear_spread_8w_ma` was chosen over the unsmoothed weekly spread (too noisy for a precedence-bearing label) and over a cross-era percentile rank (adds a 252-week warm-up the engine doesn't otherwise need). Put-call ratio and Investors Intelligence sentiment remain valid alternative sources for a future calibration revision but require fetchers not yet built.

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

Feature:
```text
breadth_thrust_feature = 10-session moving average of pct_advancing
```

Label predicate at session t (low-to-high transition within the trailing
10-session window, ADR 0003 / Log #69 closure):
```text
breadth_thrust fires at session t when:
  EXISTS b in [t-10, t-1] with breadth_thrust_feature[b] < 0.40
  AND breadth_thrust_feature[t] > 0.615
```

Both inequalities are strict per Zweig's canonical formulation (the
1986 *Winning on Wall Street* definition). The thresholds 0.40 and
0.615 are spec-fixed (not configurable). NaN at
`breadth_thrust_feature[t]` or at every `b` in `[t-10, t-1]`
falsifies the rule (V1 §2.7 cold-start contract).

V2 adds new breadth labels:
- `breadth_thrust` (bullish initiation — predicate above)
- `broadening_breadth` (recovery confirmation: NH/NL ratio rising AND ad_line_slope_20d > 0)
- `recovery_breadth` (improvement starting, not yet confirmed: NH/NL ratio rising AND ad_line_slope_20d <= 0; ADR 0003 / Log #70 closure)
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

## Implementation Ambiguity Log

Per `docs/v2_slice_gate_checklist.md` §8 (framework), every ambiguity discovered
during slice implementation that is resolved in code (rather than re-spec'd) is
recorded here with: spec citation, the ambiguity, the pinned resolution, and
the slice/commit that resolved it. Entries are append-only.

1. **§3.2 line 577 — `effective_rank` log base.**
   Spec pseudocode wrote `log` without naming the base.
   Resolution: natural log (`ln`, base e); identity correlation matrix yields
   `effective_rank = N`. Pinned in spec line 581 and in
   `regime_detection.network_fragility.compute_features`.
   Resolved by Slice 1.2 cleanup (commit `ef08eb0`).

2. **§3.2 — `min_universe_size` and `min_window_completeness`.**
   Spec did not specify minimum universe size or per-window completeness floor
   for the 63d correlation window.
   Resolution: pinned at `min_universe_size = 20` and
   `min_window_completeness = 0.9`, exposed in v2 config under
   `network_fragility` for §9.1 calibration. See
   `NetworkFragilityConfig` in `regime_detection.config`.
   Resolved by Slice 1.2 cleanup (commit `ef08eb0`).

3. **§3.5 line 634 / line 656 — `narrowing_breadth` enum gap.**
   v2 §3.5 names `narrowing_breadth` in the accepted breadth sets for
   `rising_fragility` and `systemic_stress`, but V1's `BreadthLabel` enum
   (`regime_detection.breadth_state`) did not contain that literal at the
   time of Slice 1.3.
   Resolution (Slice 1.3): pin the accepted sets to what V1 could express
   then — `rising_fragility` accepts `{weak_breadth, divergent_fragile}` and
   `systemic_stress` accepts `{weak_breadth}`. Both call sites in
   `regime_detection.network_fragility_rules` carried breadth-enum
   follow-up marker comments so they could be relinked when the enum
   was extended.
   Resolved by Slice 1.3 (commit `c3badfc`).

   Status update (post Slice 2.8c + Log #3 follow-up): fully closed.
   `BreadthLabel` was widened in Slice 2.8c, and the breadth-enum
   follow-up marker comments in
   `regime_detection.network_fragility_rules` (`rising_fragility` and
   `systemic_stress` accepted_breadth sets) have since been actioned
   and removed. `rising_fragility` now accepts
   `{weak_breadth, narrowing_breadth, divergent_fragile}` (matches
   §3.5 line 634 verbatim) and `systemic_stress` now accepts
   `{weak_breadth, narrowing_breadth}` (matches §3.5 line 656
   verbatim). The §3.5 rule semantics are unchanged; only the
   code-side mapping widened to match the now-canonical spec set.

4. **§3.5 line 620 — `effective_rank_stability_threshold`.**
   Spec wrote "21d std < 5% of mean" inline.
   Resolution: 0.05 pinned as a configurable threshold under
   `network_fragility.rules.effective_rank_stability_threshold` (v2
   calibration §9.1 may retune).
   Resolved by Slice 1.3 (commit `c3badfc`).

5. **§3.5 line 632 — `rising_fragility` "positive slope" definition.**
   Spec wrote "rising over 21d (positive slope)" without naming the
   regression form or strictness.
   Resolution: strictly-positive OLS slope (`numpy.polyfit(x, y, deg=1)`) over
   the trailing 21 sessions with a unit trading-day x-index. The 21d window is
   spec-fixed (`_SPEC_SLOPE_WINDOW_DAYS` constant, not configurable); only the
   threshold (`> 0.0`) is part of the rule.
   Resolved by Slice 1.3 (commit `c3badfc`).

6. **§3.7 lines 675–680 — partial hysteresis spec.**
   Spec lists de-escalation-day defaults for only 4 of the 7 §3.3 labels
   (`rising_fragility=3`, `correlation_concentration=3`,
   `correlation_to_one=5`, `systemic_stress=5`).
   Resolution: the other three labels (`diversified_normal`,
   `stock_picker_dispersion`, `unknown`) default to `0` (immediate
   de-escalation), consistent with their low §3.6 risk-rank. Pinned in the v2
   config under `network_fragility.hysteresis.deescalation_days`.
   Resolved by Slice 1.4 (commit `f82eeb0`).

7. **§3.6 line 667 — `systemic_stress` risk_rank.**
   Spec pins `systemic_stress: 3`. A legacy local fixture in
   `tests/test_per_label_hysteresis.py` had used `4`, which silently bypassed
   the now-canonical config value.
   Resolution: import `NETWORK_FRAGILITY_RISK_RANK` from
   `regime_detection.network_fragility_rules` (the spec-aligned constant) in
   tests rather than re-declaring locally.
   Resolved by Slice 1.4 cleanup.

8. **§3.7 line 675 — `unknown` flicker risk.**
   `unknown` carries `risk_rank=2` (§3.6 line 668), lower than
   `correlation_to_one=3` and `systemic_stress=3`. With `unknown` defaulting
   to `default_deescalation_days=0` (entry #6), a single-day data-quality
   flicker through `unknown` while stable is `correlation_to_one` would
   immediately fast-track de-escalation to `unknown`.
   Resolution: pin `deescalation_days_by_label.unknown = 5` (equal to
   `correlation_to_one`) in the v2 yaml so single-day flickers cannot relax
   the axis. Exposed `NetworkFragilityConfig.default_deescalation_days` so
   the §9.1 calibration can re-tune both the listed and default cohorts
   without code changes.
   Resolved by Slice 1.4 cleanup.

9. **V1↔V2 axis date alignment (`axis_series.py` v2 classifier).**
   The classifier consumes V1 breadth/volatility `active_labels_by_date`
   dicts. The pre-cleanup code used `dict.get(day, "unknown")`, which
   silently downgraded any drifted session to `"unknown"` — defanging
   `systemic_stress`/`rising_fragility` (both gated on breadth).
   Resolution: when the v1 dict is supplied (non-None), a missing session
   raises `KeyError` (loud failure). The `"unknown"` fallback is reachable
   only when the caller explicitly passes `None` for the v1 dict
   (unit-test path).
   Resolved by Slice 1.4 cleanup.

10. **§2.8 data-quality helper — pure-quality vs label-aware paths.**
    `assess_series_input_quality` historically short-circuited on
    `raw_label == "unknown"` to mark an `insufficient_history` status. V2
    classifiers (NetworkFragility) compute the raw label AFTER quality, so
    the V1 short-circuit forced a magic-string workaround at the call site.
    Resolution: add `skip_raw_label_short_circuit: bool = False` to the
    helper. V1 callers keep default semantics; V2 callers opt in.
    Resolved by Slice 1.4 cleanup.

11. **§1A line 79 — Hurst exponent estimator.**
    Spec lists "Hurst Exponent (250d)" with the H>0.55 / H<0.45 bands but
    does not specify the estimator (R/S, DFA, DMA, periodogram, ...).
    Resolution: classical Mandelbrot–Wallis Rescaled-Range (R/S) over a
    single 250-session window (no chunk-averaging). H = log(R/S) /
    log(N) where N = lookback - 1 log-returns. Pinned in
    `regime_detection.trend_direction_v2._rs_hurst_window`.
    Resolved by Slice 2.1.

12. **§1A line 79 — Hurst input series (price vs log-returns).**
    Spec is silent on whether the 250d Hurst window operates on price
    levels or on returns. Resolution: log-returns (literature standard
    for R/S on financial time series; Lo 1991, Mandelbrot–Wallis 1969).
    Pinned in `regime_detection.trend_direction_v2._rs_hurst_window`.
    Resolved by Slice 2.1.

13. **§1A line 116 — `drawdown_252d` peak-window inclusivity.**
    Spec writes "prior 252d drawdown <= -0.15" without naming whether
    the trailing-peak window includes session `t`. Resolution: window is
    `close[t-251..t]` (inclusive of `t`), so the drawdown equals 0
    exactly at a fresh 252d high and is strictly negative otherwise.
    Matches the slice-1.3 convention in
    `regime_detection.network_fragility_rules._trailing_drawdown`.
    Resolved by Slice 2.1.

14. **§1A lines 105–108 — SMA / slope NaN handling at cold-start.**
    Spec is silent on cold-start. Resolution: pandas
    `.rolling(N, min_periods=N).mean()` for SMA; slope is NaN until
    `t >= sma_period - 1 + slope_lookback_days` (so slope_sma_50 first
    non-NaN at t=69, slope_sma_200 at t=219). Standard V1 cold-start
    contract (no warm-up). Pinned in
    `regime_detection.trend_direction_v2._slope_of_sma`.
    Resolved by Slice 2.1.

15. **§1C line 142 — ATR estimator (Wilder vs simple-mean true range).**
    Spec names "ATR_14 / ATR_50" without naming the estimator.
    Resolution: classical Wilder recursive smoothing (the textbook /
    industry default since Wilder 1978 — seed = simple-mean(TR) over the
    first `period` observations, then
    `ATR[t] = (ATR[t-1] * (period - 1) + TR[t]) / period`). Implemented
    once in the shared helper `regime_detection.volatility_state.wilders_atr`
    so the V2 §1C `atr_ratio` feature (slice 2.2) and the future
    `rising_vol` / volatility-rules labels slice both consume one
    implementation.
    Resolved by Slice 2.2.

    **Amendment (Slice 2.4):** v1's `regime_detection.trend_character`
    already contains a `_wilder_ewm(series, n)` helper that uses
    pandas-EWM-style seeding (first-value of the TR series), which is
    NOT byte-equivalent to the textbook mean-seeded `wilders_atr` here at
    cold-start. The two implementations intentionally coexist (option
    (b) from slice 2.2 review): v1 ADX cold-start values are frozen, and
    V2 §1C ATR ratio uses the more faithful textbook form. Both
    converge for large `t` but differ at cold-start. A future cleanup
    may unify them after V2 walk-forward validation per §9.1. A
    cross-reference docstring line on `wilders_atr` in
    `volatility_state.py` calls out the v1 EWM smoother in
    `trend_character.py` so future authors find both via grep.

16. **§1C lines 176–181 — `gap_frequency_20d` window inclusion.**
    Spec writes `count(gap > 0.005) / 20` without naming whether the
    20-session window includes session `t` itself.
    Resolution: window is `[t-19..t]` inclusive of `t`. First valid
    index is **t = 20 (NOT t = 19)**, because `gap[0]` is NaN by
    construction (no `close[-1]` available) and `min_periods=20`
    requires 20 non-NaN observations in the window. This differs from
    `efficiency_ratio_20d` (first valid at t = 19) by exactly one
    session due to the gap-input NaN propagation — the earlier slice
    2.2 note that said the convention "matches slice 2.1's
    efficiency_ratio_20d 'ending at t' convention" was off by one and
    is corrected here. Strictly `> threshold` per spec text — a gap
    exactly equal to the threshold is NOT counted. Pinned in
    `regime_detection.volatility_state_v2._gap_frequency`.
    Resolved by Slice 2.2; first-valid-index documentation amended by
    Slice 2.4.

17. **§1C lines 183–187 — `intraday_range_percentile_252d` rank direction.**
    Spec writes `percentile_rank(intraday_range, lookback=252)` without
    naming `ascending` vs `descending`. Resolution: ascending rank (1.0 =
    current value is the maximum within the trailing 252-session window),
    so a rising intraday-range maps to a rising percentile. Mirrors slice
    1.2's `pd.Series.rolling(N).rank(pct=True)` pattern in
    `regime_detection.network_fragility`. Pinned in
    `regime_detection.volatility_state_v2._intraday_range_percentile`.
    Resolved by Slice 2.2.

18. **§1C line 181 — `gap_threshold_pct` "configurable per market" with
    V2's US-only universe.** Spec text notes the 0.5% threshold is
    "configurable per market", but V2 markets at this point are US-only.
    Resolution: expose a single `VolatilityV2Config.gap_threshold_pct`
    knob (default `0.005`) rather than per-market branching. When
    additional markets land, the knob promotes to a per-market dict
    without changing the compute path.
    Resolved by Slice 2.2.

19. **§1C lines 151–155 — IV/RV-spread feature deferral.**
    Spec defines `iv_rv_spread = implied_vol_30d - realized_vol_21d` and
    notes "Requires options data feed". The V2 repo does not yet ingest
    an options/implied-vol series. Per v2 §10 absolute rule
    ("do not invent component score formulas — use the exact formulas in
    §4.2"; same rule for §3.5, §2A/§2B/§2C, …) we will NOT synthesize an
    implied-vol proxy. Resolution: defer the `iv_rv_spread` feature, the
    `euphoria`/`vol_crush`/`event_window` evidence wiring it feeds, and
    the updated §1C volatility precedence at line 191 until an options /
    implied-vol ingestion slice lands alongside §2D event-calendar work.
    Slice 2.2 explicitly ships only the three §1C features that depend
    on OHLC alone (`atr_ratio`, `gap_frequency_20d`,
    `intraday_range_percentile_252d`).
    Deferred by Slice 2.2.

20. **§1C lines 157–174 — `vol_crush` rule deferral.**
    Spec rule:
    ```
    vol_crush:
      realized_vol_10d < realized_vol_21d * 0.75
      AND implied_vol_falling_sharply
      AND event_window_just_passed
    ```
    Two of the three inputs (`implied_vol_falling_sharply`,
    `event_window_just_passed`) require data the V2 repo does not yet
    ingest: an implied-vol time series (entry #19) and the §2D event
    calendar. Per v2 §10 we do NOT invent either. Resolution: defer the
    `vol_crush` LABEL and its rule wiring; the placeholder
    `VolCrushConfig` in `regime_detection.config` remains a stub until
    the prerequisite slices land.
    Deferred by Slice 2.2.

21. **§1D lines 207–210 — `pct_above_200dma` deferral.**
    Spec formula `mean(member.close > member.sma_200)` requires a
    point-in-time (PIT) constituent-membership universe with
    delisted-symbol handling. §1D lines 198–205 explicitly require
    "delisted and removed symbols included when they were members on
    `as_of_date`" and "explicit rejection of survivorship-biased
    universes". The V2 repo does not yet ingest a PIT membership table.
    Per v2 §10 absolute rule we do NOT silently substitute the current
    SPX universe (which would be survivorship-biased). Resolution: defer
    `pct_above_200dma` until the PIT membership ingestion slice lands.
    Deferred by Slice 2.3.
    Resolved by Slice 2.8c — PIT inputs (`fja05680/sp500` intervals +
    762-stock SQLite OHLCV) ingested; price-field, NaN-SMA, and
    full-history pins recorded in Ambiguity Log #54, #58, #59.

22. **§1D lines 213–216 — `ad_line` / `ad_line_slope_20d` deferral.**
    Cumulative advance/decline line and its 20d slope both require
    per-stock daily advance/decline counts over the PIT universe (entry
    #21). Resolution: defer the feature and its `broadening_breadth`
    label dependency.
    Deferred by Slice 2.3.
    Resolved by Slice 2.8c — feature ships; the `broadening_breadth`
    label remains deferred per Ambiguity Log #26.

23. **§1D lines 218–221 — `nh_nl_ratio` deferral.**
    52-week new highs / new lows ratio requires per-stock 52w
    high/low tracking across the PIT universe (entry #21). Resolution:
    defer the feature and its `broadening_breadth` / `narrowing_breadth`
    label dependencies.
    Deferred by Slice 2.3.
    Resolved by Slice 2.8c — feature ships with the 252-session
    lookback pinned in Ambiguity Log #55 and the NaN-history exclusion
    pinned in #58; the dependent labels remain deferred per #26.

24. **§1D lines 223–226 — `upvol_downvol_ratio` deferral.**
    Up/Down-volume ratio requires per-stock daily volume × advance/decline
    over the PIT universe (entry #21). Resolution: defer.
    Deferred by Slice 2.3.
    Resolved by Slice 2.8c — feature ships with the strict-inequality
    direction pin (Ambiguity Log #56) and `adjusted_close` price-field
    pin (#54). Volume reads the SQLite `volume` column (raw integer
    shares, unadjusted).

25. **§1D lines 231–237 — `breadth_thrust` feature deferral.**
    Zweig-style breadth thrust requires `pct_advancing`, a per-stock
    advance count over the PIT universe (entry #21). Resolution: defer
    the feature; the related `breadth_thrust` LABEL is also deferred
    (entry #26).
    Deferred by Slice 2.3.
    Resolved by Slice 2.8c — the FEATURE ships as the 10-session moving
    average of `pct_advancing` (per Ambiguity Log #56 strict-inequality
    direction). The LABEL ("moves from < 0.40 to > 0.615 within 10
    sessions") remains deferred per entry #26 — label wiring belongs in
    a future breadth-axis-classifier slice.

26. **§1D lines 239–246 — New V2 breadth labels deferral.**
    V2 §1D adds three breadth labels (`breadth_thrust`,
    `broadening_breadth`, `narrowing_breadth`) plus an updated
    precedence ordering at lines 244–246. Every rule input requires
    PIT-dependent features (entries #21–#25). Per v2 §10 we will NOT
    define rules over only the subset we can compute today (sector_breadth
    alone is insufficient to fire any of the three new labels per the
    spec rule text). Resolution: defer the new labels, leave V1's
    `BreadthLabel` enum unchanged, and ship `sector_breadth` as
    evidence-only. Models / classifier remain untouched.
    Deferred by Slice 2.3.

27. **§1D line 229 — `sector_breadth` denominator policy when a sector
    ETF is absent from `MarketContext.sector_etf_closes`.**
    Spec writes "divided by 11" (explicit denominator). Real-world
    feeds occasionally drop a single sector (e.g., XLRE before its 2015
    inception). Two policies are available: (A) NaN the entire output
    series when ANY of the 11 are missing; (B) rebase the denominator
    to the number of sectors present. Resolution: policy (A) — fail
    NaN. Rebasing to a partial denominator silently changes the feature's
    semantics (e.g., 5/10 = 0.5 vs 5/11 ≈ 0.45) and would mask the
    upstream data-quality gap. The fail-NaN policy is also consistent
    with V1 cold-start contract (missing input → NaN, not a synthesized
    value). Implemented in
    `regime_detection.breadth_state_v2.compute_breadth_v2_features`.
    Resolved by Slice 2.3.

28. **§1E line 256 — `volume_zscore_20d` standard-deviation `ddof` choice.**
    Spec writes `z = (volume - rolling_mean) / rolling_std` over a 20-day
    window without naming population vs sample standard deviation.
    Resolution: sample standard deviation (`ddof=1`), pandas /
    `Series.rolling(20).std()` default. This is the standard convention
    for z-scores on financial time series. Constant-volume windows
    yield `std == 0` ⇒ output masked to NaN (`0 / 0`), matching the V1
    cold-start contract (no synthesized values). Pinned in
    `regime_detection.volume_liquidity_v2._volume_zscore` and exposed
    as `VolumeLiquidityV2Config.volume_zscore_ddof` so §9.1 calibration
    can retune without code changes.
    Resolved by Slice 2.4.

29. **§1E — Volume / Liquidity axis classifier deferral.**
    v2 §1E defines three labels (`normal_volume`, `panic_volume`,
    `liquidity_gap_behavior`; lines 260–286), a rule engine (lines
    268–286), a risk-rank table (lines 288–294), and per-label
    hysteresis. The features required by those rules already exist
    (`volume_zscore_20d` from this slice; `gap_frequency_20d` and
    `intraday_range_percentile_252d` from slice 2.2; `return_1d` from
    the V1 volatility feature path), but per the slice-by-slice
    rhythm established for §1A/§1C/§1D the feature compute lands
    BEFORE the classifier wiring. Resolution: ship `volume_zscore_20d`
    as evidence-only in slice 2.4; defer the labels, rule engine,
    risk-rank table, hysteresis, and `axis_series.py`
    `VolumeLiquidityV2SeriesClassifier` to a follow-up
    volume-axis-classifier slice. That slice will consume
    `volume_zscore_20d` from `FeatureStore.volume_liquidity_v2` AND
    `gap_frequency_20d` + `intraday_range_percentile_252d` from
    `FeatureStore.volatility_state_v2` (the two §1E features that
    already live on slice 2.2's seam — they are NOT recomputed in
    `volume_liquidity_v2.py`).
    Deferred by Slice 2.4.

30. **§1E feature placement — `gap_frequency_20d` / `intraday_range_percentile_252d`.**
    Spec §1E lines 257–258 list `gap_frequency_20d` and
    `intraday_range_percentile_252d` as part of the Volume / Liquidity
    feature set, but slice 2.2 had already implemented them under the
    §1C Volatility feature compute (`volatility_state_v2.py`) because
    §1C lines 176–187 also reference them. Resolution: keep the
    one-home-per-concept rule (AGENTS rule B) — those two features
    continue to live in `volatility_state_v2.py` and surface through
    `FeatureStore.volatility_state_v2`. The new slice 2.4 module
    `volume_liquidity_v2.py` ships ONLY `volume_zscore_20d` and exposes
    a separate `FeatureStore.volume_liquidity_v2` seam. The future §1E
    axis classifier reads its three feature inputs from BOTH seams. No
    feature is computed twice.
    Resolved by Slice 2.4.

31. **§1A line 116-118 — `recovery` rule inequality strictness.**
    Spec writes three inequalities with intentionally mixed forms:
    line 116 `prior 252d drawdown <= -0.15` (non-strict), line 117
    `return_63d > 0.10` (strict), line 118 `close > SMA_50` (strict).
    Resolution: pin verbatim — `drawdown_252d` exactly at `-0.15`
    satisfies the rule; `return_63d` exactly at `0.10` does NOT;
    `close == SMA_50` does NOT. Each boundary has a dedicated unit test
    in `tests/test_trend_direction_v2_recovery_rule.py`. Pinned in
    `regime_detection.trend_direction_v2.evaluate_recovery`.
    Resolved by Slice 2.5.

32. **§1A lines 121-127 — `euphoria` label deferral.**
    Spec rule requires `sentiment_score >= configured_threshold`
    (line 126) where `sentiment_score` is sourced from AAII bull-bear,
    put-call ratio percentile, or Investors Intelligence sentiment
    (line 129). The V2 repo did not yet ingest any of those feeds at
    Slice 2.5.
    Per v2 §10 absolute rule we did NOT synthesize a sentiment proxy.
    Initial resolution: defer the `euphoria` label until a sentiment
    ingestion slice lands. The §1A line 132 precedence reserves the
    `euphoria` slot above `bull` so the slice that lands sentiment can
    drop the rule in without re-ordering. The precedence-evaluation
    table in
    `regime_detection.trend_direction_v2._V2_TREND_PRECEDENCE` includes
    `"euphoria"` at index 0 but the rule predicate did not fire at
    Slice 2.5.
    Deferred by Slice 2.5.

    Status update — fully resolved by spec amendment and the
    euphoria-wiring code slice. Three open sub-questions had to be
    pinned (recorded in `docs/decisions/0004-euphoria-sentiment-score-
    and-vol-rising-pins.md` and amended into §1A):

    - `sentiment_score = bull_bear_spread_8w_ma` (AAII 8-week MA).
      AAII fetcher (commit `8c04fae`) supplies the underlying weekly
      `bullish` / `bearish` rows; `bull_bear_spread_8w_ma` is computed
      in `regime_data_fetch.aaii_sentiment._compute_derived`.
      Weekly-to-daily alignment uses the latest publication-date
      `<= as_of_date` per V1 §2.2 stateless replay; cold-start (fewer
      than 4 weekly readings) falsifies the rule per V1 §2.7.
    - `realized_vol_21d rising = vol[t] > vol[t-5]` (strict 5-session
      change), mirroring Log #68's pin for §1D breadth `rising` /
      `falling` qualifiers — single 5-session memory horizon across
      "rate of change" predicates.
    - `euphoria_sentiment_threshold = +20` (points of bull-bear-spread
      8w-MA). V2 §9.1 walk-forward calibration placeholder; configurable
      via the `trend_direction_v2.euphoria_sentiment_threshold` yaml
      key. The Yardeni / Stovall conventional "high optimism" anchor
      sits in the +18 to +22 range; +20 also corresponds to the
      historical top-decile of the AAII bull-bear 8w-MA distribution
      (1987–present).

    Implemented in `regime_detection.trend_direction_v2.evaluate_euphoria`
    and tested by per-conjunct boundary cases in
    `tests/test_trend_direction_v2_euphoria.py`. Side-effect:
    `euphoria_specialist` in `regime_detection.cohort_routing` is now
    reachable (Log entry tracking item 29 in the partial-blocker
    audit also unblocks).

33. **§1A line 90 — `breakout_expansion` label deferral.**
    Spec rule references a `followthrough_rate` metric configurable
    threshold, but the spec text never defines the metric numerically
    (count over what window? what does "follow-through" mean
    operationally?). Per v2 §10 absolute rule we do NOT invent a
    formula. Resolution: defer the `breakout_expansion` label until
    the spec pins `followthrough_rate` or until the user supplies a
    concrete definition.
    Deferred by Slice 2.5.

    Status update — fully resolved.
    Entry #46 pinned the `followthrough_rate >= 0.60` threshold and
    entry #47 pinned the remaining three rule clauses plus the
    `followthrough_rate` windowing metadata (504-session trailing
    lookback, 20 most-recent past upside breakouts, 5-day continuous
    hold). The label is implemented in
    `regime_detection.trend_character` and covered by
    `tests/test_trend_character_v2_labels.py`
    (`test_breakout_expansion_fires_on_4_conditions` +
    four negative-case tests).

34. **§1A line 98 — `range_bound` label deferral.**
    Spec rule writes "price oscillates inside the 20d range" without
    defining "oscillates" operationally (e.g., # of touches against
    the range walls? % of sessions inside the range? Bollinger-style
    band?). Per v2 §10 we do NOT invent a definition. Resolution:
    defer the `range_bound` label until the spec pins the
    oscillation metric.
    Deferred by Slice 2.5.

    Status update — fully resolved.
    Entry #46 pinned the operational form
    `max_midpoint_excursion_20d <= 0.05` (where the 20d midpoint is
    `(max + min) / 2` and the excursion is
    `max(|close[i] - midpoint| / midpoint)` for `i in t-19..t`).
    The label is implemented in `regime_detection.trend_character`
    and covered by `tests/test_trend_character_v2_labels.py`
    (`test_range_bound_fires_on_tight_oscillation` + three
    negative-case tests).

35. **§1A line 132-134 — precedence-ordering enforcement.**
    Spec lists the V2 trend precedence as
    `euphoria > bull > recovery > bear > sideways > transition > unknown`
    but does not explicitly address what happens when multiple rules
    fire on the same session. Resolution: pin precedence-by-rank — the
    HIGHEST-ranked label whose rule fires wins, and a fired rule cannot
    OVERRIDE a higher-ranked v1 label. Concretely: if v1 emits `bull`
    AND the v2 `recovery` predicate is true, the day stays `bull`
    (bull outranks recovery). If v1 emits `bear`/`sideways`/`transition`/
    `unknown` AND the v2 `recovery` predicate fires, the day becomes
    `recovery`. Implemented in
    `regime_detection.trend_direction_v2.evaluate_v2_trend_label`.
    Resolved by Slice 2.5.

36. **§1C line 147-148 — `rising_vol` rule inequality strictness +
    partial-NaN handling.**
    Spec writes "ATR_ratio > 1.15" and "realized_vol_10d > realized_vol_63d
    * 1.25" — both clauses use strict `>` verbatim, and the combined rule
    uses `OR`. Spec is silent on partial-NaN behavior. Resolution:
    (a) pin both limbs to strict `>` — an `atr_ratio == 1.15` session is
    NOT rising_vol; a `realized_vol_10d == realized_vol_63d * 1.25`
    session is NOT rising_vol;
    (b) pin the cold-start contract: if ANY of the three rule inputs is
    NaN, the rule is False (no silent "partial-input OR → True"
    substitution). This mirrors slice 2.5's recovery cold-start and is
    conservative — a partially-warmed-up session cannot trigger a
    risk-up override. Implemented in
    `regime_detection.volatility_state_v2.evaluate_rising_vol`.
    Resolved by Slice 2.6.

37. **§1C line 148 — `realized_vol` shared helper exposure.**
    Slice 2.2 left `realized_vol` as inline pandas calls in two
    independent sites (`volatility_state.py` v1 compute_features and
    `network_fragility.py` _dispersion_ratio_series). Slice 2.6 needed
    a third site (rising_vol rule inputs) and CLAUDE.md Code-Reuse rule
    has ZERO TOLERANCE for a second system. Resolution: expose the
    shared helper `regime_detection.volatility_state.realized_vol(close,
    window, *, ddof=1)` — annualises via `* sqrt(252)`. The v1
    compute_features path was refactored to consume the helper (byte-
    identical output: same window, same default `ddof`, same
    annualisation constant). The network_fragility dispersion ratio path
    retains its DataFrame-based call (different shape contract — a
    per-symbol matrix) and a future cleanup may unify after v2 §9.1.
    The slice 2.6 RV inputs (`realized_vol_short` window=10,
    `realized_vol_long` window=63) consume the helper. `ddof=1` (sample
    std) is recorded explicitly here because §1C is silent — pandas /
    numpy financial-time-series convention. Resolved by Slice 2.6.

38. **§1C line 157-174 — `vol_crush` deferral re-confirmation.**
    Ambiguity Log entry #20 (slice 2.2) already records `vol_crush` as
    deferred (requires `implied_vol_5d_change` + the §2D event-window
    calendar, neither of which is ingested). Slice 2.6 re-confirms this
    deferral when landing the §1C precedence: the §1C line 191 ranking
    `crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol >
    unknown` carries a reserved `vol_crush` slot in
    `_V2_VOLATILITY_PRECEDENCE` so future authors can land it without
    re-ordering, but the predicate never fires today. Resolved by
    Slice 2.6 (re-confirmation; original deferral by Slice 2.2).

39. **§1C line 191 — precedence-ordering enforcement (volatility).**
    Spec lists `crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
    normal_vol > unknown` but does not explicitly address multi-rule
    fire. Resolution: mirror Slice 2.5's trend-precedence pattern — the
    HIGHEST-ranked label whose rule fires wins, and a fired v2 rule
    cannot OVERRIDE a higher-ranked v1 label. Concretely: if v1 emits
    `crisis_vol` or `high_vol` AND the v2 `rising_vol` predicate fires,
    the day keeps the v1 label (both outrank rising_vol). If v1 emits
    `low_vol` / `normal_vol` / `unknown` AND the predicate fires, the
    day becomes `rising_vol`. Implemented in
    `regime_detection.volatility_state_v2.evaluate_v2_volatility_label`.
    Resolved by Slice 2.6.

40. **§1E lines 276-280 — `liquidity_gap_behavior` deferral.**
    Spec rule requires `gap_frequency_20d percentile_252d > 0.75 AND
    intraday_range_percentile_252d > 0.75`. The intraday-range
    percentile already lives on `volatility_state_v2` (slice 2.2), but
    the 252d percentile of `gap_frequency_20d` is NOT yet computed by
    any feature module — the slice-2.2 compute exposes only the raw
    `gap_frequency_20d` series, not its 252d percentile rank. Per
    v2 §10 absolute rule we do NOT invent the missing input.
    Resolution: defer the `liquidity_gap_behavior` rule until a
    follow-up slice adds the 252d percentile of `gap_frequency_20d`
    to `volatility_state_v2`. The `evaluate_liquidity_gap_behavior`
    predicate in `regime_detection.volume_liquidity_rules`
    short-circuits to `False` today; the function signature already
    accepts the two percentile inputs (carrying NaN today) so a future
    slice can flip the implementation without changing any call site.
    The `VolumeLiquidityLabel` Literal still defines
    `liquidity_gap_behavior` so the spec's full label set is honored at
    the type level. Risk-rank slot 2 from §1E line 291 is reserved for
    the future flip. Deferred by Slice 2.7.

41. **§1E — per-label hysteresis days NOT in spec.**
    The §1E text (lines 251-294) lists labels, rules, and risk_rank but
    is SILENT on per-label de-escalation days. The §3.7 spec for
    network_fragility provides the only worked analogue
    (`correlation_to_one=5`, `correlation_concentration=3`,
    `systemic_stress=5`, `rising_fragility=3` — high-risk labels hold
    multi-day; low-risk labels release immediately). Resolution: pin
    defaults by §1E-risk-rank analogy:
      - `panic_volume = 3` (risk_rank 3, analogous to §3.7
        `correlation_to_one`/`rising_fragility` holds — a single-day
        normal_volume flicker after a panic must not fast-track
        de-escalation).
      - `normal_volume = 0` (risk_rank 0, lowest — immediate
        de-escalation matches §3.7 `diversified_normal` pattern).
      - `unknown = 2` (risk_rank 1, modest hold so a single-day NaN
        flicker through `unknown` does not strand the axis at the
        data-quality fallback — same intent as Implementation
        Ambiguity Log entry #8 for network_fragility, scaled down to
        match `unknown`'s lower §1E risk_rank).
      - `liquidity_gap_behavior = 2` (risk_rank 2, deferred — pinned
        so the future slice that flips the rule needs no config edit).
    All four defaults live on `VolumeLiquidityConfig` in
    `regime_detection.config` and in `configs/core3-v2.0.0.yaml`. The
    v2 §9.1 calibration may retune via yaml.
    Resolved by Slice 2.7.

42. **§1E line 273 — `return_1d` source.**
    Spec text references `return_1d` without naming its source. The V1
    `regime_detection.volatility_state.compute_features` already
    computes `return_1d = close / close.shift(1) - 1` and exposes it
    on `VolatilityFeatures.return_1d`. Resolution: the v2 §1E volume/
    liquidity classifier consumes that V1 series rather than
    recomputing — one source of truth per AGENTS rule B. Wired in
    `regime_detection.axis_series.VolumeLiquidityStateSeriesClassifier`.
    Resolved by Slice 2.7.

43. **§4.1–§4.3 — Layer 4 V2 transition score is blocked: no
    spec-defined weighting exists for the subset of components currently
    available.**
    v2 §4.1 composes the continuous `transition_score` from six
    components and §4.3 publishes weights for exactly two cases:
    "With HMM" (all six, weights sum to 1.0) and "Without HMM"
    (the five non-HMM components, weights sum to 1.0). §8 line 1595
    permits shipping Layer 4 "without HMM using the renormalized
    weights", which refers to the §4.3 "Without HMM" row verbatim —
    not to ad-hoc renormalization over an arbitrary subset.

    Component availability audit performed at the start of Slice 3
    (HEAD `f53760c`):

    - `volatility_acceleration_score` (§4.2 line 1238): AVAILABLE.
      `realized_vol(close, window)` exposed by
      `regime_detection.volatility_state` since Slice 2.6 (entry #37).
    - `breadth_deterioration_score` (§4.2 line 1244): BLOCKED.
      Requires `pct_above_50dma`, a point-in-time (PIT) constituent
      feature. v1 `regime_detection.breadth_state` uses an
      RSP/SPY ETF-proxy and does not expose `pct_above_50dma`; v2
      `regime_detection.breadth_state_v2` (Slice 2.3) explicitly
      defers all PIT pct_above_*dma features per entry #21 and v2
      §1D lines 198–205 ("V2 PIT breadth must not silently fall back
      to biased current constituents").
    - `correlation_concentration_score` (§4.2 line 1249): AVAILABLE.
      `avg_pairwise_corr_percentile_504d` exposed on
      `FeatureStore.network_fragility` since Slice 1.2.
    - `trend_break_score` (§4.2 line 1255): AVAILABLE.
      `drawdown_252d` exposed by
      `regime_detection.trend_direction_v2` since Slice 2.1.
    - `macro_event_score` (§4.2 line 1260): AVAILABLE.
      `regime_detection.event_calendar.classify_event_calendar`
      already emits the spec-named labels `fed_week`, `cpi_week`,
      and `nfp_week`.
    - `hmm_probability_shift_score` (§4.2 line 1265): BLOCKED.
      HMM module per v2 §6.1 is unscoped; v2 §8 places HMM at
      slice 6, after Layer 4.

    Two components are BLOCKED (`breadth_deterioration` and
    `hmm_probability_shift`). The §4.3 weight tables do not enumerate
    a "Without HMM AND Without breadth_deterioration" row. Per v2 §10
    ABSOLUTE RULE (line 1244, "when the spec is ambiguous or silent,
    stop and ask; do not invent"), and per the V2 Slice Promotion
    Checklist §1 ("no formulas, thresholds, or precedence invented —
    v2 spec §10: 'do not invent component score formulas — use the
    exact formulas in §4.2'; same rule for §3.5, §2A/§2B/§2C, etc."),
    Slice 3 is blocked: renormalizing the four available weights
    (`volatility_acceleration`, `correlation_concentration`,
    `trend_break`, `macro_event`) to sum to 1.0 would be a spec
    invention.

    Resolution (Slice 3): defer Layer 4 V2 transition score until either
    (a) PIT constituent membership ingestion lands (unblocks
    `pct_above_50dma`, then ship the §4.3 "Without HMM" row
    verbatim over the five non-HMM components), or (b) HMM ships
    (entry deferred to v2 §8 slice 6, after which the §4.3 "With
    HMM" row applies if PIT membership has also landed). Until
    then, the v1 `transition_risk` named-warning path remains
    authoritative (per §4.5 "V1 warning labels remain
    authoritative; the score adds gradation").

    Status update (post Slice 8 change-point): both unblocking paths
    have since landed and the entry is fully resolved.

    - PIT constituent membership now ships through the engine
      end-to-end (`market_context.py` accepts
      `pit_constituent_intervals` + `constituent_ohlcv`;
      `breadth_state_v2._compute_pit_features` materialises
      `pct_above_50dma`), unblocking `breadth_deterioration_score`.
    - HMM shipped in Slice 6 (`regime_detection.hmm_state`),
      unblocking `hmm_probability_shift_score`.
    - Change-point shipped in Slice 8
      (`regime_detection.change_point`), adding a 7th component
      and a third weight row.

    `configs/core3-v2.0.0.yaml` now publishes THREE weight tables
    consumed by `regime_detection.transition_score`:

    - `weights_without_hmm` — 5-component fallback (V1 byte-identity
      path when HMM seam returns None or is disabled).
    - `weights_with_hmm` — 5-component-plus-HMM, used when the HMM
      seam is lit.
    - `weights_with_hmm_with_change_point` — 6-component-plus-
      change_point, used when both HMM and change-point seams are
      lit (per Log #66 for the `change_point_score` 7th-component
      addition to §4.2).

    `regime_detection.transition_score.compute_transition_score`
    selects among the three tables based on which seams returned
    non-None evidence on the as-of date (per the per-day PIT-correct
    masking added in commit 19e395d). When neither HMM nor
    change-point is lit, the without-HMM 5-component path runs and
    V1 byte-identity is preserved.

    Resolved by Slices 3 + 6 + 8 + Slice 2.8c (PIT) combined.

44. **§2A lines 882–913 — Layer 2A Monetary/Liquidity V2 axis is blocked:
    spec defines rule predicates but omits the structural scaffolding
    (label set, precedence, risk-rank, hysteresis) and several feature
    formulas.**

    §2A provides:
      - Source contract (lines 887–889): `2y yield = FRED DGS2`,
        `10y yield = FRED DGS10`, `broad_usd_index = FRED DTWEXBGS`.
      - One feature formula (line 896): `yield_change_zscore =
        (yield_change_63d - mean_5y) / std_5y`.
      - Three rule predicates (lines 901–913) referencing five distinct
        z-score inputs (`yield_change_zscore_2y`,
        `yield_change_zscore_10y`, `broad_usd_index_zscore_63d`,
        `yield_change_zscore_21d_2y`, `yield_change_zscore_21d_10y`).

    §2A is SILENT on every other scaffolding element that the §3
    network-fragility template (which the slice prompt directs us to
    mirror) provides explicitly:

    - **Label set (analogous to §3.3).** The three rule predicates name
      `tightening_pressure`, `easing_pressure`, `rate_shock`, but no
      `Literal[...]` set is declared and no `neutral_*` / `unknown`
      fallback label is named. Two of the three rules use OR-logic, so
      both `tightening_pressure` AND `rate_shock` can fire on the same
      session — the spec does not name a tie-breaker label or a
      precedence ordering.
    - **Precedence ordering (analogous to §3.4 / §1A line 132).** Not
      stated. Slice 2.5 / 2.6 precedence-by-rank pattern (Ambiguity
      Log entries #35 and #39) requires a spec-given ordering to
      pin to. None exists for §2A.
    - **Risk-rank table (analogous to §3.6 / §1E line 291).** Not
      stated. Slice 1.4 / 2.7 hysteresis design requires a risk-rank
      input.
    - **Per-label de-escalation days (analogous to §3.7).** Not
      stated. Ambiguity Log entry #41 pinned §1E hysteresis defaults
      by §3.7 analogy, but only after §1E itself defined a complete
      risk-rank table — which §2A lacks.
    - **Missing feature formulas.** §2A gives the yield-z-score formula
      for the 63d window only. The USD-index z-score
      (`broad_usd_index_zscore_63d`) and the two 21d yield z-scores
      consumed by `rate_shock` have NO formula in §2A — neither the
      mean/std window length nor the change-window definition for the
      21d variant is stated. Generalizing the 63d formula
      (5y mean/std on the 63d-change series) to either the USD index
      or the 21d window would be a spec invention.

    Per V2 §10 ABSOLUTE RULE (line 1721 in v2 spec, "When the spec is
    ambiguous or silent, stop and ask; do not invent") and the V2 Slice
    Promotion Checklist §1 ("no formulas, thresholds, or precedence
    invented — same rule for §3.5, §2A/§2B/§2C, etc."), Slice 4 cannot
    ship a Monetary/Liquidity V2 axis classifier. Inventing the label
    set, precedence, risk-rank, hysteresis days, and three of the five
    feature formulas would be six interleaved spec inventions.

    The two §2A features that ARE spec-given as formula
    (`yield_change_zscore_2y` over 63d, `yield_change_zscore_10y` over
    63d, both using the line-896 formula) cannot ship as
    "evidence-only" either, because the only consumers named by the
    spec are the three rule predicates — and the slice-2.4 precedent
    (entry #29) for shipping features-before-classifier requires that
    the feature has a determinate downstream consumer. Without label
    set / precedence / risk-rank / hysteresis, there is no
    `MonetaryPressureSeriesClassifier` to land in a follow-up.

    Resolution: defer Slice 4 (Monetary/Liquidity V2 axis) until §2A
    is amended with:
      (a) an explicit label set (e.g.,
          `Literal[tightening_pressure, easing_pressure, rate_shock,
          neutral_monetary, unknown]` or whatever the author of §2A
          intends);
      (b) a precedence ordering analogous to §3.4;
      (c) a risk-rank table analogous to §3.6;
      (d) per-label de-escalation days analogous to §3.7;
      (e) feature formulas for `broad_usd_index_zscore_63d`,
          `yield_change_zscore_21d_2y`, and `yield_change_zscore_21d_10y`
          — specifically: window length for the change, and window
          length / placement for the mean/std normalizer.

    Until §2A is amended, the v1 `MonetaryPressureOutput` placeholder
    (`label="unknown"`, `evidence={"reason":
    "v2_classifier_not_yet_implemented"}`,
    `data_quality.status="insufficient_history"`) remains on
    `RegimeOutput.structural_causal_state.monetary_pressure`, identical
    to the slice-1-foundation shim. The V1 frozen-replay fixtures
    (which use the separate `RegimeOutputV1Frozen` shim with
    `LabelReasonOutputV1Frozen` for `monetary_pressure`) are
    unaffected.

    Note: the existing `MonetaryPressureV2Config` in
    `regime_detection.config` (lines 417–432) was sketched before this
    audit and references "draft absolute bps thresholds" per the §2A
    line 891 deferral language. Those fields are unused at runtime
    today and are out of scope for this entry — a future
    spec-amendment slice will rewrite the config alongside the new
    §2A scaffolding.

    No code committed for this slice — doc-only Ambiguity Log entry.
    The next data slice (slice 5 = §2B inflation/growth) is blocked
    on GDPNow/Citi Surprise fetcher per the V2 Slice Promotion
    Checklist `docs/v2_slice_gate_checklist.md` row 5; the next
    non-data slice (slice 6 = HMM) is orthogonal to §2A and can
    proceed when chosen.

45. **§2A line 896 — features-only sub-slice (slice 4.1) ships the
    ONE spec-pinned z-score formula.**

    Scope decision following the entry #44 audit: although the full
    §2A axis classifier is blocked (label set, precedence, risk-rank,
    hysteresis days, and three of five feature formulas are missing),
    the ONE feature formula §2A pins verbatim at line 896

    ```python
    yield_change_zscore = (yield_change_63d - mean_5y) / std_5y
    ```

    CAN ship as evidence-only because (a) the source contract for its
    two inputs is also explicit (lines 887–889: `2y yield = FRED DGS2`,
    `10y yield = FRED DGS10`), and (b) the slice-2.4 precedent
    (Ambiguity Log entry #29) establishes that features may ship
    before their downstream axis classifier when the formula and
    inputs are unambiguous — `volume_zscore_20d` (§1E line 256)
    shipped in slice 2.4 and waited for the §1E axis classifier in
    slice 2.7.

    Entry #44's argument against an evidence-only ship rested on the
    claim that "the only consumers named by the spec are the three
    rule predicates — and the slice-2.4 precedent requires that the
    feature has a determinate downstream consumer." On re-read of
    entry #29 the precedent is weaker: it requires that the formula
    and inputs be spec-pinned, NOT that the downstream consumer
    already exist. Slice 2.4 shipped `volume_zscore_20d` four slices
    before its classifier; the same pattern applies here.

    Scope IN (slice 4.1):
      - `yield_change_zscore_2y_63d`  (FRED DGS2; v2 §2A line 896).
      - `yield_change_zscore_10y_63d` (FRED DGS10; v2 §2A line 896).

    Scope OUT (stays deferred per entry #44 and V2 §10 absolute rule):
      - `broad_usd_index_zscore_63d` (formula unspecified).
      - `yield_change_zscore_21d_2y` / `yield_change_zscore_21d_10y`
        (21d-variant formula unspecified — neither change-window nor
        mean/std window length pinned).
      - The §2A label set (`tightening_pressure`, `easing_pressure`,
        `rate_shock`, neutral, unknown).
      - Precedence ordering, risk-rank table, per-label hysteresis days.
      - `MonetaryPressureSeriesClassifier`.
      - Retype of `RegimeOutput.structural_causal_state.monetary_pressure`
        — stays as the V1 `LabelReasonOutput` placeholder.

    Sub-ambiguity resolved by slice 4.1:

    - **Sample vs population std for `std_5y`.** §2A is silent. Pinned
      to `ddof=1` (sample std) — matches the slice-2.4
      `volume_zscore_20d` convention (Ambiguity Log entry #28) and the
      pandas / numpy default. A constant-change window produces
      `std == 0` which is masked to NaN via `std.where(std > 0)`
      (avoids RuntimeWarning-laden `0/0` propagation).
    - **First valid index.** With defaults
      `yield_change_lookback_days=63` and
      `zscore_normalizer_window_days=1260`:
      `yield_change_63d` is NaN for `t < 63` (shift introduces 63 NaN
      at the head), and the 5y rolling normalizer requires
      `min_periods=1260` non-NaN observations on the change series,
      so the first non-NaN z-score lands at
      `t = 63 + 1260 - 1 = 1322`. Pinned by
      `tests/test_monetary_pressure_features.py`.
    - **DGS2 / DGS10 independence.** The two inputs are processed in
      separate compute pipelines (one `_yield_change_zscore` call per
      series) so a NaN in DGS2 cannot leak into the DGS10 z-score and
      vice versa. Pinned by a dedicated unit test.

    File / function location:
    `src/regime_detection/monetary_pressure.py` —
    `compute_monetary_pressure_features(*, dgs2, dgs10, config) -> MonetaryPressureV2Features`.
    Mirrors slice 2.1/2.2/2.3/2.4 shape (typed frozen dataclass +
    pure function). Wired through `FeatureStore.monetary` and
    `build_regime_timeline`.

    The previous draft `MonetaryPressureV2Config` in
    `regime_detection.config` (the unused-at-runtime sketch flagged by
    entry #44 as "out of scope") is DELETED in this slice and
    replaced by `MonetaryPressureV2FeaturesConfig` carrying ONLY the
    two spec-pinned lookback knobs (`yield_change_lookback_days`,
    `zscore_normalizer_window_days`), each with `Field(gt=0)` and
    `extra='forbid'`. The deferred `series_ids` /
    `tightening_threshold_bps` / `easing_threshold_bps` /
    `dxy_threshold_pct` fields are reintroduced (with proper spec
    citations) by the future spec-amendment slice that completes the
    §2A axis.

    Resolved by Slice 4.1.

46. **Spec amendment cycle — §1A line 90, §1A line 98, and §2A
    scaffolding (slice-1 of the spec-amendment work).**

    Three previously-deferred ambiguities were amended directly in the
    spec via the rewrite-existing-lines rule (Path A from the
    spec-amendment audit):

    - **§1A line 90 `followthrough_rate` threshold** (was entry #33,
      `breakout_expansion` deferral). Threshold pinned to `0.60`
      directly in the rule and the definition rewritten to be
      self-contained. Rationale: symmetric with §1D
      `nh_nl_ratio < 0.4` (i.e., `1 - 0.6`); conventional in
      breakout-quality literature (Zweig-style; O'Neil-style screens
      use the same neighborhood). Entry #33 is now resolvable: the
      `breakout_expansion` label is no longer blocked on this
      ambiguity. The remaining blocker for `breakout_expansion` is
      that the rule references an `bollinger_band_width_expanding`
      predicate whose operational definition is still implicit — to
      be pinned in the upcoming `breakout_expansion` label slice.

    - **§1A line 98 `range_bound` "oscillates inside 20d range"** (was
      entry #34). Initial amendment pinned `range_ratio_20d < 0.05`
      (total span of the 20d window). **Revised in the same
      amendment cycle** to `max_midpoint_excursion_20d <= 0.05`
      where the 20d midpoint is `(max + min) / 2` over the window
      and the excursion is `max(|close[i] - midpoint| / midpoint)
      for i in t-19..t`. Rationale for the revision: the literal
      reading of "oscillates inside" is "closes orbit a center,"
      which the midpoint-bound form captures directly; the
      range-ratio form is strictly a total-span condition and is
      ~2× tighter for symmetric oscillations (a symmetric ±5% chop
      around 100 yields range_ratio=0.10 but midpoint_excursion=0.05).
      The other two conjunctions
      (`abs(return_63d) < 0.05`, `ADX_14 < 20`) already filter for
      low directional intensity, so the third clause should encode
      the structural around-a-center property rather than double
      up on strictness. Close-prices only; fully derivable from
      existing inputs. Entry #34 is now resolved: the `range_bound`
      label is unblocked.

    - **§2A monetary scaffolding** (was entry #44 and addressed for
      one formula by entry #45). All five missing scaffolding
      elements pinned in §2A:
      (a) **Three missing feature formulas** — `broad_usd_index_zscore_63d`,
          `yield_change_zscore_21d_2y`, `yield_change_zscore_21d_10y`
          — added as mechanical generalizations of the line-896
          template `(change - mean_5y_of_changes) / std_5y_of_changes`.
          The 5y normalizer's window length (1260 trading days) is
          held constant; only the change-window length (63d vs 21d)
          and the input series (DGS2 / DGS10 / DTWEXBGS) vary. The
          line-1088 formula is also rewritten to be explicit that
          mean/std are computed over the change series, NOT the
          level series (slice 4.1 already implemented it this way;
          the rewrite removes ambiguity for future implementers).
      (b) **Label set** `{tightening_pressure, easing_pressure,
          rate_shock, neutral_monetary, unknown}`. The three rule
          names from §2A lines 1093–1104 are kept verbatim
          (`_pressure` suffix preserved per current spec text); a
          `neutral_monetary` fallback is added (no rule fired) and
          `unknown` for the data-quality gate. Pattern matches §1E
          (3 rules + normal fallback + unknown) and §3.3
          (named labels + unknown gate).
      (c) **Precedence**
          `rate_shock > tightening_pressure > easing_pressure >
          neutral_monetary > unknown`. Pattern matches §3.4. Reasoning
          documented inline: `rate_shock` (21d ±2σ) is a stronger
          signal than `tightening_pressure` (63d ±1.5σ) and must
          outrank when both fire; `tightening_pressure` and
          `easing_pressure` are opposite-sign predicates on the same
          metric and cannot co-fire (their order is for log
          readability).
      (d) **Risk rank**
          `{neutral_monetary: 0, easing_pressure: 1, unknown: 1,
          tightening_pressure: 2, rate_shock: 3}`. Pattern matches
          §3.6 and §1E lines 288–294. The
          `easing_pressure < tightening_pressure` asymmetry follows
          §3.6's "severity-of-defensive-action-required" convention,
          not strict directional symmetry (network-fragility risk-rank
          uses the same asymmetric convention).
      (e) **Per-label asymmetric hysteresis**
          `{rate_shock: 5, tightening_pressure: 3, easing_pressure: 2,
          neutral_monetary: 0, unknown: 2}` with
          `default_deescalation_days: 0`. Pattern matches §3.7
          (5-day hold for high-risk labels, 3-day for medium) and
          §1E (Ambiguity Log entry #41 for the volume axis).

    Spec amendments are confined to existing-line rewrites within §1A
    and §2A (no new sections added). Entries #33, #34, and #44 are
    now structurally resolved at the spec level; the corresponding
    code slices (label implementations for `breakout_expansion` and
    `range_bound`; full §2A axis classifier on top of slice 4.1
    features) can be dispatched as TDD slices without further spec
    blockage.

    Resolved by spec-amendment commit (this doc-only change). The
    downstream code slices that consume these pins ship in
    subsequent commits.

47. **§1A `breakout_expansion` — operational forms for the remaining
    three clauses (clauses 1–3) and `followthrough_rate` windowing
    metadata.**

    Entry #46 resolved clause 4 (the `followthrough_rate >= 0.60`
    threshold pin). Three additional clauses in the `breakout_expansion`
    rule and several pieces of `followthrough_rate` windowing metadata
    were left implicit and are now pinned in §1A:

    - **Clause 1 — `close breaks 20d or 50d range`.** Pinned as
      `breakout_20d OR breakout_50d` where
      `breakout_Nd = close[t] > max(close[t-N..t-1])`. Strict `>`
      (a true break, not a touch); compares against the prior-window
      maximum of *closes* (consistent with the rest of §1A's
      close-only inputs); the spec's word "or" reads as logical OR
      (either window suffices).

    - **Clause 2 — `Bollinger band width expanding`.** Pinned as
      textbook Bollinger Bands (period=20, multiplier=2;
      `bb_width = 4 * std(close[t-19..t], ddof=0)`) with "expanding"
      operationalised as `bb_width_20[t] > bb_width_20[t-5]`. The
      5-session comparison window matches the 5-day post-break hold
      in clause 4, keeping a single coherent timeframe through the
      rule rather than introducing another constant.

    - **Clause 3 — `volume > 20d average`.** Pinned as
      `volume[t] > mean(volume[t-20..t-1])`. Strict `>`; baseline
      excludes `t` so today's volume is genuinely above its prior
      20-session average rather than being self-included.

    - **Clause 4 metadata — `followthrough_rate` windowing.** The
      "trailing window" wording in entry #46 is operationalised as
      a 504-session capped lookback over which the 20 most-recent
      past upside breakouts are collected. "Held above breakout
      level for 5+ trading days" is operationalised as continuous
      hold — every close in `b+1..b+5` strictly above the
      `breakout_level` (= `max(close[b-N..b-1])` for whichever window
      fired at session `b`).

    Direction pin: `breakout_expansion` is upside-only, since
    `followthrough_rate` explicitly references "held above breakout
    level." Downside breakouts would require a separate label (not in
    §1A).

    Cold-start: the strictest warm-up in any V2 label — the rule
    cannot fire until at least 20 prior upside breakouts have occurred
    within the trailing 504-session window. New universes / early
    backtest dates will see this label silent. Recorded inline at §1A.

    Asymmetric-cost framing on the 0.60 threshold (added in this
    amendment): false positives route through the `breakout_specialist`
    cohort (§5.1) and cause active PnL damage in chop; false negatives
    cost only missed opportunity. 0.60 deliberately skews toward
    false-negative bias. The value is a V2 §9.1 walk-forward
    calibration placeholder: tighten to 0.65 if false-positive rate
    exceeds target, loosen to 0.55 if false-negative rate dominates.

    Entry #33 (the original `breakout_expansion` deferral) is now
    fully resolved. The label slice can be dispatched as straight
    TDD without further spec ambiguity.

    Resolved by spec-amendment commit (this doc-only change).

48. **§2B Inflation/Growth — scaffolding + operational pins.**

    Applies the §2A template to §2B. The original §2B spec listed an
    8-label set with precedence and a 7-feature / 7-rule schema, but
    every rule had prose-level predicates (e.g., "yields rising",
    "equities weak", "CPI 6m trend stable or falling") and the spec
    was missing every scaffolding element below the rule block.

    Pinned in §2B:
    - **Risk rank** `{goldilocks: 0, recovery_growth: 0,
      earnings_expansion: 0, unknown: 1, disinflation: 1,
      earnings_contraction: 2, recession_scare: 3, inflation_shock: 3}`.
      Pattern matches §3.6 / §1E / §2A.
    - **Hysteresis** per-label asymmetric days
      `{inflation_shock: 5, recession_scare: 5, earnings_contraction: 3,
      disinflation: 3, goldilocks/recovery_growth/earnings_expansion: 0,
      unknown: 2}`, `default_deescalation_days: 0`. Pattern matches §3.7
      / §2A.
    - **Unknown gate** — staleness-based (`cpi > 60d`, `pmi > 45d`,
      `dgs10 > 5 sessions`) plus `assess_series_input_quality`.
    - **Feature formulas** — operational definitions for
      `cpi_3m_change_pct`, `cpi_6m_change_pct`, `pmi_manufacturing` +
      `pmi_manufacturing_slope_21d`, `commodity_return_63d` (with DBC
      ETF substitute for Bloomberg Commodity Index — bias-warning per
      §1D PIT-source precedent), `treasury_10y_yield_slope_21d`,
      `cyclical_defensive_ratio` + `cyclical_defensive_slope_21d`.
    - **Rule predicate operational forms** — "stable" pinned to
      `|cpi_6m_change_pct[t] - cpi_6m_change_pct[t-21]| <= 0.005`
      (< 50bps drift over 21d); "PMI > 50" disambiguated as
      manufacturing-PMI > 50; "equities rising/weak/falling" pinned
      to `spy_21d_return` thresholds (`>0` / `<-0.05` / `<0`);
      "yields rising/falling" pinned to `dgs10` 21d OLS slope sign;
      `inflation_shock`'s AND/OR grouping resolved as
      `(surprise > +1.5σ) OR (composite shock signature)`.
    - **Deferred features** with documented short-circuit behavior:
      `inflation_surprise_zscore` (BLS consensus-vs-actual feed not
      ingested) and `aggregate_forward_eps_revision_direction_4w`
      (workbook snapshots only, no weekly time series per
      market_data_fetch_plan.md line 88). Both short-circuit to `False`
      until the data feeds land. The `inflation_shock` composite-shock
      limb remains active without the surprise input.
    - **Cross-axis short-circuit** — rules referencing
      `credit_funding.active_label` (§2C) short-circuit their cross-axis
      predicate to `False` until §2C ships, mirroring slice-1.3's
      systemic_stress / credit_funding=None pattern.

    Resolved by spec-amendment commit (this doc-only change). The §2B
    axis classifier can be dispatched as a TDD slice with the cross-axis
    short-circuit in place ahead of §2C.

49. **§2C Credit/Funding — scaffolding + operational pins.**

    Applies the §2A template to §2C. Same pattern as #48: §2C had
    6-label set + precedence + 5-rule schema but missing every
    scaffolding element and every operational form for "rising /
    widening / weak / falling" predicates.

    Pinned in §2C:
    - **Risk rank** `{credit_calm: 0, unknown: 1, spread_widening: 1,
      credit_stress: 2, funding_squeeze: 3, deleveraging: 4}`. The
      `deleveraging: 4` slot is the only V2 axis label with risk-rank
      above 3, reflecting that the rule fires only when five
      cross-axis stress signals coincide (§1C / §2A / §2C / §3) —
      strictly more selective than any single-axis high-risk label.
    - **Hysteresis** `{deleveraging: 5, funding_squeeze: 5,
      credit_stress: 3, spread_widening: 3, credit_calm: 0,
      unknown: 2}`, `default_deescalation_days: 0`. Pattern matches
      §3.7 / §2A / §2B.
    - **Unknown gate** — staleness-based on HYG/LQD/TLT (> 5
      sessions), NFCI (> 14 days = 2× weekly cycle), SOFR/IORB
      missing, or `assess_series_input_quality` failure.
    - **Credit-spread metric — dual source.** §2C ships TWO source
      paths for `hy_spread_proxy_63d` and `ig_spread_proxy_63d`:

      1. **Real ICE BofA OAS (preferred)** — FRED-redistributed
         `BAMLH0A0HYM2` (HY) and `BAMLC0A4CBBB` (BBB IG) under ICE
         license, free at the FRED endpoint. When both series are
         present on `MarketContext.macro_series` (keys `hy_oas` and
         `ig_bbb_oas`), `compute_credit_funding_features` uses them
         directly. Bias-warning row provenance =
         `credit_spread_ice_bofa_oas_fred`. Wired in the same
         commit that closed Log #49.
      2. **Total-return-differential proxy (fallback)** — when either
         OAS series is absent, §2C uses
         `hy_spread_proxy = tlt_total_return_63d - hyg_total_return_63d`
         and `ig_spread_proxy = tlt_total_return_63d - lqd_total_return_63d`
         with sign convention "rising proxy = spread widening." The
         total-return differentials are direction-only (not absolute
         bps), but every §2C rule consumes either `percentile_504d`
         or `slope_21d` of the spread series — both scale-invariant —
         so the proxy survives every rule predicate. Bias-warning row
         provenance = `credit_spread_proxy_total_return_differential`.

      Both paths share the same sign convention ("rising = widening").
      Rule predicates are scale-invariant by design so the swap is
      byte-equivalent at the rule level. The only behavioural
      difference between the two paths is the absolute values on the
      `_63d` series (bps vs total-return-differential units) and the
      bias-warning row provenance string.
    - **Rule predicate operational forms** — "non-rising" =
      `slope_21d <= 0`; "rising over 21d" = `slope_21d > 0`;
      "equities falling" = `spy_21d_return < -0.05`; "risk assets
      falling" = `spy_21d_return < 0`; "SOFR-IORB widening" =
      `sofr_iorb_slope_21d > 0`; "bonds weak or unstable" =
      `tlt_21d_return < 0`; "USD rising" = `broad_usd_index_zscore_21d > 0`;
      "volatility up" = `realized_vol_21d_percentile_252d > 0.75`;
      "avg_pairwise_corr rising (Layer 3)" =
      `avg_pairwise_corr_percentile_504d > 0.75`.
    - **§2A formula reuse** — `broad_usd_index_zscore_21d` is the same
      template as §2A line 1088, with change-window = 21 days instead
      of 63. Both z-scores share the 5y normalizer-on-changes contract.

    Resolved by spec-amendment commit (this doc-only change). The §2C
    axis classifier can be dispatched as a TDD slice with the proxy
    bias warning surfaced through `data_quality.evidence`.

    Status update — vendor upgrade COMPLETE. The "true OAS feeds (ICE
    BofA H0A0 / C0A0) not ingested" caveat in the original entry was
    based on an incorrect assumption that those series required a paid
    Bloomberg / vendor terminal. They are actually free on FRED under
    ICE's redistribution license: `BAMLH0A0HYM2` (HY Master II OAS)
    and `BAMLC0A4CBBB` (BBB Corporate OAS). Both series are now
    wired into `V2_FRED_SERIES` and consumed by
    `compute_credit_funding_features` via the optional `hy_oas` /
    `ig_oas` kwargs. When both real-feed series are present on
    `MarketContext.macro_series`, the function uses them directly on
    the `hy_spread_proxy_63d` / `ig_spread_proxy_63d` columns and
    flips the bias-warning row provenance to
    `credit_spread_ice_bofa_oas_fred`. The TLT-vs-HY/LQD total-return-
    differential proxy remains as the FALLBACK for operators who run
    without a FRED key. Rule predicates are unchanged (scale-invariant
    by construction). The §2C spec text amended to document the dual
    source paths (see §2C "Credit-spread metric — dual source"
    block).

50. **§2D Event Calendar V2 — operational pins + §4.2 score expansion.**

    Pinned in §2D and §4.2:
    - `election_window` default = `[-5, +10]` trading days (matches
      the §2D YAML example at the section's end; overridable per-event
      via `window_days` in the event row).
    - `global_rate_decision` source = operator-maintained YAML for
      BOE / ECB / BOJ scheduled meetings (analogous to V1 FOMC
      pre-2021 pre-fetch path).
    - `budget_week` = manual YAML flag (US has no fixed federal
      budget date; tied to operator-defined fiscal events).
    - `geopolitical_event` = manual YAML flag (war, sanctions,
      terrorism) — already pinned in §2D source text.
    - **§4.2 `macro_event_score` expansion** — set extended from
      `{fed_week, cpi_week, nfp_week}` to also include
      `{budget_week, election_window, global_rate_decision}`.
      Geopolitical events are explicitly excluded from the routine
      score because their impact manifests through cross-axis labels
      (`correlation_to_one`, `deleveraging`, `crisis_vol`) rather
      than scheduled-event scoring; including them would double-count.

    The §4.2 set expansion is the only score-impacting change in this
    amendment — `macro_event_score` will fire more often under V2
    (e.g., on ECB / BOE rate decision weeks that previously scored
    0.0). This raises the transition_score's sensitivity to
    international monetary events correctly; the §4.4 score
    interpretation bands (0.35 / 0.55 / 0.75 thresholds) absorb this
    change without modification.

    Resolved by spec-amendment commit (this doc-only change). §2D
    additions wire into the existing v1 event_calendar infrastructure
    without classifier work.

51. **§4 + §6 small pins — transition-score cleanups + HMM/GMM
    operational forms.**

    Five small but blocking pins resolved in §4 and §6:

    - **§4.2 `drawdown_from_252d_high` naming.** The spec text used
      `drawdown_from_252d_high` while slice 2.1 ships `drawdown_252d`
      in `FeatureStore.trend_direction_v2`. Pinned as the same series
      (an alias). The `trend_break_score` formula stays unchanged
      mathematically; the §4.2 code block now reads `drawdown_252d`
      with an inline comment noting the alias. Resolves the
      naming inconsistency that would have surfaced at code time.
    - **§4.4 score-interpretation boundary strictness.** Original
      text used `0.00 - 0.35` etc. without specifying which band
      owns the boundary. Pinned as half-open intervals — upper
      boundary belongs to the next band: `[0.00, 0.35)` →
      `stable`, `[0.35, 0.55)` → `weakening`, `[0.55, 0.75)` →
      `transition_warning`, `[0.75, 1.00]` → `high transition risk`.
      Also pinned the `score_interpretation` Literal short-name set
      `{"stable", "weakening", "transition_warning", "high"}` to
      match the §4.5 JSON example (which uses `"high"` not
      `"high_transition_risk"`).
    - **§6.1 HMM inputs.** Each input now cites the FeatureStore seam
      it MUST reuse (no recomputation): `realized_vol_21d` from
      slice 2.6 shared helper; `drawdown_63d` operationalised as
      slice-2.1 style with 63-day trailing-peak window;
      `volume_zscore_20d` from slice 2.4;
      `avg_pairwise_corr` from slice 1.2. Removes the risk of a fifth
      duplicate computation path emerging when the HMM module ships.
    - **§6.1 state-to-label mapping discipline.** Pinned as manual
      and config-versioned, mirroring §6.2 K-Means/GMM. Mapping
      artifact pattern (`hmm_state_label_map.yaml`) with `version`,
      `fitted_on`, `fitted_window`, `n_states`, and `mappings: {int_index:
      economic_label}`. The state ↔ label assignment is decided by
      the operator after inspecting fitted state means (typically
      `stress_crash` = lowest mean return + highest mean vol + highest
      mean correlation). Closes the V2 §10 "never auto-label" gap
      that previously applied to §6.2 only but logically applies to
      §6.1 as well.
    - **§6.1 "20% parameter drift" operational form.** Pinned as the
      maximum-across-(state × feature) relative absolute change in
      state-mean parameters, after Hungarian-algorithm permutation
      of new state indices to best match old. Transition probabilities
      and covariances are excluded from the alert metric (they drift
      naturally with refit-window shift); a separate review-flag
      fires on > 30% transition-probability shift but does not block
      deployment. Resolves the previously vague "alert on >20%
      parameter drift" line.
    - **§6.2 cluster count.** Pinned at 8 (matches the
      `gmm_8cluster_v1.0` example in the §6.2 output JSON). GMM
      preferred over K-Means because it provides membership
      probabilities; K-Means is an acceptable fallback for
      convergence-unstable cases.

    Out of scope for this amendment (still requires user decision):
    - §4.6 / §6.3 change-point algorithm choice (BOCPD / PELT /
      CUSUM) — governance question, no implementation rationale
      strong enough to pick without product preference.
    - §5 Strategy Response state→cohort / state→constraint mappings —
      governance.

    Resolved by spec-amendment commit (this doc-only change). HMM
    (slice 6) and GMM (slice 7) can now be dispatched as TDD slices;
    only their manual-mapping artifacts remain a per-fit operator
    deliverable.

52. **§5.5 PRISM — explicit V2.1 deferral.**

    PRISM (the user's signal-engine rule-discovery framework) is not
    yet producing walk-forward-validated rules. §5.5 is preserved in
    the spec for forward-reference (output schema + rule contract)
    but explicitly excluded from the initial V2 ship. V2 §8 slice 10
    is now formally V2.1 work, not V2.

    Operational implication: any classifier output, configuration
    block, or test that touches `prism_overrides_applied` MUST default
    it to the empty list `[]` and emit no warning when PRISM is
    absent. This keeps the V2 output schema stable across the
    PRISM-absent → PRISM-present transition; the future amendment
    will only need to populate the list, not introduce a new field.

    When PRISM is producing validated rules, a follow-on
    spec-amendment slice will re-activate §5.5 with explicit
    integration into §5.1 cohort routing and §5.2 family-constraints
    layers (the integration points are unambiguous because both §5.1
    and §5.2 already define the routing/constraint surface that
    PRISM overrides would modify).

    Resolved by spec-amendment commit (this doc-only change).

53. **§4.6 / §6.3 change-point algorithm + §5.1 cohort routing +
    §5.2 family constraints + §5.3 vol-crush exposure — V2 governance
    pinning.**

    Four product-strategy decisions resolved as V2 ship starters,
    each annotated as a V2 §9.1 walk-forward calibration placeholder.

    - **§4.6 + §6.3 algorithm pinned: BOCPD** (Bayesian Online Change
      Point Detection, Adams & MacKay 2007). Rejected PELT (batch-only,
      defeats streaming) and CUSUM (mean-shift step only, no
      probability output). Hazard-rate hyperparameter default = `1/250`
      (one expected break per trading year; calibration target).
      `ruptures` library implementation pinned for both online streaming
      (V2.1 ship) and the offline pilot. The change-point feature stays
      V2.1 — only the algorithm choice is pinned now.

    - **§5.1 cohort routing pinned: 9 cohorts** (8 specialist +
      `default_neutral` fallback). Precedence (fail-defensive default):
      `crisis > euphoria > bear_stress > tightening > easing > recovery
      > chop_mean_reversion > bull_low_vol > default_neutral`. Each
      cohort's routing rule defined in terms of V2-axis label
      membership (`network_fragility`, `volatility_state`,
      `trend_direction`, `monetary_pressure`, `trend_character`,
      `breadth_state`). Per-cohort `blocked_cohorts` table also pinned.
      All rules and blocks are walk-forward calibration placeholders;
      `euphoria_specialist` is silent until §1A sentiment_score data
      ships (Ambiguity Log #32).

    - **§5.2 family constraints pinned via override-on-default
      inheritance.** The §5.2 example JSON becomes the `default_neutral`
      baseline; each specialist cohort declares only the families it
      overrides. Inheritance pattern keeps the ship surface small
      (one base + per-cohort deltas) and matches the Pydantic
      config-inheritance idiom used throughout V2. All numeric
      thresholds (`max_lookback_days`, `max_holding_days`,
      `max_position_pct`, `min_adx`) are calibration placeholders.
      `easing_specialist` inherits `default_neutral` unchanged at
      V2 ship (no opinionated overrides without empirical evidence).

    - **§5.3 vol-crush exposure pinned: 50% reduction over 5-day
      cooldown.** Soft de-risk rather than hard 100% exit. Rationale:
      `vol_crush` can fire on a single-day vol drop that reverses
      within 1-2 sessions; a 100% exit would whipsaw and lock in
      execution cost; 50% provides meaningful de-risk while preserving
      optionality for label-flip. The 5-day cooldown completes
      normalization if the label persists. Asymmetric-cost framing:
      false-positive (exit when vol re-expands) = active execution +
      opportunity cost; false-negative (stay long when vol stays
      crushed) = passive opportunity cost only; 50% deliberately skews
      toward false-negative bias. `event_vol_longs: "exit_immediately"`
      stays hard exit (no partial reduction) per spec line 2301.

    With this amendment all §1, §2, §4, §5, §6 spec-blocked items are
    formally resolved at the spec level. The remaining open V2 work is
    code (the unblocked code slices) plus data sourcing (sentiment_score,
    options IV, weekly EPS revisions, true PIT vendor data, IEF/BIL).

    Resolved by spec-amendment commit (this doc-only change).

54. **§1D line 211 — price field for SMA-based PIT breadth features.**
    Spec writes `pct_above_200dma = mean(member.close > member.sma_200)`
    using the literal field `close`. For the 762-stock PIT universe,
    splits are frequent and raw `close` against an SMA of raw `close`
    false-crosses on the split day even when the economic trend has
    not changed. V1 used raw `close` for SPY safely because SPY rarely
    splits; that condition does not hold here. Per v2 §10 we do NOT
    silently swap fields, so this is a pin, not an invention.
    Resolution: PIT breadth features (`pct_above_50dma`,
    `pct_above_200dma`, the per-stock advance/decline used by
    `ad_line`, `nh_nl_ratio`, `upvol_downvol_ratio`, and `breadth_thrust`)
    read `adjusted_close` from the 762-stock SQLite store
    (`local_daily_ohlcv_sqlite.py` column `adjusted_close`).
    The §1D `close` field name is the *concept* (a price observation per
    stock per day); `adjusted_close` is the operational realization that
    preserves the concept across corporate actions. The `sma_50` /
    `sma_200` reductions are computed off the same `adjusted_close`
    series. The §1D `52-week new highs / new lows` predicate (Ambiguity
    Log #55) also uses `adjusted_close` for the same reason. Pinned in
    `regime_detection.breadth_state_v2` PIT-feature compute.
    Resolved by Slice 2.8c.

55. **§1D lines 218–221 — `nh_nl_ratio` lookback window.**
    Spec writes "52-week new highs / new lows" without naming a
    trading-day count. Calendar 52 weeks ≈ 252 NYSE sessions; using
    a calendar-week window would force calendar-day rolls that V1
    rejected by design (V1 §14 NYSE-only convention). Resolution:
    trailing 252 NYSE sessions inclusive of `as_of_date`, computed
    against `adjusted_close` (Ambiguity Log #54). "New high at D"
    means `adjusted_close[D] == max(adjusted_close[D-251..D])`;
    "new low at D" means `adjusted_close[D] == min(adjusted_close[D-251..D])`.
    Ties resolved by the equality (a ticker at its trailing-max can
    be both a member of the high count and unchanged from a prior
    high). Exposed as `BreadthV2Config.nh_nl_lookback_sessions = 252`
    so §9.1 calibration can retune without code changes. Pinned in
    `regime_detection.breadth_state_v2` PIT-feature compute.
    Resolved by Slice 2.8c.

56. **§1D lines 213–214 + §1D `pct_advancing` — `advances` / `declines`
    on equal-close days.**
    Spec writes `ad_line[t] = ad_line[t-1] + (advances[t] - declines[t])`
    and the breadth-thrust rule references `pct_advancing` but neither
    defines the per-stock predicate operationally. Three options exist:
    (A) strict — advance = `adjusted_close[t] > adjusted_close[t-1]`,
    decline = strict `<`, equality is neither; (B) tie-breaks to
    advance; (C) tie-breaks to decline. Resolution: option (A),
    strict inequality on `adjusted_close` (Ambiguity Log #54).
    Rationale: equality on `adjusted_close` after split/dividend
    adjustment is rare but non-zero (low-priced stocks with
    sub-penny moves rounded by the data vendor); biasing either way
    introduces a small directional drift that compounds in the
    cumulative `ad_line`. Treating equality as a no-event preserves
    `advances + declines + unchanged = N` exactly. Pinned in the
    PIT-feature compute helper `_per_stock_daily_direction`.
    Resolved by Slice 2.8c.

57. **§1D line 213 — `ad_line` t=0 anchor.**
    Spec defines the recurrence `ad_line[t] = ad_line[t-1] +
    (advances[t] - declines[t])` but leaves the t=0 value unspecified.
    `ad_line` is a cumulative integer-valued series; only its slope
    has economic meaning (the level is anchor-relative). Resolution:
    `ad_line[0] = 0` at the first session of the computation window
    (standard convention). Downstream consumers must read
    `ad_line_slope_20d`, not the level. The level is exposed for
    diagnostic inspection only; no rule predicate references it.
    Pinned in `regime_detection.breadth_state_v2._compute_ad_line`.
    Resolved by Slice 2.8c.

58. **§1D line 211 + line 230 — newly-listed members lacking SMA
    history at `as_of_date`.**
    A ticker that joined the PIT universe N < 50 (or N < 200) trading
    days before `as_of_date` has no `sma_50` (or `sma_200`) value at
    `as_of_date`. The pandas expression `close > sma` returns `False`
    when `sma` is NaN — silently biasing `pct_above_50dma` /
    `pct_above_200dma` downward. Per v2 §10 we do NOT silently treat
    NaN as `False`. Resolution: tickers with NaN SMA at `as_of_date`
    are excluded from BOTH the numerator AND the denominator. The
    denominator is `count(member with valid SMA at as_of_date)`, not
    `count(member at as_of_date)`. This mirrors the §1D sector-breadth
    fail-NaN policy (Ambiguity Log #27) at the per-ticker level
    rather than the per-axis level. When the denominator collapses to
    zero (no member has 50 days of history, e.g. first session of
    SQLite coverage), the feature output is NaN — consistent with V1
    cold-start. Same rule applies to per-stock new-52w-high /
    new-52w-low predicates (Ambiguity Log #55): a ticker with fewer
    than 252 sessions of history is excluded from both numerator and
    denominator at `as_of_date`. Pinned in
    `regime_detection.breadth_state_v2` PIT-feature compute.
    Resolved by Slice 2.8c.

59. **§1D — PIT membership semantics for backward-looking
    technical-state computations.**
    Resolving Ambiguity Log #21–#25 requires a pin on the interaction
    between two windows: (a) the PIT universe at `as_of_date` D and
    (b) the per-ticker historical window (50d, 200d, 252d, 20d, 10d)
    each technical indicator pulls. Two interpretations exist:
    (X) restrict each ticker's historical window to only sessions
    during which it was a member; (Y) use each ticker's full continuous
    `adjusted_close` history regardless of past membership status.
    Resolution: option (Y). The PIT universe at D answers the question
    "which 762 stocks count today?"; the technical state at D answers
    "where is this stock today vs its own history?". The latter is a
    property of the stock itself, not of its index-membership timeline.
    Option (X) would force-NaN the SMA of a stock the day after it was
    added to the S&P 500 (no in-membership history), even though the
    stock has a full price history in `daily_ohlcv_rows`. This is the
    standard backtest convention and matches CRSP/Compustat consumer
    practice. The combination of #58 (NaN-SMA exclusion) and #59
    (full-history SMA computation) is what unblocks the §1D PIT
    features without inventing a definition. Pinned in
    `regime_detection.breadth_state_v2` PIT-feature compute and
    asserted by the integration test in slice 2.8d.
    Resolved by Slice 2.8c.

60. **§1D — `pct_above_50dma` 50-session SMA window source.**
    Spec §1D explicitly writes the `pct_above_200dma` formula at lines
    207–210 with a 200-session SMA. The §1D new-breadth-labels
    precedence at lines 239–246 also references `pct_above_50dma`
    (the rule for `narrowing_breadth`: "pct_above_50dma falling AND
    pct_above_200dma falling AND nh_nl_ratio < 0.4"), but the spec
    never restates the `mean(member.close > member.sma_50)` formula
    for the 50-session sibling. Two interpretations: (X) the 50dma
    feature is implicit / parallel to pct_above_200dma with the only
    change being the SMA window; (Y) the 50dma feature is a label
    input only and not itself a feature-store series. Resolution:
    option (X). Rationale: the V1 spec consistently defines pairs of
    SMA-window features (e.g. SMA_50 / SMA_200 both used by trend
    rules) and §1D treats both `pct_above_50dma` and `pct_above_200dma`
    interchangeably in label predicates. Implementing one but not the
    other would force `narrowing_breadth` to short-circuit, which would
    contradict the §1D label-set ship target. Pinned: same formula
    `mean(member.close > member.sma_50)` with 50-session SMA,
    `adjusted_close` price field (Ambiguity Log #54), and the same
    NaN-SMA exclusion (#58) and full-history convention (#59).
    Exposed as `BreadthV2Config.sma_lookback_50 = 50` for §9.1
    calibration retunes.
    Resolved by Slice 2.8c.

61. **§1D lines 218–221 — `nh_nl_ratio` flat-series at trailing
    extremum.**
    Spec writes "52-week new highs / new lows" with the equality
    predicates `adj[D] == max(window)` / `adj[D] == min(window)`
    (Ambiguity Log #55). A ticker whose `adjusted_close` is constant
    across the full 252-session window satisfies BOTH conditions
    simultaneously (the value equals its own max AND its own min).
    Three options: (X) ticker counts toward BOTH `new_highs` AND
    `new_lows`; (Y) ticker counts toward neither (treat flat as a
    non-event); (Z) ticker counts toward `new_highs` only (preference
    rule). Resolution: option (X), implicit in the equality predicate.
    Rationale: option (X) is the only choice that keeps `new_highs`
    and `new_lows` as INDEPENDENT counts of the equality predicate,
    not a coupled either/or category. The downstream
    `ratio = new_highs / max(new_highs + new_lows, 1)` then returns
    0.5 for an all-flat universe, which is the correct neutral
    interpretation. Options (Y) and (Z) would require special-casing
    the flat-series detection (an extra `adj[D] == min == max` check)
    that adds a hidden invariant to the predicate. Pinned in
    `regime_detection.breadth_state_v2._compute_nh_nl_ratio` and
    asserted by `test_nh_nl_ratio_zero_when_no_new_high_or_low` (the
    truly-no-extremum case) plus the structural-counting design of
    the helper.
    Resolved by Slice 2.8c.

62. **§4.6 + §6.3 library correction — `ruptures` does NOT ship online
    BOCPD; substitute `bayesian_changepoint_detection`.**
    Ambiguity Log #53 pinned `ruptures` for both offline pilot and
    online streaming BOCPD per §6.3 line 2871. Audit shows `ruptures`
    ships only OFFLINE batch algorithms (Binseg, PELT, Dynp, Window,
    BottomUp). There is no `ruptures.online` module. The Adams-MacKay
    2007 online BOCPD algorithm is implemented in
    `bayesian_changepoint_detection` (PyPI package, last release
    2023, pure-Python, MIT-licensed) — same algorithm spec cites,
    actual implementation available. Resolution: substitute the
    library. The algorithm choice (BOCPD), the hazard hyperparameter
    default (`1/250`), and the §6.3 output schema (`score`,
    `days_since_last_break`, `method`) remain as pinned in #53; only
    the library binding changes. The `method` string in the output
    schema stays `"BOCPD"` — it identifies the algorithm, not the
    library. Pinned in `regime_detection.change_point` and declared
    as `bayesian-changepoint-detection` in pyproject.toml.
    Resolved by Slice 8.

63. **§6.3 — input observation series for BOCPD.**
    Spec §6.3 line 2864 says "Detect statistical break points in
    returns or volatility series" without naming one. Two options:
    (X) `return_1d` — high-frequency single-day returns, noisier,
    captures sentiment-driven event-day breaks; (Y) `realized_vol_21d`
    — smoother, captures regime-level volatility shifts.
    Resolution: option (Y), `realized_vol_21d`. Rationale: BOCPD's
    StudentT observation likelihood (the canonical Adams-MacKay
    conjugate prior for Gaussian-with-unknown-mean-and-variance)
    handles smoother series with more numerical stability — daily
    returns have heavy tails that violate the Gaussian-emission
    assumption and yield spurious change-points on single-day spikes.
    `realized_vol_21d` is already in the FeatureStore via the slice
    1.x volatility seam — no new compute. The slice-6 HMM uses the
    same series as one of its five inputs, so this is consistent
    with the V2 "evidence-layer regime classifiers share input
    primitives" pattern. Pinned in
    `regime_detection.change_point.compute_change_point_features`.
    Resolved by Slice 8.

64. **§6.3 line 2880 — `score` field formula.**
    Spec output JSON shows `score: 0.78` but no formula. BOCPD emits
    a per-session posterior P(run_length=0 at t) = P(change-point at
    t given data up to t). Three operational choices: (A) raw
    per-session probability; (B) max over a trailing N-day window;
    (C) sum/integral over a trailing window. Resolution: option (B)
    with N=5: `score[t] = max(posterior_changepoint_prob[t-4..t])`.
    Rationale: matches the §4.2 line 2396 `hmm_probability_shift_score`
    5-day-lag convention — both transition-evidence components share
    a 5-NYSE-session memory horizon so they're comparable as
    weighted-sum inputs (even though change_point doesn't yet enter
    the §4.1 composition — that's V2.1 spec-amendment work).
    Pinned in `regime_detection.change_point._rolling_max_changepoint_prob`.
    Window length exposed as `ChangePointConfig.score_window_days = 5`
    so calibration can retune without code changes.
    Resolved by Slice 8.

65. **§6.3 line 2881 — `days_since_last_break` operational definition.**
    Spec leaves "break" undefined. BOCPD's natural threshold question
    is "at what posterior probability do we call a session a break".
    Resolution: a break occurs at session t when
    `posterior_changepoint_prob[t] >= break_threshold` (default 0.5;
    `ChangePointConfig.break_threshold` for calibration tuning).
    `days_since_last_break[t]` = number of NYSE sessions since the
    most recent session that crossed the threshold. When no such
    session exists in the available history (cold-start or genuinely
    quiet period), the value is `None` per V1 §2.7 cold-start NaN
    contract — not zero, not `inf`. Pinned in
    `regime_detection.change_point._days_since_last_break`.
    Resolved by Slice 8.

66. **§4.2 transition_score — `change_point_score` 7th component
    formula + 4-table weight system.**
    Spec §4.6 line 2472 says change-point evidence "Feeds
    transition_score" but §4.2 has no `change_point_score` component
    formula and §4.3 has no weight table for it. Slice 8 shipped the
    evidence layer (`ChangePointOutput.score` per Log #64) without
    the transition_score consumer because the weight table was
    unpinned. Resolution: pin the formula and 4-table weight system.

    Formula:
    ```python
    change_point_score = float(change_point.score)
    # No clip — score is already in [0, 1] by construction
    # (5-session rolling max of a posterior probability per Log #64).
    ```

    Weight tables — 4-table system gated on (HMM seam lit, CP seam lit):

    Without HMM, without change_point (5 components — current):
    ```yaml
    weights_without_hmm:
      volatility_acceleration: 0.225
      breadth_deterioration:   0.225
      correlation_concentration: 0.225
      trend_break:             0.225
      macro_event:             0.10
    ```

    With HMM, without change_point (6 components — current):
    ```yaml
    weights_with_hmm:
      volatility_acceleration: 0.20
      breadth_deterioration:   0.20
      correlation_concentration: 0.20
      trend_break:             0.20
      macro_event:             0.10
      hmm_probability_shift:   0.10
    ```

    Without HMM, with change_point (6 components — NEW):
    ```yaml
    weights_with_change_point:
      volatility_acceleration: 0.20
      breadth_deterioration:   0.20
      correlation_concentration: 0.20
      trend_break:             0.20
      macro_event:             0.10
      change_point:            0.10
    ```

    With HMM, with change_point (7 components — NEW, full V2 evidence):
    ```yaml
    weights_with_hmm_with_change_point:
      volatility_acceleration: 0.175
      breadth_deterioration:   0.175
      correlation_concentration: 0.175
      trend_break:             0.175
      macro_event:             0.10
      hmm_probability_shift:   0.10
      change_point:            0.10
    ```

    Rationale: keep the secondary-evidence components (macro_event,
    hmm_probability_shift, change_point) at the same 0.10 weight to
    reflect parity between calendar-time evidence, latent-state
    evidence, and structural-break evidence. Renormalize the four
    primary deterministic components downward proportionally as
    additional evidence layers come online (4 × 0.225 = 0.90 → 4 ×
    0.20 = 0.80 → 4 × 0.175 = 0.70). All weights are V2 §9.1
    walk-forward calibration placeholders.

    Composer behavior (`compose_transition_score_for_session`):
    select the weight table by inspecting which evidence-component
    inputs were passed:
    - hmm_top_state_prob_now/_5d_ago present + change_point_score
      present → `weights_with_hmm_with_change_point` (7 components)
    - hmm_* present only → `weights_with_hmm` (6 components, current)
    - change_point_score present only → `weights_with_change_point`
      (6 components, NEW)
    - neither → `weights_without_hmm` (5 components, current)

    All four cases compute a normalized score in [0, 1]; the bands
    interpretation (§4.4) applies identically. V1 byte-identity
    preserved by the "neither present → current weights_without_hmm
    path" branch — the same code path that has shipped since slice 3.

    Pinned in `regime_detection.transition_score.compute_transition_score`
    + the yaml weights tables in `core3-v2.0.0.yaml`. Resolved by the
    transition_score change-point wire-in slice.

67. **§1B Trend Character — precedence with the 2 new V2 labels
    (`breakout_expansion`, `range_bound`).**
    §1B introduces two new V2 labels (`breakout_expansion` per Log
    #33/#46/#47, `range_bound` per Log #34/#46) but never restates the
    full §1B precedence ordering with all seven labels. The existing
    V1 precedence is
    `recovery_attempt > trending > chop > transition > unknown` (5
    labels); adding the 2 V2 labels requires a precedence pin.

    Resolution: pin the full V2 §1B precedence as
    `breakout_expansion > recovery_attempt > trending > range_bound >
    chop > transition > unknown`. Rationale:

    - **`breakout_expansion` outranks everything** including
      `recovery_attempt`: a 4-condition strict breakout (close above
      prior 20d/50d high AND BB-width expanding AND volume above 20d
      avg AND followthrough_rate >= 0.60 per Log #47) is a more
      specific signal than any single-axis recovery / trend predicate.
      Per §1B asymmetric-cost framing (Log #47), false positives route
      through `breakout_specialist` cohort (§5.1) and produce active
      PnL damage; outranking ensures the cohort routing is reached.

    - **`recovery_attempt` outranks `trending`** preserves the V1
      ordering (V1 §1B picks recovery_attempt first when both fire;
      keep V1-compat unless a spec amendment says otherwise).

    - **`trending` outranks `range_bound`**: a high-ADX directional
      move that ALSO satisfies range_bound's `abs(return_63d) < 0.05`
      AND `max_midpoint_excursion_20d <= 0.05` would be a degenerate
      input — the predicates are nearly mutually exclusive (ADX >= 20
      VS ADX < 20), but in the corner case where both bands fire (an
      ADX-just-crossing-20 day with a tight close cluster), trending
      wins because the directional intensity signal is stronger
      evidence.

    - **`range_bound` outranks `chop`**: range_bound is a STRICTER
      conjunction than chop (the midpoint-excursion clause adds a
      structural around-a-center constraint that chop's `abs(ret10)`
      and `abs(ret21)` predicates don't capture). When both fire,
      prefer the more specific label.

    - Tail order (`chop > transition > unknown`) is preserved from V1.

    Risk-rank extension:
    `{trending: 0, breakout_expansion: 0, recovery_attempt: 1,
    range_bound: 1, chop: 1, transition: 2, unknown: 2}`.
    Rationale: `breakout_expansion` is a benign / opportunity signal
    (risk-rank 0 alongside trending — both indicate directional flow).
    `range_bound` is risk-rank 1 alongside chop (low-directional-
    intensity states that prefer mean-reversion over trend-following).

    Per-label asymmetric hysteresis defaults (V2 §9.1 calibration
    placeholders, matching the §1B / §3.7 5-day / 3-day / 0-day
    pattern):
    `{breakout_expansion: 3, recovery_attempt: 3, trending: 0,
    range_bound: 3, chop: 0, transition: 2, unknown: 2}`.
    `breakout_expansion` holds 3 days post-event (matches the 5-day
    followthrough_rate definition's coherence window from Log #47).
    `range_bound` holds 3 days to avoid flickering on single-day
    midpoint-excursion spikes within an otherwise-bound regime.

    Pinned in `regime_detection.trend_character` precedence walker
    and the yaml `trend_character` config block. Resolved by the
    §1B V2 character labels slice.

68. **§1D V2 breadth labels — "rising" / "falling" operational
    definitions for `narrowing_breadth` + `broadening_breadth`.**

    Spec lines 279-280 define `narrowing_breadth` and
    `broadening_breadth` predicates in terms of "rising" / "falling"
    rate-of-change qualifiers without naming a window length. The
    `nh_nl_ratio < 0.4` term in the narrowing predicate is strict; the
    "rising"/"falling" terms need pinning.

    Resolution: pin "rising" / "falling" as STRICT 5-session change
    on the underlying feature:

    ```python
    pct_above_50dma_falling  = pct_above_50dma[t]  <  pct_above_50dma[t-5]
    pct_above_200dma_falling = pct_above_200dma[t] <  pct_above_200dma[t-5]
    nh_nl_ratio_rising       = nh_nl_ratio[t]      >  nh_nl_ratio[t-5]
    ```

    `ad_line_slope_20d > 0` (the other conjunct in `broadening_breadth`)
    is already strict by spec.

    Rationale for N=5: matches the §1B Bollinger band-width expanding
    lookback (Log #47) and the §4.2 `hmm_probability_shift_score`
    5-NYSE-session memory horizon. A coherent 5-session memory window
    across all "change over time" V2 predicates keeps the cross-axis
    timeframes aligned and simplifies operator interpretation. NaN
    in either endpoint (`t` or `t-5`) falsifies the rule — V1 §2.7
    cold-start contract.

    Exposed as `BreadthV2Config.label_rate_of_change_lookback_sessions = 5`
    so V2 §9.1 walk-forward calibration can retune. Pinned in
    `regime_detection.breadth_state` V2 rule predicates and the yaml
    `breadth_state.label_rate_of_change_lookback_sessions` config.
    Resolved by the §1D V2 breadth classifier slice.

69. **§1D `breadth_thrust` LABEL — operational predicate pinned.**

    Spec lines 273-275 defined `breadth_thrust` as the "10d MA of
    pct_advancing moves from < 0.40 to > 0.615 within 10 trading
    days" — a multi-session STATEFUL event detector, not a per-day
    predicate. Three candidate operational forms were considered in
    `docs/decisions/0003-breadth-thrust-and-recovery-breadth-
    predicates.md`:

    (X) at session t, fire if EXISTS b in [t-10, t-1] with
        breadth_thrust_feature[b] < 0.40 AND
        breadth_thrust_feature[t] > 0.615;
    (Y) fire if MAX in [t-10, t] > 0.615 AND MIN in [t-10, t]
        < 0.40 (window contains both regimes);
    (Z) fire if MIN in [t-10, t-N] < 0.40 AND
        breadth_thrust_feature[t] > 0.615 for some N pinning the
        "low-then-high" ORDERING.

    Resolution: pin (X). Rationale (per ADR 0003):

    - "moves **from** < 0.40 **to** > 0.615" is directional;
      interpretation (Y) fails the directional reading — it allows
      max-first / min-after which is not a low-to-high move.
    - "within 10 trading days" pins the window precisely;
      interpretation (Z) requires an extra parameter N the spec
      does not provide (V2 §10 "do not invent" violation).
    - (X) introduces no new parameters and exactly maps the literal
      spec text: "the low occurs somewhere in the trailing
      10-session window, the high occurs at session t". It matches
      Zweig's canonical Breadth Thrust formulation the spec cites at
      line 269.
    - (X) is stateless-per-day computable from
      `breadth_thrust_feature[t-10..t]` alone — preserves V1 §2.2
      stateless replay.

    Boundary semantics: both inequalities are strict (`< 0.40` and
    `> 0.615`) per spec text. The thresholds 0.40 and 0.615 are
    spec-fixed (not configurable). NaN at `breadth_thrust_feature[t]`
    or at every `b` in `[t-10, t-1]` falsifies the rule (V1 §2.7
    cold-start). The pinned spec form lives in §1D "Breadth Thrust
    (Zweig-style)" predicate block.

    Resolved by spec-amendment commit (this doc-only change). The
    code-wiring slice in `regime_detection.breadth_state` ships in
    a subsequent TDD commit.

70. **§1D `recovery_breadth` LABEL — operational predicate pinned.**

    Spec line 284 placed `recovery_breadth` in the V2 §1D breadth
    precedence (between `narrowing_breadth` and `broadening_breadth`)
    but never defined its predicate. Two candidate interpretations
    were considered in
    `docs/decisions/0003-breadth-thrust-and-recovery-breadth-
    predicates.md`:

    (X) "Initial recovery" — NH/NL ratio rising (per Log #68 pin)
        AND ad_line_slope_20d not yet strictly positive
        (i.e. breadth strength improving but cumulative AD not yet
        confirming).
    (Y) "Recovery confirmation precursor" — pct_above_50dma rising
        AND pct_above_200dma not yet rising (short-term breadth
        picking up, long-term still lagging).

    Resolution: pin (X). Rationale (per ADR 0003):

    - **Reuses already-pinned features.** (X) operates on
      `nh_nl_ratio` (rising-of pinned in Log #68 — strict 5-session
      change) and `ad_line_slope_20d` — the exact two inputs of
      `broadening_breadth` per spec §1D line 279. (Y) introduces a
      `pct_above_50dma` + `pct_above_200dma` pair that is not used
      by either bracketing label's predicate. Per Log #46's
      spec-amendment pattern, prefer the form that reuses analogues
      from already-pinned semantics over the form that introduces
      new dependencies.
    - **Clean "halfway" semantics.** (X) is exactly
      `broadening_breadth` with one conjunct relaxed: same
      "nh_nl_ratio rising" first clause, plus
      "ad_line_slope_20d <= 0" instead of "> 0". This encodes
      "improvement starting, not yet confirmed by the cumulative
      advance-decline line".
    - **Disjoint from `broadening_breadth` by construction.**
      `recovery_breadth` fires when `ad_line_slope_20d <= 0`;
      `broadening_breadth` fires when `ad_line_slope_20d > 0`. They
      cannot co-fire — no precedence collision. The §1D precedence
      chain (line 284) becomes monotone in slope:
      `narrowing_breadth` (slope falling) → `recovery_breadth`
      (slope ≤ 0 with NH/NL rising) → `broadening_breadth`
      (slope > 0 with NH/NL rising).
    - **Operator-useful early-turn signal.** Recovery sits ABOVE
      broadening in the §1D precedence (line 284) so it surfaces
      the earliest improvement signal rather than waiting for the
      lagging cumulative-AD confirmation.

    Pinned predicate at session t:

    ```text
    recovery_breadth fires at session t when:
      nh_nl_ratio[t] > nh_nl_ratio[t-5]      (rising NH/NL, Log #68)
      AND ad_line_slope_20d[t] <= 0          (not yet broadening)
    ```

    NaN in any of `nh_nl_ratio[t]`, `nh_nl_ratio[t-5]`, or
    `ad_line_slope_20d[t]` falsifies the rule (V1 §2.7 cold-start).
    The 5-session lookback for NH/NL rising-of inherits the
    `BreadthV2Config.label_rate_of_change_lookback_sessions = 5`
    config pinned in Log #68 (operator-tunable via v2 §9.1
    calibration).

    Resolved by spec-amendment commit (this doc-only change). The
    code-wiring slice in `regime_detection.breadth_state` ships in
    a subsequent TDD commit (jointly with the §69 `breadth_thrust`
    predicate).

---

## 2. Layer 2 V2 — Full Structural-Causal State

### 2A. Monetary / Liquidity V2

Monetary pressure was explicitly not implemented in V1. V2 is the first release allowed to implement it, and must lock a clean data contract for 2y yield, 10y yield, and `broad_usd_index` before coding begins.

US V2 source contract:
- `2y yield` = FRED `DGS2`
- `10y yield` = FRED `DGS10`
- `broad_usd_index` = FRED `DTWEXBGS`

V1's draft absolute bps thresholds were deferred because they are rate-era dependent. V2 must adapt to rate era.

#### Rate-Era Recalibration

Each z-score normalizer's *window length* is 5 trading years (1260 days). The *series being normalized* must match the metric's own change-window — the mean and std are computed over a rolling history of that metric's change series, NOT over the level series.

```python
# 63d-change z-scores (used by tightening_pressure / easing_pressure)
yield_change_63d                = yield[t] - yield[t-63]
yield_change_zscore             = (yield_change_63d - mean_5y_of_yield_changes_63d) / std_5y_of_yield_changes_63d

# Applied to DGS2 → yield_change_zscore_2y_63d
# Applied to DGS10 → yield_change_zscore_10y_63d

broad_usd_index_change_63d      = level[t] - level[t-63]
broad_usd_index_zscore_63d      = (broad_usd_index_change_63d - mean_5y_of_level_changes_63d) / std_5y_of_level_changes_63d

# 21d-change z-scores (used by rate_shock)
yield_change_21d                = yield[t] - yield[t-21]
yield_change_zscore_21d         = (yield_change_21d - mean_5y_of_yield_changes_21d) / std_5y_of_yield_changes_21d

# Applied to DGS2 → yield_change_zscore_21d_2y
# Applied to DGS10 → yield_change_zscore_21d_10y
```

Each formula reuses the same template (`(change - mean_5y_of_changes) / std_5y_of_changes`); only the change-window length (63d vs 21d) and the input series (DGS2 / DGS10 / DTWEXBGS) vary.

Updated rules:
```text
tightening_pressure:
  yield_change_zscore_2y_63d > +1.5
  OR yield_change_zscore_10y_63d > +1.5
  OR broad_usd_index_zscore_63d > +1.5

easing_pressure:
  yield_change_zscore_2y_63d < -1.5
  OR yield_change_zscore_10y_63d < -1.5

rate_shock:
  yield_change_zscore_21d_2y > +2.0
  OR yield_change_zscore_21d_10y > +2.0
```

#### Labels

```text
tightening_pressure
easing_pressure
rate_shock
neutral_monetary
unknown
```

`neutral_monetary` is the fallback when no rule fires. `unknown` is forced by the data-quality gate (cold-start or NaN inputs).

#### Precedence

```text
rate_shock > tightening_pressure > easing_pressure > neutral_monetary > unknown
```

Reasoning: `rate_shock` (21d, ±2.0σ) is a stronger move than `tightening_pressure` (63d, ±1.5σ) and must outrank when both fire. `tightening_pressure` and `easing_pressure` are opposite-sign predicates on the same yield-change z-score and cannot co-fire; their relative order is for log readability only.

#### Risk Rank

```yaml
monetary_pressure_risk_rank:
  neutral_monetary: 0
  easing_pressure: 1
  unknown: 1
  tightening_pressure: 2
  rate_shock: 3
```

Matches the §3.6 / §1E convention: states that do not require defensive treatment are 0; severity rises with rank. The `easing_pressure / tightening_pressure` asymmetry reflects that — for downstream strategy responses — tightening is more constraining than easing.

#### Hysteresis

Per-label asymmetric de-escalation, analogous to §3.7 and §1E:

```yaml
monetary_pressure:
  deescalation_days_by_label:
    rate_shock: 5             # matches §3.7 systemic_stress / correlation_to_one
    tightening_pressure: 3    # matches §3.7 rising_fragility / correlation_concentration
    easing_pressure: 2
    neutral_monetary: 0       # immediate de-escalation
    unknown: 2                # matches slice 2.7 volume-axis unknown hold
  default_deescalation_days: 0
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

#### Features (operational definitions)

```python
# CPI trend — trailing-3m / trailing-6m inflation rates (not annualized; matches BLS convention)
cpi_3m_change_pct = (cpi[t] - cpi[t - 3_months]) / cpi[t - 3_months]
cpi_6m_change_pct = (cpi[t] - cpi[t - 6_months]) / cpi[t - 6_months]

# Inflation surprise — z-score of actual-vs-consensus on BLS release dates.
#   inflation_surprise_zscore = (actual_release - consensus_estimate) / std_of_surprise_history_5y
# DEFERRED: requires consensus-vs-actual feed (not yet ingested). See Ambiguity Log #48
# for the short-circuit behavior used by `inflation_shock` until the feed lands.

# PMI — ISM Manufacturing PMI is the primary signal. ISM Services is a separate
# input only when both are available. "PMI > 50" in rule predicates refers to the
# manufacturing index.
pmi_manufacturing = ism_manufacturing_pmi[t]
pmi_manufacturing_slope_21d = ols_slope(pmi_manufacturing, window=21)

# Aggregate forward EPS revision direction
#   revision_4w = (forward_eps[t] - forward_eps[t - 4_weeks]) / forward_eps[t - 4_weeks]
# DEFERRED: workbook snapshot path does not expose weekly time series (data plan
# line 88). Until weekly EPS data lands, `earnings_expansion` / `earnings_contraction`
# short-circuit to False — see Ambiguity Log #48.

# Commodity returns — DBC ETF substitute for Bloomberg Commodity Index (paid feed
# unavailable). Documented as proxy with bias-warning (same precedent as §1D PIT
# constituent CSV).
commodity_return_63d = (dbc_close[t] / dbc_close[t - 63]) - 1

# Bond yield trend — DGS10 from FRED (slice 4.1 already loads this)
treasury_10y_yield_slope_21d = ols_slope(dgs10, window=21)

# Cyclical vs defensive relative strength — close-price ratio + 21d OLS slope
cyclical_defensive_ratio = (xly_close + xli_close) / (xlp_close + xlu_close)
cyclical_defensive_slope_21d = ols_slope(cyclical_defensive_ratio, window=21)
```

#### Rules (operational definitions)

```text
goldilocks:
  (abs(cpi_6m_change_pct[t] - cpi_6m_change_pct[t-21]) <= 0.005      # "stable" = <50bps drift over 21d
   OR cpi_6m_change_pct 21d slope <= 0)                              # OR "falling"
  AND pmi_manufacturing > 50
  AND spy_21d_return > 0                                             # "equities rising"
  AND credit_funding.active_label == "credit_calm"                   # cross-ref §2C

inflation_shock:
  (inflation_surprise_zscore > +1.5)                                  # "positive AND large"
  OR (commodity_return_63d > 0.15
      AND treasury_10y_yield_slope_21d > 0
      AND spy_21d_return < 0
      AND tlt_21d_return < 0)                                         # "equities AND bonds both weak"

disinflation:
  cpi_6m_change_pct 21d slope < 0
  AND treasury_10y_yield_slope_21d < 0
  AND pmi_manufacturing > 45

recession_scare:
  treasury_10y_yield_slope_21d < 0
  AND cyclical_defensive_slope_21d < 0
  AND credit_funding.active_label in {spread_widening, credit_stress}
  AND spy_21d_return < -0.05                                          # "equities weak"

recovery_growth:
  pmi_manufacturing_slope_21d > 0 AND pmi_manufacturing > 50
  AND cyclical_defensive_slope_21d > 0
  AND credit_funding.active_label == "credit_calm"

earnings_expansion:
  aggregate_forward_eps_revision_direction_4w > +0.02
  # short-circuit to False until weekly EPS revision time series ships

earnings_contraction:
  aggregate_forward_eps_revision_direction_4w < -0.02
  # short-circuit to False until weekly EPS revision time series ships
```

#### Risk Rank

```yaml
inflation_growth_risk_rank:
  goldilocks: 0
  recovery_growth: 0
  earnings_expansion: 0
  unknown: 1
  disinflation: 1
  earnings_contraction: 2
  recession_scare: 3
  inflation_shock: 3
```

Pattern matches §3.6 / §1E / §2A: benign states at 0, mild/unknown at 1, medium severity at 2, high-risk states at 3.

#### Hysteresis

Per-label asymmetric de-escalation, analogous to §3.7 / §2A:

```yaml
inflation_growth:
  deescalation_days_by_label:
    inflation_shock: 5             # high-risk hold
    recession_scare: 5
    earnings_contraction: 3
    disinflation: 3
    goldilocks: 0
    recovery_growth: 0
    earnings_expansion: 0
    unknown: 2                     # match slice 2.7 / §2A unknown hold
  default_deescalation_days: 0
```

#### Unknown Gate

`unknown` is forced when:
- CPI series stale > 60 days (2× monthly release cycle)
- PMI series stale > 45 days (1.5× monthly release cycle)
- DGS10 stale > 5 sessions
- `assess_series_input_quality` fails on any required series

#### Cross-Axis Short-Circuit

Rules referencing `credit_funding.active_label` (`goldilocks`, `recession_scare`, `recovery_growth`) short-circuit the cross-axis predicate to `False` when the §2C axis is unbuilt (slice-4 deferral). Precedence walker then falls through to the next-rank rule. Mirrors slice 1.3's systemic_stress / credit_funding=None pattern (Ambiguity Log #1.3 inline TODO).

`earnings_expansion` / `earnings_contraction` short-circuit to `False` until the weekly aggregate forward EPS revision direction time series ships (currently snapshot-only per market_data_fetch_plan.md line 88).

`inflation_shock`'s single-signal limb (`inflation_surprise_zscore > +1.5`) short-circuits to `False` until the BLS consensus-vs-actual feed is ingested. The composite-shock limb remains active.

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

#### Features (operational definitions)

True OAS feeds (ICE BofA H0A0 / C0A0) are not ingested. §2C uses
total-return-differential proxies on the available ETFs (HYG, LQD, TLT)
and the FRED short-rate series (SOFR, IORB, NFCI). Documented as proxy,
not as absolute spread level — the §2C rules only consume percentile and
slope of these series, both of which are scale-invariant, so the proxy
survives every rule predicate. Vendor upgrade noted as a TODO in code.

```python
# Total-return series for HY, IG, Treasury proxies (close-to-close cumulative)
#   sign convention: rising proxy = spread widening (Treasury outperforming HY/IG)
hy_spread_proxy_63d = tlt_total_return_63d - hyg_total_return_63d
ig_spread_proxy_63d = tlt_total_return_63d - lqd_total_return_63d

# Percentile rank (504d window, mirrors §3.2 / §1E percentile convention)
hy_spread_proxy_percentile_504d = rolling(hy_spread_proxy_63d, window=504).rank(pct=True)

# Slope (21d OLS, mirrors §2A / §2B slope convention)
hy_spread_proxy_slope_21d = ols_slope(hy_spread_proxy_63d, window=21)
ig_spread_proxy_slope_21d = ols_slope(ig_spread_proxy_63d, window=21)

# Bank index relative strength
kre_spy_ratio       = kre_close / spy_close
kre_spy_slope_63d   = ols_slope(kre_spy_ratio, window=63)

# Chicago Fed NFCI — weekly release; carry forward to daily via last-known-value
nfci_weekly_carried = forward_fill(nfci_weekly, to_daily=True)

# Broad dollar index — reuses §2A z-score; explicit 21d-change variant
broad_usd_index_zscore_21d = (
    broad_usd_index_change_21d - mean_5y_of_level_changes_21d
) / std_5y_of_level_changes_21d
# (Same template as §2A line 1088, change-window = 21 days instead of 63.)

# Short-rate funding stress
sofr_iorb_spread       = sofr - iorb              # both FRED series, done-live-verified
sofr_iorb_slope_21d    = ols_slope(sofr_iorb_spread, window=21)
```

#### Rules (operational definitions)

```text
credit_calm:
  hy_spread_proxy_percentile_504d < 0.50
  AND hy_spread_proxy_slope_21d <= 0                # "non-rising" = non-positive slope

spread_widening:
  hy_spread_proxy_slope_21d > 0
  AND ig_spread_proxy_slope_21d > 0                 # strict rising on BOTH HY and IG

credit_stress:
  hy_spread_proxy_percentile_504d > 0.80
  AND spy_21d_return < -0.05                        # "equities falling" = >5% drop over 21d

funding_squeeze:
  broad_usd_index_zscore_21d > +1.5                 # reuses §2A formula
  AND sofr_iorb_slope_21d > 0                       # "SOFR-IORB widening" = strictly positive
  AND spy_21d_return < 0                            # "risk assets falling"

deleveraging:                                       # 5-condition composite
  spy_21d_return < -0.05                            # equities down
  AND tlt_21d_return < 0                            # bonds weak or unstable
  AND broad_usd_index_zscore_21d > 0                # USD rising
  AND realized_vol_21d_percentile_252d > 0.75       # volatility up
  AND avg_pairwise_corr_percentile_504d > 0.75      # Layer 3 cross-ref
```

#### Risk Rank

```yaml
credit_funding_risk_rank:
  credit_calm: 0
  unknown: 1
  spread_widening: 1
  credit_stress: 2
  funding_squeeze: 3
  deleveraging: 4         # most severe — multi-system composite collapse signal
```

The `deleveraging: 4` slot is the only V2 axis label with risk-rank above 3 — reflects that the rule fires only when five distinct stress signals coincide across §1C / §2A / §2C / §3, making it strictly more selective than any single-axis high-risk label.

#### Hysteresis

Per-label asymmetric de-escalation, analogous to §3.7 / §2A / §2B:

```yaml
credit_funding:
  deescalation_days_by_label:
    deleveraging: 5            # most severe — long hold
    funding_squeeze: 5
    credit_stress: 3
    spread_widening: 3
    credit_calm: 0
    unknown: 2
  default_deescalation_days: 0
```

#### Unknown Gate

`unknown` is forced when:
- HYG / LQD / TLT stale > 5 sessions
- NFCI stale > 14 days (2× weekly release cycle)
- SOFR or IORB missing
- `assess_series_input_quality` fails on any required series

#### Proxy Bias Warning

The `hy_spread_proxy_63d` / `ig_spread_proxy_63d` are total-return differentials, not yield-curve spreads. They preserve *direction* of spread changes (rising = widening) but **cannot be read as bps-level absolutes**. Slice that consumes them MUST stick to percentile / slope predicates and MUST emit a bias-warning row in any feature-store output (same pattern as §1D PIT-constituent bias warning). A future spec-amendment slice will replace these with direct OAS feeds (ICE BofA H0A0 / C0A0) when vendor sourcing is approved.

---

### 2D. Event Calendar V2

Add labels to V1's calendar:
- `budget_week` — manual YAML flag (relevant for India when extended; US has no fixed federal budget date, so tied to operator-defined fiscal events)
- `election_window` — default trading-day window `[-5, +10]` around the result date (matches the §2D YAML example below); configurable via `window_days` in the event row
- `geopolitical_event` — manual YAML flag (war, sanctions, terrorism)
- `global_rate_decision` — manual YAML for BOE / ECB / BOJ scheduled meetings; operator maintains the calendar (analogous to V1 FOMC pre-2021 pre-fetch path)

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
shannon_entropy = -sum(p_i * ln(p_i) for p_i in p if p_i > 0)   # natural log (base e)
effective_rank = exp(shannon_entropy)
```

`log` here is the natural log (base e); identity correlation matrix → `effective_rank = N`.
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
# `drawdown_from_252d_high` is the same series as slice-2.1's `drawdown_252d`
# in `FeatureStore.trend_direction_v2.drawdown_252d` (per Ambiguity Log #13).
# Values are <= 0; 0 at fresh 252d high, negative below.
distance_from_high = drawdown_252d            # negative (alias retained for spec readability)
score = clip(-distance_from_high / 0.15, 0, 1)  # 0 at top, 1 at -15%
```

`macro_event_score`:
```python
score = 1.0 if event_calendar.label in [
    "fed_week", "cpi_week", "nfp_week",
    # V2 §2D additions:
    "budget_week", "election_window", "global_rate_decision",
] else 0.0
```

`geopolitical_event` is treated separately (high-impact ad-hoc — not part of the routine `macro_event_score`; expected to manifest through `correlation_to_one` / `deleveraging` / `crisis_vol` labels rather than through scheduled-event scoring).

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

**With change_point evidence (Ambiguity Log #66):** when the §4.6 / §6.3
change_point evidence layer is lit alongside the 5 deterministic
components, the composer selects from two additional weight tables:

```yaml
# 6 components — change_point lit, HMM unlit
weights_with_change_point:
  volatility_acceleration: 0.20
  breadth_deterioration:   0.20
  correlation_concentration: 0.20
  trend_break:             0.20
  macro_event:             0.10
  change_point:            0.10

# 7 components — both HMM and change_point lit (full V2 evidence)
weights_with_hmm_with_change_point:
  volatility_acceleration: 0.175
  breadth_deterioration:   0.175
  correlation_concentration: 0.175
  trend_break:             0.175
  macro_event:             0.10
  hmm_probability_shift:   0.10
  change_point:            0.10
```

The composer (`compose_transition_score_for_session`) selects one of
the four weight tables per session by inspecting which evidence
components are non-None and non-NaN. See Ambiguity Log #66 for the
full 4-table gating rationale and the parity-at-0.10 design for
secondary-evidence components.

### 4.4 Score Interpretation

Boundaries are half-open: the upper boundary belongs to the next band. Exactly `0.35` is `weakening` (not `stable`); exactly `0.55` is `transition_warning`; exactly `0.75` is `high transition risk`.

```text
[0.00, 0.35)  →  stable
[0.35, 0.55)  →  weakening
[0.55, 0.75)  →  transition_warning
[0.75, 1.00]  →  high transition risk
```

`score_interpretation` Literal: `{"stable", "weakening", "transition_warning", "high"}` (the JSON example at §4.5 uses `"high"` as the short name for the top band; pin that name to keep the JSON contract consistent).

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

Implementation: **BOCPD (Bayesian Online Change Point Detection, Adams & MacKay 2007)**. Online streaming algorithm matches V2's `classify_window` evaluation pattern; native probability output feeds `transition_score` as evidence rather than as a hard label (per V2 §10 evidence-not-label discipline). See §6.3 for the full method contract.

Output:
```json
{
  "change_point": {
    "score": 0.78,
    "days_since_last_break": 4,
    "method": "BOCPD"
  }
}
```

Feeds `transition_score` as additional evidence (V2.1 — not in initial V2 ship; change-point detection ships in V2.1 alongside §6.3 implementation).

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
- `default_neutral` (fallback when no specialist rule matches)

#### Cohort Precedence (V2 ship starter, V2 §9.1 walk-forward calibration placeholder)

```text
crisis_specialist > euphoria_specialist > bear_stress_specialist
> tightening_specialist > easing_specialist > recovery_specialist
> chop_mean_reversion_specialist > bull_low_vol_specialist > default_neutral
```

Reasoning: defensive cohorts (`crisis_specialist`, `bear_stress_specialist`) outrank optimistic ones (`bull_low_vol_specialist`, `chop_mean_reversion_specialist`) so a bullish trend with a single crisis signal routes to crisis — fail-defensive default. Monetary cohorts outrank generic bull/chop to ensure rate regime drives strategy choice when it's the dominant signal.

#### Routing Rules (V2 ship starter, walk-forward calibration placeholder)

```yaml
cohort_routing:
  crisis_specialist:
    network_fragility.active_label in [correlation_to_one, systemic_stress]
    OR volatility_state.active_label == "crisis_vol"

  euphoria_specialist:
    trend_direction.active_label == "euphoria"
    # Note: euphoria label is deferred until sentiment_score data ships
    # (Ambiguity Log #32). Until then this rule is silent.

  bear_stress_specialist:
    trend_direction.active_label == "bear"
    AND breadth_state.active_label in [weak_breadth, divergent_fragile, narrowing_breadth]

  tightening_specialist:
    monetary_pressure.active_label in [tightening_pressure, rate_shock]

  easing_specialist:
    monetary_pressure.active_label == "easing_pressure"

  recovery_specialist:
    trend_direction.active_label == "recovery"

  chop_mean_reversion_specialist:
    trend_character.active_label == "range_bound"
    AND volatility_state.active_label in [normal_vol, low_vol]

  bull_low_vol_specialist:
    trend_direction.active_label == "bull"
    AND volatility_state.active_label in [low_vol, normal_vol]

  default_neutral:
    # falls through when no rule above matches
```

#### Blocked Cohorts (per active cohort)

The `blocked_cohorts` JSON field at the top of §5.1 is populated by the active cohort's blocklist:

```yaml
blocked_cohorts:
  crisis_specialist:        [short_vol, leveraged_long, breakout]
  euphoria_specialist:      [mean_reversion]    # don't fade strength
  bear_stress_specialist:   [short_vol, breakout, leveraged_long]
  tightening_specialist:    []                  # constraints applied via §5.2 instead
  easing_specialist:        []
  recovery_specialist:      [short_vol]
  chop_mean_reversion_specialist: [trend_following, breakout]
  bull_low_vol_specialist:  []
  default_neutral:          []
```

The starter routing rules + blocked-cohorts table are V2 §9.1 walk-forward calibration placeholders (same pattern as §1A `0.60` threshold). Operator refines after walk-forward evidence reveals false-positive / false-negative rates per cohort.

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

#### Per-Cohort Override Pattern (V2 ship starter, V2 §9.1 calibration placeholder)

The JSON above is the **`default_neutral` cohort's** base constraint set. Each specialist cohort declares only `overrides`; unspecified families inherit the `default_neutral` values. This avoids the combinatorial `N_cohorts × N_families` table and lets walk-forward calibration tune per-cohort deviations without rewriting full constraint sets.

```yaml
strategy_family_constraints:
  default_neutral:                                # baseline, inherited unless overridden
    # uses the §5.2 example JSON above as-is

  crisis_specialist:
    trend_following: {allowed: false, reason: "false_signals_in_chop"}
    mean_reversion:  {allowed: false, reason: "knife_catching"}
    breakout:        {allowed: false, reason: "false_breakout_rate_high"}
    short_vol:       {allowed: false, reason: "vol_can_keep_expanding"}
    long_vol:        {allowed: true,  event_window_only: false}    # always-on long vol

  euphoria_specialist:
    mean_reversion:  {allowed: false, reason: "do_not_fade_strength"}

  bear_stress_specialist:
    trend_following: {allowed: false, reason: "directional_in_stressed_chop"}
    breakout:        {allowed: false, reason: "false_breakout_rate_high"}
    short_vol:       {allowed: false, reason: "stress_can_persist"}
    long_vol:        {allowed: true,  event_window_only: false}

  tightening_specialist:
    trend_following: {allowed: true,  max_lookback_days: 50, require_breadth_confirmation: true}
    breakout:        {allowed: true,  require_volume_confirmation: true}
    long_vol:        {allowed: true,  event_window_only: true}

  easing_specialist:
    # inherits default_neutral (no overrides at V2 ship)

  recovery_specialist:
    short_vol:       {allowed: false, reason: "recovery_can_relapse"}

  chop_mean_reversion_specialist:
    trend_following: {allowed: false, reason: "chop_kills_trend"}
    breakout:        {allowed: false, reason: "false_breakouts_dominant"}
    mean_reversion:  {allowed: true,  max_holding_days: 10, require_volume_confirmation: false}

  bull_low_vol_specialist:
    trend_following: {allowed: true,  max_lookback_days: 200, min_adx: 15}
    mean_reversion:  {allowed: true,  max_holding_days: 10}
    breakout:        {allowed: true}
    short_vol:       {allowed: true,  max_position_pct: 0.25}     # cautious enable
```

The override-on-default inheritance pattern keeps the ship surface small (one base constraint set + per-cohort deltas) and matches Pydantic's config-inheritance idiom that the rest of V2 uses. All thresholds (`max_lookback_days`, `max_holding_days`, `max_position_pct`, `min_adx`) are V2 §9.1 walk-forward calibration placeholders.

### 5.3 Vol-Crush Exit Rules

When `volatility_state.active_label = vol_crush`:
- Exit all event-vol longs immediately
- Reduce long-vol exposure by `long_vol_position_reduction_pct = 0.50` (V2 ship default; V2 §9.1 walk-forward calibration placeholder)
- Normalize risk after `cooldown_days = 5`

```yaml
vol_crush_exit_rules:
  event_vol_longs: "exit_immediately"          # hard exit; no partial reduction
  long_vol_position_reduction_pct: 0.50        # soft 50% de-risk; calibration target
  cooldown_days: 5                             # full normalization horizon
```

Rationale for the soft 50% reduction (not hard 100% exit): `vol_crush` can fire on a single-day vol drop that reverses within 1-2 sessions. A 100% exit would whipsaw exposure and lock in execution cost; 50% provides meaningful de-risk while preserving optionality for label-flip. The 5-day cooldown then completes the normalization if `vol_crush` persists.

Asymmetric-cost framing (same pattern as §1A 0.60 threshold): false-positive (exit when vol re-expands) has active execution cost + opportunity cost; false-negative (stay long when vol stays crushed) has only passive opportunity cost. 0.50 deliberately skews toward false-negative bias.

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

### 5.5 Learned PRISM Rules — DEFERRED TO V2.1

**Status: deferred to V2.1; out of scope for the initial V2 ship.**

PRISM (the user's signal-engine rule-discovery framework) is not yet producing validated rules. §5.5 is preserved in this spec for forward-reference but is explicitly excluded from V2 §8 slice 10 in the initial implementation order.

When PRISM produces walk-forward-validated rules, a future spec-amendment slice will re-activate §5.5 with the contract below, the `prism_overrides_applied` output schema, and explicit integration with the §5.1 cohort routing and §5.2 family-constraint layers.

PRISM rule contract (for the future amendment):
- Walk-forward validated on at least 3 years of data
- Versioned (`prism_rule_id`, `prism_rule_version`)
- Logged for review
- Reversible via single config flag
- Each rule includes: condition, modifier, expected effect, validation metrics

Forward-reference output schema:
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

V2 §8 slice 10 (PRISM rule integration) does not ship in the initial V2 release. Any classifier output, configuration block, or test that references `prism_overrides_applied` must therefore default it to the empty list `[]` and emit no warning when PRISM is absent.

---

## 6. Probabilistic Models

These are **evidence layers**, not final regime labels. Outputs feed into transition_score and other classifiers as additional input. **Never used as standalone label.**

### 6.1 HMM (Hidden Markov Model)

#### Purpose
Infer latent market states from returns and volatility.

#### Inputs
- `return_1d`
- `realized_vol_21d` — same series as `volatility_state.realized_vol(close, window=21)` (slice 2.6 shared helper)
- `drawdown_63d` — analogous to `drawdown_252d` (slice 2.1) but with a 63d trailing-peak window: `close[t] / max(close[t-62..t]) - 1`
- `volume_zscore_20d` — same series as `FeatureStore.volume_liquidity_v2.volume_zscore_20d` (slice 2.4)
- `avg_pairwise_corr` (Layer 3 V2) — `FeatureStore.network_fragility.avg_pairwise_corr` (slice 1.2)

All HMM inputs reuse existing FeatureStore seams. The HMM module MUST NOT recompute any of them.

#### Model
- Gaussian HMM
- 3 states (recommended): `calm_bull`, `choppy_normal`, `stress_crash`
- Optionally 4 states (split bull into trending vs euphoric) once 3-state version validates

#### State-to-Label Mapping (Manual, Config-Versioned)

Same discipline as §6.2 K-Means/GMM: the HMM emits states `0`, `1`, `2` (raw integer indices from `hmmlearn`); these are then manually mapped to economic labels via a versioned config artifact. **Never auto-map.**

Mapping artifact (`hmm_state_label_map.yaml`):
```yaml
hmm_state_label_map:
  version: "1.0"
  fitted_on: "2026-01-15"
  fitted_window: "2020-01-01..2025-12-31"
  n_states: 3
  mappings:
    0: "calm_bull"
    1: "choppy_normal"
    2: "stress_crash"
```

Mapping is decided by the operator after inspecting fitted state means and persistence patterns — typically `stress_crash` is the state with the lowest mean `return_1d` + highest mean `realized_vol_21d` + highest mean `avg_pairwise_corr`. The mapping is reviewed and re-versioned each time the HMM is refit (per quarterly cadence below).

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
- Compare new model parameters to prior version; alert when **state-mean parameter drift** exceeds 20%.

Drift operational definition:

```python
# After aligning new states to old states by closest-mean matching (so
# state index permutations across refits are not counted as drift):
relative_drift_per_state = max(
    abs(new_state_mean[s][i] - old_state_mean[s][i])
    / max(abs(old_state_mean[s][i]), 1e-9)
    for i in range(n_features)
)
parameter_drift = max(relative_drift_per_state for s in range(n_states))
alert_threshold = 0.20
```

The drift metric is the **maximum across (state × feature)** of the relative absolute change in state-mean parameters, after Hungarian-algorithm permutation of new state indices to best match old. State-transition probabilities and covariance parameters are not included in the drift alert (they're typically noisier than means and drift naturally with refit-window shift); a separate review-flag fires when transition-probability shifts exceed 30% but does not block deployment.

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

#### Model
- Algorithm: GMM (Gaussian Mixture Model) preferred over hard K-Means because it provides per-day cluster membership probabilities (useful as evidence). K-Means is an acceptable fallback when GMM convergence is unstable.
- **Number of clusters: 8** (matches the `gmm_8cluster_v1.0` example in the output JSON below; pinned as the V2 ship default).
- Like the HMM (§6.1), cluster index → economic label mapping is **manual and config-versioned**. Never auto-map.

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

#### Method (V2 ship choice)

**BOCPD (Bayesian Online Change Point Detection, Adams & MacKay 2007).** Pinned alongside §4.6. Rationale:
- Online streaming evaluation matches V2's `classify_window` pattern (no batch re-run on every classify call required, unlike PELT)
- Native probability output ("posterior probability that a change-point occurred at session t") satisfies V2 §10's evidence-not-label discipline
- Standard implementation available via the `bayesian_changepoint_detection` PyPI library (corrects an earlier reference to `ruptures` in Ambiguity Log #53; `ruptures` ships only offline algorithms — Binseg, PELT, Dynp, Window, BottomUp — and has no `online` module. See Ambiguity Log #62 for the library substitution. The algorithm choice (BOCPD), hazard rate default, and output schema are unchanged.)
- Hazard-rate hyperparameter is the only tuning knob; V2 ship default = `1/250` (one expected change-point per trading year, calibration placeholder for V2 §9.1)

PELT and CUSUM are rejected for V2 ship: PELT is batch-only (would require re-running on every classify call, defeating streaming); CUSUM lacks the probabilistic output and only detects mean-shift step changes (not variance regime changes).

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
