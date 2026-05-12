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
   (`regime_detection.breadth_state`) does not yet contain that literal.
   Resolution: pin the accepted sets to what V1 can express today —
   `rising_fragility` accepts `{weak_breadth, divergent_fragile}` and
   `systemic_stress` accepts `{weak_breadth}`. Both call sites carry
   `# TODO(v2.1-breadth-enum)` markers in
   `regime_detection.network_fragility_rules` so they can be relinked when the
   enum is extended.
   Resolved by Slice 1.3 (commit `c3badfc`).

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

22. **§1D lines 213–216 — `ad_line` / `ad_line_slope_20d` deferral.**
    Cumulative advance/decline line and its 20d slope both require
    per-stock daily advance/decline counts over the PIT universe (entry
    #21). Resolution: defer the feature and its `broadening_breadth`
    label dependency.
    Deferred by Slice 2.3.

23. **§1D lines 218–221 — `nh_nl_ratio` deferral.**
    52-week new highs / new lows ratio requires per-stock 52w
    high/low tracking across the PIT universe (entry #21). Resolution:
    defer the feature and its `broadening_breadth` / `narrowing_breadth`
    label dependencies.
    Deferred by Slice 2.3.

24. **§1D lines 223–226 — `upvol_downvol_ratio` deferral.**
    Up/Down-volume ratio requires per-stock daily volume × advance/decline
    over the PIT universe (entry #21). Resolution: defer.
    Deferred by Slice 2.3.

25. **§1D lines 231–237 — `breadth_thrust` feature deferral.**
    Zweig-style breadth thrust requires `pct_advancing`, a per-stock
    advance count over the PIT universe (entry #21). Resolution: defer
    the feature; the related `breadth_thrust` LABEL is also deferred
    (entry #26).
    Deferred by Slice 2.3.

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
    (line 129). The V2 repo does not yet ingest any of those feeds.
    Per v2 §10 absolute rule we do NOT synthesize a sentiment proxy.
    Resolution: defer the `euphoria` label until a sentiment ingestion
    slice lands. The §1A line 132 precedence reserves the `euphoria`
    slot above `bull` so the slice that lands sentiment can drop the
    rule in without re-ordering. The precedence-evaluation table in
    `regime_detection.trend_direction_v2._V2_TREND_PRECEDENCE` includes
    `"euphoria"` at index 0 but the rule predicate never fires today.
    Deferred by Slice 2.5.

33. **§1A line 90 — `breakout_expansion` label deferral.**
    Spec rule references a `followthrough_rate` metric configurable
    threshold, but the spec text never defines the metric numerically
    (count over what window? what does "follow-through" mean
    operationally?). Per v2 §10 absolute rule we do NOT invent a
    formula. Resolution: defer the `breakout_expansion` label until
    the spec pins `followthrough_rate` or until the user supplies a
    concrete definition.
    Deferred by Slice 2.5.

34. **§1A line 98 — `range_bound` label deferral.**
    Spec rule writes "price oscillates inside the 20d range" without
    defining "oscillates" operationally (e.g., # of touches against
    the range walls? % of sessions inside the range? Bollinger-style
    band?). Per v2 §10 we do NOT invent a definition. Resolution:
    defer the `range_bound` label until the spec pins the
    oscillation metric.
    Deferred by Slice 2.5.

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

    Resolution: defer Layer 4 V2 transition score until either
    (a) PIT constituent membership ingestion lands (unblocks
    `pct_above_50dma`, then ship the §4.3 "Without HMM" row
    verbatim over the five non-HMM components), or (b) HMM ships
    (entry deferred to v2 §8 slice 6, after which the §4.3 "With
    HMM" row applies if PIT membership has also landed). Until
    then, the v1 `transition_risk` named-warning path remains
    authoritative (per §4.5 "V1 warning labels remain
    authoritative; the score adds gradation").

    No code committed for this slice — doc-only Ambiguity Log entry.

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

```python
yield_change_zscore = (yield_change_63d - mean_5y) / std_5y
```

Updated rules:
```text
tightening_pressure:
  yield_change_zscore_2y > +1.5
  OR yield_change_zscore_10y > +1.5
  OR broad_usd_index_zscore_63d > +1.5

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
- Aggregate forward EPS revision direction (4w change in 12m forward EPS estimate)
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
aggregate_forward_eps_revision_direction_4w > +0.02
```

`earnings_contraction`:
```text
aggregate_forward_eps_revision_direction_4w < -0.02
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
- Broad dollar index (`broad_usd_index`, FRED `DTWEXBGS`)
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
broad_usd_index_zscore_21d > +1.5
AND SOFR-IORB spread widening
AND risk assets falling
```

`deleveraging`:
```text
equities down
AND bonds weak or unstable
AND broad_usd_index rising
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
