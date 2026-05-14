# Decision 0004: V2 §1A `euphoria` LABEL — `sentiment_score` definition, `realized_vol_21d rising` lookback, threshold

**Status:** accepted — picks Q1=A, Q2=X, Q3=+20, Q4=publication-date forward-fill, Q5=4-week cold-start.

Spec owner delegated the picks to the assistant with the explicit instruction "tell me the best answer when spec is not clear, then edit/log the spec then code it." The recommendations in this ADR became the chosen pins; §1A spec text amended and Log #32 closed in the same commit as this status flip.

**Context:**
The V2 §1A `euphoria` label has four rule conjuncts (spec lines 159–165):

```text
euphoria fires when:
  close > SMA_200
  AND return_126d > 0.20
  AND realized_vol_21d rising
  AND sentiment_score >= configured threshold
```

Three of the four are blocked by spec-owner decisions, not by data:

- **`sentiment_score`** — spec line 167 explicitly says the operational definition "is required before implementation. Candidate sources: AAII bull-bear, put-call ratio percentile, Investors Intelligence sentiment." None of the three has been picked.
- **`realized_vol_21d rising`** — undefined lookback. Same family of question Log #68 resolved for breadth (`rising` / `falling` over how many sessions).
- **Threshold value** — spec line 164 says "configured threshold" without naming a default. V2 §9.1 walk-forward calibration is intended to retune; the question is the initial pin.

The AAII sentiment fetcher landed in commit `8c04fae` and exposes the weekly columns `bullish`, `neutral`, `bearish`, `bull_bear_spread`, `bull_bear_spread_8w_ma`. The fetcher's data is *present*; what's missing is the spec-pinned mapping from fetcher columns to `sentiment_score`.

Per V2 §10 ABSOLUTE RULE ("When the spec is ambiguous or silent, stop and ask. Do not invent."), the coding agent must escalate rather than pin. This document presents the candidate interpretations for the three open sub-questions.

Related Log entries: #32 (`euphoria` deferred pending sentiment_score data — fetcher has since landed but operational definition still open).

---

## Question 1 — `sentiment_score` operational definition

The AAII fetcher exposes weekly columns. Spec line 167 names three CANDIDATE sources. Possible operational forms:

| | Form | Notes |
|---|---|---|
| **(A)** | `sentiment_score = bull_bear_spread_8w_ma` | Already computed by the fetcher (`bullish − bearish`, 8-week MA). Smooths week-to-week noise. Range typically [-30, +40] historically; threshold values are direct bull-bear-spread points. |
| (B) | `sentiment_score = bull_bear_spread` (raw, no MA) | Weekly raw spread. Noisier than (A). Matches the spec's literal "AAII bull-bear" phrasing most strictly. |
| (C) | `sentiment_score = percentile_rank(bull_bear_spread_8w_ma, lookback=252w)` | Cross-era normalization — `0.95` means current sentiment is in the top 5% of the past 5y. Robust to regime shifts in the absolute spread distribution. Threshold becomes a percentile (0..1) rather than a points value. |
| (D) | `sentiment_score = bullish_pct` (just the bull share) | Spec mentions "AAII bull-bear" so this loses information; probably out. |
| (E) | Put-call ratio percentile (different source) | Requires CBOE put-call data not yet ingested. Spec lists it as a candidate but defers fetcher work. |
| (F) | Investors Intelligence sentiment (different source) | Requires II newsletter feed not yet ingested. Spec lists it as a candidate but defers fetcher work. |

### Reasoning (assistant's recommendation: **A**)

1. **Already computed by the fetcher**, no new feature code needed.
2. **Smoother than raw spread** — matches the literature's preference for AAII 8-week smoothing (e.g., Stovall, Larkin) for regime signals; the unsmoothed weekly is too noisy for a precedence-bearing label.
3. **Same source class as the spec example** ("AAII bull-bear" — spec line 167) — no source-substitution needed.
4. **Defers (E)/(F) explicitly** — they require new fetchers and don't unblock anything (A) wouldn't already.

If you prefer (C), the cross-era percentile form, the engine work is identical but the threshold becomes a percentile, not a points value (affects Question 3 below).

### Open sub-questions

- **Weekly → daily forward-fill semantics.** AAII is weekly-Thursday-released. For per-NYSE-day classification, the spec is silent on how to align: forward-fill from the last Thursday release? Take Thursday-of-week-containing-`as_of_date`? *Recommendation: forward-fill from the latest publication date `≤ as_of_date` (V1 §2.2 stateless replay — never use future-dated readings).*
- **Cold-start.** First non-NaN observation: with the 8-week MA, the fetcher's `min_periods=1` produces values from week 1. *Recommendation: require at least 4 weeks of history before `sentiment_score` is considered lit; below that, the euphoria rule falsifies on the NaN.*

---

## Question 2 — `realized_vol_21d rising` lookback

Spec text: `AND realized_vol_21d rising` (line 163). No lookback named.

Direct analogue: Log #68 resolved the same form for §1D breadth (`pct_above_50dma falling`, `nh_nl_ratio rising`) by pinning a **strict 5-session change**: `feature[t] > feature[t-5]`.

| | Form | Notes |
|---|---|---|
| **(X)** | `realized_vol_21d[t] > realized_vol_21d[t-5]` (strict 5-session, mirrors Log #68) | Matches the cross-axis 5-session memory horizon pinned for breadth. |
| (Y) | `realized_vol_21d[t] > realized_vol_21d[t-21]` (one-window-length lookback) | Same window as the rolling vol itself; symmetric but doubles the warm-up to ~42 sessions. |
| (Z) | OLS slope on `realized_vol_21d[t-21..t]` strictly positive | Statistical rather than two-point; more robust to single-day noise; slower to fire. |

### Reasoning (assistant's recommendation: **X**)

- **Cross-axis consistency** — Log #68 already pinned 5 sessions for `rising` / `falling` qualifiers on §1D breadth features. Same memory horizon for §1A means operators don't track different lookbacks per axis.
- **No new config key needed** — can reuse the same `label_rate_of_change_lookback_sessions = 5` convention or introduce `TrendDirectionV2RulesConfig.realized_vol_rising_lookback_sessions = 5` for tunability.
- **Statelessness** — direct two-point check, V1 §2.2-compatible.

---

## Question 3 — `euphoria_sentiment_threshold` default

Spec text: `sentiment_score >= configured threshold` (line 164). No default value.

If **Q1=A** (`bull_bear_spread_8w_ma`):
- Historical AAII bull-bear spread 8w-MA distribution (1987-2024) has mean ≈ +6, std ≈ 12. Top-5% threshold ≈ +24, top-10% ≈ +19. Euphoria literature (Yardeni, Stovall) typically cites bull-bear-spread > +20 as "high optimism".
- **Recommendation: `+20` as the V2 §9.1 walk-forward calibration placeholder.** Comment in yaml to retune via §9.1.

If **Q1=C** (cross-era percentile):
- Recommendation: `0.90` (top 10%) as placeholder.

### Reasoning

The threshold is explicitly a **V2 §9.1 walk-forward calibration knob**, not a fixed spec constant. The default value just needs to be defensible from the literature; calibration will adjust. Both `+20` and `0.90` are conventional anchors for "euphoria" in the respective forms.

---

## Cross-cutting impact if (A, X, +20) is accepted

### Spec amendments (would land in a follow-up doc-only commit, per Log #46 pattern)

1. **§1A line 164** — extend the rule predicate with the operational form:
   ```text
   sentiment_score = bull_bear_spread_8w_ma  (8-week MA of AAII bullish - bearish)
                     forward-filled from the latest publication date ≤ as_of_date
                     (V1 §2.2 stateless replay)
   AND sentiment_score >= euphoria_sentiment_threshold  (default +20)
   ```
2. **§1A line 163** — pin the rising-of operational form:
   ```text
   realized_vol_21d rising over 5 sessions:
     realized_vol_21d[t] > realized_vol_21d[t-5]   (strict, matches Log #68 §1D pattern)
   ```
3. **Log entries #32** — close with "Resolved by spec-amendment commit (this doc-only change) + sentiment-wiring code slice."

### Code wiring (separate TDD slice, would follow the spec amendment)

- Plumb `aaii_sentiment_path` (or pre-loaded `aaii_sentiment: pd.DataFrame | None`) through `MarketContext` → `feature_store` → `TrendDirectionV2Features.sentiment_score`.
- Add `EuphoriaConfig(threshold: float = 20.0, vol_rising_lookback: int = 5)` to `TrendDirectionV2RulesConfig`.
- Add `evaluate_euphoria(features, close, sentiment_score, dt, rules_config)` to `trend_direction_v2.py`.
- Activate the `"euphoria"` slot in `_V2_TREND_PRECEDENCE` (currently reserved-but-inert).
- Per-predicate unit tests with boundary cases (sentiment exactly at +20, exactly at +20 + 1e-9, weekly-to-daily forward-fill at NYSE-only gaps).
- V1 byte-identity preserved: euphoria fires only when `sentiment_score` is supplied AND ≥ threshold; otherwise the precedence walker behaves identically to today.
- Side-effect: **item 29 unblocks** — `euphoria_specialist` strategy-family routing (`cohort_routing.py:24`) can now actually be reached.

---

## Decision

**Spec owner action required:**
1. Pick a form for `sentiment_score` (A / B / C / D / E / F — recommend A).
2. Pick a lookback for `realized_vol_21d rising` (X / Y / Z — recommend X).
3. Pick a default for `euphoria_sentiment_threshold` (recommend +20 if Q1=A, 0.90 if Q1=C).

Reply with the chosen pins and I'll land the spec amendment + TDD code wiring as separate commits in that order.
