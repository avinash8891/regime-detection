# Decision 0003: V2 §1D `breadth_thrust` and `recovery_breadth` LABEL Predicates

**Status:** accepted — `breadth_thrust` LABEL = (X), `recovery_breadth` LABEL = (X).

Spec owner delegated the picks to the assistant with the standing instruction "tell me the best answer when spec is not clear, then edit/log the spec then code it." The recommendations in this ADR became the chosen pins; §1D spec text amended and Log #69 + Log #70 closed in the same commit as this status flip.

**Context:**
Two V2 §1D breadth LABELs have no operational predicate in the spec. Per V2 §10 ABSOLUTE RULE ("When the spec is ambiguous or silent, stop and ask. Do not invent."), the coding agent must escalate rather than pin. This document presents the candidate interpretations and a recommendation for the spec owner to accept, override, or amend.

The corresponding Ambiguity Log entries are:
- **#69** — `breadth_thrust` LABEL (multi-session stateful event detection undefined)
- **#70** — `recovery_breadth` LABEL (no operational definition in spec)

Both Log entries currently end with "Resolved when spec amends ..." — i.e., explicit "owner must pick" state.

---

## Question 1 — `breadth_thrust` LABEL predicate

### Spec text (V2 §1D lines 269–275)

```text
Breadth Thrust (Zweig-style)
breadth_thrust:
  10d moving average of pct_advancing
  moves from < 0.40 to > 0.615
  within 10 trading days
```

The FEATURE (`breadth_thrust_feature` = 10-session MA of `pct_advancing`) ships from slice 2.8c. Only the LABEL predicate is undefined.

### Candidate interpretations

| | Predicate at session t | Implications |
|---|---|---|
| **(X)** | `EXISTS b ∈ [t-10, t-1] with breadth_thrust_feature[b] < 0.40 AND breadth_thrust_feature[t] > 0.615` | Low first (somewhere in trailing 10), high now. Stateless-per-day. Zero invented parameters. |
| (Y) | `MAX(t-10..t) > 0.615 AND MIN(t-10..t) < 0.40` | Window contains both regimes — no ordering enforced. |
| (Z) | `MIN(t-10..t-N) < 0.40 AND breadth_thrust_feature[t] > 0.615` for some N | Stricter low-then-high ordering with a tunable lookback split N. |

### Reasoning (assistant's recommendation: **X**)

1. **Directionality.** The spec phrase "moves **FROM** < 0.40 **TO** > 0.615" is directional. Interpretation (Y) does not enforce ordering — a peak-then-trough series would qualify. (X) and (Z) enforce low-first / high-now.
2. **Parameter inventory.** (Z) introduces a new parameter `N` that the spec does not provide. Per V2 §10 ("do not invent"), this is the strongest objection. (X) and (Y) use only the constants in the spec text (0.40, 0.615, the 10-trading-day window).
3. **Canonical literature reference.** Spec line 269 cites "Breadth Thrust (Zweig-style)". Zweig's original 1986 *Winning on Wall Street* definition is exactly form (X): "the 10-day MA of the advances-issues ratio rises from below 0.40 to above 0.615 within 10 trading days." Technician-community references (Investopedia, StockCharts, Edwards-Magee) reproduce this form.
4. **Statelessness.** (X) is computable per session from `feature[t-10..t]` alone — preserves V1 §2.2 stateless replay.

### Open questions (please answer)

- **Window inclusivity at the low end** — does "exists b in trailing 10 days" mean `b ∈ [t-10, t-1]` (10 candidate sessions before t) or `b ∈ [t-9, t-1]` (9 sessions, today excluded)? Zweig's literal text supports either; the spec doesn't pin. *Recommendation: `[t-10, t-1]`.*
- **NaN policy** — if `feature[t]` is NaN, the rule trivially falsifies. If every `feature[b]` in the trailing window is NaN (cold-start), what does the rule emit? *Recommendation: falsify the rule (V1 §2.7 cold-start contract — propagate `unknown` via the data-quality gate rather than co-fire).*
- **Threshold tunability** — should 0.40 and 0.615 be configurable for V2 §9.1 walk-forward calibration, or spec-fixed? *Recommendation: spec-fixed (Zweig literature anchor + analogous treatment of §1D `nh_nl_ratio` threshold 0.4 which is spec-fixed).*

---

## Question 2 — `recovery_breadth` LABEL predicate

### Spec text (V2 §1D line 284)

```text
breadth_thrust > divergent_fragile > narrowing_breadth > recovery_breadth > broadening_breadth > weak_breadth > healthy_breadth > neutral_breadth > unknown
```

`recovery_breadth` is named ONLY in the precedence chain. No rule block defines it.

The two bracketing labels (per Log #68 pin):
- `narrowing_breadth` (higher precedence): `pct_above_50dma falling AND pct_above_200dma falling AND nh_nl_ratio < 0.4`
- `broadening_breadth` (lower precedence): `nh_nl_ratio rising AND ad_line_slope_20d > 0`

Semantic role: "improving but not yet fully confirmed", sitting between deterioration and full recovery.

### Candidate interpretations

| | Predicate at session t | Implications |
|---|---|---|
| **(X)** | `nh_nl_ratio[t] > nh_nl_ratio[t-5] AND ad_line_slope_20d[t] <= 0` | Reuses both inputs of `broadening_breadth`; just relaxes the slope conjunct. Disjoint from broadening by construction. |
| (Y) | `pct_above_50dma[t] > pct_above_50dma[t-5] AND pct_above_200dma[t] <= pct_above_200dma[t-5]` | Uses features from `narrowing_breadth`'s predicate inverted — "50dma rising, 200dma not yet rising." Introduces a new feature pairing for this label. |

### Reasoning (assistant's recommendation: **X**)

1. **Feature reuse.** (X) operates on the exact two features that already define `broadening_breadth` (Log #68: `nh_nl_ratio_rising` + spec §1D: `ad_line_slope_20d`). (Y) introduces a `pct_above_50dma` + `pct_above_200dma` pairing that no other label predicate uses. Per Log #46's spec-amendment pattern (reuse pinned analogues over new dependencies), (X) is the lighter pin.
2. **Disjointness with adjacent labels.** (X) fires when `ad_line_slope_20d <= 0`; `broadening_breadth` fires when `> 0`. Strictly disjoint — no precedence collision. The §1D chain becomes monotone in slope:
   - `narrowing_breadth` — slope falling (deterioration)
   - `recovery_breadth` — slope ≤ 0 with NH/NL rising (improving, unconfirmed)
   - `broadening_breadth` — slope > 0 with NH/NL rising (confirmed)
3. **Precedence-ordering rationale.** Recovery sits ABOVE broadening in the §1D precedence (line 284). The operator-useful interpretation: surface the EARLY turning-point signal before the LAGGING cumulative-AD confirmation. (X) directly encodes "early turning point". (Y) is a different concept ("short-term breadth picking up, long-term lagging") that doesn't naturally fit the "between narrowing and broadening" semantics.
4. **Inherited lookback.** (X) uses the same `label_rate_of_change_lookback_sessions = 5` config already pinned by Log #68 for `nh_nl_ratio_rising`. No new config keys needed.

### Open questions (please answer)

- **Slope boundary at zero** — should `ad_line_slope_20d == 0` route to recovery (`<= 0`) or broadening (`>= 0`)? *Recommendation: route to recovery (`<= 0`), so broadening is strictly `> 0` per the existing spec text "ad_line_slope > 0".*
- **NaN policy** — same V1 §2.7 cold-start contract: falsify on NaN in any input. *Recommendation: confirm.*
- **Is (Y) worth considering?** (Y) is internally consistent but uses different inputs. Worth surfacing in case the spec owner's intent was the "50dma vs 200dma" reading.

---

## Cross-cutting impact if (X) is accepted for both

### Spec amendments (would land in a follow-up doc-only commit, per Log #46 pattern)

1. **§1D Breadth Thrust block** — extend the existing pseudo-code block (lines 269–275) to separate FEATURE from LABEL predicate:
   ```text
   Feature:
     breadth_thrust_feature = 10-session MA of pct_advancing

   Label at session t:
     EXISTS b ∈ [t-10, t-1] with breadth_thrust_feature[b] < 0.40
     AND breadth_thrust_feature[t] > 0.615
   ```
2. **§1D new-labels list** (lines 277–280) — add a `recovery_breadth` bullet with the pinned predicate.
3. **Log entries #69 and #70** — close with "Resolved by spec-amendment commit (this doc-only change)" + reasoning.

### Code wiring (separate TDD slice, would follow the spec amendment)

- Add `breadth_thrust` and `recovery_breadth` predicate evaluators in `regime_detection.breadth_state` (V2 rule predicate table — entries already preserved per Log #69/#70 deferral notes).
- Per-predicate unit tests with boundary cases (`feature[t] == 0.615`, `slope == 0`, NaN cold-start, exact-low-day-window-edge).
- Golden-date verification entries in `tests/fixtures/derived/golden_dates_v2.yaml`.
- Update `_V2_BREADTH_PRECEDENCE` and risk-rank tables if entries don't already carry the labels.

---

## Decision

**Spec owner action required:** confirm (X)/(X), pick alternatives, or request additional candidates. Reply on this doc (or in the spec PR) with the chosen pins, and I'll land the spec amendment + code wiring as separate commits.
