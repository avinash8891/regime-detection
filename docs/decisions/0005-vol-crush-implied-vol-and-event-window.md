# Decision 0005: V2 §1C `vol_crush` — `implied_vol_5d_change` units, `implied_vol_30d` source, `event_window_just_passed`

**Status:** accepted — Q1 = relative 5-session change, Q2 = VIXCLS÷100, Q3 = event-window-end + 3 trailing NYSE sessions, Q4 = 21-session realized vol via the shared helper.

Spec owner delegated the picks under the standing rule "when spec is not clear, tell me the best answer, then edit/log the spec then code it." The recommendations below became the chosen pins; §1C spec text amended and Ambiguity Log #20 closed in the same commit as this status flip.

**Context:**
The V2 §1C `vol_crush` label has a three-conjunct rule (spec lines ~222-240):

```text
vol_crush:
  realized_vol_10d < realized_vol_21d * 0.75
  AND implied_vol_falling_sharply        (implied_vol_5d_change <= -0.20)
  AND event_window_just_passed           (as_of_date within 3 NYSE trading days AFTER configured event end)
```

Two of the three conjuncts were data-blocked and one feature input was undefined. The user-prompted FRED-availability audit established that `implied_vol_30d` is free on FRED (`VIXCLS` — the CBOE VIX, which IS the canonical model-free 30-day implied vol on SPX). That removes the data block; what remains is spec-interpretation, which this ADR pins.

Related Log entries: #19 (IV/RV-spread feature deferral), #20 (`vol_crush` rule deferral — "needs implied_vol_5d_change AND the §2D event-window calendar").

---

## Question 1 — `implied_vol_5d_change <= -0.20`: relative or absolute?

The spec writes the bare threshold `-0.20`. `implied_vol_5d_change` is a 5-NYSE-session change in `implied_vol_30d`.

| | Form | At VIX≈25 | Fireability |
|---|---|---|---|
| **(A)** | Relative: `(iv[t] - iv[t-5]) / iv[t-5] <= -0.20` (a 20% drop) | VIX 25→20 fires | Several times/year — matches the recurring post-event IV-deflation `vol_crush` is meant to catch |
| (B) | Absolute, decimal units: `iv[t] - iv[t-5] <= -0.20` | VIX 35→15 (decimal 0.35→0.15) | Once a year or less — would make `vol_crush` near-unfireable |
| (C) | Absolute, VIX-point units: `iv[t] - iv[t-5] <= -0.20` (0.20 VIX points) | VIX 25→24.8 fires | Constantly — defangs the rule (VIX moves 0.20 points on quiet days) |

### Reasoning (pick: **A**)

1. **Fireability matches intent.** `vol_crush` detects the routine post-FOMC / post-CPI implied-vol collapse. (B) makes it near-unfireable; (C) makes it fire on noise. (A) is the only reading where the rule does what its name says.
2. **Unit-agnostic.** A relative change is a ratio — units cancel. The rule works identically whether `implied_vol_30d` is stored as VIX-points (18.0) or decimal (0.18). That eliminates a unit-mismatch bug class.
3. **Consistent with the codebase's other "change" pins.** §1A `euphoria` uses a strict N-session change on `realized_vol_21d` (ADR 0004); §1D breadth uses strict 5-session change (Log #68). Relative-5-session here keeps the cross-axis "5-session memory" convention.

`implied_vol_5d_change = (implied_vol_30d[t] - implied_vol_30d[t-5]) / implied_vol_30d[t-5]`. NaN at either endpoint falsifies the rule (V1 §2.7 cold-start).

---

## Question 2 — `implied_vol_30d` source + units

**Source: FRED `VIXCLS`** (CBOE VIX Index, daily close). VIX is the CBOE-defined model-free 30-day implied volatility on SPX options — exactly what the spec's `implied_vol_30d` denotes. Free at the FRED endpoint; added to `V2_FRED_SERIES`.

**Units: divide by 100.** VIX is quoted in annualized-vol *points* (e.g. `18.0` = 18%). The codebase's `realized_vol` helper returns *decimal* annualized vol (~`0.16`). The spec's `iv_rv_spread = implied_vol_30d - realized_vol_21d` subtraction requires both operands in the same units, so `implied_vol_30d = VIXCLS / 100` to land in decimal form. (Q1's relative `implied_vol_5d_change` is unit-agnostic and unaffected by this choice; the ÷100 matters only for the `iv_rv_spread` evidence feature.)

---

## Question 3 — `event_window_just_passed`

Spec text: "as_of_date within 3 NYSE trading days AFTER configured event end."

"Configured event end" = the trailing edge of the event's window. The §1D event-calendar already pins per-type windows (`_WINDOWS` in `regime_detection.event_calendar`: `fed_week` ±2, `cpi_week` ±1, `nfp_week` ±1). For an event of type T on date D, the window end is `D + end_offset(T)` in NYSE trading days.

`event_window_just_passed` at session `t` fires iff there EXISTS a calendar event whose window end `E` satisfies `1 <= trading_days_between(E, t) <= 3` — i.e. `t` is one of the 3 NYSE sessions immediately AFTER an event window closed. Strict "after": `t == E` does not fire (that's still inside the window).

Cold-start / no-calendar: when `MarketContext.normalized_event_calendar` is `None` (V1-only callers, or a context built without an event calendar), `event_window_just_passed` is `False` everywhere — `vol_crush` then cannot fire, which is the correct deg, V1-byte-identity-preserving behavior.

---

## Question 4 — `realized_vol_21d`

The `vol_crush` rule's first conjunct needs `realized_vol_10d` and `realized_vol_21d`. The §1C `rising_vol` rule already ships `realized_vol_short` (10d default) and `realized_vol_long` (63d default) on `VolatilityV2Features`. The 10d window is reusable; **the 21d window is new** — add `realized_vol_21d` to `VolatilityV2Features`, computed via the same `regime_detection.volatility_state.realized_vol` helper (log-returns, ddof=1, sqrt(252) annualization — one home per concept).

---

## Cross-cutting impact (if A/B/C/D accepted)

### Spec amendments (§1C, doc-only)

- §1C "Vol Crush" block: pin `implied_vol_5d_change` as the relative-5-session form, `implied_vol_30d` source = `VIXCLS÷100`, `event_window_just_passed` operational definition.
- §1C "IV vs RV Spread" block: drop "Requires options data feed" — VIXCLS is the feed.
- Log #19 + #20: close with status updates.

### Code wiring

- `V2_FRED_SERIES` += `"implied_vol_30d": "VIXCLS"`.
- `MarketContext` += optional `implied_vol_30d: pd.Series | None`. `build_market_context` + `engine.classify`/`classify_window` thread it through (same pattern as the AAII sentiment seam).
- `VolatilityV2Features` += `implied_vol_30d`, `implied_vol_5d_change`, `iv_rv_spread`, `realized_vol_21d`.
- `compute_volatility_v2_features` gains an optional `implied_vol_30d` param; when absent, the three IV-derived features are all-NaN (vol_crush falsifies — V1 byte-identity preserved).
- `VolatilityV2RulesConfig` += `vol_crush_*` thresholds (relative-change threshold −0.20, realized-vol ratio 0.75, event-window-trailing-sessions 3).
- New `event_window_just_passed` compute consuming `context.normalized_event_calendar`.
- `vol_crush` rule predicate + precedence wiring (`crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown`).
- Item #25 (`event_window_just_passed`) ships in the same slice — it has no other consumer.

### Tests

Per-conjunct boundary cases; relative-change exactly at −0.20 (non-strict `<=`); event-window-just-passed at E+1/E+3 (fires) vs E (does not) vs E+4 (does not); no-calendar → False everywhere; full vol_crush integration through `engine.classify`.

---

## Decision

Accepted as above. Spec amendment + code wiring land as the follow-on commits.
