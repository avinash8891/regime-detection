# ADR 0022 — `correlation_to_one`'s 504d-percentile predicate cannot detect a single-day shock

**Status:** Accepted
**Date:** 2026-06-01
**Resolves:** §9.4 golden-date dispute `2024-08-05:transition_evidence:correlation_to_one`
(the yen-carry unwind). One of the six disputes settled by the 2026-06 golden-set
measurement pass.
**Owner decision:** keep §9.4's `correlation_to_one` expectation for 2024-08-05 in the
self-policing DISPUTED set as a documented spec limitation; do **not** relabel it GREEN
and do **not** widen the rule today.

## Context

Spec §9.4 names **2024-08-05** (the yen-carry unwind) as a date that should exercise
`correlation_to_one`. The engine emits `diversified_normal` for network_fragility on
that session. The 2026-06 measurement pass established — by reproducing the engine's
own rule evidence from the raw fixture with two independent implementations (a blind
recompute and a from-scratch §3.4/§3.5 ladder derivation) — that this is **not an
engine bug**.

The §3.5 `correlation_to_one` predicate is:

```text
avg_pairwise_corr_percentile_504d > 0.90
AND realized_vol_percentile_252d   > 0.80
AND drawdown_21d                   < 0
```

Independently measured on 2024-08-05 over the §3.1 24-ETF universe:

| input | measured | threshold | met? |
|---|---|---|---|
| `avg_pairwise_corr_percentile_504d` | **0.349** | > 0.90 | **no** |
| `realized_vol_percentile_252d` | 1.000 | > 0.80 | yes |
| `drawdown_21d` | −0.084 | < 0 | yes |

The volatility and drawdown limbs fire decisively; only the correlation-percentile
limb fails — and it fails by a wide margin (0.349 vs 0.90).

### Why the predicate structurally cannot fire here

`avg_pairwise_corr_percentile_504d` is the **504-trading-day percentile rank** of a
**63-day rolling** average pairwise correlation. The yen-carry unwind was a violent
**one-day** shock. A single session moves a 63-day-window correlation only marginally
(`avg_pairwise_corr_63d` = 0.324 on the day), and one mildly-elevated reading sits near
the **middle** of its own trailing-504-day distribution, not the top decile. The
methodology is, by construction, a slow-moving structural-regime detector; it cannot
represent an acute single-day correlation spike. No threshold re-tuning of the existing
504d/63d feature fixes this without changing what the feature measures.

This is a genuine **spec-internal tension**: §9.4's scenario expectation outruns what
§3.5's own feature definition can produce. Neither the hand-label nor the engine is
"wrong" against the other — the spec asks a slow feature to catch a fast event.

## Decision

1. **Document the limitation here and keep the dispute visible.** The §9.4 expectation
   for 2024-08-05 (`correlation_to_one`) remains in
   `_VALUE_ASSERT_DISPUTED` (`tests/test_fixture_verification.py`) with a `spec gap
   (ADR 0022)` rationale, so the self-policing oracle continues to assert the gap
   neither silently appears nor silently resolves.
2. **Do not relabel it GREEN.** Forcing a match would require either blessing the
   engine output or asserting a label the spec feature cannot emit — both prohibited
   by the golden-set provenance (`tests/fixtures/derived/golden_dates.yaml`).
3. **Do not widen the rule now.** A fast-window shock-correlation path (e.g. a 5–10
   session correlation spike branch, analogous to the §3.5 cold-start fallback in
   ADR 0021) is the candidate future fix, but it is a feature-design change with its
   own false-positive surface and must be calibrated against the full backtest, not
   bolted on to satisfy one golden date.

The sibling token on the same date, `funding_squeeze`, is a separate dispute and was
**corrected** to `credit_stress` (the yen-carry unwind was USD-flat, not a USD funding
spike: `broad_usd_index_zscore_21d` = −0.017, and the §2C credit_stress predicate fires
independently). See `golden_dates.yaml`.

## Consequences

- No code or config change. The §3.5 `correlation_to_one` rule and the
  `core3-v2.0.0.yaml` thresholds are unchanged.
- `2024-08-05:transition_evidence:correlation_to_one` stays in the DISPUTED set; the
  golden row retains `correlation_to_one` alongside the corrected `credit_stress`.
- If a future fast-window shock path is added, this ADR is the open item it closes; the
  dispute entry would then either go GREEN (the new path fires) or be re-evaluated.
- Scope note: the engine **does** flag this session as stressed through other axes —
  `volatility_state` = crisis_vol, `volume_liquidity_state` = panic_volume,
  `transition_risk` = crisis — so the regime is not silently benign; only the specific
  `correlation_to_one` network-fragility label is unreachable.
