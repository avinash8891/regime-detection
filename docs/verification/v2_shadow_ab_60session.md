# V2 60-Session Shadow A/B (§9.3)

- Window: 2026-02-12 → 2026-05-08 (60 NYSE sessions)
- Engine version: regime-engine-v2.0.0
- v1-mode errors (sessions): 0
- v2-mode errors (sessions): 0
- Generated: 2026-05-13T12:53:44.830044+00:00

## Gate intent (docs/v2_slice_gate_checklist.md item 7)

Zero unexpected wire diffs in v1 fields; v2 enrichments match
expectations. The two tables below separate the regression-class
(v1 fields — MUST be zero) from the activation-class (v2 fields —
EXPECTED to be non-zero when V2 inputs are wired in).

## v1-field disagreements (v1-mode vs v2-mode)

These fields belong to the V1 wire contract and MUST remain
identical when V2 kwargs are added. Any non-zero count here is a
regression.

| v1 field | disagreement count |
|---|---|
| trend_direction | 0 |
| trend_character | 0 |
| volatility_state | 0 |
| breadth_state | 0 |
| transition_risk_label | 0 |

### trend_direction — most recent disagreement examples

_(none)_

### trend_character — most recent disagreement examples

_(none)_

### volatility_state — most recent disagreement examples

_(none)_

### breadth_state — most recent disagreement examples

_(none)_

### transition_risk_label — most recent disagreement examples

_(none)_

## v2-field activations (expected non-zero deltas)

These fields are NEW in v2 — under v1-mode they are typically
``None``/omitted and under v2-mode they populate when the
corresponding seam is lit. Non-zero counts here are the v2
wins, not regressions.

| v2 field | activation/diff count |
|---|---|
| transition_risk_score | 0 |
| agent_routing | 0 |
| change_point | 0 |
| credit_funding_state | 60 |
| inflation_growth_state | 60 |
| cluster | 0 |
| monetary_pressure_state | 60 |
| volume_liquidity_state | 0 |
| network_fragility | 32 |

### transition_risk_score — most recent activation examples

_(none)_

### agent_routing — most recent activation examples

_(none)_

### change_point — most recent activation examples

_(none)_

### credit_funding_state — most recent activation examples

| session | v1-mode | v2-mode |
|---|---|---|
| 2026-05-04 | `None` | `'unknown'` |
| 2026-05-05 | `None` | `'unknown'` |
| 2026-05-06 | `None` | `'unknown'` |
| 2026-05-07 | `None` | `'unknown'` |
| 2026-05-08 | `None` | `'credit_calm'` |

### inflation_growth_state — most recent activation examples

| session | v1-mode | v2-mode |
|---|---|---|
| 2026-05-04 | `None` | `'unknown'` |
| 2026-05-05 | `None` | `'unknown'` |
| 2026-05-06 | `None` | `'unknown'` |
| 2026-05-07 | `None` | `'unknown'` |
| 2026-05-08 | `None` | `'unknown'` |

### cluster — most recent activation examples

_(none)_

### monetary_pressure_state — most recent activation examples

| session | v1-mode | v2-mode |
|---|---|---|
| 2026-05-04 | `None` | `'unknown'` |
| 2026-05-05 | `None` | `'unknown'` |
| 2026-05-06 | `None` | `'unknown'` |
| 2026-05-07 | `None` | `'unknown'` |
| 2026-05-08 | `None` | `'unknown'` |

### volume_liquidity_state — most recent activation examples

_(none)_

### network_fragility — most recent activation examples

| session | v1-mode | v2-mode |
|---|---|---|
| 2026-05-04 | `'unknown'` | `'correlation_concentration'` |
| 2026-05-05 | `'unknown'` | `'correlation_concentration'` |
| 2026-05-06 | `'unknown'` | `'correlation_concentration'` |
| 2026-05-07 | `'unknown'` | `'correlation_concentration'` |
| 2026-05-08 | `'unknown'` | `'correlation_concentration'` |

