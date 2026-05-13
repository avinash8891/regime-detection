# V2 Walk-forward Performance Gate (§9.1)

- Window: 2025-02-07 → 2026-05-08 (314 NYSE sessions)
- Engine versions: v1=regime-engine-v2.0.0, v2=regime-engine-v2.0.0
- v1-mode errors (sessions): 0
- v2-mode errors (sessions): 0
- Generated: 2026-05-13T13:34:02.585211+00:00

## Wire-level metrics

| metric | v1 | v2 | delta |
|---|---|---|---|
| sessions classified | 314 | 314 | 0 |
| sessions with crisis_override fired | 37 | 37 | 0 |
| sessions with bear_stress fired | 0 | 0 | 0 |
| sessions with bull_fragile fired | 0 | 0 | 0 |
| sessions with recovery_attempt | 0 | 0 | 0 |
| sessions with score_components dict | 0 | 0 | 0 |
| sessions with agent_routing field | 314 | 314 | 0 |
| sessions with change_point.score | 314 | 314 | 0 |
| sessions with hmm evidence on score | 0 | 0 | 0 |
| sessions with credit_funding_state | 0 | 314 | 314 |
| sessions with inflation_growth_state | 0 | 314 | 314 |
| sessions with cluster output | 0 | 0 | 0 |

## §9.1 Gate Conditions (v2 §9.1 + docs/v2_slice_gate_checklist.md item 6)

Per the spec, AT LEAST ONE of:

- LOWER_DRAWDOWN
- HIGHER_SHARPE
- EARLIER_CRISIS_DETECTION
- LOWER_FALSE_SWITCH_RATE

must show v2 improvement. Note: this script ships the per-session wire
comparison; the strategy-PnL metrics (drawdown/sharpe/false-switch) are
operator concerns when v2 outputs route into a backtester (e.g.
vectorbt). The gate currently asserts the wire-level lit-vs-unlit
deltas as a precondition to the strategy gate.

## Per-axis activation rate (v2 mode only)

| axis | sessions lit | activation rate |
|---|---|---|
| network_fragility (non-unknown) | 225 | 71.7% |
| credit_funding (non-unknown) | 235 | 74.8% |
| inflation_growth (non-unknown) | 4 | 1.3% |
| monetary_pressure_v2 (non-unknown) | 0 | 0.0% |
| volume_liquidity_state (non-unknown) | 314 | 100.0% |
| agent_routing != default | 314 | 100.0% |
| change_point >= 0.5 | 0 | 0.0% |

