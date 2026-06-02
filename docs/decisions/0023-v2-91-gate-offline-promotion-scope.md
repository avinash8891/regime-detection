# 0023 — The V2 §9.1 performance gate is an offline promotion gate, not a runner step

- Status: accepted
- Date: 2026-06-02
- Finding: F-022 (spec_review.md), milestone M3
- Supersedes the prior `comparison.py` module docstring claim that
  `evaluate_v2_gate` is "consumed by the walk-forward runner (`--engine-profile both`)".

## Context

The V2 spec §9.1 requires every V2 component to demonstrate, in walk-forward
backtest, at least one of: lower max drawdown, higher Sharpe, earlier crisis
detection, or lower false-switch rate than V1. `regime_detection.comparison`
implements this as the pure function `evaluate_v2_gate(v1_metrics, v2_metrics)
-> GateResult`, where `GateResult.passed` is True iff V2 beats V1 on ≥1 metric.

F-022 flagged a contradiction: the module docstring asserted the gate was
"consumed by the walk-forward runner (`--engine-profile both`)", but:

1. No runner (`run_historical_walkforward.py`, `run_shadow_regime.py`) imports
   or calls `evaluate_v2_gate`, and no `--engine-profile both` flag exists.
2. The gate's inputs are `StrategyMetrics` — `max_drawdown`, `sharpe`,
   `mean_crisis_detection_lag_days`, `false_switch_rate` — which are **whole-
   backtest strategy-performance aggregates**. The per-session regime engine
   never computes them; they originate in the downstream strategy-eval layer
   (the F-014 ledger producer). The walk-forward / shadow runners classify
   regimes only; they have no strategy P&L to summarize.

So the gate **cannot** be a per-session runner step without inventing a strategy
backtest inside the classification runner — which would conflate the regime
engine with the strategy layer it is deliberately decoupled from
(`StrategyMetrics` docstring: "The engine itself never computes these").

## Decision

`evaluate_v2_gate` is an **offline promotion gate**, invoked by the offline
strategy-eval / promotion harness that consumes the strategy ledger, not by the
per-session walk-forward or shadow runner. A `GateResult.passed == False` means
"do not promote the V2 candidate". The runners' own promotion control remains the
§6 walk-forward replay gate and the shadow qualification window; the §9.1 gate is
a separate, later, offline check against realized strategy metrics.

We therefore corrected the false "consumed by the walk-forward runner" claim in
`comparison.py` rather than wiring a gate call the runner has no data to feed.
This pairs with F-014: when the reproducible strategy-metrics producer lands, it
is the component that calls `evaluate_v2_gate`.

## Consequences

- The gate's block/allow semantics are tested as a pure-function contract
  (`tests/test_v2_comparison.py`: passes on any single-metric win; fails on
  identical or uniformly-worse metrics). No runner wiring test is needed because
  there is, by decision, no runner wiring.
- If a future change does make a runner produce `StrategyMetrics`, the call site
  belongs in that producer/promotion harness, and this ADR should be revisited.
- `compute_v1_v2_diff` (the per-session A/B label diff for the §9.3 shadow
  review) is unaffected — it remains runner/shadow-consumable.
