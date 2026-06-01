# 0025 — Reproducible shadow strategy success metrics (§10)

- Status: accepted
- Date: 2026-06-02
- Finding: F-014 (spec_review.md), milestone M3
- Code: `src/regime_detection/shadow_strategy_metrics.py`,
  `scripts/build_shadow_metrics_report.py`

## Context

`docs/shadow_runner_spec.md` §10 (L190-203) requires the forward shadow run to
accumulate six strategy success metrics — strategy return, max drawdown, Sharpe,
false switch rate, average detection lag, wrong-environment trades avoided — and
states the V2 measurable-strategy-improvement prerequisite "is not satisfied until
those metrics exist in a **reproducible report**". §11 forbids "strategy-backtest-
specific orchestration copied from another repo": the runner must stay "small,
durable, reproducible".

The shadow ledger already persists everything needed to compute these
deterministically: the `runs` table (success dates), the per-date classification
output JSON (regime labels), and the archived `market_data.parquet` (prices). What
was missing was a definition of the strategy mapping and each metric, plus the
reporter. This ADR pins those definitions so the report is reproducible; it is
deliberately a small in-repo reducer over the ledger, not a backtest platform.

## Decisions

**Strategy mapping (defensive overlay).** The regime engine emits labels, not
positions; the §10 metrics need a position rule. We pin the minimal defensive
overlay: exposure is `0.0` (flat) when the session is risk-off — `transition_risk.
state == "crisis"` OR `volatility_state.active_label == "crisis_vol"` — and `1.0`
(fully invested in the SPY universe close) otherwise. The **no-regime baseline** is
exposure `1.0` on every session. This is the simplest mapping that makes "regime vs
no-regime" measurable and matches the engine's own §3 emergency-override semantics
(crisis is the reactive risk-off trigger).

**No lookahead.** The position held into session *i* is the exposure implied by the
**prior** session's labels (`exposure(label[i-1])`). Returns are close-to-close SPY
returns between consecutive success sessions. The first success session seeds the
series and contributes no return.

**Metric definitions (all over the success-session return series):**

1. `strategy_return` — terminal value of `cumprod(1 + exposure_{i-1}·r_i) − 1`.
2. `max_drawdown` — most-negative `equity/cummax − 1` of the strategy equity curve.
3. `sharpe` — `mean(strat_r)/std(strat_r, ddof=1)·sqrt(252)`; `0.0` if std is 0.
4. `false_switch_rate` — of all exposure switches, the fraction that revert to the
   pre-switch exposure within `H = 3` sessions; `0.0` if there are no switches.
5. `average_detection_lag` — for each configured crash window (the F-050
   `CRASH_WINDOWS`) covered by success sessions, the session lag from the window's
   first covered session to the first flat (risk-off) session within the window;
   full-window length if never flat. Averaged over covered windows; `None` if no
   window is covered.
6. `wrong_environment_trades_avoided` — count of sessions the strategy held flat
   (prior-session risk-off) while the realized SPY return was negative — i.e. losing
   sessions the no-regime baseline took but the strategy dodged.

The report also emits the no-regime baseline's `baseline_return`,
`baseline_max_drawdown`, and `baseline_sharpe` so "improvement vs no-regime baseline"
is directly readable. `evaluate_v2_gate` (§9.1, ADR 0023) consumes the four
gate-relevant metrics from this report.

## Consequences

- The report is a pure function of the ledger: re-running the reporter on the same
  `output_root` yields byte-identical metrics (no clock, no randomness). A test
  asserts all six keys are present and reproduced on a second run.
- The strategy mapping is intentionally minimal and documented here; a richer
  strategy router, if ever needed, lives in the external strategy repo (§10/§11) and
  would supersede this mapping via a new ADR.
- This satisfies F-014's "add a reproducible report" branch; it does not import any
  external backtest orchestration (§11 non-goal respected).
