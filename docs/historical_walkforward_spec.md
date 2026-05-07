# Historical Walk-Forward Spec

Logic-validation gate for V1 before forward shadow qualification begins.

This document defines what counts as a qualifying historical walk-forward for the V2 activation gate. It exists because historical walk-forward and forward shadow test different failure modes and must be judged by different standards.

## 0. Purpose

Historical walk-forward answers:

- "Does the frozen V1 engine produce stable, explainable, and economically defensible outputs on unseen historical data when fed only as-of inputs?"

It does **not** answer:

- whether daily operational runs are reliable;
- whether the data source fails cleanly under live conditions;
- whether the system misses sessions or silently skips writes.

Those operational questions belong to `docs/shadow_runner_spec.md`.

## 1. Role in V2 Activation

V2 activation requires both:

1. Historical walk-forward passes first.
2. Forward shadow then runs for 252 consecutive NYSE trading sessions.

Historical walk-forward is the fast logic gate. Forward shadow is the slow operational gate.

## 2. Frozen-Version Rule

The historical walk-forward must run on a frozen V1 engine/config pair.

Requirements:

- Freeze `engine_version` and `config_version` before the first walk-forward run.
- Do not change thresholds, labels, precedence, or classification logic during the evaluation.
- If a classification bug is found and fixed, restart the walk-forward from the beginning under the new frozen version.

This freeze is what makes the walk-forward meaningfully out-of-sample.

## 3. Data Discipline

The walk-forward must respect strict as-of data boundaries.

Rules:

- Each historical session is classified using only data available as of that historical date.
- No future event outcomes, future constituent information, or future price rows may leak into the run.
- Historical re-runs must be reproducible from versioned archived inputs.

Recommended archived input layout:

```text
walkforward/
├── regime_walkforward.db
├── outputs/
│   └── YYYY-MM-DD.json
├── input_archives/
│   └── YYYY-MM-DD/
│       ├── market_data.parquet
│       ├── events.yaml
│       └── checksums.json
└── reports/
    └── walkforward_report.md
```

The same "archive inputs before classify" rule used for shadow mode applies here.

## 4. Minimum Coverage

Minimum qualifying period:

- at least one full out-of-sample year of NYSE trading sessions.

Recommended stronger baseline:

- multi-year walk-forward, such as 2017-01-01 through 2024-12-31, with the qualifying holdout clearly separated from any calibration period.

The minimum one-year requirement is the gate. Longer windows are preferred because they cover more regime types.

## 5. Required Output Artifacts

The walk-forward is not complete unless it leaves behind reusable artifacts.

Minimum required artifacts:

- immutable JSON output per `as_of_date`;
- archived daily input snapshot per `as_of_date`;
- checksums for every archived input file;
- one summary report, for example `walkforward_report.md`;
- one machine-readable summary table, for example `walkforward_summary.csv` or a SQLite table.

Every output artifact must preserve:

- `engine_version`
- `config_version`
- `as_of_date`
- run timestamp
- input archive path

## 6. Pass/Fail Criteria

The historical walk-forward passes only if all of the following hold:

- no engine crashes across the entire qualifying window;
- no silent skips of required NYSE sessions;
- no NaN leakage beyond the explicit V1 `unknown` / `insufficient_history` contract;
- all 10 golden test dates still pass under the frozen version;
- no replay mismatch between stored outputs and archived-input recomputation for the sampled verification set;
- label distributions and transition behavior are reviewed and considered economically defensible;
- the strategy-improvement comparison versus the no-regime baseline is produced and is not materially worse on every tracked metric.

Any deterministic classification bug found during the walk-forward is a failure, not a warning.

## 7. Golden-Date Contract

The golden-date regression remains a hard gate inside walk-forward qualification.

Rules:

- run the current 10 golden dates under the frozen version before and after the historical walk-forward batch;
- a walk-forward pass is invalid if the golden-date suite fails at any point;
- if fixture expectations change, the fixture regeneration must be explicit, reviewed, and completed before the historical walk-forward qualification run begins.

## 8. Defensible Label Distribution

"Defensible label distribution" must be concrete enough to reject obviously broken engines.

At minimum, review:

- fraction of time spent in each top-level label;
- longest uninterrupted run length per label;
- switch count per year;
- false-switch rate where the label changes and reverses within the configured hysteresis horizon;
- crisis labels around known stress periods;
- prolonged unknown-label stretches;
- whether breadth / volatility / transition-risk labels cluster where the market narrative makes sense.

Red flags that fail review unless explicitly justified:

- one label dominates nearly the whole evaluation window with no economic reason;
- repeated one-day flip-flops in calm periods;
- crisis labels missing during obvious crash windows;
- long unknown runs after history warm-up;
- transition-risk warnings that almost never fire or fire almost every day.

This review is partly quantitative and partly operator judgment, but it must be written down in the report instead of left implicit.

## 9. Strategy Improvement vs No-Regime Baseline

The walk-forward must include a baseline comparison. Without that, the V2 prerequisite of measurable benefit is not satisfied.

Track at minimum:

- strategy return;
- max drawdown;
- Sharpe;
- hit rate;
- average trade duration;
- false switch rate;
- average detection lag;
- time spent in each regime;
- strategy PnL improvement from regime gating.

Comparison rules:

- use the same strategy universe and execution assumptions for both arms;
- compare `with_regime_gating` vs `no_regime_baseline`;
- do not retune thresholds on the holdout window;
- report both absolute and relative differences.

Pass interpretation:

- the regime engine does not need to beat baseline on every metric;
- it must demonstrate clear benefit on at least one material dimension, such as lower drawdown, fewer wrong-environment trades, or improved Sharpe;
- if it is materially worse on all tracked dimensions, the walk-forward fails as a qualification gate.

## 10. Report Contents

The final historical walk-forward report should contain:

1. frozen `engine_version` and `config_version`;
2. evaluation date range and NYSE session count;
3. data source and archive policy;
4. golden-date results;
5. label-distribution tables and charts;
6. transition and false-switch summaries;
7. strategy-vs-baseline comparison table;
8. incidents, anomalies, and reruns;
9. explicit conclusion: `pass` or `fail`.

The report must be reproducible from archived inputs and committed outputs.

## 11. Non-Goals

Do not turn the historical walk-forward gate into:

- a threshold-tuning exercise;
- a hyperparameter sweep over the same holdout period;
- an operational reliability test;
- a substitute for the forward shadow qualification window.

The point of this gate is to reject broken or economically useless frozen logic quickly before spending a calendar year on shadow mode.
