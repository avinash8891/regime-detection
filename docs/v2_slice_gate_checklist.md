# V2 Slice Promotion Checklist

Operating contract for promoting any v2 vertical slice (per v2 spec §8 order) from feature branch to main. Each slice merges only when every item below is checked, in order.

A slice = one of the ten units listed in v2 spec §8 (Network Fragility, Layer 1 V2 incremental features, Transition Score, Credit/Funding, Inflation/Growth, HMM, K-Means/GMM, Change-Point, Cohort/Family Routing, PRISM).

## Pre-merge checklist

### 1. Slice scope

- [ ] Slice maps to exactly one v2 spec §8 row.
- [ ] No changes to v1 production code outside what the slice's v2 spec section explicitly authorizes.
- [ ] No formulas, thresholds, or precedence invented (v2 spec §10: "do not invent component score formulas — use the exact formulas in §4.2"; same rule for §3.5, §2A/§2B/§2C, etc.).

### 2. Config

- [ ] The slice's v2 sub-config block exists in `configs/core3-v2.0.0.yaml` with reviewed defaults (each value cited to its v2 spec line).
- [ ] The corresponding Pydantic class in `src/regime_detection/config.py` has `extra="forbid"` and `Field(...)` constraints on every numeric range.

### 3. Models

- [ ] Wire-level type evolution (if any) preserves v1 byte-identity: `tests/test_v1_frozen_replay.py::test_v1_frozen_outputs_parse_through_v1_frozen_models` still passes.
- [ ] Optional `RegimeOutput` fields default `None` so consumers that don't enable the slice see no diff.

### 4. Feature store + axis_series

- [ ] New feature dataclass lives in the axis module (e.g., `network_fragility.py`), not in `feature_store.py`.
- [ ] `feature_store.py` is extended with `Optional[X] = None` and only computes when the corresponding `MarketContext` data input is present.
- [ ] New `XYZSeriesClassifier` follows the existing protocol: `build(context, feature_store) -> AxisSeriesResult | dict[date, OutputType]`.
- [ ] Hysteresis routed through the appropriate helper: `apply_asymmetric_hysteresis` (single-int de-escalation, v1 axes) or `apply_per_label_asymmetric_hysteresis` (per-label de-escalation, v2 §3.7).

### 5. Tests

- [ ] Unit tests for the new feature compute (synthetic inputs, hand-computed expected values; no toy names per AGENTS rule).
- [ ] Unit tests for the rule engine (one test per rule precedence position).
- [ ] Unit tests for the hysteresis wrapper (escalation immediate; de-escalation honors per-label thresholds).
- [ ] Integration test invoking `engine.classify` end-to-end with the slice's data input.
- [ ] At least one v2 golden date passes (`tests/fixtures/derived/golden_dates_v2.yaml` row's `expected.<slice_field>`).

### 6. v2 §9.1 performance gate

- [ ] `scripts/run_historical_walkforward.py --engine-profile both` run completed over ≥1 year of out-of-sample NYSE sessions.
- [ ] `scripts/run_v2_performance_gate.py` (or equivalent gate evaluator) reports `passed=True` on `evaluate_v2_gate(v1_metrics=..., v2_metrics=...)` with at least one of:
  - `LOWER_DRAWDOWN`
  - `HIGHER_SHARPE`
  - `EARLIER_CRISIS_DETECTION`
  - `LOWER_FALSE_SWITCH_RATE`
- [ ] Gate output committed to `docs/verification/v2_slice_<n>_perf_gate.md` with v1 vs v2 metric tables.

### 7. v2 §9.3 shadow A/B (post-walkforward, pre-routing)

- [ ] `scripts/run_shadow_regime.py --engine-profile both` has run on 60 consecutive NYSE sessions.
- [ ] `scripts/run_v1_v2_diff_report.py` output reviewed: zero unexpected wire diffs in v1 fields; v2 enrichments match expectations.
- [ ] Any disagreement day has a documented rationale in `docs/verification/v2_slice_<n>_disagreements.md`.

### 8. Documentation

- [ ] If a v2 spec ambiguity surfaced during implementation, it is recorded in the slice PR description and (if material) added to the spec via the rewrite-existing-lines rule from File 3.
- [ ] No new top-level docs sections added; spec edits are inline.

### 9. Commit + CI

- [ ] Single commit per slice (or single commit per sub-step if the slice has ≥3 sub-steps).
- [ ] Commit message identifies the slice (v2 §8 row) and the v2 spec sections implemented.
- [ ] CI green: `pytest -m "not v2_shadow"` (unit + integration + v2_gate; v2_shadow is long-running and runs separately).

## After merge

- [ ] Slice tagged in repo: `v2-slice-<n>-<short-name>`.
- [ ] Next slice's branch can be cut from the new main.

## Slices that are blocked

| Slice | v2 §8 row | Currently blocked on |
|---|---|---|
| Layer 5 V2 PRISM (§5.5) | 10 | PRISM framework not in repo; deferred to V2.1 per spec §5.5. |

Slices not in the above table can proceed from foundation as-is.

### Recently unblocked (no longer in the table above)

- **Layer 1 V2 incremental features (§1A/§1C/§1D), v2 §8 row 2** — was blocked on the `sentiment_score` AAII/put-call/IIA fetcher. Resolved: the AAII fetcher shipped (`regime_data_fetch.aaii_sentiment`), `sentiment_score = bull_bear_spread_8w_ma`, and `euphoria` fires (Ambiguity Log #32). Put-call / Investors Intelligence remain *optional* calibration-revision sources only — they block no label. `vol_crush` (ADR 0005), the PIT breadth features (Slice 2.8c), and `breakout_expansion` (Log #46/#47) also landed.
- **Layer 2B Inflation/Growth (§2B), v2 §8 row 5** — all 8 labels are now reachable. `earnings_expansion` / `earnings_contraction` are wired end-to-end via the `aggregate_eps` weekly-snapshot accumulator (Log #48; silent only during the >4-week cold-start). The single-signal `inflation_shock` limb consumes `inflation_surprise_zscore`, computed from the free Cleveland Fed inflation nowcast as the `consensus_estimate` substitute (ADR 0006); the dedicated `cleveland_fed_nowcast` fetch path is built. No paid feed required. Note: `GDPNow` IS free on FRED (`GDPNOW`, in `V2_FRED_SERIES`); `GDPNow` / `Citi Surprise` are NOT in any §2B rule predicate.
