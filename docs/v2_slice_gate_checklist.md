# V2 Slice Promotion Checklist

Operating contract for promoting any v2 vertical slice (per v2 spec §8 order) from feature branch to main. Each slice merges only when every item below is checked, in order.

A slice = one of the ten units listed in v2 spec §8 (Network Fragility, Layer 1 V2 incremental features, Transition Score, Credit/Funding, Inflation/Growth, HMM, K-Means/GMM, Change-Point, Cohort/Family Routing, PRISM).

## Pre-merge checklist

### 1. Slice scope

- [x] Slice maps to exactly one v2 spec §8 row.
- [x] No changes to v1 production code outside what the slice's v2 spec section explicitly authorizes.
- [x] No formulas, thresholds, or precedence invented (v2 spec §10: "do not invent component score formulas — use the exact formulas in §4.2"; same rule for §3.5, §2A/§2B/§2C, etc.).

### 2. Config

- [x] The slice's v2 sub-config block exists in `configs/core3-v2.0.0.yaml` with reviewed defaults (each value cited to its v2 spec line).
- [x] The corresponding Pydantic class in `src/regime_detection/config.py` has `extra="forbid"` and `Field(...)` constraints on every numeric range.

### 3. Models

- [x] Wire-level type evolution (if any) preserves v1 byte-identity: `tests/test_v1_frozen_replay.py::test_v1_frozen_outputs_parse_through_v1_frozen_models` still passes.
- [x] Optional `RegimeOutput` fields default `None` so consumers that don't enable the slice see no diff.

### 4. Feature store + axis_series

- [x] New feature dataclass lives in the axis module (e.g., `network_fragility.py`), not in `feature_store.py`.
- [x] `feature_store.py` is extended with `Optional[X] = None` and only computes when the corresponding `MarketContext` data input is present.
- [x] Any new live-axis builder follows the current axis-series pattern: compute the slice-specific feature object in its home module, expose the axis output through `AxisSeriesResult` or the established per-axis output mapping, and wire it from `axis_series.py`/`engine.classify` only when the slice's required `MarketContext` inputs are present.
- [x] Hysteresis routed through `apply_per_label_asymmetric_hysteresis` with `deescalation_days_by_label` from neutral axis-level config sections. Per-label hysteresis is mandatory for all 9 label axes (ADR 0010); missing config raises. Both V1 and V2 configs ship per-label blocks on the axis sections, and V2 feature/rule sections reject hysteresis keys so calibration changes cannot become silent no-ops. The old flat `apply_asymmetric_hysteresis` helper has been removed.

### 5. Tests

- [x] Unit tests for the new feature compute (synthetic inputs, hand-computed expected values; no toy names per AGENTS rule).
- [x] Unit tests for the rule engine (one test per rule precedence position).
- [x] Unit tests for the hysteresis wrapper (default immediate escalation, configurable delayed escalation where used, and de-escalation honors per-label thresholds).
- [x] Integration test invoking `engine.classify` end-to-end with the slice's data input.
- [x] At least one v2 golden date passes (`tests/fixtures/derived/golden_dates.yaml` row's `expected.<slice_field>`).

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

- [x] If a v2 spec ambiguity surfaced during implementation, it is recorded in the slice PR description and (if material) added to the spec via the rewrite-existing-lines rule from File 3.
- [x] No new top-level docs sections added; spec edits are inline.

### 9. Commit + CI

- [x] Single commit per slice (or single commit per sub-step if the slice has ≥3 sub-steps).
- [x] Commit message identifies the slice (v2 §8 row) and the v2 spec sections implemented.
- [x] CI green: `pytest -m "not v2_shadow"` (unit + integration + v2_gate; v2_shadow is long-running and runs separately).

### 10. Safety invariants (added by audit)

- [x] V1 config does NOT light V2 feature seams — `build_regime_timeline` gates V2 feature configs on `config_version`.
- [x] Golden fixtures are hand-labeled, not engine-generated — `golden_dates.yaml` carries `provenance: hand_labeled`.
- [x] HMM/GMM label maps are NOT shipped in default config — auto-labeling requires reviewed operator artifact.
- [x] HMM/GMM label maps are validated against fitted model metadata (n_states/n_clusters, model_version) before populating `mapped_label`.
- [x] HMM `state_persistence_days` is computed across the full history, not just the output window.
- [x] `AxisEvidencePayload` type is preserved through `model_copy` (no plain-dict leakage).
- [x] Core axis evidence is typed and rejects undeclared evidence keys.
- [x] Cross-axis dependency payload and failure semantics are declared in `AXIS_DEPENDENCY_CONTRACTS`.
- [x] Feature-store optional seams emit `FeatureStore.availability` with policy, required inputs, and missing inputs.
- [x] V2 operator artifacts expose dependency payload contracts, classification coverage, and rule provenance in profile, shadow, and walk-forward outputs.
- [x] Central-bank-text `max_release_age_days` filter applies to aggregation (pass filtered `working` frame, not original `scored_releases`).
- [x] Available-sector breadth proxy is reachable when some (but not all) sector ETFs are present.

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
- **Layer 2B Inflation/Growth (§2B), v2 §8 row 5** — all current labels are reachable, including the ADR 0019 valid-data partitions (`contractionary_disinflation`, `late_cycle_inflation_stress`, `recovery_growth_unconfirmed`, `macro_neutral`). `earnings_expansion` / `earnings_contraction` are wired end-to-end via the `aggregate_eps` weekly-snapshot accumulator (Log #48; silent only during the >4-week cold-start). The single-signal `inflation_shock` limb consumes `inflation_surprise_zscore`, computed from the free Cleveland Fed inflation nowcast as the `consensus_estimate` substitute (ADR 0006); the dedicated `cleveland_fed_nowcast` fetch path is built. No paid feed required. Note: `GDPNow` IS free on FRED (`GDPNOW`, in `V2_FRED_SERIES`); `GDPNow` / `Citi Surprise` are NOT in any §2B rule predicate.
