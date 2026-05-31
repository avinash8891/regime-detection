# ADR 0020 - V2 Prerequisite and Shadow Scope

**Status:** Accepted
**Date:** 2026-05-31
**Context:** The spec-compliance review found several low-risk mismatches where
the implementation was already deterministic, but the written contract implied
a different process or source boundary. This ADR records those decisions so the
V2 extension remains one engine with explicit activation gates, not a second
system with implicit assumptions.

## Decision

- **F-019 — 9-slice prerequisite.** The 9-slice prerequisite is a process gate,
  not an engine runtime assertion. V2 activation remains blocked until the
  slice checklist, walk-forward gate, golden-date gate, and forward-shadow gate
  are all complete. The repository enforces this through
  `docs/v2_slice_gate_checklist.md`, `docs/historical_walkforward_spec.md`,
  `docs/shadow_runner_spec.md`, and their gate tests rather than by adding a
  classifier branch.
- **F-020 — "All 10 V1 golden dates pass" activation gate.** This prerequisite
  is enforced as a regression suite, not an engine runtime branch. The citable
  gate is the pair `tests/test_fixture_verification.py::test_classified_golden_outputs_cover_every_row_without_silent_skips`
  (exactly the full `golden_dates.yaml` set classifies with no silent skips) and
  `::test_golden_dates_match_live_labels_without_data_quality_bypass` (every axis
  matches the hand-labeled expectation with `data_quality == "ok"`). The
  per-slice ">= 1 v2 golden date" checklist item is the incremental slice check;
  these two pytests are the all-10 prerequisite gate, now cited from
  `docs/v2_slice_gate_checklist.md` so the gate wiring is explicit rather than
  implied by prose.
- **F-021 — PIT membership row schema.** The shipped PIT constituent artifact
  uses a ticker / start_date / end_date interval schema. That is the canonical
  row-level representation for membership validity because every trading-date
  row can be derived deterministically from the interval. Replacing the free
  `fja05680/sp500` source with CRSP / Compustat / FactSet / Norgate should keep
  the same interval shape unless the loader contract is explicitly amended.
- **F-025 — HMM parameter drift.** HMM parameter-drift flags are not part of the
  capital-protection transition score contract. The runtime transition score
  consumes point-in-time HMM probability movement, GMM cluster flips, and
  change-point evidence. The §6.1 operator calibration-review monitor now
  ships as `regime_detection.hmm_state.compute_hmm_parameter_drift` AND is wired
  into `compute_hmm_features`: at each PIT refit checkpoint the fitted state
  means are de-standardized back to raw feature units and compared to the prior
  checkpoint, surfacing the result on `HMMFeatures.parameter_drift` (None when
  fewer than two checkpoints were fit). The wiring is fail-open observability —
  it never blocks the fit and never feeds the runtime transition score. The
  report carries the 20% state-mean parameter-drift alert
  (max relative state-mean change after Hungarian alignment of new state
  indices to old, per the spec line 4456-4466 operational form) and the
  separate non-blocking 30% transition-probability review flag. The
  transition-probability shift is measured as the maximum **absolute** change
  in any aligned transition entry — transition probabilities are bounded
  `[0, 1]`, so an absolute "30 percentage point" move is the only stable
  reading (relative change is ill-defined for near-zero probabilities); this
  resolves the spec's otherwise-unspecified transition-prob metric. The helper
  is advisory: it does not block deployment and is distinct from the loud
  runtime model-evidence requirement (F-006).
- **F-053 — Vol-crush exposure response.** V2 §5.3 vol-crush exposure response
  is a downstream strategy-layer contract, not `regime_detection` runtime
  logic. The engine's responsibility is to emit the `vol_crush` volatility
  label and evidence correctly. The 50% long-vol exposure reduction over the
  5-day cooldown belongs to the position-management layer that consumes engine
  outputs, not to this classifier package.
- **F-045 — CPI vintage scope.** The V2 §2A dual-vintage implementation is a
  CPI-only dual-vintage store for first-release historical replay. The current
  code loads `CPIAUCSL` realtime observations into
  `cpi_all_items_vintages.parquet`, derives first-release CPI through
  `loaders.load_cpi_vintages_first_release`, and preserves the latest-revision
  path when the vintage seam is absent. There is no generic all-macro vintage
  store in this slice.
- **F-049 — Shadow source of truth.** For this repository's current forward
  shadow implementation, local/Alpaca archived parquet is the shadow source of truth.
  The earlier Stooq wording is superseded by the May 2026 data-source
  plan because the engine-facing ETF and constituent OHLCV artifacts were
  re-fetched and verified from Alpaca/local parquet. Replays read archived
  inputs only.
- **F-050 — Daily fetch boundary.** The daily fetch is upstream of the runner.
  The runner consumes already-fetched inputs, archives them before
  classification, writes checksums, and then calls the engine. This keeps the
  runner small and replayable; fetch failures belong to the acquisition layer
  before a shadow `runs` row is promoted to classification.

- **F-041 — V1-config transition-risk path.** The V1 config path
  (`transition_score is None`) computing a transition-risk state is **intended
  behavior, not a non-goal violation.** V1 §10 strategy response gates nine
  capital-protection modifiers — `crisis`, `bear_stress`, `high_transition_risk`,
  `fragile_bull`, `recovery_attempt`, `weakening`, `transition_warning`, `watch`
  (`strategy_response.py:135-202`) — on `transition_risk.state`, so V1 MUST emit
  a meaningful state or it would silently lose crisis/bear-stress protection.
  The V1 path (`transition_risk_series.py:108-140`) runs a flag-driven, debounced
  rule state machine derived from the base axes (trend / character / volatility /
  breadth stable labels) with `transition_score_config=None`; it emits
  `score = None` and `score_components = None` and returns BEFORE the
  model-evidence requirement, so it needs no HMM / change-point / clustering
  evidence. The non-goal V1OUT-029 ("transition_risk is V2-owned; do not add V1
  transition-risk labels / precedence / scoring") forbids re-implementing the V2
  WEIGHTED-SCORE algorithm (HMM / GMM / change-point) in V1 — which the V1 path
  does not do — not the rule-based state the V1 strategy contract requires. The
  V1/V2 asymmetry (V1 needs no model evidence; V2 raises loudly when it is
  missing, F-006) is pinned by
  `tests/test_transition_risk.py::test_v1_transition_risk_fallback_preserves_flag_only_stable_state`
  and `::test_v1_transition_risk_requires_no_model_evidence`.

## Consequences

- V2 remains an extension of the same engine and contracts; V1 replay and V2
  activation are separated by gates, not by separate codebases.
- Runtime failures still stay loud where capital protection depends on the
  evidence, especially transition-score model evidence.
- Documentation that refers to shadow input sources or fetch ownership must
  cite this ADR and `docs/shadow_runner_spec.md` rather than resurrecting Stooq
  as the current source-of-truth.
