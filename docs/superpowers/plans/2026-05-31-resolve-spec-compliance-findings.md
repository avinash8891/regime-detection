# Resolve Spec Compliance Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the verified unresolved findings from `pasted_text_2026-05-31_08-23-28.txt` in dependency order, with failing tests before behavior changes and verification before each commit.

**Architecture:** Treat resolved findings as a baseline and finish the remaining work in contract-sized commits. Capital-protection fixes already landed in this branch; next tasks harden wire schemas, walk-forward/shadow gates, golden-date coverage, and spec/documentation drift. Each task has a focused test surface and must not relax deterministic predicates to pass fixtures.

**Tech Stack:** Python 3.14, pydantic models, pytest, YAML fixtures, SQLite-backed shadow/walk-forward ledgers, markdown specs.

---

## Verified Baseline Already Landed

- `F-005`, `F-006`: transition-score model evidence required and cluster flip included.
- `F-011`: breadth golden fixture split into raw and active labels.
- `F-015`, `F-016`: walk-forward NaN leakage and golden version/date validation.
- `F-038`, `F-039`: DQ hysteresis and cohort precedence.
- `F-046`, `F-047`: shadow replay incidents and deadman exit code.

## Task 1: V1/V2 Wire Contract Alignment (`F-001`, `F-002`, `F-003`, `F-023`)

**Files:**
- Modify: `src/regime_detection/strategy_models.py`
- Modify: `src/regime_detection/legacy_v1_wire.py`
- Modify: `docs/regime_engine_v1_final_spec.md`
- Test: `tests/test_models.py`

- [x] Step 1: Write failing tests asserting `StructuralCausalState` accepts `monetary_pressure`, V1 projection emits `state` for `network_fragility`, and `transition_risk` is present in the V1 canonical contract shape.
- [x] Step 2: Run `python3.14 -m pytest -o addopts='' tests/test_models.py::test_regime_output_legacy_v1_projection_preserves_archived_wire_shape tests/test_models.py::test_regime_output_non_v1_dump_keeps_native_shape_and_json_mode -q; echo "EXIT:$?"` and confirm failure.
- [x] Step 3: Add a small label/reason output model, add `monetary_pressure` to `StructuralCausalState`, and project V1 `network_fragility` / `transition_risk` using `state`.
- [x] Step 4: Update the V1 spec §11 JSON and frozen shim expectations only after the model tests prove the contract.
- [x] Step 5: Run targeted model tests plus frozen replay tests.
- [x] Step 6: Commit `fix(wire): align v1 structural causal contract`.

## Task 2: Live V1 Frozen Replay (`F-004`)

**Files:**
- Modify: `tests/test_v1_frozen_replay.py`
- Modify fixtures under `tests/fixtures/v1_frozen_outputs/` only if live-engine evidence shows the archived fixture is stale.

- [x] Step 1: Write a failing test that instantiates `RegimeEngine`, classifies each archived fixture date with fixture market data, projects through V1 wire mode, and diffs against the archived JSON.
- [x] Step 2: Run only `tests/test_v1_frozen_replay.py`; confirm the existing round-trip-only enforcement is insufficient.
- [x] Step 3: Wire fixture data helpers into the test without network calls.
- [x] Step 4: Run `python3.14 -m pytest -o addopts='' tests/test_v1_frozen_replay.py -q; echo "EXIT:$?"`.
- [ ] Step 5: Commit `fix(test): replay v1 frozen fixtures through live engine`.

## Task 3: Walk-Forward Gate Invariants (`F-014`, `F-017`, `F-053`, `F-054`, `F-055`, `F-056`, `F-057`)

**Files:**
- Modify: `scripts/build_walkforward_report.py`
- Test: `tests/test_build_walkforward_report.py`

- [x] Step 1: Add failing tests for version inconsistency, replay mismatch failure, `<252` successful OOS sessions, red-flag failures, missing transition-risk report fields, unknown baseline metrics, relative deltas, and missing per-date provenance.
- [x] Step 2: Run the focused test file and verify the intended failures.
- [x] Step 3: Implement helpers for frozen-version validation, replay mismatch ingestion, OOS count, red flags, report completeness, metric direction validation, relative deltas, and provenance checks.
- [x] Step 4: Run `python3.14 -m pytest -o addopts='' tests/test_build_walkforward_report.py -q; echo "EXIT:$?"`.
- [x] Step 5: Commit `fix(walkforward): enforce qualification gate invariants`.

## Task 4: Shadow Qualification Window (`F-018`, `F-048`)

**Files:**
- Create or modify: `src/regime_detection/shadow_qualification.py`
- Modify: `scripts/run_shadow_deadman_check.py`
- Test: `tests/test_shadow_deadman_check.py`
- Test: create `tests/test_shadow_qualification.py`

- [ ] Step 1: Add failing tests for missing sessions inside a qualification window, qualification-breaking incidents restarting the counter, and exactly 252 consecutive successful NYSE sessions qualifying.
- [ ] Step 2: Run the new/focused shadow tests and confirm failure.
- [ ] Step 3: Implement a ledger scanner over NYSE sessions that counts consecutive successful, incident-free runs for one frozen engine/config pair.
- [ ] Step 4: Call the scanner from deadman/qualification tooling without changing existing single-day alert semantics.
- [ ] Step 5: Run focused shadow tests.
- [ ] Step 6: Commit `fix(shadow): qualify contiguous 252 session window`.

## Task 5: Golden-Date Integrity (`F-008`, `F-009`, `F-010`, `F-012`, `F-013`, `F-058`)

**Files:**
- Modify: `tests/fixtures/derived/golden_dates.yaml`
- Create: `tests/fixtures/derived/golden_dates_v2.yaml`
- Modify: `tests/conftest.py`
- Modify: axis golden tests under `tests/test_*state.py`, `tests/test_trend_character.py`, `tests/test_trend_direction.py`
- Test: create or extend `tests/test_fixture_verification.py`

- [ ] Step 1: Add failing tests that assert the V1 §12.2 10 dates exist, no golden row is skipped, non-ok DQ does not bypass equality, all 10 V1 rows pass, V2 §9.4 rows are registered, and stateless replay is independent of history length.
- [ ] Step 2: Run focused fixture tests and confirm failure.
- [ ] Step 3: Re-anchor or justify the V1 golden dates using independent fixture evidence; do not relax predicates.
- [ ] Step 4: Extend fixture coverage back far enough for 2017/2018 rows or explicitly block completion if raw data is absent.
- [ ] Step 5: Add V2 §9.4 fixture registration.
- [ ] Step 6: Run focused golden fixture tests.
- [ ] Step 7: Commit `fix(fixtures): enforce complete golden date coverage`.

## Task 6: V1 Path Hygiene Guard (`F-035`)

**Files:**
- Modify: `tests/test_source_hygiene.py`
- Optionally modify: `.github/workflows/ci.yml` if the new pytest is not covered by existing test selection.

- [ ] Step 1: Add a failing path-scoped test that rejects forbidden V2 scaffolding tokens in V1-only files.
- [ ] Step 2: Run the focused source hygiene test and confirm failure.
- [ ] Step 3: Implement an allowlist for documented compatibility comments and remove or reword true violations.
- [ ] Step 4: Run source hygiene tests.
- [ ] Step 5: Commit `test: guard v1 paths against v2 scaffolding`.

## Task 7: Schema and Spec Drift (`F-022`, `F-024`, `F-031`, `F-037`)

**Files:**
- Modify: `src/regime_detection/hysteresis.py`
- Modify: config models/YAML if escalation config is added.
- Modify: `docs/regime_engine_v2_spec.md`
- Test: `tests/test_per_label_hysteresis.py`, `tests/test_v2_config.py`, `tests/test_network_fragility_classifier.py`

- [ ] Step 1: Add failing tests for configurable escalation days defaulting to byte-identical immediate escalation.
- [ ] Step 2: Decide by evidence whether `systemic_stress_unconfirmed` is formalized in spec or removed; add test for the chosen label set.
- [ ] Step 3: Close Ambiguity Log #26 in the spec and update classification-status docs to match shipped statuses.
- [ ] Step 4: Run focused hysteresis/config/network tests.
- [ ] Step 5: Commit `fix(spec): close v2 schema drift`.

## Task 8: V1 Config and Strategy Cleanup (`F-032`, `F-033`, `F-034`, `F-042`, `F-043`)

**Files:**
- Modify: `src/regime_detection/configs/core3-v1.0.0.yaml`
- Modify: `src/regime_detection/trend_character.py`
- Modify: `src/regime_detection/strategy_response.py`
- Test: focused config, trend-character, strategy-response tests.

- [ ] Step 1: Add failing tests for V1 config not lighting V2 blocks, V1 modifier predicates excluding V2-only labels, and scalar/vector Layer-1 rule parity.
- [ ] Step 2: Run focused tests and confirm failure.
- [ ] Step 3: Remove dead V2 blocks from V1 config or document retained compatibility fields if config validation requires them.
- [ ] Step 4: Collapse duplicate Layer-1 rule encoding only if parity tests prove a safe path.
- [ ] Step 5: Pin ADX seeding convention in docs/tests.
- [ ] Step 6: Commit `refactor(v1): tighten config and rule parity`.

## Task 9: Decision Records and Scope Notes (`F-019`, `F-021`, `F-025`, `F-045`, `F-049`, `F-050`)

**Files:**
- Modify or create docs under `docs/decisions/` and relevant specs.

- [ ] Step 1: Add doc checks where feasible for prerequisite tracking, PIT interval schema, HMM drift scope, CPI vintage scope, shadow source-of-truth, and upstream fetch responsibility.
- [ ] Step 2: Update the decision records with exact current implementation references.
- [ ] Step 3: Run docs/source hygiene checks.
- [ ] Step 4: Commit `docs: close remaining scope-note findings`.

## Final Verification

- [ ] Run all focused tests touched by this plan.
- [ ] Run `python3.14 -m black --check src tests scripts`.
- [ ] Run `python3.14 -m ruff check .`.
- [ ] Run `python3.14 -m pyright`.
- [ ] Run `git diff --check`.
- [ ] Push branch and report CI status if GitHub Actions runs are created.
