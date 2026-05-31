# Pasted Coverage Gap Closeout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining open or partial test-coverage gaps from `.context/attachments/TwRcYM/pasted_text_2026-05-31_12-28-40.txt`.

**Architecture:** Add focused tests around existing code paths before changing production code. Only add production code if a failing test proves the behavior has no current implementation surface. Keep each coverage cluster in a separate commit.

**Tech Stack:** Python, pytest, pandas, existing regime engine fixtures, existing walk-forward/shadow scripts.

---

## Verification Summary

Fresh verification on 2026-05-31 found the pasted review is **partially still correct**. Several original gaps are now covered, but not all.

## Evidence-Checked Task List

- [x] Golden V2 dates run live end-to-end for fixture-supported dates, not registration-only.
  Current evidence: `tests/test_fixture_verification.py::test_v2_section_9_4_golden_dates_are_registered` validates `golden_dates_v2.yaml` registration only. No live classification/equality test was found for the 9 V2 dates.
  Completion evidence: `tests/test_fixture_verification.py::test_v2_golden_dates_classify_expected_fields` now live-classifies the fixture-supported V2 golden rows and asserts the exact unsupported set (`2010-05-06`, `2011-08-08`, `2015-08-24`, `2018-10-10`, `2020-08-15`) so fixture/date gaps cannot silently disappear.
- [ ] Golden-date data quality is asserted, not only label equality.
  Current evidence: `test_golden_dates_match_live_labels_without_data_quality_bypass` compares labels but does not assert each golden axis `data_quality.status == "ok"`.
- [ ] V1 stateless replay/window-length independence has an explicit live-engine regression.
  Current evidence: live V1 frozen replay exists, but no test compares the same V1 `as_of_date` under different supplied history lengths.
- [ ] V1/V2 extension contract has one comprehensive all-fields absence/preservation test.
  Current evidence: frozen live replay and slice-local checks exist, but no single comprehensive all-V2-fields absence test was found.
- [ ] Walk-forward JSON-output NaN leakage is tested.
  Current evidence: `build_walkforward_report.py` scans JSON payloads, but tests cover summary NaN leakage only.
- [ ] Walk-forward golden-before/after batch contract is executable or explicitly documented as validation-only.
  Current evidence: report validation accepts an external golden-results payload; no code was found that runs golden dates before and after the walk-forward batch.
- [ ] Walk-forward red flags cover long unknown stretches and repeated one-day flip-flops.
  Current evidence: tests cover label dominance and transition-risk-never-fires, but not long unknown stretches or repeated one-day flip-flops.
- [ ] HMM drift / transition-probability review flags are implemented and tested, or explicitly moved out of runtime scope with a test-backed decision.
  Current evidence: docs define 20% state-mean drift and 30% transition-probability review flags, but implementation/test search found only docs/ADR references.
- [ ] Vol-crush exposure response is implemented and tested in the proper downstream strategy layer, or explicitly tested as out of `regime_detection` runtime scope.
  Current evidence: engine emits `vol_crush` labels, but no `vol_crush_exit_rules`, `long_vol_position_reduction_pct`, or 5-day exposure cooldown implementation/test was found.

---

## Task 1: V2 Golden-Date Live Execution

**Files:**
- Modify: `tests/test_fixture_verification.py`
- Possibly modify fixtures only if the failing test proves expected V2 outputs are missing or stale.

- [x] **Step 1: Add a failing test that classifies every fixture-supported `golden_dates_v2.yaml` row**

Add a test that loads all 9 rows, classifies each fixture-supported `as_of_date` with real V2 fixture kwargs, asserts all expected V2 fields named in `expected_v2_fields` are present and non-empty, and asserts the exact unsupported set for dates outside the real V2 fixture/trading-day surface.

- [x] **Step 2: Run targeted test**

```bash
python3 -m pytest tests/test_fixture_verification.py::test_v2_golden_dates_classify_expected_fields -q ; echo "EXIT:$?"
```

- [x] **Step 3: Implement only what the failing test proves is missing**

Prefer fixture/test wiring fixes over production changes unless classification itself cannot produce the expected V2 field.

Evidence: RED failed on `2010-05-06` because the real V2 daily OHLCV fixture has no VIX rows before 2019; GREEN passes with explicit unsupported-date assertions and live classification for the supported rows.

---

## Task 2: Golden Data-Quality Assertions

**Files:**
- Modify: `tests/test_fixture_verification.py`

- [ ] **Step 1: Add data-quality checks to golden-date live-label test**

For each V1 golden output, assert transition-risk `data_quality.status == "ok"` and assert any axis exposing data-quality does not silently bypass expected-label equality.

- [ ] **Step 2: Run targeted tests**

```bash
python3 -m pytest tests/test_fixture_verification.py::test_golden_dates_match_live_labels_without_data_quality_bypass -q ; echo "EXIT:$?"
```

---

## Task 3: V1 Stateless Replay Window Independence

**Files:**
- Modify: `tests/test_v1_frozen_replay.py`

- [ ] **Step 1: Add live-engine V1 same-date/different-history test**

Classify the same V1 `as_of_date` with full market data and a shorter still-sufficient market-data slice, then assert the V1 labels and serialized V1 output match except provenance fields that are intentionally history-dependent.

- [ ] **Step 2: Run targeted test**

```bash
python3 -m pytest tests/test_v1_frozen_replay.py::test_v1_live_engine_replay_is_independent_of_extra_history_length -q ; echo "EXIT:$?"
```

---

## Task 4: Comprehensive V1/V2 Extension Contract

**Files:**
- Modify: `tests/test_v1_frozen_replay.py`

- [ ] **Step 1: Add all-V2-fields absence/preservation test**

For each frozen V1 fixture, run live `RegimeEngine.classify` with `core3-v1.0.0.yaml`, serialize with `exclude_none=True`, and assert V2 extension fields are absent while the V1 base fields match the fixture.

- [ ] **Step 2: Run targeted test**

```bash
python3 -m pytest tests/test_v1_frozen_replay.py::test_v1_live_replay_omits_all_v2_extension_fields -q ; echo "EXIT:$?"
```

---

## Task 5: Walk-Forward JSON NaN Leakage

**Files:**
- Modify: `tests/test_build_walkforward_report.py`
- Possibly modify: `scripts/build_walkforward_report.py`

- [ ] **Step 1: Add JSON-output NaN leakage test**

Create a fake archived output JSON containing a JSON NaN and assert `build_walkforward_report` fails with `nan_leakage_detected`.

- [ ] **Step 2: Run targeted test**

```bash
python3 -m pytest tests/test_build_walkforward_report.py::test_build_walkforward_report_rejects_json_output_nan_leakage -q ; echo "EXIT:$?"
```

---

## Task 6: Walk-Forward Golden Before/After Contract

**Files:**
- Modify: `scripts/build_walkforward_report.py`
- Modify: `tests/test_build_walkforward_report.py`
- Possibly create a small helper under `scripts/` only if the failing test proves current report validation cannot represent before/after golden runs.

- [ ] **Step 1: Add failing coverage for before/after golden result sets**

Assert the walk-forward report rejects a payload that lacks either pre-batch or post-batch golden results, or document and test that this gate is intentionally validation-only and must be supplied by an external runner.

- [ ] **Step 2: Run targeted test**

```bash
python3 -m pytest tests/test_build_walkforward_report.py::test_build_walkforward_report_requires_before_and_after_golden_results -q ; echo "EXIT:$?"
```

---

## Task 7: Walk-Forward Red-Flag Completeness

**Files:**
- Modify: `tests/test_build_walkforward_report.py`
- Possibly modify: `scripts/build_walkforward_report.py`

- [ ] **Step 1: Add long-unknown red-flag test**

Construct a successful walk-forward summary with a label column exceeding `UNKNOWN_STRETCH_THRESHOLD` and assert `red_flags_detected`.

- [ ] **Step 2: Add one-day flip-flop red-flag test or document why current `_false_switch_count` is the intended implementation**

If current code already counts repeated one-day flip-flops through `_false_switch_count`, add a direct test. If not, add the missing detector first.

- [ ] **Step 3: Run targeted tests**

```bash
python3 -m pytest tests/test_build_walkforward_report.py::test_build_walkforward_report_rejects_long_unknown_stretch tests/test_build_walkforward_report.py::test_build_walkforward_report_rejects_repeated_one_day_flip_flops -q ; echo "EXIT:$?"
```

---

## Task 8: HMM Drift Scope

**Files:**
- Modify: `tests/test_source_hygiene.py` if documenting non-runtime scope is enough.
- Otherwise create implementation/tests in the HMM calibration-review area.

- [ ] **Step 1: Decide and encode scope**

Either implement the 20% state-mean drift / 30% transition-probability review metrics, or add a test-backed decision that these are calibration-review tooling, not runtime engine coverage.

- [ ] **Step 2: Run targeted tests**

```bash
python3 -m pytest tests/test_source_hygiene.py::test_spec_scope_decisions_are_documented -q ; echo "EXIT:$?"
```

---

## Task 9: Vol-Crush Exposure Scope

**Files:**
- Modify strategy-layer tests if this repo owns the downstream strategy response.
- Otherwise modify scope-decision docs/tests.

- [ ] **Step 1: Decide and encode scope**

Either implement/test 50% long-vol reduction over 5-day cooldown, or add a test-backed decision that §5.3 exposure response is outside `regime_detection` runtime scope.

- [ ] **Step 2: Run targeted tests**

```bash
python3 -m pytest tests/test_volatility_state_v2_vol_crush.py tests/test_source_hygiene.py -q ; echo "EXIT:$?"
```

---

## Verification Before Completion

Before claiming the pasted coverage gaps are fully closed:

1. Run every targeted command above.
2. Run the relevant combined groups:

```bash
python3 -m pytest tests/test_fixture_verification.py tests/test_v1_frozen_replay.py tests/test_build_walkforward_report.py tests/test_shadow_qualification.py tests/test_shadow_deadman_check.py tests/test_shadow_replay_check.py tests/test_transition_score_v2.py tests/test_transition_score_v2_weights_outputs.py tests/test_volatility_state_v2_vol_crush.py -q ; echo "EXIT:$?"
```

3. Report any items that remain out of runtime scope as explicit non-goals with code/test-backed documentation.
