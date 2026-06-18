# Plan 002: Replace production assert guards with explicit runtime errors

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving on. If a STOP condition occurs, stop and report.
>
> **Drift check (run first)**: `git diff --stat 7f8608fe..HEAD -- src/regime_data_fetch/acquisition_store.py src/regime_detection/transition_risk_series.py scripts/_v2_calibration_helpers.py scripts/publish_canonical_snapshot.py scripts/upload_missing_ohlcv_to_manifest.py`
> If any in-scope file changed, compare the current excerpts below to live code before editing.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: security
- **Planned at**: commit `7f8608fe`, 2026-06-18
- **Status**: DONE

## Why this matters

Production code has `assert` guards for runtime invariants. Python removes asserts under `-O`, which can turn clear invariant failures into later `AttributeError`s or wrong-path behavior. The fix is boring: replace each production assert with an explicit `if ...: raise ...`.

## Current state

`rg -n "^\\s*assert\\b" src scripts` reports these production asserts:

- `scripts/publish_canonical_snapshot.py:638` — `assert canon is not None`
- `scripts/upload_missing_ohlcv_to_manifest.py:177` — `assert store is not None`
- `scripts/_v2_calibration_helpers.py:47` — `assert spec.default_relpath is not None`
- `scripts/_v2_calibration_helpers.py:126` — `assert label is not None`
- `src/regime_data_fetch/acquisition_store.py:806` and `:815` — `assert self.artifact_store is not None`
- `src/regime_detection/transition_risk_series.py:172-176` — post-validation type-narrowing asserts after the explicit `missing` RuntimeError block

`python3 -m ruff check . --select S` currently emits many test `assert` warnings plus these production `S101` hits. Tests are not in scope.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Production assert scan | `rg -n "^\\s*assert\\b" src scripts` | no real assert statements in `src` or `scripts` |
| Ruff security assert check | `python3 -m ruff check src scripts --select S101` | exit 0 |
| Targeted tests | `python3 -m pytest tests/test_acquisition_artifact_ledger.py tests/test_transition_risk.py tests/test_v2_calibration_helpers.py -q ; echo "EXIT:$?"` | `EXIT:0` |

## Scope

**In scope**
- `src/regime_data_fetch/acquisition_store.py`
- `src/regime_detection/transition_risk_series.py`
- `scripts/_v2_calibration_helpers.py`
- `scripts/publish_canonical_snapshot.py`
- `scripts/upload_missing_ohlcv_to_manifest.py`

**Out of scope**
- Test-file asserts
- Broad Ruff security cleanup beyond `S101` in `src` and `scripts`
- Refactoring shared helpers

## Steps

### Step 1: Replace each assert with the nearest explicit error

Use minimal local checks:

- For impossible internal state, raise `RuntimeError` with the missing field named.
- For a violated helper contract like missing `default_relpath`, raise `RuntimeError("pmi_path manifest input has no default_relpath")`.
- In `transition_risk_series.py`, keep the existing aggregated `missing` guard, then replace the five type-narrowing asserts with `if ... is None: raise RuntimeError(...)` only if Pyright still needs narrowing. If no extra checks are needed after the existing guard, delete the asserts.

**Verify**: `rg -n "^\\s*assert\\b" src scripts` -> no real assert statements in source/scripts.

### Step 2: Run focused checks

Run the three commands in "Commands you will need". If a targeted test file is absent, replace it with the closest existing test for the touched module and record that in the commit message.

**Verify**: all commands return exit 0.

## Done criteria

- [x] `python3 -m ruff check src scripts --select S101` exits 0.
- [x] No production `assert` statements remain under `src` or `scripts`.
- [x] Targeted tests pass.
- [x] No behavior changes except clearer invariant failures.

## STOP conditions

- Removing an assert requires changing public output shape or CLI behavior.
- Pyright cannot narrow after the replacement without a larger refactor.
- Ruff reports non-assert security findings and fixing them would expand scope.

## Maintenance notes

Keep pytest asserts. This plan is only about production code that can run under optimized Python.
