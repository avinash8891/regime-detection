# Strict Pyright Coverage Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand strict Pyright coverage from the current four-file slice to full `src/` plus the runtime `scripts/` surface, then fix the resulting diagnostics without weakening strict mode.

**Architecture:** Treat this as a contract-ratchet, not a behavior refactor. First widen the Pyright include set and the readiness guardrail, then remove the biggest classes of diagnostics in layers: public helper exports for cross-script imports, pandas/index narrowing for shared calibration/profile helpers, and finally script-local typing cleanup. Keep CI on the existing `python -m pyright` command the whole time.

**Tech Stack:** Python 3.11, Pyright strict mode, pandas, pytest, GitHub Actions.

---

## Current Evidence

Current repo gate:

```toml
[tool.pyright]
typeCheckingMode = "strict"
include = [
  "src/regime_detection/observability.py",
  "src/regime_detection/loaders.py",
  "scripts/detect_flaky_tests.py",
  "scripts/validate_agents_md.py",
]
```

Current readiness test anchor:

```python
assert {
    "src/regime_detection/observability.py",
    "src/regime_detection/loaders.py",
    "scripts/detect_flaky_tests.py",
    "scripts/validate_agents_md.py",
} <= include_paths
```

Broad baseline sample from `python3 -m pyright src scripts` shows three dominant failure classes:

1. **cross-script private helper imports**
   - `scripts/profile_engine.py`
   - `scripts/audit_layer2_30d.py`
2. **pandas unknown / DatetimeIndex narrowing**
   - `scripts/_v2_calibration_helpers.py`
   - `scripts/profile_engine.py`
   - `scripts/profile_engine_reporting.py`
   - `scripts/approve_group_b_candidate.py`
   - `scripts/audit_layer2_30d.py`
3. **script-local unknowns / unused-private export noise**
   - `scripts/fetch_regime_engine_v1_data.py`
   - helper re-export modules that intentionally expose testing/runtime helpers

## File Structure

- Modify: `pyproject.toml`
  - Expand strict Pyright include coverage from file slice to full runtime ownership.
- Modify: `tests/test_readiness_contracts.py`
  - Replace the four-file assertion with package/runtime-scope assertions.
- Modify: `scripts/profile_engine_reporting.py`
  - Promote private cross-module helper exports to public names and tighten summary helpers.
- Modify: `scripts/profile_engine.py`
  - Import the public reporting helpers and narrow pandas/index types.
- Modify: `scripts/_v2_calibration_helpers.py`
  - Add typed dataframe/index helpers reused by profile/walkforward/shadow scripts.
- Modify: `scripts/audit_layer2_30d.py`
  - Import public helpers instead of private names; tighten evidence typing.
- Modify: `scripts/approve_group_b_candidate.py`
  - Narrow parquet row/match types so strict Pyright can validate the manual-approval flow.
- Modify as needed after rerun: `scripts/fetch_regime_engine_v1_data.py`, `scripts/profile_engine_timers.py`, `scripts/profile_engine_reporting.py`, and any runtime script surfaced by the widened gate.
- Validate with:
  - `tests/test_readiness_contracts.py`
  - `tests/test_approve_group_b_candidate.py`
  - `tests/test_profile_engine.py`
  - `tests/test_profile_engine_reporting_loaders.py`
  - `tests/test_v2_gate_scripts.py`

## Runtime Script Coverage List

Use this exact runtime script set in the widened Pyright include list:

```toml
[
  "scripts/_fetch_regime_engine_v1_args.py",
  "scripts/_v2_calibration_helpers.py",
  "scripts/approve_group_b_candidate.py",
  "scripts/audit_layer2_30d.py",
  "scripts/audit_step1_harness.py",
  "scripts/build_walkforward_report.py",
  "scripts/consolidate_regime_acquisition.py",
  "scripts/detect_flaky_tests.py",
  "scripts/fetch_aaii_sentiment.py",
  "scripts/fetch_regime_engine_v1_data.py",
  "scripts/materialize_constituent_ohlcv_tree.py",
  "scripts/materialize_regime_data.py",
  "scripts/normalize_s3_daily_to_sqlite_layout.py",
  "scripts/profile_engine.py",
  "scripts/profile_engine_reporting.py",
  "scripts/profile_engine_timers.py",
  "scripts/publish_canonical_snapshot.py",
  "scripts/run_historical_walkforward.py",
  "scripts/run_shadow_deadman_check.py",
  "scripts/run_shadow_regime.py",
  "scripts/run_shadow_replay_check.py",
  "scripts/run_v2_calibration.py",
  "scripts/run_v2_shadow_ab_gate.py",
  "scripts/run_v2_walkforward_gate.py",
  "scripts/upload_missing_ohlcv_to_manifest.py",
  "scripts/validate_agents_md.py",
  "scripts/validate_central_bank_text_lexicon.py",
  "scripts/verify_fixtures.py",
]
```

Do not include blanket `scripts/**` globs in the readiness assertion; keep the runtime list deliberate.

---

### Task 1: Widen Pyright Coverage and Update the Readiness Guardrail

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_readiness_contracts.py`

- [ ] **Step 1: Write the failing readiness test**

Replace the current four-file assertion in `tests/test_readiness_contracts.py` with:

```python
def test_pyright_strict_scope_covers_main_runtime_packages_and_scripts() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)

    include_paths = set(payload["tool"]["pyright"]["include"])

    assert {
        "src/regime_detection",
        "src/regime_data_fetch",
        "src/regime_shared",
        "scripts/profile_engine.py",
        "scripts/run_v2_walkforward_gate.py",
        "scripts/run_v2_shadow_ab_gate.py",
        "scripts/fetch_regime_engine_v1_data.py",
        "scripts/materialize_regime_data.py",
        "scripts/validate_agents_md.py",
    } <= include_paths


def test_pyright_strict_scope_is_not_a_tiny_file_allowlist() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)

    include_paths = payload["tool"]["pyright"]["include"]

    assert "src/regime_detection/observability.py" not in include_paths
    assert len(include_paths) >= 10
```

- [ ] **Step 2: Run the readiness test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_readiness_contracts.py -q
```

Expected before config change: FAIL because `pyproject.toml` still uses the four-file allowlist.

- [ ] **Step 3: Widen `[tool.pyright].include`**

In `pyproject.toml`, replace the current include list with:

```toml
[tool.pyright]
typeCheckingMode = "strict"
include = [
  "src/regime_detection",
  "src/regime_data_fetch",
  "src/regime_shared",
  "scripts/_fetch_regime_engine_v1_args.py",
  "scripts/_v2_calibration_helpers.py",
  "scripts/approve_group_b_candidate.py",
  "scripts/audit_layer2_30d.py",
  "scripts/audit_step1_harness.py",
  "scripts/build_walkforward_report.py",
  "scripts/consolidate_regime_acquisition.py",
  "scripts/detect_flaky_tests.py",
  "scripts/fetch_aaii_sentiment.py",
  "scripts/fetch_regime_engine_v1_data.py",
  "scripts/materialize_constituent_ohlcv_tree.py",
  "scripts/materialize_regime_data.py",
  "scripts/normalize_s3_daily_to_sqlite_layout.py",
  "scripts/profile_engine.py",
  "scripts/profile_engine_reporting.py",
  "scripts/profile_engine_timers.py",
  "scripts/publish_canonical_snapshot.py",
  "scripts/run_historical_walkforward.py",
  "scripts/run_shadow_deadman_check.py",
  "scripts/run_shadow_regime.py",
  "scripts/run_shadow_replay_check.py",
  "scripts/run_v2_calibration.py",
  "scripts/run_v2_shadow_ab_gate.py",
  "scripts/run_v2_walkforward_gate.py",
  "scripts/upload_missing_ohlcv_to_manifest.py",
  "scripts/validate_agents_md.py",
  "scripts/validate_central_bank_text_lexicon.py",
  "scripts/verify_fixtures.py",
]
```

- [ ] **Step 4: Run the readiness test to verify the guardrail passes**

Run:

```bash
python3 -m pytest tests/test_readiness_contracts.py -q
```

Expected: PASS.

---

### Task 2: Remove Cross-Module Private Helper Usage

**Files:**
- Modify: `scripts/profile_engine_reporting.py`
- Modify: `scripts/profile_engine.py`
- Modify: `scripts/audit_layer2_30d.py`

- [ ] **Step 1: Write targeted failing tests for public helper names**

Add to `tests/test_readiness_contracts.py`:

```python
def test_profile_engine_reporting_exposes_public_reporting_helpers() -> None:
    module_text = Path("scripts/profile_engine_reporting.py").read_text()

    assert "def format_stage_rows(" in module_text
    assert "def input_status(" in module_text
    assert "def profile_input_seam_values(" in module_text


def test_profile_engine_uses_public_reporting_imports() -> None:
    module_text = Path("scripts/profile_engine.py").read_text()

    assert "from scripts.profile_engine_reporting import (" in module_text
    assert "format_stage_rows" in module_text
    assert "_format_stage_rows" not in module_text
```

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_readiness_contracts.py -q
```

Expected before implementation: FAIL because the helpers are still named/imported with leading underscores.

- [ ] **Step 3: Promote private exports to public names**

In `scripts/profile_engine_reporting.py`, rename the helper definitions:

```python
def input_status(name: str, value: Any) -> str:
    ...


def profile_input_seam_values(inputs: ProfileInputBundle) -> dict[str, Any]:
    ...


def format_stage_rows(
    stage_names: list[str], timer: StageTimer, total: float
) -> list[str]:
    ...
```

In `scripts/profile_engine.py`, replace the imports:

```python
from scripts.profile_engine_reporting import (
    PROFILE_INPUT_SEAM_NAMES,
    build_json_report,
    compact_timeline_rows,
    format_stage_rows,
    input_status,
    profile_input_seam_values,
    reporting_label,
    trailing_v2_status,
    verify_invariants,
    write_json_report,
)
```

and replace callsites like:

```python
for row in format_stage_rows(...):
    print(row)

input_values = profile_input_seam_values(inputs)
print(input_status(name, input_values[name]))
```

In `scripts/audit_layer2_30d.py`, replace private helper imports from `profile_engine` with public wrapper/helper names. Preferred pattern:

```python
from scripts.profile_engine import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONSTITUENT_TREE,
    DEFAULT_DAILY_DIR,
    build_required_sessions,
    load_constituent_ohlcv_from_tree,
    load_optional_aaii_sentiment,
    load_optional_central_bank_text_releases,
    load_optional_cpi_first_release,
    load_event_calendar,
    load_optional_news_sentiment,
)
```

and define these public names in `scripts/profile_engine.py` as simple runtime-supported aliases around the current helper implementations.

- [ ] **Step 4: Run Pyright to verify the private-usage class is removed**

Run:

```bash
python3 -m pyright
```

Expected: the earlier `reportPrivateUsage` errors for `profile_engine.py` and `audit_layer2_30d.py` are gone, with remaining failures now concentrated in pandas narrowing and script-local unknowns.

---

### Task 3: Add Typed pandas / DatetimeIndex Narrowing Helpers

**Files:**
- Modify: `scripts/_v2_calibration_helpers.py`
- Modify: `scripts/profile_engine.py`
- Modify: `scripts/profile_engine_reporting.py`

- [ ] **Step 1: Write failing tests for index/dataframe normalization helpers**

Add to `tests/test_profile_engine_reporting_loaders.py`:

```python
def test_profile_input_status_report_counts_series_rows() -> None:
    from scripts.profile_engine_reporting import input_status_report

    series = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-05-01", "2026-05-02"]))

    report = input_status_report("macro", series)

    assert report["status"] == "present"
    assert report["rows"] == 2


def test_datetime_index_normalizer_returns_datetime_index() -> None:
    from scripts._v2_calibration_helpers import normalize_datetime_index

    raw = pd.Index(["2026-05-01", "2026-05-02"])
    idx = normalize_datetime_index(raw)

    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx.strftime("%Y-%m-%d")) == ["2026-05-01", "2026-05-02"]
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_profile_engine_reporting_loaders.py -q
```

Expected before implementation: FAIL because `input_status_report` / `normalize_datetime_index` do not exist publicly.

- [ ] **Step 3: Add the shared narrowers**

In `scripts/_v2_calibration_helpers.py`, add:

```python
def normalize_datetime_index(index: pd.Index[Any]) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index, errors="raise")).normalize()
    return normalized


def require_dataframe_column(frame: pd.DataFrame, column: str) -> pd.Series[Any]:
    if column not in frame.columns:
        raise KeyError(column)
    return frame[column]
```

In `scripts/profile_engine_reporting.py`, expose the status helper publicly:

```python
def input_status_report(name: str, value: Any) -> dict[str, Any]:
    ...
```

Update existing code to use the new helpers instead of raw ambiguous pandas calls. Examples:

```python
selected_index = normalize_datetime_index(pd.Index(selected_dates))
aligned = series.reindex(selected_index)
```

```python
dates = normalize_datetime_index(pd.Index(pd.to_datetime(dates, errors="coerce").dropna()))
```

- [ ] **Step 4: Apply the narrowers in `profile_engine.py`**

Replace ambiguous index construction patterns like:

```python
observed = pd.DatetimeIndex(
    pd.to_datetime(dates).dt.normalize().sort_values().unique()
)
```

with:

```python
normalized_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
observed = normalize_datetime_index(pd.Index(normalized_dates.sort_values().unique()))
```

Replace dict creation from untyped `load_timer.totals` with:

```python
load_timings: dict[str, float] = {key: float(value) for key, value in load_timer.totals.items()}
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_profile_engine_reporting_loaders.py tests/test_profile_engine.py -q
```

Expected: PASS.

---

### Task 4: Fix Script-Local Unknowns in Approval and Audit Paths

**Files:**
- Modify: `scripts/approve_group_b_candidate.py`
- Modify: `scripts/audit_layer2_30d.py`
- Test: `tests/test_approve_group_b_candidate.py`
- Test: `tests/test_layer2_audit.py`

- [ ] **Step 1: Write a failing approval-path typing regression test**

Add to `tests/test_approve_group_b_candidate.py`:

```python
def test_candidate_lookup_uses_single_typed_row(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "abc",
                "event_type": "budget",
                "promotion_outcome": "withhold",
                "requires_manual_review": True,
                "source_count": 2,
                "date": "2026-05-01",
                "importance": "high",
            }
        ]
    ).to_parquet(candidates_path)

    frame = pd.read_parquet(candidates_path)
    matches = frame.loc[frame["candidate_id"] == "abc"]

    assert len(matches) == 1
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_approve_group_b_candidate.py tests/test_layer2_audit.py -q
```

Expected: current tests may pass functionally; keep this step as the pre-change baseline before the typing cleanup.

- [ ] **Step 3: Narrow the approval candidate row explicitly**

In `scripts/approve_group_b_candidate.py`, replace the untyped row access with:

```python
matches = candidates.loc[candidates["candidate_id"] == args.candidate_id].copy()
if matches.empty:
    raise SystemExit(f"candidate_id not found: {args.candidate_id}")

row = matches.iloc[0]
event_type = str(row["event_type"])
promotion_outcome = str(row.get("promotion_outcome", ""))
requires_manual_review = bool(row.get("requires_manual_review"))
source_count_value = row["source_count"] if "source_count" in row else None
source_count = int(source_count_value) if pd.notna(source_count_value) else 1
```

Then pass only narrowed variables into `append_approval_record`.

- [ ] **Step 4: Narrow `audit_layer2_30d.py` evidence dicts**

Replace loose evidence access patterns like:

```python
evidence = axis_output.evidence or {}
for metric, value in evidence.get("rule_evidence", {}).items():
    ...
```

with:

```python
raw_evidence = axis_output.evidence
evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
rule_evidence_obj = evidence.get("rule_evidence", {})
rule_evidence: dict[str, Any] = (
    rule_evidence_obj if isinstance(rule_evidence_obj, dict) else {}
)
for metric, value in rule_evidence.items():
    ...
```

- [ ] **Step 5: Run focused tests again**

Run:

```bash
python3 -m pytest tests/test_approve_group_b_candidate.py tests/test_layer2_audit.py -q
```

Expected: PASS.

---

### Task 5: Finish the Broad Pyright Cleanup and Lock the Gate

**Files:**
- Modify as surfaced by rerun: `scripts/fetch_regime_engine_v1_data.py` and any remaining runtime script with strict failures
- Validate: `pyproject.toml`, `tests/test_readiness_contracts.py`

- [ ] **Step 1: Run the full strict gate and capture remaining errors**

Run:

```bash
python3 -m pyright
```

Expected at this point: remaining failures should be a smaller tail in a few scripts, not a repo-wide flood.

- [ ] **Step 2: Fix the remaining script-local diagnostics without suppressions**

Apply the same patterns already introduced:

```python
typed_value = str(raw_value) if raw_value is not None else ""
typed_items: list[str] = [str(item) for item in raw_items]
```

```python
if not isinstance(candidate_modes_obj, list):
    raise ValueError("candidate modes must be a list")
candidate_modes = [str(item) for item in candidate_modes_obj]
```

```python
def public_helper(...) -> ReturnType:
    return _private_helper(...)
```

Do not add repo-wide `reportUnknown* = false` settings.

- [ ] **Step 3: Run the final local validator set**

Run:

```bash
python3 -m pyright
python3 -m pytest tests/test_readiness_contracts.py tests/test_approve_group_b_candidate.py tests/test_profile_engine.py tests/test_profile_engine_reporting_loaders.py tests/test_v2_gate_scripts.py -q
```

Expected:

```text
0 errors, 0 warnings, 0 informations
```

and pytest exits `0`.

- [ ] **Step 4: Commit**

Run:

```bash
git add pyproject.toml tests/test_readiness_contracts.py scripts/_v2_calibration_helpers.py scripts/approve_group_b_candidate.py scripts/audit_layer2_30d.py scripts/profile_engine.py scripts/profile_engine_reporting.py tests/test_approve_group_b_candidate.py tests/test_profile_engine.py tests/test_profile_engine_reporting_loaders.py tests/test_v2_gate_scripts.py
git commit -m "fix: expand strict pyright runtime coverage"
```

Expected: commit succeeds with the strict-coverage ratchet and targeted typing cleanup.

---

## Self-Review

- Spec coverage: covered config widening, readiness enforcement, narrow exclusions bar, and runtime typing cleanup.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: plan uses public helper names consistently (`format_stage_rows`, `input_status`, `profile_input_seam_values`, `normalize_datetime_index`).
