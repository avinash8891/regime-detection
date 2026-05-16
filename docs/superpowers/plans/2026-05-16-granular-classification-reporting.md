# Granular Classification Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop metrics/reports from showing `unknown` as a counted outcome except for an explicit catch-all axis label; show `no_rule_fired`, `data_unavailable`, `stale_data`, `insufficient_history`, or `not_wired` instead.

**Architecture:** Keep classifier labels backward-compatible inside core rule engines for now, but make all reporting/audit/timeline summaries use a display outcome derived from `classification_status`. `active_label` remains the model's raw axis label; `reporting_label` / `display_label` becomes the only value used in metric splits. This is the lowest-risk long-term path because it separates classifier semantics from operator-facing status reporting and can be enforced with tests over all current Layer 1/2/3 output fields.

**Tech Stack:** Python, Pydantic models, pandas audit artifacts, pytest.

---

## Current Evidence

Current 30-day reconstruction showed this split:

```text
credit_funding_state_proxy
active_label: unknown 23, credit_calm 7
classification_status: no_rule_fired 23, classified 7
data_quality.status: ok 30
```

So the bug is not data calculation. The bug is that reports still count `active_label` directly when `active_label == "unknown"`.

Important code anchors:

- `src/regime_detection/models.py:30` derives granular `classification_status`.
- `src/regime_detection/credit_funding.py:676` still falls through to `unknown` when no credit rule fires.
- `scripts/audit_layer2_30d.py` currently writes active-label splits and exposes the mismatch.
- `scripts/profile_engine_30d.py` currently prints active-label daily lines/splits.

## File Structure

- Modify `src/regime_detection/models.py`
  - Add one canonical helper/property for operator-facing outcome names.
  - Responsibility: every axis output exposes the same display/reporting outcome.
- Modify `scripts/audit_layer2_30d.py`
  - Use the reporting outcome for label splits.
  - Preserve raw `active_label` counts in a separate debug section.
- Modify `scripts/profile_engine_30d.py`
  - Print/report reporting outcomes instead of bare `active_label` for metric splits.
- Modify `scripts/run_v2_walkforward_gate.py`
  - Use reporting outcomes for activation/split metrics.
- Modify `scripts/run_v2_shadow_ab_gate.py`
  - Same reporting outcome behavior as walk-forward.
- Modify `src/regime_detection/comparison.py`
  - Ensure comparison summaries do not surface `unknown` as the operator-facing value when granular status exists.
- Add/modify tests:
  - `tests/test_classification_status.py`
  - `tests/test_layer2_audit.py`
  - `tests/test_profile_engine_30d.py`
  - `tests/test_v2_gate_scripts.py`

## Reporting Contract

For any axis output with `active_label`, `classification_status`, and `data_quality`:

```text
if classification_status == "classified":
    reporting_label = active_label
else:
    reporting_label = classification_status
```

Examples:

```text
active_label=unknown, data_quality=ok -> classification_status=no_rule_fired -> reporting_label=no_rule_fired
active_label=unknown, data_quality=stale_data -> classification_status=stale_data -> reporting_label=stale_data
active_label=unknown, data_quality=insufficient_history -> classification_status=insufficient_history -> reporting_label=insufficient_history
active_label=unknown, data_quality=insufficient_data -> classification_status=data_unavailable -> reporting_label=data_unavailable
active_label=credit_calm, data_quality=ok -> classification_status=classified -> reporting_label=credit_calm
```

Only a future explicit catch-all label may display as `unknown`. Until such a label is deliberately modeled, reports should not count `unknown` as an outcome when a granular status exists.

---

### Task 1: Add Canonical Reporting Outcome Helper

**Files:**
- Modify: `src/regime_detection/models.py`
- Test: `tests/test_classification_status.py`

- [ ] **Step 1: Write failing tests**

Add tests to `tests/test_classification_status.py`:

```python
def test_reporting_label_uses_no_rule_fired_for_unknown_with_ok_quality() -> None:
    out = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "no_rule_fired"
    assert out.reporting_label == "no_rule_fired"


def test_reporting_label_preserves_classified_active_label() -> None:
    out = AxisOutput(
        raw_label="credit_calm",
        stable_label="credit_calm",
        active_label="credit_calm",
        evidence={},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "classified"
    assert out.reporting_label == "credit_calm"


def test_reporting_label_uses_data_unavailable_for_insufficient_data() -> None:
    out = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "missing_required_series"},
        data_quality=DataQuality(
            status="insufficient_data",
            freshness_days=None,
            completeness=0.0,
            reason="missing_required_series",
        ),
    )

    assert out.classification_status == "data_unavailable"
    assert out.reporting_label == "data_unavailable"
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python3 -m pytest tests/test_classification_status.py -q
```

Expected before implementation: failure because `AxisOutput.reporting_label` does not exist.

- [ ] **Step 3: Implement `reporting_label`**

In `src/regime_detection/models.py`, add this property to `AxisOutput`:

```python
    @property
    def reporting_label(self) -> str:
        if self.classification_status == "classified":
            return self.active_label
        return self.classification_status or "not_wired"
```

Add equivalent property to `VolumeLiquidityOutput`:

```python
    @property
    def reporting_label(self) -> str:
        if self.classification_status == "classified":
            return self.label
        return self.classification_status or "not_wired"
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_classification_status.py -q
```

Expected: all tests pass.

---

### Task 2: Update Layer 2 Audit to Report Granular Outcomes

**Files:**
- Modify: `scripts/audit_layer2_30d.py`
- Modify: `tests/test_layer2_audit.py`

- [ ] **Step 1: Write failing audit test**

Add to `tests/test_layer2_audit.py`:

```python
def test_layer2_label_summary_uses_reporting_label_for_no_rule_fired() -> None:
    selected = [dt.date(2026, 5, 1)]
    output = SimpleNamespace(
        active_label="unknown",
        raw_label="unknown",
        stable_label="unknown",
        reporting_label="no_rule_fired",
        classification_status="no_rule_fired",
        data_quality=SimpleNamespace(status="ok", reason=None),
        evidence={"rule_evidence": {"hy_spread_percentile_504d": 0.25}},
    )
    axis_bundle = SimpleNamespace(
        monetary_pressure_state=None,
        credit_funding=None,
        credit_funding_proxy={selected[0]: output},
        credit_funding_effective=None,
        inflation_growth=None,
    )

    summary = build_label_rule_summary(
        axis_bundle=axis_bundle,
        selected_dates=selected,
        missing_constituent_files=0,
    )

    proxy = summary["axes"]["credit_funding_state_proxy"]
    assert proxy["reported"] == {"no_rule_fired": 1}
    assert proxy["active"] == {"unknown": 1}
    assert proxy["classification_status"] == {"no_rule_fired": 1}
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python3 -m pytest tests/test_layer2_audit.py::test_layer2_label_summary_uses_reporting_label_for_no_rule_fired -q
```

Expected before implementation: failure because summary has no `reported` bucket.

- [ ] **Step 3: Implement reported bucket**

In `scripts/audit_layer2_30d.py`, inside `_summarize_output_series`, add:

```python
    reported: Counter[str | None] = Counter()
```

For `output is None`:

```python
            reported["not_wired"] += 1
```

For real outputs:

```python
        reported[getattr(output, "reporting_label", output.active_label)] += 1
```

Return it in `summary`:

```python
        "reported": _json_counter(reported),
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_layer2_audit.py -q
```

Expected: pass.

- [ ] **Step 5: Regenerate Layer 2 audit**

Run:

```bash
python3 scripts/audit_layer2_30d.py \
  --lookback-days 30 \
  --daily-dir data/raw/daily_ohlcv \
  --constituent-tree data/raw/daily_ohlcv_762 \
  --macro-parquet data/raw/macro/fred_macro_series.parquet \
  --pit-parquet data/raw/pit_constituents/sp500_ticker_intervals.parquet \
  --pmi-path data/raw/pmi/us_ism_pmi.parquet \
  --event-calendar configs/events/us_events.yaml \
  --aaii-sentiment-parquet data/raw/sentiment/aaii_sentiment.parquet \
  --news-sentiment-parquet data/raw/news_sentiment/sf_fed_news_sentiment.parquet \
  --fomc-minutes-parquet data/raw/fomc_minutes/fomc_minutes.parquet \
  --powell-speeches-parquet data/raw/powell_speeches/powell_speeches.parquet \
  --cpi-vintages-parquet data/raw/macro_vintages/cpi_all_items_vintages.parquet \
  --stamp 20260516
```

Expected proof:

```text
credit_funding_state_proxy.reported = {"credit_calm": 7, "no_rule_fired": 23}
credit_funding_state_proxy.active = {"credit_calm": 7, "unknown": 23}
```

---

### Task 3: Update 30-Day Profile Reporting

**Files:**
- Modify: `scripts/profile_engine_30d.py`
- Test: `tests/test_profile_engine_30d.py`

- [ ] **Step 1: Locate active-label reporting**

Run:

```bash
rg -n "active_label|raw_label|stable_label|credit_funding_state_proxy|label split|Counter" scripts/profile_engine_30d.py tests/test_profile_engine_30d.py
```

Expected: identify every table/print path using `active_label`.

- [ ] **Step 2: Add helper test**

If `profile_engine_30d.py` already has a summary helper, test it directly. If not, add a small helper `_reporting_label(output)` and test:

```python
def test_profile_reporting_label_uses_classification_status_for_unknown() -> None:
    output = SimpleNamespace(active_label="unknown", classification_status="no_rule_fired")

    assert profile_engine_30d._reporting_label(output) == "no_rule_fired"
```

- [ ] **Step 3: Implement helper**

Add to `scripts/profile_engine_30d.py`:

```python
def _reporting_label(output: Any) -> str | None:
    if output is None:
        return None
    return getattr(output, "reporting_label", None) or (
        output.active_label
        if getattr(output, "classification_status", "classified") == "classified"
        else output.classification_status
    )
```

Replace operator-facing split counts from:

```python
out.credit_funding_state_proxy.active_label
```

to:

```python
_reporting_label(out.credit_funding_state_proxy)
```

Do not remove raw/active debug fields if they are explicitly labeled as raw debug.

- [ ] **Step 4: Run profile tests**

Run:

```bash
python3 -m pytest tests/test_profile_engine_30d.py -q
```

Expected: pass.

---

### Task 4: Update V2 Gate Reports

**Files:**
- Modify: `scripts/run_v2_walkforward_gate.py`
- Modify: `scripts/run_v2_shadow_ab_gate.py`
- Test: `tests/test_v2_gate_scripts.py`

- [ ] **Step 1: Write test expectation**

Extend the existing fixture tests to assert that a no-rule output would be counted by granular status if the helper is given an output with:

```python
active_label="unknown"
classification_status="no_rule_fired"
```

If there is no helper, add one and unit test it in `tests/test_v2_gate_scripts.py`:

```python
def test_gate_reporting_label_uses_granular_status() -> None:
    output = SimpleNamespace(active_label="unknown", classification_status="no_rule_fired")

    assert run_v2_walkforward_gate._reporting_label(output) == "no_rule_fired"
    assert run_v2_shadow_ab_gate._reporting_label(output) == "no_rule_fired"
```

- [ ] **Step 2: Implement shared or duplicated tiny helper**

Preferred: add the same helper to both scripts to avoid a new import cycle:

```python
def _reporting_label(output: Any) -> str | None:
    if output is None:
        return None
    return getattr(output, "reporting_label", None) or (
        output.active_label
        if getattr(output, "classification_status", "classified") == "classified"
        else output.classification_status
    )
```

Use this helper for any non-unknown activation/split counts. Replace checks like:

```python
lbl = (output.credit_funding_effective_state.active_label or "").lower()
if lbl and lbl != "unknown":
```

with:

```python
lbl = (_reporting_label(output.credit_funding_effective_state) or "").lower()
if lbl and lbl not in {"not_wired", "data_unavailable", "stale_data", "insufficient_history"}:
```

For display/split reports, count `no_rule_fired` explicitly rather than discarding it.

- [ ] **Step 3: Run gate tests**

Run:

```bash
python3 -m pytest tests/test_v2_gate_scripts.py -q
```

Expected: pass.

---

### Task 5: Update Comparison Summaries

**Files:**
- Modify: `src/regime_detection/comparison.py`
- Test: `tests/test_v2_comparison.py`

- [ ] **Step 1: Search comparison active-label usage**

Run:

```bash
rg -n "active_label|classification_status|unknown|label" src/regime_detection/comparison.py tests/test_v2_comparison.py
```

- [ ] **Step 2: Add test for no-rule display**

Add a comparison test that builds an output with `active_label="unknown"` and `classification_status="no_rule_fired"` and asserts the comparison summary emits `no_rule_fired`, not `unknown`.

- [ ] **Step 3: Implement reporting label in comparison**

Add a local helper or import the canonical model property:

```python
def _axis_reporting_label(output: Any) -> str | None:
    if output is None:
        return None
    return getattr(output, "reporting_label", None) or (
        output.active_label
        if getattr(output, "classification_status", "classified") == "classified"
        else output.classification_status
    )
```

Use it for all operator-facing summaries.

- [ ] **Step 4: Run comparison tests**

Run:

```bash
python3 -m pytest tests/test_v2_comparison.py -q
```

Expected: pass.

---

### Task 6: End-to-End Proof Across All Layers

**Files:**
- No required code file.
- Artifacts: `.context/profile_engine_30d_*.txt`, `.context/layer2_label_rule_summary_20260516.json`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m pytest \
  tests/test_classification_status.py \
  tests/test_layer2_audit.py \
  tests/test_profile_engine_30d.py \
  tests/test_v2_gate_scripts.py \
  tests/test_v2_comparison.py \
  tests/test_credit_funding.py \
  tests/test_inflation_growth.py \
  -q
```

Expected: all pass.

- [ ] **Step 2: Run 30-day profile**

Run:

```bash
python3 scripts/profile_engine_30d.py \
  --lookback-days 30 \
  --daily-dir data/raw/daily_ohlcv \
  --constituent-tree data/raw/daily_ohlcv_762 \
  --macro-parquet data/raw/macro/fred_macro_series.parquet \
  --pit-parquet data/raw/pit_constituents/sp500_ticker_intervals.parquet \
  --pmi-path data/raw/pmi/us_ism_pmi.parquet \
  --event-calendar configs/events/us_events.yaml \
  --aaii-sentiment-parquet data/raw/sentiment/aaii_sentiment.parquet \
  --news-sentiment-parquet data/raw/news_sentiment/sf_fed_news_sentiment.parquet \
  --fomc-minutes-parquet data/raw/fomc_minutes/fomc_minutes.parquet \
  --powell-speeches-parquet data/raw/powell_speeches/powell_speeches.parquet \
  --cpi-vintages-parquet data/raw/macro_vintages/cpi_all_items_vintages.parquet \
  2>&1 | tee .context/profile_engine_30d_20260516_granular_reporting.txt
```

Expected: runner exits 0 and reports no verification issues.

- [ ] **Step 3: Regenerate Layer 2 audit**

Run the `scripts/audit_layer2_30d.py` command from Task 2 Step 5.

Expected:

```text
credit_funding_state_proxy.reported = {"credit_calm": 7, "no_rule_fired": 23}
```

- [ ] **Step 4: Prove no operator-facing unknown remains**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path(".context/layer2_label_rule_summary_20260516.json").read_text())
bad = {}
for axis, node in summary["axes"].items():
    reported = node.get("reported", {})
    if "unknown" in reported:
        bad[axis] = reported
print(bad)
raise SystemExit(1 if bad else 0)
PY
```

Expected output:

```text
{}
```

- [ ] **Step 5: Prove raw debug still preserves source truth**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path(".context/layer2_label_rule_summary_20260516.json").read_text())
proxy = summary["axes"]["credit_funding_state_proxy"]
print("active", proxy["active"])
print("reported", proxy["reported"])
print("classification_status", proxy["classification_status"])
PY
```

Expected output shape:

```text
active {'credit_calm': 7, 'unknown': 23}
reported {'credit_calm': 7, 'no_rule_fired': 23}
classification_status {'classified': 7, 'no_rule_fired': 23}
```

This proves the implementation did not hide classifier internals; it fixed operator-facing metrics.

- [ ] **Step 6: Whitespace and status**

Run:

```bash
git diff --check
git status --short --untracked-files=all
```

Expected: no whitespace errors. Status shows only intended files.

---

## Self-Review

Spec coverage:

- No operator-facing metrics should show catch-all `unknown`: covered by Tasks 2-6.
- Granular outcomes should be used: covered by `reporting_label` and report updates.
- Proof across all layers: covered by reconstructed 30-day profile and Layer 2 audit proof. For Layer 1/3 fields, profile summary must use `reporting_label`; if any field lacks `classification_status`, Task 1 exposes that gap.

Known deliberate boundary:

- This plan does not remove `unknown` from core classifier enums in the same slice. That would be a deeper model migration across V1 replay fixtures, rule precedence, hysteresis configs, and frozen contracts. The immediate target is reports/metrics. The model still keeps raw `active_label` for internal/debug truth, while `reporting_label` becomes the operator-facing outcome.

No placeholders:

- Every task names exact files, test commands, and expected proof.
