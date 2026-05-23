# Rich GPR Geopolitical Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the richer daily Caldara-Iacoviello GPR workbook fields to generate better approval-gated geopolitical candidates without auto-promoting geopolitical events.

**Architecture:** `GPRSignalGenerator` is the Group B GPR/AI-GPR source entry point and remains separate from GDELT, ACLED, UCDP, and HDX generators. Expand `parse_gpr_table` to retain headline, acts, threats, moving averages, article count, and optional event text; score spikes from headline/acts/threats and encode the result into candidate subtype, importance, confidence, and review evidence. The event-calendar renderer remains unchanged: `geopolitical_event` rows still reach `us_events.yaml` only through the approval overlay.

**Tech Stack:** Python, pandas, existing `EventCandidate` dataclass, existing RTK/pytest test harness.

---

### Task 1: Parse Daily GPR Components

**Files:**
- Modify: `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py`
- Test: `tests/test_event_source_group_b.py`

- [ ] **Step 1: Write failing parser coverage**

Add a test that passes a CSV shaped like the daily workbook:

```python
def test_parse_gpr_table_keeps_daily_components() -> None:
    payload = """DAY,N10D,GPRD,GPRD_ACT,GPRD_THREAT,date,GPRD_MA30,GPRD_MA7,event
20220223,300,100,90,110,2022-02-23,95,98,
20220224,900,500,650,420,2022-02-24,130,180,Russia invasion of Ukraine
"""

    frame = parse_gpr_table(payload)

    assert frame.to_dict(orient="records") == [
        {
            "date": dt.date(2022, 2, 23),
            "gpr": 100,
            "gpr_act": 90,
            "gpr_threat": 110,
            "gpr_ma7": 98,
            "gpr_ma30": 95,
            "article_count": 300,
            "event": "",
        },
        {
            "date": dt.date(2022, 2, 24),
            "gpr": 500,
            "gpr_act": 650,
            "gpr_threat": 420,
            "gpr_ma7": 180,
            "gpr_ma30": 130,
            "article_count": 900,
            "event": "Russia invasion of Ukraine",
        },
    ]
```

- [ ] **Step 2: Run the failing parser test**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_parse_gpr_table_keeps_daily_components -q; echo "EXIT:$?"
```

Expected: fails because `parse_gpr_table` currently returns only `date` and `gpr`.

- [ ] **Step 3: Implement component parsing**

Update `parse_gpr_table` so it returns stable columns:

```python
date, gpr, gpr_act, gpr_threat, gpr_ma7, gpr_ma30, article_count, event
```

For missing optional columns, fill numeric fields with `pd.NA` and `event` with `""`. Required columns remain date plus headline GPR (`gpr`, `GPRD`, or `gpr_daily`).

- [ ] **Step 4: Run the parser test**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_parse_gpr_table_keeps_daily_components -q; echo "EXIT:$?"
```

Expected: `EXIT:0`.

### Task 2: Classify GPR Spike Evidence

**Files:**
- Modify: `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py`
- Test: `tests/test_event_source_group_b.py`

- [ ] **Step 1: Write failing spike classification coverage**

Add tests for:

```python
def test_detect_gpr_spikes_classifies_acts_and_threats() -> None:
    frame = parse_gpr_table("""date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,700,125,220,150,900
""")

    rows = detect_gpr_spikes(frame, min_history_days=3, stddev_threshold=2.0)

    assert rows == [
        {
            "date": dt.date(2022, 2, 24),
            "value": 500.0,
            "threshold": pytest.approx(102.21895141649746),
            "act_value": 700.0,
            "threat_value": 125.0,
            "ma7": 220.0,
            "ma30": 150.0,
            "article_count": 900.0,
            "event": "",
            "spike_components": ("headline", "acts", "persistent_7d", "persistent_30d"),
            "dominant_component": "acts",
        }
    ]
```

- [ ] **Step 2: Run the failing spike test**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_detect_gpr_spikes_classifies_acts_and_threats -q; echo "EXIT:$?"
```

Expected: fails because spike rows currently only include `date`, `value`, and `threshold`.

- [ ] **Step 3: Implement spike classification**

Keep the existing headline threshold logic. Add component evidence:

- `headline` when `gpr > threshold`
- `acts` when `gpr_act > threshold`
- `threats` when `gpr_threat > threshold`
- `persistent_7d` when `gpr_ma7 > threshold`
- `persistent_30d` when `gpr_ma30 > threshold`

Only emit a row when headline GPR exceeds the threshold. Set `dominant_component` to `acts` or `threats` if that component is the largest available component and also above threshold; otherwise use `headline`.

- [ ] **Step 4: Run the spike test**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_detect_gpr_spikes_classifies_acts_and_threats -q; echo "EXIT:$?"
```

Expected: `EXIT:0`.

### Task 3: Enrich GPR Candidates

**Files:**
- Modify: `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py`
- Test: `tests/test_event_source_group_b.py`

- [ ] **Step 1: Write failing candidate evidence coverage**

Update existing GPR generator tests so the GPR candidate has:

```python
event_subtype == "gpr_acts_spike"
importance == "high"
confidence == "high"
raw_title == "GPR acts-driven geopolitical risk spike"
raw_snippet == (
    "GPR daily value 500.00 exceeded trailing threshold 102.22; "
    "components=headline,acts,persistent_7d,persistent_30d; "
    "acts=700.00; threats=125.00; ma7=220.00; ma30=150.00; articles=900."
)
```

Add a second test for a threat-dominant spike and expect `event_subtype == "gpr_threats_spike"`.

- [ ] **Step 2: Run the failing candidate tests**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_gpr_gdelt_generator_flags_real_geopolitical_spike_date tests/test_event_source_group_b.py::test_gpr_generator_fetches_gdelt_daily_exports_for_spike_dates -q; echo "EXIT:$?"
```

Expected: fails because candidate evidence still uses the old generic GPR text.

- [ ] **Step 3: Implement candidate enrichment**

Change `_gpr_candidate` only. Do not change the `EventCandidate` schema. Encode review evidence into existing fields:

- `event_subtype`: `gpr_acts_spike`, `gpr_threats_spike`, or `gpr_headline_spike`
- `importance`: `high` when at least three components are present or `article_count >= 500`; otherwise `medium`
- `confidence`: `high` when both headline and one of acts/threats are present and `article_count >= 500`; otherwise `medium`
- `raw_title`: use `event` text when non-empty; otherwise component-driven title
- `raw_snippet`: include headline threshold, component list, acts, threats, moving averages, and article count when available

- [ ] **Step 4: Run the candidate tests**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py::test_gpr_gdelt_generator_flags_real_geopolitical_spike_date tests/test_event_source_group_b.py::test_gpr_generator_fetches_gdelt_daily_exports_for_spike_dates -q; echo "EXIT:$?"
```

Expected: `EXIT:0`.

### Task 4: Update Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-05-14-layer2d-event-calendar-group-b-design.md`
- Modify: `docs/regime_engine_v2_spec.md`
- Optionally modify: `README.md` if the event-calendar section needs one clarifying sentence.

- [ ] **Step 1: Update Group B spec**

In §6.3 and §8, replace “GPR value + trailing-window threshold” with the richer parser contract:

```markdown
GPR daily parser retains `GPRD`, `GPRD_ACT`, `GPRD_THREAT`, `GPRD_MA7`, `GPRD_MA30`, `N10D`, and optional `event` text. The detector still requires a headline `GPRD` spike before emitting a candidate, then uses acts/threats/persistence/article-count evidence to set candidate subtype, confidence, importance, and review snippets. GPR evidence never auto-promotes a `geopolitical_event`.
```

- [ ] **Step 2: Update v2 spec mention**

Clarify that GPR is quantitative evidence for approval-gated candidates, not a qualitative event source and not an automatic event-calendar renderer.

- [ ] **Step 3: Grep docs for stale wording**

Run:

```bash
rg -n "GPR value \\+ trailing-window|date`, GPR value|GPR daily-index spikes" docs README.md
```

Expected: no stale statement that says only headline GPR is retained.

### Task 5: Verification

**Files:**
- No source files unless tests reveal a real bug.

- [ ] **Step 1: Run focused Group B tests**

Run:

```bash
rtk pytest tests/test_event_source_group_b.py tests/test_event_source_group_b_conflict_budget.py::test_gpr_gdelt_generator_includes_hdx_conflict_candidates -q; echo "EXIT:$?"
```

Expected: `EXIT:0`.

- [ ] **Step 2: Run event-calendar approval tests**

Run:

```bash
rtk pytest tests/test_event_sources_orchestrator.py tests/test_event_calendar.py::test_group_b_approved_geopolitical_event_renders_with_overlay -q; echo "EXIT:$?"
```

Expected: `EXIT:0`.

- [ ] **Step 3: Review diff for approval-gate integrity**

Run:

```bash
git diff -- src/regime_data_fetch/event_sources/validators_gpr_gdelt.py src/regime_data_fetch/event_sources/orchestrator.py src/regime_data_fetch/event_calendar.py
```

Expected: only `validators_gpr_gdelt.py` changed among those source files; no change to promotion logic.
