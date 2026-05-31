# Trend Character Vectorized Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen regression coverage after making `raw_label_for_day` a wrapper over `build_raw_outputs`, so vectorized behavior is tested against expected labels instead of only against the scalar wrapper.

**Architecture:** Production code stays unchanged. Tests will assert expected outputs directly from `build_raw_outputs` for V1 labels, V2 labels, and `allow_v2_labels=False` fallback behavior. Existing scalar parity tests can remain as smoke coverage, but they are no longer the main proof because scalar and vectorized paths now share one implementation.

**Tech Stack:** Python, pytest, pandas, numpy, existing `TrendCharacterFeatures`, `build_raw_outputs`, and `_synthetic_features` test helpers.

---

## File Structure

- Modify: `tests/test_trend_character.py`
  - Add direct expected-label assertions to the existing V1 vectorized raw-output test.
  - Keep scalar parity as a secondary guard.
- Modify: `tests/test_trend_character_v2_labels.py`
  - Add one vectorized V2 expected-label test using synthetic rows.
  - Add one vectorized `allow_v2_labels=False` fallback test for V2-only predicate rows.
- No production files should change.

## Evidence-Checked Task List

- [x] V1 vectorized raw-output test asserts direct expected labels from `build_raw_outputs`.
  Evidence: `tests/test_trend_character.py::test_trend_character_vectorized_raw_outputs_match_scalar_rules`; `python3 -m pytest ... -q ; echo "EXIT:$?"` returned `EXIT:0`.
- [x] V1 vectorized raw-output test still checks scalar wrapper parity after direct expected labels.
  Evidence: same test keeps scalar comparison after direct expected label/evidence assertions; targeted pytest returned `EXIT:0`.
- [x] V2 vectorized raw-output test asserts direct expected labels for reachable V2-default outputs: `breakout_expansion`, `recovery_attempt`, `trending`, `mild_trend`, `range_bound`, `chop`, `volatile_chop`, and `unknown`.
  Evidence: `tests/test_trend_character_v2_labels.py::test_build_raw_outputs_direct_expected_v2_labels`; `python3 -m pytest ... -q ; echo "EXIT:$?"` returned `EXIT:0`.
- [x] V2 vectorized fallback test asserts `allow_v2_labels=False` suppresses V2-only labels to V1 outcomes, including `transition`.
  Evidence: `tests/test_trend_character_v2_labels.py::test_build_raw_outputs_v1_path_suppresses_v2_only_labels_vectorized`; `python3 -m pytest ... -q ; echo "EXIT:$?"` returned `EXIT:0`.
- [x] Targeted tests pass with explicit `EXIT:0`.
  Evidence: all three targeted tests, combined touched test files, `ruff`, and `black --check` returned `EXIT:0`.

---

### Task 1: Pin V1 Vectorized Expected Labels

**Files:**
- Modify: `tests/test_trend_character.py`

- [x] **Step 1: Update the existing V1 vectorized test**

In `tests/test_trend_character.py`, update `test_trend_character_vectorized_raw_outputs_match_scalar_rules` after the `build_raw_outputs(...)` call:

```python
    assert vector_labels == [
        "recovery_attempt",
        "trending",
        "chop",
        "transition",
        "unknown",
    ]
    assert vector_evidence[-1] == {"reason": "insufficient_history"}
    assert vector_evidence[0]["recovery_attempt"] is True
    assert vector_evidence[1]["trending"] is True
    assert vector_evidence[2]["chop"] is True

    scalar = [raw_label_for_day(features, ts, allow_v2_labels=False) for ts in idx]
    assert vector_labels == [label for label, _ in scalar]
    assert vector_evidence == [evidence for _, evidence in scalar]
```

- [x] **Step 2: Run the targeted V1 test**

Run:

```bash
python3 -m pytest tests/test_trend_character.py::test_trend_character_vectorized_raw_outputs_match_scalar_rules -q ; echo "EXIT:$?"
```

Expected: one passing test and `EXIT:0`.

- [x] **Step 3: Commit if this task is implemented separately**

```bash
git add tests/test_trend_character.py
git commit -m "test(trend-character): pin vectorized v1 raw labels"
```

---

### Task 2: Add Direct V2 Vectorized Expected Labels

**Files:**
- Modify: `tests/test_trend_character_v2_labels.py`

- [x] **Step 1: Add the V2 vectorized expected-label test**

Add this test near `test_build_raw_outputs_matches_per_day`:

```python
def test_build_raw_outputs_direct_expected_v2_labels() -> None:
    idx = _trading_index(8)
    f = _synthetic_features(
        close=pd.Series(
            [120.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            index=idx,
            dtype=float,
        ),
        sma_50=pd.Series(
            [100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            index=idx,
            dtype=float,
        ),
        return_10d=pd.Series(
            [0.08, 0.06, 0.01, 0.01, 0.02, 0.02, 0.04, float("nan")],
            index=idx,
            dtype=float,
        ),
        return_21d=pd.Series(
            [0.10, 0.04, 0.06, 0.01, 0.02, 0.04, 0.04, 0.01],
            index=idx,
            dtype=float,
        ),
        return_63d=pd.Series(
            [0.20, 0.10, 0.10, 0.01, 0.01, 0.10, 0.10, 0.01],
            index=idx,
            dtype=float,
        ),
        prior_63d_drawdown=pd.Series(
            [-0.20, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=idx,
            dtype=float,
        ),
        adx_14=pd.Series(
            [25.0, 15.0, 25.0, 25.0, 15.0, 15.0, 15.0, 25.0],
            index=idx,
            dtype=float,
        ),
        midpoint_excursion_20d=pd.Series(
            [0.20, 0.20, 0.20, 0.10, 0.03, 0.10, 0.10, 0.10],
            index=idx,
            dtype=float,
        ),
        breakout_20d_or_50d=pd.Series(
            [True, False, False, False, False, False, False, False],
            index=idx,
        ),
        bb_width_expanding=pd.Series(
            [True, False, False, False, False, False, False, False],
            index=idx,
        ),
        volume_above_20d_average=pd.Series(
            [True, False, False, False, False, False, False, False],
            index=idx,
        ),
        followthrough_rate=pd.Series(
            [
                0.80,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            index=idx,
            dtype=float,
        ),
    )

    labels, evidence = build_raw_outputs(f)

    assert labels == [
        "breakout_expansion",
        "recovery_attempt",
        "trending",
        "mild_trend",
        "range_bound",
        "chop",
        "volatile_chop",
        "unknown",
    ]
    assert evidence[0]["breakout_expansion"] is True
    assert evidence[4]["range_bound"] is True
    assert evidence[7] == {"reason": "insufficient_history"}
```

- [x] **Step 2: Run the new V2 test**

Run:

```bash
python3 -m pytest tests/test_trend_character_v2_labels.py::test_build_raw_outputs_direct_expected_v2_labels -q ; echo "EXIT:$?"
```

Expected: one passing test and `EXIT:0`.

- [x] **Step 3: Commit if this task is implemented separately**

```bash
git add tests/test_trend_character_v2_labels.py
git commit -m "test(trend-character): pin vectorized v2 raw labels"
```

---

### Task 3: Pin Vectorized V1 Fallbacks for V2-Only Labels

**Files:**
- Modify: `tests/test_trend_character_v2_labels.py`

- [x] **Step 1: Add a vectorized fallback test**

Add this test after `test_build_raw_outputs_direct_expected_v2_labels`:

```python
def test_build_raw_outputs_v1_path_suppresses_v2_only_labels_vectorized() -> None:
    idx = _trading_index(4)
    f = _synthetic_features(
        close=pd.Series([120.0, 100.0, 100.0, 100.0], index=idx, dtype=float),
        sma_50=pd.Series([99.0, 99.0, 99.0, 99.0], index=idx, dtype=float),
        return_10d=pd.Series([0.01, 0.01, 0.02, 0.04], index=idx, dtype=float),
        return_21d=pd.Series([0.06, 0.01, 0.02, 0.04], index=idx, dtype=float),
        return_63d=pd.Series([0.20, 0.10, 0.01, 0.10], index=idx, dtype=float),
        prior_63d_drawdown=pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype=float),
        adx_14=pd.Series([25.0, 25.0, 15.0, 15.0], index=idx, dtype=float),
        midpoint_excursion_20d=pd.Series([0.20, 0.10, 0.03, 0.10], index=idx, dtype=float),
        breakout_20d_or_50d=pd.Series([True, False, False, False], index=idx),
        bb_width_expanding=pd.Series([True, False, False, False], index=idx),
        volume_above_20d_average=pd.Series([True, False, False, False], index=idx),
        followthrough_rate=pd.Series([0.80, float("nan"), float("nan"), float("nan")], index=idx, dtype=float),
    )

    v2_labels, _ = build_raw_outputs(f)
    v1_labels, _ = build_raw_outputs(f, allow_v2_labels=False)

    assert v2_labels == [
        "breakout_expansion",
        "mild_trend",
        "range_bound",
        "volatile_chop",
    ]
    assert v1_labels == [
        "trending",
        "transition",
        "chop",
        "transition",
    ]
```

- [x] **Step 2: Run the fallback test**

Run:

```bash
python3 -m pytest tests/test_trend_character_v2_labels.py::test_build_raw_outputs_v1_path_suppresses_v2_only_labels_vectorized -q ; echo "EXIT:$?"
```

Expected: one passing test and `EXIT:0`.

- [x] **Step 3: Run all trend-character tests touched by this plan**

Run:

```bash
python3 -m pytest tests/test_trend_character.py tests/test_trend_character_v2_labels.py -q ; echo "EXIT:$?"
```

Expected: all selected tests pass and `EXIT:0`.

- [x] **Step 4: Commit if this task is implemented separately**

```bash
git add tests/test_trend_character_v2_labels.py
git commit -m "test(trend-character): pin vectorized v1 fallbacks"
```

---

## Verification Before Completion

Before claiming this plan is implemented:

1. Run the targeted test from Task 1.
2. Run the targeted test from Task 2.
3. Run the targeted test from Task 3.
4. Run the combined trend-character test command from Task 3 Step 3.
5. Report command output excerpts including `EXIT:0`.

Full-suite verification should remain in GitHub Actions per `AGENTS.md`; do not run the full local pytest suite unless explicitly requested.

## Subagent Handoff

Use one implementer subagent for Task 1 and one implementer subagent for Tasks 2-3, because both V2 tasks edit the same file and should not run in parallel. After each implementer:

1. Review the diff locally.
2. Run the task-specific targeted command.
3. Dispatch a read-only review subagent to check that tests assert direct expected labels rather than scalar parity only.
4. Only then mark the task complete in the evidence-checked task list.
