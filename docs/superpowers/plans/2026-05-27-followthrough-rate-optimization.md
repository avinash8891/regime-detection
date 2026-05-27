# Followthrough Rate Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(n × lookback_sessions) backward-walk loop in `_compute_followthrough_rate` with an O(n log b) cumulative-sum + bisect algorithm that produces bit-for-bit identical output.

**Architecture:** Three-step TDD cycle. (1) Pin current output as a YAML golden snapshot by running the slow implementation against the SPY market_data fixture. (2) Implement the replacement and verify it still passes the snapshot. (3) Verify with a throwaway equivalence/benchmark script that the new code is element-wise identical across the full series and measure speedup.

**Tech Stack:** Python 3.12, NumPy, pandas, pytest, PyYAML. No new dependencies. The `bisect` module is in the standard library.

**Spec:** `docs/superpowers/specs/2026-05-27-followthrough-rate-optimization-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/regime_detection/trend_character.py` | `_compute_followthrough_rate` function body (lines ~210-228). Add `from bisect import bisect_right` if not already imported. | Modify |
| `tests/test_trend_character.py` | Add one new test: `test_followthrough_rate_matches_pinned_output_on_realistic_close_series` | Modify |
| `tests/fixtures/derived/followthrough_rate_pinned.yaml` | Per-date pinned `followthrough_rate` values computed from current implementation against SPY market_data fixture | Create |
| `scripts/generate_followthrough_pinned_fixture.py` | One-shot script that builds the YAML fixture by running the CURRENT implementation. Committed for reproducibility. | Create |
| `scripts/verify_followthrough_equivalence.py` | Throwaway script comparing old vs new implementations element-wise and timing both. Created during development, **deleted before final commit**. | Create then delete |

---

## Task 1: Pin current output as a golden YAML fixture

**Files:**
- Create: `scripts/generate_followthrough_pinned_fixture.py`
- Create: `tests/fixtures/derived/followthrough_rate_pinned.yaml`
- Modify: `tests/test_trend_character.py` (add one test)

- [ ] **Step 1.1: Verify the SPY close series is reachable from a script**

Run:
```bash
python3 -c "
from tests.conftest import _load_market_data, _close_series_from_market_data
md = _load_market_data()
close = _close_series_from_market_data(md, 'SPY')
print(len(close), close.index.min().date(), close.index.max().date())
"
```
Expected: prints the row count and date range (multi-year SPY history). If this fails, stop and fix the import path before continuing.

- [ ] **Step 1.2: Write the fixture-generation script**

Create `scripts/generate_followthrough_pinned_fixture.py`:

```python
"""Generate the pinned followthrough_rate YAML fixture.

Run once to capture the CURRENT (slow) implementation's output against the
SPY market_data fixture. The committed YAML fixture is what
``test_followthrough_rate_matches_pinned_output_on_realistic_close_series``
asserts against, both before and after the algorithmic change.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import _close_series_from_market_data, _load_market_data  # noqa: E402

from regime_detection.trend_character import (  # noqa: E402
    _compute_breakout_20d_or_50d,
    _compute_followthrough_rate,
    _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
)


def main() -> None:
    market_data = _load_market_data()
    close = _close_series_from_market_data(market_data, "SPY")
    breakout = _compute_breakout_20d_or_50d(close)
    ft_rate = _compute_followthrough_rate(
        close,
        breakout,
        lookback_sessions=_DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
        window_count=_DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
        hold_sessions=_DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    )

    rows: list[dict[str, object]] = []
    for ts, value in ft_rate.items():
        if isinstance(value, float) and math.isnan(value):
            rows.append({"date": ts.date().isoformat(), "value": None})
        else:
            rows.append({"date": ts.date().isoformat(), "value": float(value)})

    fixture_path = (
        REPO_ROOT / "tests" / "fixtures" / "derived" / "followthrough_rate_pinned.yaml"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "symbol": "SPY",
                "lookback_sessions": _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
                "window_count": _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
                "hold_sessions": _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
                "rows": rows,
            },
            sort_keys=False,
        )
    )
    print(f"Wrote {fixture_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
```

**Note:** This script uses `_compute_breakout_20d_or_50d` and `_compute_followthrough_rate`, which are module-private (underscore-prefixed). They are imported here intentionally — this script lives in `scripts/` and exists only to capture pinned output, not as production consumer code.

- [ ] **Step 1.3: Verify the names this script imports actually exist in trend_character.py**

Run:
```bash
python3 -c "
from regime_detection.trend_character import (
    _compute_breakout_20d_or_50d,
    _compute_followthrough_rate,
    _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
)
print('imports ok')
"
```
Expected: prints `imports ok`. If any name fails, open `src/regime_detection/trend_character.py` and find the correct private name (e.g. it might be `_compute_breakout_levels` instead of `_compute_breakout_20d_or_50d`); update the script accordingly before continuing.

- [ ] **Step 1.4: Run the fixture-generation script**

Run:
```bash
python3 scripts/generate_followthrough_pinned_fixture.py
```
Expected: prints `Wrote tests/fixtures/derived/followthrough_rate_pinned.yaml with NNNN rows.` and the file exists.

- [ ] **Step 1.5: Write the pinned-snapshot test**

Add to `tests/test_trend_character.py` (after the imports block, alongside the existing tests):

```python
import math

import numpy as np

from regime_detection.trend_character import (
    _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
    _compute_breakout_20d_or_50d,
    _compute_followthrough_rate,
)


def test_followthrough_rate_matches_pinned_output_on_realistic_close_series(
    raw_market_data,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_path = (
        repo_root / "tests" / "fixtures" / "derived" / "followthrough_rate_pinned.yaml"
    )
    pinned = yaml.safe_load(fixture_path.read_text())

    spy = raw_market_data[raw_market_data["symbol"] == "SPY"].sort_values("date")
    close = pd.Series(
        spy["close"].astype(float).to_numpy(),
        index=pd.to_datetime(spy["date"]),
        name="SPY",
    )
    breakout = _compute_breakout_20d_or_50d(close)
    ft_rate = _compute_followthrough_rate(
        close,
        breakout,
        lookback_sessions=pinned["lookback_sessions"],
        window_count=pinned["window_count"],
        hold_sessions=pinned["hold_sessions"],
    )

    expected_by_date = {row["date"]: row["value"] for row in pinned["rows"]}
    assert len(expected_by_date) == len(ft_rate), (
        f"row count mismatch: fixture has {len(expected_by_date)}, "
        f"computed has {len(ft_rate)}"
    )

    for ts, actual in ft_rate.items():
        key = ts.date().isoformat()
        assert key in expected_by_date, f"unexpected date in computed output: {key}"
        expected = expected_by_date[key]
        if expected is None:
            assert math.isnan(actual), f"{key}: expected NaN, got {actual}"
        else:
            assert not math.isnan(actual), f"{key}: expected {expected}, got NaN"
            assert np.isclose(actual, expected, rtol=0.0, atol=0.0), (
                f"{key}: expected {expected}, got {actual}"
            )
```

**Note:** If `Path`, `pd`, or `yaml` are not already imported at the top of `tests/test_trend_character.py`, they already are (see lines 3-7). No additional import block needed beyond the names listed above.

- [ ] **Step 1.6: Run the new test against the CURRENT (slow) implementation**

Run:
```bash
python3 -m pytest tests/test_trend_character.py::test_followthrough_rate_matches_pinned_output_on_realistic_close_series -v
```
Expected: **PASS**. The pinned fixture was just generated from the current implementation, so this must pass before any algorithmic change. If it fails, stop — there is a fixture/test mismatch to fix first.

- [ ] **Step 1.7: Run the rest of `test_trend_character.py` to confirm no collateral breakage**

Run:
```bash
python3 -m pytest tests/test_trend_character.py -v
```
Expected: all tests pass.

- [ ] **Step 1.8: Commit the test, fixture, and generator script**

```bash
git add scripts/generate_followthrough_pinned_fixture.py \
        tests/fixtures/derived/followthrough_rate_pinned.yaml \
        tests/test_trend_character.py
git commit -m "$(cat <<'EOF'
test: pin followthrough_rate output as golden snapshot

Adds a YAML fixture capturing _compute_followthrough_rate output against
the SPY market_data fixture using the current (O(n*lookback)) implementation.
A new test asserts bit-for-bit equality, locking the behavior in before
the upcoming algorithmic replacement.
EOF
)"
```

---

## Task 2: Build the throwaway equivalence + benchmark script

**Files:**
- Create: `scripts/verify_followthrough_equivalence.py` (deleted in Task 4)

- [ ] **Step 2.1: Capture the CURRENT function body as a reference implementation in the throwaway script**

Open `src/regime_detection/trend_character.py`, locate `_compute_followthrough_rate` (around line 159 onwards), and copy its current body verbatim into `scripts/verify_followthrough_equivalence.py` as `_compute_followthrough_rate_reference`. Keep the signature identical to the production function.

Create `scripts/verify_followthrough_equivalence.py`:

```python
"""Throwaway: verify old vs new _compute_followthrough_rate are element-wise
identical and measure speedup. NOT COMMITTED — deleted at end of the
optimization PR. See plan
docs/superpowers/plans/2026-05-27-followthrough-rate-optimization.md Task 4.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import _close_series_from_market_data, _load_market_data  # noqa: E402

from regime_detection.trend_character import (  # noqa: E402
    _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
    _compute_breakout_20d_or_50d,
    _compute_followthrough_rate,
)


def _compute_followthrough_rate_reference(
    close: pd.Series,
    breakout: pd.Series,
    *,
    lookback_sessions: int,
    window_count: int,
    hold_sessions: int,
) -> pd.Series:
    """PASTE THE CURRENT BODY OF _compute_followthrough_rate HERE VERBATIM.

    Copy from src/regime_detection/trend_character.py BEFORE making any
    changes to that file. This is the oracle the new implementation must
    match element-wise.
    """
    raise NotImplementedError("Paste the current body here before running")


def main() -> None:
    market_data = _load_market_data()
    close = _close_series_from_market_data(market_data, "SPY")
    breakout = _compute_breakout_20d_or_50d(close)

    kwargs = dict(
        lookback_sessions=_DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
        window_count=_DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
        hold_sessions=_DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    )

    t0 = time.perf_counter()
    old = _compute_followthrough_rate_reference(close, breakout, **kwargs)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = _compute_followthrough_rate(close, breakout, **kwargs)
    t_new = time.perf_counter() - t0

    assert np.array_equal(old.to_numpy(), new.to_numpy(), equal_nan=True), (
        "Old and new outputs differ"
    )
    print(f"OK: {len(close)} sessions, old={t_old:.3f}s new={t_new:.3f}s "
          f"speedup={t_old / max(t_new, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Paste the current function body into the placeholder**

Open `src/regime_detection/trend_character.py`, copy the entire body of `_compute_followthrough_rate` (everything between the signature and `return pd.Series(out, index=close.index)`), and paste it into `_compute_followthrough_rate_reference` in the throwaway script, replacing the `raise NotImplementedError` line.

- [ ] **Step 2.3: Run the throwaway script to confirm baseline equivalence**

Run:
```bash
python3 scripts/verify_followthrough_equivalence.py
```
Expected: **OK** line printed with `speedup=1.0x` or near it (the new function is still the old algorithm — we haven't changed it yet). If `assert np.array_equal` fails here, the reference paste is wrong; fix and rerun before Task 3.

- [ ] **Step 2.4: Do NOT commit the throwaway script**

Confirm:
```bash
git status --short scripts/verify_followthrough_equivalence.py
```
Expected: `?? scripts/verify_followthrough_equivalence.py` (untracked). If it shows `A` (staged), unstage with `git restore --staged scripts/verify_followthrough_equivalence.py`.

---

## Task 3: Replace `_compute_followthrough_rate` with O(n log b) algorithm

**Files:**
- Modify: `src/regime_detection/trend_character.py:210-228` (function body of `_compute_followthrough_rate`)

- [ ] **Step 3.1: Verify `bisect_right` import status**

Run:
```bash
grep -n "^from bisect\|^import bisect" src/regime_detection/trend_character.py
```
Expected: either nothing (in which case Step 3.2 adds the import) or `from bisect import ...` already present.

- [ ] **Step 3.2: Add the `bisect_right` import if missing**

If Step 3.1 returned nothing, open `src/regime_detection/trend_character.py` and add this line to the import block at the top of the file (alphabetically before `from dataclasses import ...` or wherever the `from X import Y` block is):

```python
from bisect import bisect_right
```

If a `from bisect import ...` line already exists, append `bisect_right` to its name list instead of duplicating the import.

- [ ] **Step 3.3: Replace the function body**

Open `src/regime_detection/trend_character.py`. Locate the inner block of `_compute_followthrough_rate` starting at the line that constructs `breakout_level` (around line 183) through the final `return pd.Series(out, index=close.index)` (around line 228).

The blocks computing `breakout_level` (lines ~183-193) and `held` (lines ~196-208) MUST remain unchanged — they are correctness-critical and the new algorithm depends on `held` having identical contents to the old version.

Replace ONLY the final block (the per-session forward loop starting at the comment `# Now compute followthrough_rate per session t`, around line 210) with this:

```python
    # Vectorized lookup: index every breakout, precompute a cumulative sum of
    # held breakouts, then for each session t fetch the most-recent
    # window_count breakouts strictly before t via bisect_right and the
    # held_count via two cumulative-sum reads.
    #
    # The lookback bound is enforced by a single check on the oldest element
    # of the window — older breakouts than `t - lookback_sessions` were
    # unreachable in the original backward walk, so requiring
    # breakout_idx[k - window_count] >= t - lookback_sessions reproduces the
    # original semantics exactly.
    out = np.full(n, np.nan, dtype=float)
    breakout_idx = np.flatnonzero(~np.isnan(breakout_level))
    if breakout_idx.size >= window_count:
        held_at_breakouts = held[breakout_idx].astype(np.int64)
        held_cum = np.empty(breakout_idx.size + 1, dtype=np.int64)
        held_cum[0] = 0
        np.cumsum(held_at_breakouts, out=held_cum[1:])
        breakout_idx_list = breakout_idx.tolist()
        for t in range(n):
            k = bisect_right(breakout_idx_list, t - 1)
            if k < window_count:
                continue
            j = k - window_count
            if breakout_idx_list[j] < t - lookback_sessions:
                continue
            out[t] = (held_cum[k] - held_cum[j]) / window_count
    return pd.Series(out, index=close.index)
```

**Why `.tolist()` on `breakout_idx`?** `bisect_right` operates on Python sequences in C; a Python `list` is faster here than a NumPy array because each `bisect_right` call would otherwise box/unbox NumPy scalars. The `held_cum` array stays NumPy because it's indexed once per session, not bisected.

**Variables used (`n`, `breakout_level`, `held`, `lookback_sessions`, `window_count`, `close`) are all in scope from the outer function and the unchanged upstream blocks. Do not rename or redeclare them.**

- [ ] **Step 3.4: Run the pinned-snapshot test**

Run:
```bash
python3 -m pytest tests/test_trend_character.py::test_followthrough_rate_matches_pinned_output_on_realistic_close_series -v
```
Expected: **PASS**. The new implementation must produce bit-for-bit identical output. If it fails, do NOT proceed. Inspect the first failing date in the assertion message and debug — the most common causes are an off-by-one in `bisect_right` (use `bisect_left` if the old code used `<= t-1` semantics, but it used `range(t-1, ...)` which is `< t`, so `bisect_right(..., t - 1)` is correct), or an off-by-one in the lookback bound check.

- [ ] **Step 3.5: Run the full trend_character test files**

Run:
```bash
python3 -m pytest tests/test_trend_character.py tests/test_trend_character_v2_labels.py -v
```
Expected: all tests pass. `test_trend_character_v2_labels.py` exercises `followthrough_rate` through the v2 trend-character classifier, so any drift in output would surface here.

- [ ] **Step 3.6: Run the throwaway equivalence + benchmark script**

Run:
```bash
python3 scripts/verify_followthrough_equivalence.py
```
Expected: **OK** line printed with `speedup>1.0x`, ideally 20-100x or more. The `assert np.array_equal(..., equal_nan=True)` must pass. Record the speedup number — it goes in the commit message in Step 3.8.

- [ ] **Step 3.7: Run the wider trend-direction test suite**

Run:
```bash
python3 -m pytest tests/test_trend_direction.py tests/test_trend_direction_v2_features.py tests/test_trend_direction_v2_euphoria.py tests/test_trend_direction_v2_recovery_rule.py -v
```
Expected: all pass. These are the downstream consumers most likely to be sensitive to `followthrough_rate` drift.

- [ ] **Step 3.8: Commit the optimization**

Substitute `<NNN>` below with the speedup number printed by Step 3.6.

```bash
git add src/regime_detection/trend_character.py
git commit -m "$(cat <<'EOF'
perf: vectorize _compute_followthrough_rate from O(n*lookback) to O(n log b)

Replaces the per-session backward-walk loop with a cumulative-sum +
bisect_right lookup over the precomputed breakout index. End-of-series
held semantics and the lookback bound are preserved exactly; the
committed YAML fixture asserts bit-for-bit identical output.

Measured speedup on the SPY market_data fixture: <NNN>x.
EOF
)"
```

---

## Task 4: Cleanup throwaway script

**Files:**
- Delete: `scripts/verify_followthrough_equivalence.py`

- [ ] **Step 4.1: Delete the throwaway script**

```bash
rm scripts/verify_followthrough_equivalence.py
```

- [ ] **Step 4.2: Confirm it is gone and the working tree is clean**

```bash
git status --short
```
Expected: empty output (working tree clean). If `scripts/verify_followthrough_equivalence.py` still shows up, it was accidentally added to git in Task 2 — run `git rm scripts/verify_followthrough_equivalence.py` and commit with message `chore: remove throwaway equivalence script`.

- [ ] **Step 4.3: Final full-suite run**

Run:
```bash
python3 -m pytest tests/ -x --tb=short
```
Expected: all tests pass. This is the last safety net before the change is considered done.

---

## Done Criteria

- `test_followthrough_rate_matches_pinned_output_on_realistic_close_series` passes against the new implementation.
- All of `test_trend_character.py`, `test_trend_character_v2_labels.py`, `test_trend_direction*.py` pass.
- `scripts/verify_followthrough_equivalence.py` is deleted and working tree is clean.
- Two commits exist on the branch: one for the pinned fixture + test, one for the algorithmic replacement.
- Speedup measured and recorded in the second commit message.
