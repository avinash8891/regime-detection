# Followthrough Rate Optimization — Design Spec

**Date:** 2026-05-27
**Author:** avinash8891
**Status:** Draft, awaiting user review
**Scope:** Single-function algorithmic optimization in `src/regime_detection/trend_character.py`.

## Problem

`_compute_followthrough_rate` in `src/regime_detection/trend_character.py` computes,
for each session `t`, the fraction of the most-recent `window_count` breakouts (within
`lookback_sessions`) that "held" their breakout level for `hold_sessions` sessions.

The current implementation (lines ~213-228) walks backward from `t-1` for every `t`,
skipping non-breakout sessions until it has collected `window_count` breakouts or
exhausts the lookback window. Worst case is **O(n × lookback_sessions)** — when
breakouts are sparse, the inner loop runs the full `lookback_sessions` (default 504)
for every `t`. For multi-year daily series (~6,300 sessions), that is roughly 3M
Python-level iterations on the hot path.

This is the most expensive remaining nested-loop pattern in the trend_character
module and a measurable contributor to engine wall-clock time on full historical
runs.

## Goal

Replace the nested backward-walk with an **O(n log b)** algorithm (b = number of
breakouts) that produces **bit-for-bit identical output** to the current
implementation across all input series. No signature change, no caller changes.

## Non-Goals

- Changing the upstream computation of `breakout_level` or `held` (lines ~183-208).
  Those vectorized passes remain unchanged.
- Changing the followthrough semantics (definitions of "breakout", "held", or the
  cold-start `NaN` rule).
- Optimizing other functions in `trend_character.py` or any other module. Other
  hotspots flagged in prior review (`acquisition_consolidation`, `event_calendar`)
  are out of scope for this spec.

## Algorithm

### Inputs (already computed upstream, unchanged)

- `n` — number of sessions.
- `breakout_level: np.ndarray[float]` — length `n`. NaN where session is not a
  breakout; otherwise the level (20d or 50d prior high) that was crossed.
- `held: np.ndarray[bool]` — length `n`. True iff `close[b+1..b+hold_sessions]`
  is strictly above `breakout_level[b]`. False for breakouts where the hold window
  runs past the end of the series (unchanged "not yet validated" rule).

### Replacement body

```python
out = np.full(n, np.nan, dtype=float)

# 1. Index of every session that is a breakout, in ascending order.
breakout_idx = np.flatnonzero(~np.isnan(breakout_level))

# 2. Cumulative count of held breakouts. held_cum[k] - held_cum[j] equals the
#    number of held breakouts among breakout_idx[j:k] in O(1).
held_at_breakouts = held[breakout_idx].astype(np.int64)
held_cum = np.empty(len(breakout_idx) + 1, dtype=np.int64)
held_cum[0] = 0
np.cumsum(held_at_breakouts, out=held_cum[1:])

# 3. For each session t, find how many breakouts occurred strictly before t.
#    bisect_right against the sorted breakout_idx array.
for t in range(n):
    k = bisect_right(breakout_idx, t - 1)
    if k < window_count:
        continue  # insufficient history -> NaN
    j = k - window_count
    # The window_count-th most recent breakout must be within lookback_sessions.
    if breakout_idx[j] < t - lookback_sessions:
        continue  # most-recent window spills past lookback -> NaN
    out[t] = (held_cum[k] - held_cum[j]) / window_count

return pd.Series(out, index=close.index)
```

### Why this is equivalent to the old code

The old loop iterates `b` from `t-1` downward, skipping non-breakouts. It
collects up to `window_count` breakouts and exits early once it has them. The set
of breakouts it counts is therefore exactly:

- the most recent `window_count` breakouts strictly before `t`,
- provided the oldest of those is at position `>= t - lookback_sessions`.

`bisect_right(breakout_idx, t - 1)` gives the count of breakouts strictly before
`t`. Slicing `breakout_idx[k - window_count : k]` gives those exact most-recent
breakouts. The lookback check on the oldest element of that slice is identical to
the old `start = max(0, t - lookback_sessions)` bound, because the old loop only
ever exited via `collected >= window_count` or via running out of indices —
never via the lookback check failing in the middle of a window. (Breakouts older
than `t - lookback_sessions` were unreachable; the new bound enforces the same
unreachability.)

The cumulative-sum lookup of `held_count` is mathematically identical to the
inner running sum.

### Complexity

- Setup: O(n) for `flatnonzero`, `held[breakout_idx]`, and `cumsum`.
- Main loop: O(n log b) using `bisect_right` per session.
- Total: **O(n log b)** vs the old **O(n × lookback_sessions)** worst case.

A strict O(n) variant using a two-pointer sweep (advance `k` as `t` advances) is
possible but offers diminishing returns at typical n; the bisect form is preferred
for readability and because `b ≤ n`. We may revisit if profiling warrants.

## Correctness Invariants to Preserve

| Old behavior | Preserved by |
|---|---|
| Breakouts strictly before `t` (`range(t-1, ...)`) | `bisect_right(breakout_idx, t - 1)` |
| Skip non-breakout sessions | `breakout_idx` excludes them by construction |
| Early exit at `collected >= window_count` → take the most recent | Take `breakout_idx[k - window_count : k]` directly |
| `collected < window_count` → `NaN` | `k < window_count` → `NaN` |
| End-of-series rule: breakouts with `end >= n` get `held = False` (line 200-205) | `held` array unchanged upstream |
| Lookback bound `start = max(0, t - lookback_sessions)` | `breakout_idx[j] < t - lookback_sessions` → `NaN` |

## Testing Strategy

### Durable (committed) — golden snapshot

Add one new test to `tests/test_trend_character.py`:

- `test_followthrough_rate_matches_pinned_output_on_realistic_close_series`
- Loads a realistic multi-year close series from existing test fixtures
  (`tests/fixtures/derived/`, same source as `test_trend_character_matches_pinned_fixtures`).
  No synthetic toy series — per project testing rules.
- Computes `breakout_20d_or_50d` and feeds it to `_compute_followthrough_rate`
  with default parameters (504 lookback, 20 window_count, 5 hold_sessions).
- Pins the output as a per-index expected array stored in a YAML fixture under
  `tests/fixtures/derived/`. YAML over a hash so that diff output is debuggable
  when the test fails.
- Test is **written and committed BEFORE the algorithm change**, against the
  current slow implementation. After the change it must still pass without
  edits — this is the regression guard.

Edge cases the snapshot must cover (achieved by choosing a long-enough realistic
series):

- Early indices where `k < window_count` → NaN.
- Sparse breakout regions where the most-recent window spills past lookback → NaN.
- Dense breakout regions where the window is well within lookback.
- End-of-series sessions where some breakouts have `held = False` due to the
  `end >= n` rule.

Existing tests that must remain green without modification:

- `test_trend_character_matches_pinned_fixtures` (integration via
  `classified_golden_outputs`).
- `test_trend_character_rolling_features_match_legacy_inline_formulas`.
- All of `tests/test_trend_character_v2_labels.py` (depends on
  `followthrough_rate` for v2 trend labels).
- All of `tests/test_trend_direction_v2_*` that consume trend character output.

### Throwaway (NOT committed) — equivalence script

A local script `scripts/verify_followthrough_equivalence.py`:

- Keeps a copy of the OLD function body inline as `_compute_followthrough_rate_reference`.
- Runs both implementations against the full historical fixture close series.
- Asserts `np.array_equal(old, new, equal_nan=True)`.
- Times both with `time.perf_counter()` and prints the speedup.
- Deleted before commit. The commit message records the measured speedup.

This script is for developer confidence during the change. The committed golden
snapshot test is what guards against regression long-term.

## Files Touched

- `src/regime_detection/trend_character.py` — function body of
  `_compute_followthrough_rate` replaced. No signature change. Imports may add
  `from bisect import bisect_right` if not already present.
- `tests/test_trend_character.py` — one new test added.
- `tests/fixtures/derived/` — one new YAML fixture holding the pinned expected
  followthrough_rate output keyed by ISO date.
- `scripts/verify_followthrough_equivalence.py` — created during development,
  deleted before commit.

## Risk Assessment

**Low.** The function is pure, no I/O, no global state, single caller
(`compute_features` at line 263), output is fully pinnable. The main risk is a
subtle off-by-one in the lookback or window boundary; the golden snapshot and
the equivalence script both guard against that.

## Performance Expectation

Worst case before: O(n × 504) Python-level iterations. After: O(n log b) with
NumPy-level cumulative-sum setup. Expect **50–200× wall-clock speedup** on this
function for multi-year daily series. Measured speedup will be reported in the
commit message via the throwaway script.

## Rollout

Single PR, single commit. No feature flag. No migration. The function output
is unchanged by design, so callers and downstream classifiers are unaffected.
