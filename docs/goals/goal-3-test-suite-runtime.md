# Codex Goal: Test Suite Runtime Optimization

## Goal prompt (paste into `/goal`)

```
Reduce the default local test-suite wall clock from the current direct baseline of 7m39s (`python3.14 -m pytest`, 1328 passed / 1 skipped on 2026-05-22) to under 3m00s on the same Mac workspace, without losing behavioral coverage, weakening assertions, hiding failures, or changing production behavior only to make tests faster.

Objective:
- Preserve the full correctness signal of the default suite: the same tests must still run by default unless a test is explicitly proven to be slow-only coverage and moved behind an existing or documented marker with an equivalent fast default guard.
- Keep all deterministic rule, V1 frozen replay, V2 gate, config validation, data-routing, and fixture provenance coverage intact.
- Eliminate avoidable test overhead through measurement-backed changes only.

Verification surface:
- Establish baseline with `python3.14 -m pytest --durations=100 ; echo "EXIT:$?"` and save the full output summary in the thread.
- After each change, run the smallest affected test command with explicit `EXIT:$?`.
- Before completion, run:
  - `python3.14 -m pytest ; echo "EXIT:$?"`
  - `rtk pytest ; echo "EXIT:$?"`
  - `git diff --check ; echo "EXIT:$?"`
  - ruff on changed Python files.
- Completion requires the full direct pytest summary to show no new failures, no new skips, and no reduced collected-test count unless the difference is explicitly justified by a marker split and covered by a documented slow-suite command.

Allowed optimization strategies:
- Use `pytest --durations=100`, per-file timing, and xdist scheduling evidence to find bottlenecks before changing code.
- Convert repeated expensive fixture setup into scoped fixtures only when tests remain isolated and cannot share mutable state unsafely.
- Replace repeated fixture construction with checked golden fixtures when they preserve exact values and failure modes.
- Split genuine long-running confidence tests behind an explicit marker only if the default suite keeps a faster equivalent behavior check and the slow command is documented.
- Improve xdist scheduling, collection-time imports, parametrization blowups, and redundant subprocess/profile invocations.
- Cache immutable test data inside the test process when the cache key includes every input that affects behavior.

Forbidden changes:
- Do not loosen assertions (`==` to ranges, exact counts to non-null, strict type checks to truthiness).
- Do not mock internal code to bypass behavior; only mock external services using captured real fixtures.
- Do not remove V1 replay, V2 gate, cold-start/NaN, config unknown-key, PIT, or provenance tests from the default suite without an explicit user decision.
- Do not introduce silent skips, broad `pytest.importorskip`, hidden network calls, or order-dependent shared state.
- Do not change production classifier logic unless a measured test bottleneck proves the production path has avoidable duplicate work and the behavioral output remains byte-identical or numerically equivalent under an explicit tolerance.

Investigation order:
1. Capture baseline: total wall clock, collected count, skipped count, top 100 durations, slowest files, and collection time.
2. Classify the top bottlenecks as fixture setup, subprocess/script integration, pandas/numpy compute, I/O, xdist imbalance, import/collection, or overly broad parametrization.
3. For each candidate, write down why the change preserves the same correctness signal before editing.
4. Make one optimization per commit-sized unit, then rerun affected tests and record before/after timing.
5. Stop after reaching <3m00s full direct pytest, or if remaining time is irreducible without dropping coverage. If blocked, report the measured floor and the exact tests responsible.

Done when:
- `python3.14 -m pytest` completes in <3m00s on the same local machine with no new failures, no new skips, and no unjustified collection-count reduction.
- `rtk pytest` returns `EXIT:0` under failures-only mode.
- The repo documents any new marker or slow-suite command in `pytest.ini` and the relevant docs.
- The final summary includes a before/after timing table, changed files, and the evidence that coverage was not weakened.
```

## Context for Codex
- Repo: regime-detection (manila-v2 workspace)
- Branch: avinash8891/regime-detection-audit
- Current known full-suite evidence: `1328 passed, 1 skipped in 459.05s (0:07:39)` from `python3.14 -m pytest ; echo "EXIT:$?"` on 2026-05-22.
- Current pytest config: `pytest.ini` uses `addopts = -q -m "not slow" -n auto`.
- RTK note from AGENTS.md: RTK runs pytest in failures-only mode. Always append `; echo "EXIT:$?"`; `Pytest: No tests collected` with `EXIT:0` means no failures were found, not that the full suite was enumerated.
- Key files likely involved: `pytest.ini`, `tests/conftest.py` if present, slow test files identified by `--durations`, and any source modules only when production duplicate work is proven by measurement.
- Initial commands:
  - `python3.14 -m pytest --durations=100 ; echo "EXIT:$?"`
  - `python3.14 -m pytest --collect-only -q ; echo "EXIT:$?"`
  - `python3.14 -m pytest --durations=50 --dist=loadfile ; echo "EXIT:$?"` if xdist scheduling imbalance is suspected.
