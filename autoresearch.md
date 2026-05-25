# Autoresearch: Full Test Suite Runtime

## Objective
Reduce the full direct pytest wall-clock for this repository's entire suite by at least 50% without weakening coverage, adding skips, or changing production behavior solely for test speed.

The workload under study is the full suite command documented in the repo goal notes:
`PYTEST_ADDOPTS='' python3 -m pytest -m "" -q -n auto`

## Metrics
- **Primary**: `wall_seconds` (seconds, lower is better)
- **Secondary**: `collected_tests` (count, higher/equal is better), `passed_tests` (count, higher/equal is better), `skipped_tests` (count, no unjustified increase), `slowest_test_seconds` (seconds, lower is better)

## How to Run
`./autoresearch.sh` — runs the full suite and prints `METRIC name=value` lines.

## Files in Scope
- `pytest.ini` — pytest defaults and scheduling policy
- `tests/conftest.py` — shared fixture and session bootstrap behavior
- `tests/**/*.py` — only for measured fixture/caching/runtime-safe test-harness optimizations
- `Makefile` — only if a measured, equivalent canonical full-suite command needs wiring

## Off Limits
- `src/**/*.py` unless profiling proves duplicate production work is the dominant bottleneck and output remains behaviorally identical
- assertion weakening, marker-based silent coverage drops, new broad skips
- existing untracked files `conftest.py` and `pytest_runtime_policy.py` unless the user explicitly asks to fold them in or remove them

## Constraints
- Full suite must preserve the same correctness signal
- No degraded assertions, no hidden skips, no reduced collected count without explicit justification
- No new dependencies
- Every accepted optimization must be measurement-backed
- Final validation must include the full direct suite, `rtk pytest`, and `git diff --check`

## Termination
Stop when the measured full-suite wall clock is reduced by at least 50% versus baseline, or when remaining bottlenecks are proven irreducible without dropping coverage or larger architectural changes.

## What's Been Tried
- In a prior integration-only pass, `-n 0` beat xdist for a tiny 5-test integration subset, but that result does not transfer to the full suite and must not be generalized without measurement.
- The repo already documents an earlier full-suite baseline from 2026-05-22: `1328 passed, 1 skipped in 459.05s`.
- Initial full-suite collection in this workspace reports 134 lines from `--collect-only -q | wc -l`; this is only a quick probe, not the authoritative collected-test count from a real run.
