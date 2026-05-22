# Claude Goal: Test Suite Runtime Optimization

Reduce this repo's default local test-suite runtime without losing coverage or weakening correctness.

Current baseline from this workspace on 2026-05-22:
- Command: `python3.14 -m pytest ; echo "EXIT:$?"`
- Result: `1328 passed, 1 skipped in 459.05s (0:07:39)`, `EXIT:0`
- Pytest config: `pytest.ini` has `addopts = -q -m "not slow" -n auto`

Target:
- Get `python3.14 -m pytest` under 3m00s on the same local machine.
- Keep the same correctness signal: no new failures, no new hidden skips, no unjustified drop in collected tests.
- Preserve deterministic rule tests, V1 frozen replay, V2 gate, config validation, PIT/data-routing, cold-start/NaN handling, and provenance coverage.

Hard constraints:
- Do not loosen assertions: no `==` to broad ranges, exact counts to non-null, strict types to truthiness.
- Do not mock internal code to bypass behavior. Mock only external services, using captured real fixtures.
- Do not remove important tests from the default suite unless there is a documented slow-suite command and a fast default test that preserves the same behavior signal.
- Do not add silent skips, order-dependent shared state, hidden network calls, or broad import skips.
- Do not change production classifier behavior just to speed tests. Production changes are allowed only when profiling proves duplicate work and outputs stay byte-identical or explicitly tolerance-equivalent.

Workflow:
1. Baseline first. Run:
   `python3.14 -m pytest --durations=100 ; echo "EXIT:$?"`
   Capture total time, collected count, skipped count, and top slow tests.
2. Also inspect collection/scheduling if needed:
   `python3.14 -m pytest --collect-only -q ; echo "EXIT:$?"`
   Try `--dist=loadfile` only if xdist imbalance is visible.
3. Classify bottlenecks before editing: fixture setup, subprocess integration, pandas/numpy compute, I/O, xdist imbalance, import/collection, or parametrization blowup.
4. Make one measured optimization at a time. Before editing, state why the change preserves coverage.
5. After each change, run the smallest affected test command with `; echo "EXIT:$?"` and record before/after timing.

Allowed optimizations:
- Scope expensive immutable fixtures safely.
- Cache immutable test data in-process with complete cache keys.
- Replace repeated fixture construction with checked golden fixtures that preserve exact values and failure modes.
- Reduce duplicate subprocess/profile invocations.
- Improve xdist scheduling or parametrization when behavior is unchanged.
- Move genuinely long confidence tests behind an explicit marker only with documented slow command and retained fast default coverage.

Completion checks:
- `python3.14 -m pytest ; echo "EXIT:$?"`
- `rtk pytest ; echo "EXIT:$?"`
- `git diff --check ; echo "EXIT:$?"`
- Ruff on changed Python files.

RTK note from AGENTS.md: RTK is failures-only. `Pytest: No tests collected` with `EXIT:0` means no failures were found, not that the full suite was enumerated.

Done only when full direct pytest is under 3m00s with no weakened coverage. If blocked, report the measured floor, the exact remaining slow tests, and why further speedups would require dropping coverage or larger design work.
