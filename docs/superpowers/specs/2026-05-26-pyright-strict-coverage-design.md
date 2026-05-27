## Strict Pyright coverage expansion design

### Goal

Expand strict Pyright coverage from the current four-file allowlist to the main runtime codepaths: full `src/` plus runtime scripts that drive operator workflows, while preserving strict mode and avoiding broad suppressions.

### Why strict Pyright is worth doing

Strict Pyright is not just theatre when it is used as a real merge gate on runtime code:

- It catches interface drift early: wrong argument types, missing attributes, bad optional handling, and invalid dict/object assumptions before runtime.
- It hardens refactors: when shared models, loaders, or observability payloads change, dependent callsites fail fast in CI instead of silently diverging.
- It improves agent safety: autonomous edits are much less likely to introduce shape mismatches when the repo has an enforceable static contract.
- It raises the floor on runtime code quality in places tests may not fully enumerate, especially for error paths and rarely hit branches.

It becomes theatre only if the checked scope is too small, if suppressions are broad, or if the team ignores the results. The current repo state is partially in that danger zone because strict mode exists, but only for a tiny slice. This task fixes that by making the gate cover the real runtime surface.

### Current state

- `[tool.pyright]` uses `typeCheckingMode = "strict"`.
- `include` currently names only:
  - `src/regime_detection/observability.py`
  - `src/regime_detection/loaders.py`
  - `scripts/detect_flaky_tests.py`
  - `scripts/validate_agents_md.py`
- CI, release, and full-verification workflows already run `python -m pyright`.
- `tests/test_readiness_contracts.py` asserts the current four-file slice.

### Desired end state

Pyright strict mode should cover:

- `src/regime_detection`
- `src/regime_data_fetch`
- `src/regime_shared`
- runtime scripts used by operator and validation workflows

This should be the default repo gate, using the existing `python -m pyright` command in CI.

### Recommended approach

Use a full-scope ratchet with narrow exclusions only where justified.

1. Change the Pyright include set from a file-level slice to package/runtime-scope coverage.
2. Run Pyright on that broader scope and collect the true error set.
3. Fix strict typing issues in the main runtime paths without changing behavior.
4. Add only narrowly scoped exclusions if a file is not part of the runtime contract or is blocked by known external-stub limitations.
5. Update readiness tests so they enforce the broader scope and prevent regression back to a tiny allowlist.

### Scope rules

Included:

- all Python packages under `src/`
- runtime scripts that are part of profiling, materialization, acquisition, qualification, replay, and validation flows

Excluded only if justified:

- one-off local utilities that are not part of the runtime or CI contract
- paths whose failures are dominated by third-party stub limitations, but only after verifying the issue is not local typing debt

Not allowed:

- changing `typeCheckingMode` away from `strict`
- repo-wide disabling of `report*` diagnostics
- broad excludes such as all `scripts/` without path-level reasoning

### Implementation details

#### Pyright config

Update `[tool.pyright]` so the include list expresses the intended runtime ownership rather than a small hand-picked slice.

Expected pattern:

- package directories instead of individual files for `src/`
- explicit runtime script paths or a curated runtime-script grouping

#### Readiness guardrail

Replace the current readiness assertion that checks for four specific files with one that verifies:

- the main runtime packages are included
- the repo does not regress back to a tiny allowlist model

#### Error-fix strategy

Fixes should follow these patterns:

- add precise optional narrowing instead of `cast` when possible
- prefer typed helper functions over repeated inline shape assumptions
- tighten model and collection types where the runtime contract is already known
- keep import-bootstrapping scripts minimal and typed without weakening the rest of the repo

#### Exclusion bar

Any exclusion must be:

- narrow
- documented inline or obvious from the config
- justified by non-runtime status or tooling limitations

If a file is part of the operator path, the default assumption is that it should be fixed rather than excluded.

### Validation plan

Use cheap local validation only:

1. `python -m pyright`
2. `python -m pytest tests/test_readiness_contracts.py -q`

If typing fixes touch behavior-sensitive runtime code, run only the smallest relevant targeted tests for those modules.

### Risks

- Broadening strict coverage may surface many existing errors at once.
- Some errors may come from pandas or third-party stubs rather than real logic problems.
- Scripts that manipulate `sys.path` may need careful typing cleanup.

### Risk controls

- keep strict mode unchanged
- fix runtime packages first
- exclude only narrow, justified non-runtime paths
- preserve behavior; this is a typing-contract change, not a feature refactor

### Success criteria

This design is complete when:

- Pyright strict coverage includes full `src/` plus runtime scripts
- CI can continue using `python -m pyright` unchanged
- readiness tests enforce the broader scope
- no broad suppressions are introduced
- the final gate passes on the expanded scope
