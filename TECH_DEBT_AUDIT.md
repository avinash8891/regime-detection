# Tech Debt Audit - 2026-05-17

Grounding: live repository state in `/Users/avinashvankadaru/conductor/workspaces/regime-detection/vaduz` on branch `avinash8891/tech-debt-audit`. This is a repeat audit; older findings were rechecked against current code before being kept, narrowed, or marked resolved.

## Executive Summary

The repo is materially healthier than the previous audit snapshot in several places: the axis-series surface is split out of `axis_series.py`, feature-store and fetch-mode registries exist, slow/V2 gate tests are wired in CI, artifact URI handling is fixed, and several formerly skip-gated tests now use fixtures. The remaining debt is concentrated in four areas:

1. Large orchestrator modules still own too many contracts at once.
2. New registries stop short of owning typed outputs or invocation.
3. Type checking is mostly not enforceable yet: full pyright currently reports 714 errors, while CI only checks `src/regime_detection/config.py`.
4. Operational scripts are still runtime products, but several are too large, monkeypatch-heavy, or outside strict type gates.

Top implementation priorities:

1. Fix the broken `profile_engine_30d.py` inflation-growth timing hook after the axis split.
2. Expand pyright coverage incrementally beyond config-only.
3. Finish evidence model typing where timeline and classifiers still pass plain dict payloads.
4. Replace the feature-store `dict[str, Any]` builder bus with typed builder outputs.
5. Move fetch-mode invocation into the registry so `FETCH_MODE_REGISTRY` owns execution, not just classification.

## Mental Model

This repo has three coupled products:

- Runtime classifier: `src/regime_detection` builds `MarketContext`, features, axis series, timeline outputs, and Pydantic wire models.
- Data acquisition: `src/regime_data_fetch` and fetch scripts collect external market/macro/event artifacts, persist provenance, and materialize inputs.
- Operational scripts and gates: `scripts/` runs fetches, profiling, calibration, walk-forward gates, shadow checks, and audit/report generation.

Most risk appears at the boundaries between those products: typed model boundaries, feature-store handoff, script/runtime imports, and acquisition artifact contracts.

## Tool Evidence

| Command | Result |
|---|---|
| `python3 -m ruff check .` | Passed: `All checks passed!` |
| `python3 -m pyright src/regime_detection src/regime_data_fetch scripts` | Failed: 714 errors. Examples include `scripts/profile_engine_30d.py:351` / `:353`, `src/regime_detection/timeline.py:89`, `:233`, `:239`, `src/regime_detection/transition_risk.py:138`, and pandas typing errors in calibration/audit scripts. |
| `python3 -m vulture src scripts tests` | Unavailable: `No module named vulture`. |
| `python3 -m pydeps src/regime_detection --show-cycles --noshow` | Unavailable: `No module named pydeps`. |
| `python3 -m pip_audit --desc off` | Environment-level failure: 51 known vulnerabilities in 23 packages. The output also skipped local/non-PyPI packages including `regime-detection (2.0.0)`, so this is not a clean project-lock audit. |
| `PYTHONPATH=src python3 - <<'PY' ... hasattr(axis_series, ...)` | `assess_series_input_quality=True`, `build_inflation_growth_rule_inputs_by_date=False`, `evaluate_inflation_growth_rules=False`; confirms the profiler still points at moved symbols. |
| `python3 -m pytest --cov=src --cov-report=term --cov-fail-under=80 -q` | Failed: 5 tests failed, coverage still reached 88.33%. All failures are `TransitionRiskOutput` validation errors where tests still pass `{}` or `{"warnings": []}` evidence. |

CI currently runs ruff, pyright only against `src/regime_detection/config.py`, and default pytest with coverage in `.github/workflows/ci.yml:63-66`. Slow/V2 gate tests are in a separate path-sensitive job at `.github/workflows/ci.yml:71-120`.

## Repeat-Audit Status

Resolved or mostly resolved from earlier audit:

- Full axis split out of `axis_series.py`: resolved at the import surface; implementation now lives in `src/regime_detection/axis_builders/series.py`.
- Feature-store registry: partially resolved; `_FeatureStoreBuilder` and `_FEATURE_STORE_BUILDERS` exist at `src/regime_detection/feature_store.py:146-158` and `:607-627`.
- Fetch-mode registry/concurrency: partially resolved; `FETCH_MODE_REGISTRY` exists at `scripts/fetch_regime_engine_v1_data.py:74-96`, and conservative concurrency is planned/executed at `:304-324` and `:500-528`.
- Typed transition-risk evidence: partially resolved; `TransitionRiskEvidencePayload` exists at `src/regime_detection/models.py:73-80`.
- Config v1/v2 split: intentionally reverted/deferred by user decision. Current debt is "large config file", not "missing v1/v2 files".
- Timeline output builder: partially resolved; helper extraction exists, but evidence construction and output assembly still contain untyped dict payloads.
- Artifact URI contract: resolved; `StoredArtifact.uri` is a fully qualified URI contract in `src/regime_data_fetch/artifact_store.py`.
- V2 gate fail-closed behavior: resolved enough for prior finding; gate script tests cover failure behavior, and CI has a slow/V2 gate job.
- FOMC/news sentiment skip-gate concerns: resolved enough; fixture-backed tests exist.

## Findings

| ID | Severity | Status | Finding | Evidence |
|---|---:|---|---|---|
| TD-001 | P1 | New | `profile_engine_30d.py` has a real post-axis-split broken hook: it imports `regime_detection.axis_series` then dereferences inflation-growth helpers that now live in `axis_builders/series.py`. | `scripts/profile_engine_30d.py:347-353`; live import check reports `build_inflation_growth_rule_inputs_by_date=False` and `evaluate_inflation_growth_rules=False` on `axis_series`. |
| TD-002 | P1 | Still open | Full pyright is not actionable yet: 714 errors across runtime/data/scripts, while CI only checks `src/regime_detection/config.py`. This lets typed contract drift accumulate outside config. | `.github/workflows/ci.yml:63-66`; pyright examples from `_v2_calibration_helpers.py:58`, `profile_engine_30d.py:351`, `timeline.py:89`, `transition_risk.py:138`. |
| TD-003 | P1 | Still open | Evidence models are only partially typed. Most evidence payload classes still inherit `RootModel[dict[str, Any]]`, so Pydantic accepts arbitrary payloads and pyright cannot enforce field contracts. | `src/regime_detection/models.py:22-70`. |
| TD-004 | P1 | Still open | Timeline still builds typed output objects using plain dict evidence payloads. This is a direct source of pyright failures and weakens the typed evidence migration. | `src/regime_detection/timeline.py:84-90`, `:230-240`. |
| TD-005 | P1 | New | Transition-risk now has a typed payload class, but callers/tests still use old `{}` or `{"warnings": []}` evidence shapes, and the default test suite currently fails on this mismatch. Runtime code also passes a plain dict into the typed output. | `src/regime_detection/transition_risk.py:136-142`; payload class at `models.py:73-80`; failing test constructors at `tests/test_schema_and_timeline.py:155-158` and `tests/test_v2_comparison.py:205`. |
| TD-006 | P1 | Still open | The feature-store registry exists, but its internal bus is `values: dict[str, Any]`; final assembly depends on string keys and untyped values. A misspelled key or wrong feature object is caught late or only by runtime tests. | `src/regime_detection/feature_store.py:140-143`, `:667-685`. |
| TD-007 | P2 | Still open | `build_feature_store` remains the central feature assembly choke point with many optional configs and an explicit TODO to decompose it. The registry reduced the if-chain shape but did not create typed builder outputs. | `src/regime_detection/feature_store.py:630-685`, TODO at `:644-646`. |
| TD-008 | P2 | Still open | Axis implementation moved out of `axis_series.py`, but the new module is a 1455-line all-axis builder file importing every rule family and owning every axis pipeline. The "full split" is not exhausted. | `src/regime_detection/axis_builders/series.py:1-77`, axis builders at `:118`, `:351`, `:941`, `:1113`, `:1320`. |
| TD-009 | P2 | Still open | `Config` remains a 1103-line monolith with `RegimeConfig` and all V2 optional sub-configs in one file. This is accepted deferred debt after the v1/v2 split was reverted. | `src/regime_detection/config.py:1013-1055`. |
| TD-010 | P2 | Partially resolved | Fetch-mode classification/concurrency moved into a registry, but invocation still lives in a long `_run_unattended_fetch_mode` if-chain. Adding a mode requires editing both registry data and dispatch logic. | Registry at `scripts/fetch_regime_engine_v1_data.py:74-103`; dispatch at `:531-663`. |
| TD-011 | P2 | Still open | `profile_engine_30d.py` remains a 1579-line operational script that loads all input families, monkeypatches runtime modules for timing, and generates reports. It is a product surface, not just a helper script. | Timing wrapper at `scripts/profile_engine_30d.py:338-370`; input orchestration at `:1115-1155`. |
| TD-012 | P2 | Still open | Acquisition event calendar remains a large orchestration module: fetches official sources, builds curated candidates, writes YAML, and formats candidate/validation records in one file. | `src/regime_data_fetch/event_calendar.py:280-335`, `:638-669`, `:768-820`. |
| TD-013 | P2 | Still open | Aggregate EPS acquisition combines operator/manual detection, direct download, browser fallback, parsing, output/report writing, and Wayback backfill in one module. | `src/regime_data_fetch/aggregate_eps.py:185-235`, `:493-500`, `:700-745`. |
| TD-014 | P2 | Still open | Investing.com live fetch has no retry/backoff wrapper around the JSON request path; a single transient URL open failure fails that operation. | `src/regime_data_fetch/investing_live.py:897-902`. |
| TD-015 | P2 | Still open | `AcquisitionStore` owns schema creation, artifact-store selection, run lifecycle, artifact records, lineage, and ad hoc migrations. That makes persistent-state changes high blast radius. | `src/regime_data_fetch/acquisition_store.py:40-59`, `:361-423`, `:600-650`. |
| TD-016 | P2 | Still open | SQLite schema migrations are embedded as opportunistic `ALTER TABLE` checks during store initialization, with no explicit schema version/migration history. | `src/regime_data_fetch/acquisition_store.py:638-650`. |
| TD-017 | P2 | Still open | `fetch_text_result` is typed and logs, but the compatibility facade `fetch_text_url` still returns an empty string on error. Any caller using the facade can silently convert source failure into empty content. | `src/regime_data_fetch/event_sources/_common.py:67-80`. |
| TD-018 | P2 | Still open | Strategy response construction builds a loose dict and unpacks it into a typed Pydantic model, producing type-checking failures and hiding field-level contract mistakes. | `src/regime_detection/strategy_response.py:124-139`. |
| TD-019 | P2 | Still open | V2 calibration and audit helpers have broad pandas typing debt, including `.dt` on inferred `DatetimeIndex` and `sort_values`/`rename` overload mismatches. Some are probably harmless stubs noise, but CI does not force triage. | pyright examples: `scripts/_v2_calibration_helpers.py:58-77`, `scripts/audit_layer2_30d.py:167-171`. |
| TD-020 | P3 | Still open | Dev tooling is incomplete for repeatable debt audits: `vulture` and `pydeps` are not installed and not declared in `[project.optional-dependencies].dev`. | `pyproject.toml:27-42`; command failures: `No module named vulture`, `No module named pydeps`. |
| TD-021 | P3 | Still open | Supply-chain audit is not repo-scoped. Running `pip-audit` against the current environment reports many unrelated package vulnerabilities and skips local packages, so it is noisy but not enforceable. | `pyproject.toml:5-18`, `:27-42`; `pip-audit` output skipped `regime-detection (2.0.0)` and other local packages. |
| TD-022 | P3 | Accepted risk | Default local pytest excludes `slow`, and `v2_shadow` remains off by default. CI now has a path-sensitive slow/V2 gate job, so this is acceptable if shadow runs remain an explicit operational gate. | `pytest.ini:1-11`; `.github/workflows/ci.yml:71-120`. |

## Top 5 Concrete Tasks

### Task 1 - Fix profiler hook after axis split

Subtasks:

1. Move the inflation-growth timing hook in `scripts/profile_engine_30d.py` to patch `regime_detection.axis_builders.series`, or expose stable instrumentation hooks from the axis builder module.
2. Add a focused test that calls `_timed_inflation_growth_builder` and asserts it no longer raises `AttributeError` when the wrapped builder runs.
3. Run `python3 -m pytest tests/test_profile_engine_30d.py -q`.

Acceptance evidence:

- `PYTHONPATH=src` import check shows the patched module owns all referenced attributes.
- The focused profiler test fails before the fix and passes after.

### Task 2 - Typed evidence model slice

Subtasks:

1. Replace placeholder dict evidence for network fragility and monetary pressure timeline fallbacks with explicit typed payload models.
2. Update transition-risk construction to instantiate `TransitionRiskEvidencePayload` explicitly.
3. Add tests that invalid evidence fields are rejected for the typed models.
4. Run pyright on the touched files only, then add them to the CI pyright list.

Acceptance evidence:

- Pyright no longer flags `timeline.py` evidence construction for the touched outputs.
- Runtime output JSON remains backward-compatible where V1 frozen replay applies.

### Task 3 - Feature-store builder output typing

Subtasks:

1. Replace `_FeatureStoreBuildState.values: dict[str, Any]` with a typed intermediate object or one dataclass field per feature.
2. Make each `_FeatureStoreBuilder` return a typed update instead of mutating string-keyed shared state.
3. Keep `build_feature_store(...)` as the public entry point until the typed builder bus is stable.
4. Run feature-store and timeline tests.

Acceptance evidence:

- No string-key lookup remains in final `FeatureStore(...)` assembly.
- Pyright can validate the feature object types used by `FeatureStore`.

### Task 4 - Fetch-mode registry owns invocation

Subtasks:

1. Extend `FetchModeSpec` with an invocation callable or adapter object.
2. Move each branch of `_run_unattended_fetch_mode` into a mode-specific callable with owned argument validation.
3. Keep `_plan_fetch_mode_execution` unchanged except for reading registry metadata.
4. Add a test that a new fake mode can be planned and invoked through registry metadata without editing an if-chain.

Acceptance evidence:

- `_run_unattended_fetch_mode` disappears or becomes a one-line registry dispatch.
- Existing `tests/test_fetch_workflow.py` coverage still passes.

### Task 5 - Incremental pyright ratchet

Subtasks:

1. Keep the current full pyright command as a non-blocking audit target.
2. Expand CI pyright from `src/regime_detection/config.py` to the fixed files from Tasks 1-4.
3. Add a `scripts/typecheck_changed.py` or documented command if path-scoped pyright becomes too noisy.
4. Track remaining error count in this audit file or a dedicated type debt note.

Acceptance evidence:

- CI fails on type regression in at least the files touched by the debt tasks.
- Full pyright error count decreases from 714.

## Quick Wins

- Add `vulture`, `pydeps`, and `pip-audit` to a separate optional `audit` extra instead of the default `dev` extra.
- Add a tiny regression test around `profile_engine_30d._timed_inflation_growth_builder`.
- Replace `fetch_text_url` callers with `fetch_text_result`, then delete the empty-string facade.
- Add retry/backoff around `investing_live._request_json`.
- Move acquisition-store schema DDL into a dedicated schema/migration module without changing tables.

## Looks Bad But Is Fine For Now

- Keeping a single `src/regime_detection/config.py` is an explicit current decision. The debt is size/cohesion, not "missing v1/v2 split".
- `axis_series.py` no longer being the implementation home is a real improvement. The remaining problem is the new `axis_builders/series.py` size and coupling.
- Default pytest excluding `slow` is acceptable because CI has a path-sensitive slow/V2 gate job. `v2_shadow` should remain an explicit long-running operational gate unless product requirements change.
- `pip-audit` output is useful as an environment warning, but it is too noisy to treat as this repo's dependency baseline until a lockfile or isolated environment audit exists.

## Open Questions

1. Should typed evidence preserve dict-compatible access permanently, or can callers migrate to field access?
2. Should pyright be ratcheted by file list, package, or error-count budget?
3. Should fetch mode invocation support concurrent operator-assisted modes, or should concurrency stay unattended-only?
4. Should acquisition-store migrations be versioned in SQLite metadata or kept as code-only migrations?
5. Should `profile_engine_30d.py` remain a script with imports, or become a package module with a thin CLI wrapper?
