# Tech Debt Audit - regime-detection

Generated: 2026-05-17

## Executive summary

- The largest debt concentration is orchestration: `scripts/fetch_regime_engine_v1_data.py`, `scripts/profile_engine_30d.py`, `src/regime_detection/feature_store.py`, `src/regime_detection/axis_series.py`, and `src/regime_detection/timeline.py` each carry multiple responsibilities and are also the highest-churn surfaces.
- The repo has good behavioral tests, but static quality gates are not yet real gates: `ruff` currently fails with 37 findings and `pyright` reports 1,315 errors.
- The artifact-store/materialization work is directionally right, but URI semantics are still ambiguous and documented as a TODO inside the core abstraction.
- Several production-like scripts treat per-session classifier failures as report rows and still exit `0`, which can make a failed gate look like a successful CI/job run unless an operator reads the report.
- Integration coverage for some important real-data contracts is skip-gated on local `data/raw/` materialization, so default CI does not prove those live-data contracts.
- Temporal normalization is acknowledged in the docs as a next slice, but current loaders/writers still mix date-only fields, ET fields, naive `pd.to_datetime`, and UTC strings.
- Config and output models are strongly validated with Pydantic, but evidence payloads are still free-form `dict[str, Any]`, which leaves the riskiest cross-axis contract weakly typed.
- The fetch layer has many source-specific validators, but there is no single source-normalization contract yet; missing or failing optional sources often degrade to empty rows with logs rather than run-level status.
- Some code that looks suspicious is intentional: V1 wire-shape rewriting, cluster output without economic labels, and PIT breadth "biased research" labeling are load-bearing compatibility or safety choices, not debt to remove casually.

## Architectural mental model

This checkout is a unified V1+V2 regime engine. `src/regime_detection` owns the runtime classifier: `engine` builds a `MarketContext`, `feature_store` computes V1/V2 features, `axis_series` turns feature seams into per-session axis outputs, `transition_risk_series` composes transition evidence, and `timeline` emits the final Pydantic `RegimeOutput`/`RegimeTimeline` wire shape. V2 is cumulative on V1, so a lot of code intentionally preserves V1 byte identity while adding optional V2 seams.

`src/regime_data_fetch` owns source acquisition, canonical artifacts, local materialization, and SQLite provenance. `scripts/` are not thin wrappers; many are operational products themselves: fetch/backfill, materialize, historical walk-forward, shadow checks, V2 calibration, 30-day profiling, and layer-2 audits. The specs under `docs/` are normative and often more detailed than the code. The repo's practical risk is not a lack of tests; it is that the operational surface is broad, recent, and only partially enforced by static gates and real-data CI.

## Findings table

| ID | Category | File:Line | Severity | Effort | Description | Recommendation |
|---|---|---:|---|---|---|---|
| F001 | Architectural decay | `src/regime_detection/axis_series.py:91` | High | L | `AxisSeriesBundle` has become the cross-axis hub for V1 plus many V2 axes, with optional maps for network, volume, credit, proxy credit, effective credit, monetary, and inflation outputs. | Split per-axis builders into small modules with one shared interface, then keep `axis_series.py` as an assembly layer only. |
| F002 | Architectural decay | `src/regime_detection/axis_series.py:367` | High | M | `NetworkFragilitySeriesClassifier.build` is a 165-line orchestration method handling data-quality gates, cross-axis lookups, rule evaluation, hysteresis, and output construction. | Extract day-quality assessment, dependency label resolution, and output construction into tested helpers. |
| F003 | Architectural decay | `src/regime_detection/axis_series.py:716` | High | M | Credit/funding's `_build_for_spread_source` is 214 LOC and handles both source policy and per-day classifier mechanics. | Split source selection from rule evaluation so OAS/proxy policy changes cannot alter per-day scoring by accident. |
| F004 | Architectural decay | `src/regime_detection/feature_store.py:258` | High | L | `build_feature_store` is 377 LOC and computes every V1/V2 seam in one function; the file already carries a TODO to decompose it. | Convert to a registry/list of feature builders that each declare required context keys and config dependencies. |
| F005 | Architectural decay | `src/regime_detection/timeline.py:73` | High | L | `build_regime_timeline` is 315 LOC and handles minimum-history math, feature/axis building, transition risk, V2 optional fields, routing, and output emission. | Extract the history-window resolver and output-emission loop; keep final `RegimeTimeline` assembly separate from feature orchestration. |
| F006 | Architectural decay | `scripts/fetch_regime_engine_v1_data.py:79` | High | L | The main fetch CLI is 441 LOC and combines argument parsing, mode validation, source orchestration, artifact-store policy, and manifest emission. | Replace the if-chain with a fetch-mode registry whose entries own args, validation, and invocation. |
| F007 | Architectural decay | `scripts/profile_engine_30d.py:959` | Medium | M | `profile_engine_30d.main` is 276 LOC and repeats engine setup, timing, feature-store rebuild, invariant checks, and report printing. | Separate profiling instrumentation from data loading and report rendering. |
| F008 | Architectural decay | `src/regime_data_fetch/acquisition_consolidation.py:22` | Medium | S | Core package code carries machine-specific `/private/tmp/...` default consolidation DB paths. | Move these defaults into a script/config file; require explicit paths for library use. |
| F009 | Consistency rot | `src/regime_data_fetch/artifact_store.py:23` | High | M | `StoredArtifact.uri` is documented as relative for local stores but scheme-qualified for S3, so manifest consumers must know which backend produced it. | Normalize to a single URI contract, preferably fully-qualified `file://`/`s3://`, and update manifest materialization tests. |
| F010 | Consistency rot | `src/regime_detection/loaders.py:420` | High | M | Event dates are normalized with `pd.to_datetime(...).dt.date`, while `publication_date` uses `errors="coerce"` plus manual validation, and other loaders use naive `DatetimeIndex` conversion. | Implement the documented shared temporal-normalization module and route every canonical loader/writer through it. |
| F011 | Documentation drift | `docs/market_data_fetch_plan.md:87` | High | M | The docs explicitly list a temporal-normalization implementation TODO, but the current code still has mixed temporal handling. | Promote the TODO into a tracked debt item or implement it before treating artifact portability as complete. |
| F012 | Type and contract debt | `src/regime_detection/models.py:67` | High | L | Axis evidence is still `dict[str, Any]` on the shared output model. This is the highest-value wire payload but has no per-axis schema. | Add typed evidence models axis-by-axis when touching each classifier, starting with credit/funding and inflation/growth. |
| F013 | Type and contract debt | `src/regime_detection/models.py:456` | Medium | M | V1 wire compatibility rewrites raw dump payloads after Pydantic validation, which keeps compatibility but bypasses model-level schema clarity. | Keep the behavior, but wrap legacy serialization behind named serializer methods and golden tests so callers know this is a compatibility projection. |
| F014 | Type and contract debt | `src/regime_detection/market_context.py:28` | Medium | M | `constituent_ohlcv` uses `SkipValidation`, so a major PIT input bypasses Pydantic even though downstream V2 breadth depends on it. | Add a source-specific validator for the dict's required columns/index shape before constructing `MarketContext`. |
| F015 | Type and contract debt | `scripts/profile_engine_30d.py:878` | Medium | S | Pyright catches that `ProfileInputBundle` annotations do not match actual values for `central_bank_text_releases` and `pit_constituent_intervals`. | Fix the dataclass annotations or the loaded values; this makes profiling code a useful type-check target. |
| F016 | Test debt | `tests/test_central_bank_text_cycle_regression.py:148` | High | M | Real FOMC integration checks skip whenever `data/raw/fomc_minutes/fomc_minutes.parquet` is absent. Default CI therefore does not prove the live-parquet cycle behavior. | Add a small checked-in redacted/captured FOMC fixture or CI materialization step for the two cycle windows. |
| F017 | Test debt | `tests/test_news_sentiment_coverage.py:125` | High | M | News sentiment coverage/freshness checks skip unless both live news parquet and `data/raw/daily_ohlcv` exist locally. | Add a fixture-backed coverage contract in CI and reserve local-data tests for larger operational validation. |
| F018 | Test debt | `pytest.ini:2` | Medium | S | Default pytest excludes `slow` and uses xdist, so historical walk-forward and V2 gate tests are not part of the default CI proof. | Add a separate required CI job for gate/slow tests on PRs touching engine, data, or V2 config paths. |
| F019 | Tooling debt | `.github/workflows/ci.yml:59` | High | S | CI installs dev deps and runs pytest, but does not run Ruff, Pyright, package build, or audit tooling. | Add separate CI steps for `ruff check`, a scoped type-check target, and package import/build smoke tests. |
| F020 | Tooling debt | `scripts/run_v2_calibration.py:31` | Medium | S | Ruff reports unused imports and trivial f-string issues in operational scripts, showing lint is not currently a clean gate. | Clean the current 37 Ruff findings, then make Ruff required in CI. |
| F021 | Tooling debt | `scripts/approve_group_b_candidate.py:16` | Low | S | Multiple scripts use `sys.path.insert` before imports, triggering E402 and making script/package invocation inconsistent. | Prefer `python -m scripts.<name>` with package-safe imports, or configure Ruff exceptions intentionally for these scripts. |
| F022 | Error handling | `scripts/run_v2_walkforward_gate.py:198` | High | S | Per-session `engine.classify` exceptions are logged and counted, but the script still writes a report and returns `0`. | Exit non-zero when `v1_errors` or `v2_errors` is nonzero, or add an explicit `--allow-errors` flag. |
| F023 | Error handling | `scripts/run_v2_shadow_ab_gate.py:180` | High | S | The shadow A/B gate has the same fail-open classifier loop and returns success after session errors. | Make session errors a failed gate unless explicitly downgraded by operator flag. |
| F024 | Error handling | `src/regime_data_fetch/event_sources/_common.py:56` | Medium | S | Shared event-source fetch returns an empty string on URL failure; downstream parsers cannot distinguish "empty valid page" from "source failed" unless callers add side channels. | Return a typed result with status/error, or raise and let orchestrators record source-level failure explicitly. |
| F025 | Error handling | `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py:141` | Medium | M | GPR/GDELT fetch failures degrade to empty candidate lists after logging; this is acceptable for optional evidence but not surfaced as a run-level partial failure. | Record skipped source counts/status in the fetch report and propagate >threshold missing evidence as failed/partial. |
| F026 | Performance | `scripts/fetch_regime_engine_v1_data.py:221` | Medium | M | `--fetch all` runs many independent upstream fetches serially; the TODO notes wall-clock is sum(N). | Add a conservative per-source concurrency plan with rate-limit groups rather than one global serial chain. |
| F027 | Performance | `src/regime_detection/timeline.py:230` | Medium | M | Final timeline emission loops per day and performs many dict lookups plus nested Pydantic model construction; acceptable today, but it is on every classification window. | Keep the loop, but extract and benchmark an output-emission builder before adding more V2 fields. |
| F028 | Performance | `src/regime_data_fetch/alpaca_daily.py:71` | Medium | M | Alpaca daily fetch loops batches serially and uses print progress instead of structured progress logging. | Add rate-limit-aware concurrency or at least structured batch timing/row counts in stdlib logging. |
| F029 | Security hygiene | `src/regime_data_fetch/local_daily_ohlcv_sqlite_reader.py:45` | Low | S | SQL uses dynamic placeholder construction. Values are parameterized, so this is not injection-prone, but table names elsewhere are string-formatted. | Keep values parameterized; centralize safe table-name constants and avoid expanding dynamic SQL beyond fixed constants. |
| F030 | Documentation drift | `src/regime_detection/config.py:160` | Medium | S | Comments say some V2 trend labels remain deferred, but the same config class includes `euphoria_*` rule defaults and the live code wires euphoria evidence. | Update comments to separate "feature/rule config present" from "operationally qualified". |
| F031 | Documentation drift | `docs/v2_slice_gate_checklist.md:29` | Medium | S | The checklist is still blank-template style (`XYZSeriesClassifier`) while multiple V2 slices have already landed. | Replace template placeholders with a living checklist keyed to current shipped axes and required evidence. |
| F032 | Dependency/config debt | `pyproject.toml:15` | Medium | S | Runtime dependency `bayesian-changepoint-detection>=0.2.dev1` pins to a dev release range without an upper compatibility bound beyond the package family. | Add a comment/constraint explaining the exact API used, or vendor-wrap the dependency behind a small adapter with tests. |
| F033 | Dependency/config debt | `src/regime_detection/config.py:1012` | Medium | M | `RegimeConfig` is one 90-field-ish model file with many nested V2 classes; it is high churn and hard to review for spec drift. | Split config models by axis module, then re-export through `config.py` to preserve public imports. |
| F034 | Observability | `scripts/profile_engine_30d.py:1109` | Low | S | Profiling output is many `print` lines, not structured logs or a machine-readable report artifact. | Emit JSON plus markdown/text so CI and agents can diff timings and seam statuses. |
| F035 | Observability | `src/regime_data_fetch/alpaca_daily.py:74` | Low | S | Fetch progress uses `print(..., flush=True)` and omits logger context like run id/acquisition db. | Convert to module logger and include source, batch, symbol counts, and run id when available. |
| F036 | Security hygiene | `src/regime_data_fetch/alpaca_daily.py:16` | Low | S | Alpaca credentials are read directly from env inside the client helper. It does not print secrets, but missing env vars raise raw `KeyError`. | Raise a project-scoped error naming the missing env var and remediation without exposing values. |

## Top 5 if you fix nothing else

1. **F022/F023 - Make V2 gates fail closed on classifier errors.**
   Sketch: after both `_classify_*` calls, add `if v1_errors or v2_errors: logger.error(...); return 2`. Add `--allow-session-errors` only if an operator truly needs exploratory reports.

2. **F004/F005 - Decompose feature/timeline orchestration.**
   Sketch: introduce `FeatureBuilder` objects with `name`, `required_context`, `config_getter`, and `build(context)`; move one low-risk seam first, then require byte-identical V1 replay before continuing.

3. **F009/F010/F011 - Finish the artifact/temporal contract.**
   Sketch: make `StoredArtifact.uri` backend-independent, add `time_normalization.py`, and fail canonical writes that emit mixed naive datetimes, ET fields, or timestamp/date ambiguity.

4. **F016/F017/F018 - Make real-data contracts visible in CI.**
   Sketch: create tiny captured fixtures for FOMC/news coverage and add a separate gate job for slow/gate markers when engine, loaders, configs, or scripts change.

5. **F019/F020 - Turn Ruff into a required gate.**
   Sketch: fix the current 37 findings, decide whether `scripts/` path shims get a deliberate per-file ignore, then add `python -m ruff check .` to CI.

## Quick wins

- [ ] F020: Remove unused imports and extraneous f-string prefixes reported by Ruff.
- [ ] F021: Add intentional Ruff ignores or package-safe invocation for scripts with import-path shims.
- [ ] F022/F023: Return non-zero from V2 gate scripts when session errors are nonzero.
- [ ] F030: Fix stale comments in `TrendDirectionV2RulesConfig`.
- [ ] F031: Replace `XYZSeriesClassifier` placeholder in the V2 gate checklist.
- [ ] F034/F035: Convert profiling/fetch progress `print` calls to logger calls or JSON report output.
- [ ] F036: Wrap missing Alpaca env vars in a clearer error.

## Things that look bad but are actually fine

- `src/regime_detection/models.py:467` rewrites V1 wire payloads after model dump. This looks like a schema smell, but it is preserving V1 byte identity while V2 fields exist in the internal model. Do not remove it without `tests/test_v1_frozen_replay.py` coverage.
- `src/regime_detection/models.py:294` deliberately omits mapped economic labels for clusters. That matches the V2 requirement that HMM/GMM clusters are not auto-labeled.
- `src/regime_detection/axis_series.py:292` labels PIT breadth mode as `pit_constituent_biased_research`. The wording is noisy but useful: it prevents biased constituent evidence from looking like a production-grade PIT vendor feed.
- `src/regime_detection/market_context.py:145` passes PIT intervals/constituent OHLCV through when slicing. That looks inconsistent with reindexing other inputs, but the comments explain why slicing them there would silently disable PIT breadth.
- `src/regime_detection/config.py:1095` dispatches default config from package `__version__`. It is easy to dislike implicit version dispatch, but here it supports the V1/V2 packaged-config split and should be changed only with a migration plan.

## Open questions for the maintainer

- Should V2 walk-forward/shadow A/B gate scripts ever exit `0` with classifier errors, or should all error counts be hard failures?
- Are the `/private/tmp/...` acquisition consolidation defaults still intended to be runnable, or are they post-migration archaeology that should move out of `src/`?
- Which static tool should become authoritative first: Ruff only, Pyright scoped to `src/`, or a smaller hand-written type check target for runtime modules?
- Should `data/raw`-dependent integration tests become required through fixture snapshots, or should CI materialize a small object-store manifest?
- Is `event_calendar_fetch_report.json` a durable repo artifact or a generated local report? Its placement affects whether docs can cite it as current evidence.

## Verification and tooling evidence

- `python3 -m ruff check .` failed with 37 findings. Main classes: E402 script path-shim imports, F401 unused imports, and F541 f-strings without placeholders.
- `python3 -m pyright` failed with 1,315 errors. Most are pandas typing/annotation drift in scripts and tests; examples include mismatched `ProfileInputBundle` fields in `scripts/profile_engine_30d.py`.
- `python3 -m pip_audit` was unavailable: `No module named pip_audit`.
- `python3 -m vulture` was unavailable: `No module named vulture`.
- `python3 -m pydeps src/regime_detection --show-cycles --noshow` was unavailable: `No module named pydeps`.
- `python3 -m pytest -q` passed under default pytest options (`-q -m "not slow" -n 2`): progress reached `[100%]`; one warning remained from `tests/test_hmm_state.py::test_compute_hmm_features_returns_none_when_hmm_fit_fails` (`sklearn.base.ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (4)`).
