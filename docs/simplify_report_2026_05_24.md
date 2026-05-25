# /simplify Report — branch `avinash8891/run-regime-2016-present` vs `main`

Generated 2026-05-24. Branch diff: 352 files, ~91k insertions / ~16k deletions.

Four parallel reviewers were run, one per subsystem (`scripts/`, `src/regime_data_fetch/`, `src/regime_detection/`, `tests/` + configs/CI), each evaluating three dimensions (code reuse, code quality, efficiency).

No code was changed. This report is plan-only.

---

## Tier 1 — High-leverage, mechanical fixes (recommended)

| # | Area | Change | LoC delta |
|---|------|--------|-----------|
| 1 | `scripts/run_historical_walkforward.py` L41–243 | Replace local `RUNS_SCHEMA` + `_ensure_layout`/`_open_db`/`_insert_run_row`/`_update_run_row_success`/`_update_run_row_failure`/`_write_archived_inputs` with the public `regime_detection.shadow_storage` API. | −150 |
| 2 | `scripts/{verify_fixtures,audit_step1_harness,run_historical_walkforward}.py` | Use `regime_data_fetch.artifact_store.sha256_file`, `regime_detection.shadow_storage.utc_iso_now`, `regime_data_fetch.cli_common.parse_date` instead of 3–4 private re-implementations. | −50 |
| 3 | `scripts/_v2_calibration_helpers.py` L452–469 | Drop local `CROSS_ASSET_SYMBOLS`; import from `regime_detection.fragility_universe`. | −20 |
| 4 | `scripts/publish_canonical_snapshot.py` L126–156 + `scripts/upload_missing_ohlcv_to_manifest.py` L67–93 | Merge `_canonicalize_parquet_bytes` / `_canonicalize` into one helper in `regime_data_fetch/`. | −60 |
| 5 | `tests/conftest.py` L206–280 | Replace synthetic-OHLCV `wobble` builder + `VIXY → VIX` rename with real fixture rows. Violates global "no stubs/synthetic data when real constants exist" rule. | small but rule-critical |
| 6 | `scripts/profile_engine_timers.py` L113–349 | Replace 236-line list of `(module, attr, _timed_wrapper(...))` patches with a declarative `(module, attr, stage_name)` table. Also delete duplicate `_timed_method_wrapper`. | −150 |
| 7 | `src/regime_detection/axis_builders/breadth.py` L97–179 | Refactor onto `build_per_label_axis_outputs` like its 4 sibling axes (`credit_funding`, `inflation_growth`, `network_fragility`, `volume_liquidity`). | −80 |
| 8 | `src/regime_detection/axis_series.py` L16–20 | Make `_STALENESS_SENTINEL` / `_calendar_staleness_days_series` / `_trading_staleness_series` public in `axis_builders/staleness.py`; drop the re-export indirection. | small |

**Total Tier 1**: ~510 LoC removed, low risk, ~1 PR.

---

## Tier 2 — Meaningful refactors (worth doing, larger surface)

- **Shared HTTP helper in `regime_data_fetch/_http.py`** — collapse 13 modules' `urllib.request.Request(..., User-Agent=...)` + `urlopen` + retry into one helper; unify 5 distinct UA strings (`HTTP_USER_AGENT`, `YAHOO_USER_AGENT`, bare `Mozilla/5.0`, `regime-engine-fetcher/2.0`, `regime-detection-event-calendar/1.0`, `regime-detection-fetch/1.0`). ~150 LoC.
- **`AcquisitionStore.run(fetch_type, params)` context manager** — eliminates the start/finish-run boilerplate across ~12 fetchers (`aggregate_eps`, `fetch_workflow` × 2, `aaii_sentiment`, `pmi` × 2, `event_calendar`, `local_daily_ohlcv_sqlite` × 2, `investing_archive`, `cleveland_fed_nowcast`, `sf_fed_news_sentiment`, `fomc_minutes`, `powell_speeches`, `pit_constituents`). ~300 LoC.
- **`MarkupFetcherMixin` + shared `_record_html`/`_dedupe`** across `event_sources/{official_boe,official_ecb,official_boj}.py`. ~80 LoC.
- **Normalized-import dispatch table** in `acquisition_consolidation_normalized.py` — 11 near-identical `_import_*_rows` collapse to one spec-driven function. ~250 LoC.
- **Parametrize `tests/fixtures/audit_step1/{current,historical}/*.json`** (18 nearly-identical fixtures, 15 KB each) — fold to 2 templates + runner-name set.
- **`tests/fixtures/configs/core3-v2-fast.yaml` (751 lines, 99% canonical copy)** — replace with runtime overlay via `model_copy(update={...})`; delete file. Eliminates drift trap.
- **Drop `getattr(config, "fieldname", default)`** in `inflation_growth_rules.py` (L24, 52, 116, 131, 160), `network_fragility_rules.py` (L377–381), `credit_funding.py` (L626–648). Pydantic configs already have explicit defaults; `getattr` silently hides typos.
- **Per-axis bulk-reindex** in `axis_builders/credit_funding.py` (per-session `.loc[dt]` × 8 series) and `axis_builders/inflation_growth.py` — pre-materialize `.to_numpy()` arrays before the loop. Perf win in the 2016-present hot path.
- **Shared `bootstrap_runner_args` helper** to absorb the identical parse → materialize → resolve dance in `profile_engine` / `run_v2_walkforward_gate` / `run_v2_shadow_ab_gate` / `audit_layer2_30d` / `run_v2_calibration`.
- **`_path_bootstrap.py`** — extract `sys.path.insert(...)` boilerplate copied across ~8 scripts (or just rely on `pip install -e`).

---

## Tier 3 — Large refactors (out of scope for /simplify; queue separately)

### Files exceeding the 800-line ceiling
- `src/regime_detection/feature_store.py` — 912
- `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py` — ~1210
- `src/regime_detection/axis_series.py` — 1484 (much already extracted)
- `src/regime_data_fetch/acquisition_store.py` — 752
- `src/regime_data_fetch/investing_live.py` — 765
- `src/regime_data_fetch/aggregate_eps.py` — 744
- `src/regime_data_fetch/event_calendar.py` — 808
- `src/regime_data_fetch/pmi.py` — 113 net (legacy ~700+)
- `tests/conftest.py` — 510 (near limit)
- `tests/test_aggregate_eps.py` — 1041
- `tests/test_credit_funding_axis_engine.py` — 1015
- `tests/test_inflation_growth.py` — 838
- `tests/test_event_source_group_b_conflict_budget.py` — 837
- `tests/test_inflation_growth_axis_engine.py` — 824
- `tests/test_profile_engine_reporting_loaders.py` — 818

### Functions exceeding the 50-line ceiling (sample)
- `scripts/profile_engine.py::main` — 277 lines
- `scripts/run_historical_walkforward.py` per-session loop — 95 lines
- `src/regime_detection/axis_builders/credit_funding.py::_build_credit_funding_for_spread_source` — 263 lines
- `src/regime_detection/axis_builders/inflation_growth.py::build_inflation_growth_axis_series` — 217 lines
- `src/regime_detection/transition_risk_series.py::build_transition_risk_outputs_by_date` — 184 lines
- `src/regime_detection/timeline.py::_build_timeline_output_for_day` — 138 lines
- `src/regime_data_fetch/event_calendar.py::run_us_event_calendar_fetch` — 168 lines
- `src/regime_data_fetch/fetch_workflow.py::run_market_fetch` — 189 lines, `::run_macro_fetch` — 177 lines
- `src/regime_data_fetch/aggregate_eps.py::run_aggregate_eps_fetch` — 118 lines, `::run_wayback_aggregate_eps_fetch` — 250 lines

### Performance opportunities (need perf measurement)
- Parallelize sequential network loops: FRED 17-series, Yahoo chart 762 symbols, GDELT per-day, FOMC historical years, manifest materialize.
- `timeline.py::_hmm_state_persistence_days` — O(N²) backward scan; vectorize via `groupby(cumsum(shift)).cumcount()`.
- `transition_risk_series.py::_strict_lookup_by_sessions` — extra `.copy()` per call.
- `local_daily_ohlcv_sqlite.py::_write_daily_ohlcv_symbol_tree` — re-reads existing parquet per symbol in groupby loop (762 disk reads on rerun); add sha-based change check.
- `acquisition_store.py::_connect` — fresh `sqlite3.connect()` per method call.

---

## Cross-cutting themes (root causes)

1. **No shared HTTP layer.** `event_sources/_common.py` has `fetch_text_result` but only event_sources can import it without a layering break. Promote to `regime_data_fetch/_http.py`.
2. **No shared "fetch run" lifecycle.** Every fetcher repeats the same `AcquisitionStore.start_fetch_run → try → finish_fetch_run` block. Needs a context manager on the store.
3. **No shared per-session classifier loop.** 6 axis builders re-implement the same loop. Needs a `run_per_session_classifier(sessions, gates, rule_evaluator, evidence_builder)` helper.
4. **No shared bulk-reindex pattern.** Per-session `.loc[dt]` lookups across many series appear in multiple hot paths.
5. **`getattr(config, ...)` with defaults** treats Pydantic configs as untrusted dicts, defeating typo detection.
6. **Synthetic test data where real fixtures exist.** `tests/conftest.py` builds `wobble`-perturbed OHLCV instead of using the 5,553-row real V2 fixture.

---

## Subsystem-specific reports

The four detailed reports (one per reviewer) are not persisted to disk to keep the repo clean. They covered:

- `scripts/` — 21 files, ~50 findings; biggest single deletion is `run_historical_walkforward.py` shadow_storage duplication.
- `src/regime_data_fetch/` — ~40 files, ~70 findings; dominated by HTTP + AcquisitionStore boilerplate.
- `src/regime_detection/` — ~50 files, ~70 findings; dominated by per-axis classifier loop and bulk-reindex.
- `tests/` + configs + CI — ~30 findings; dominated by toy/synthetic data violations and the canonical-config copy.

If you want the full per-reviewer transcripts, re-run `/simplify` scoped to a single subsystem.
