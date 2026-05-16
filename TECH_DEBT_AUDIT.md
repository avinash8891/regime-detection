# Tech Debt Audit — regime-detection
Generated: 2026-05-16

---

## Executive Summary

- **10 files exceed the 800-line hard limit** set in CLAUDE.md; `axis_series.py` reaches 1,426 lines.
- **55% test coverage** (from a representative test subset) with three critical modules under 25%: `loaders.py` (13%), `transition_score.py` (22%), `network_fragility_rules.py` (27%).
- **`evidence: dict[str, Any]`** on every output model erases all contract safety for the most-read field in every downstream consumer; 14 downstream `# type: ignore` comments in `transition_risk_series.py` are the visible symptom.
- **`sma_50` / `return_63d` are computed independently in three source files** — a live data-divergence bug waiting to emerge when the windows differ by a row.
- **CI does not run `ruff` or `mypy`** as standalone gates; 50 ruff violations exist today and would never fail a merge.
- **`axis_series.py` contains 11 single-method classifier classes** with no shared state — they are functions in disguise, inflating the file 7× beyond what coordination logic alone would require.
- **3 bare `except Exception:` blocks** swallow errors without logging, making production failures invisible.
- **`shadow_storage.py` has 0% coverage** (89 executable lines); the shadow-runner path is untested end-to-end.
- **README.md is 5 lines and says "V1"** — the repo is on V2 with 32 PRs.
- **No `bandit` or security scanner in CI**; no coverage gate enforced.

---

## Architectural Mental Model

`regime_detection` is an axis-based rule engine. Each "axis" (trend direction, volatility, breadth, credit, inflation/growth, etc.) is independently computed: features are extracted from market data into a `FeatureStore`, rule functions evaluate those features against config thresholds, and the results are wrapped in typed output models (`AxisOutput`, `BreadthStateOutput`, …). `axis_series.py` coordinates all classifiers and returns an `AxisSeriesBundle`. The `timeline.py` then assembles per-session `RegimeRow` outputs from the bundle.

`regime_data_fetch` is a separate package that acquires raw data from a dozen external sources (Alpaca, FRED, BLS, PMI, event calendars, EPS) and writes to SQLite acquisition stores. The two packages are loosely coupled — `regime_detection` consumes DataFrames from context objects; `regime_data_fetch` produces them.

V1 must remain byte-identical for archive replay (enforced by `test_v1_frozen_replay.py`). V2 extends V1 by adding new axes and refining rules behind config-version guards. This dual-track constraint explains many of the `v2_config is not None` guards and the `allow_v2_labels` flags throughout the core classifiers.

**Contradiction with README:** The README says "V1 regime detection engine." The codebase is firmly V2 in progress, with V2 axes, V2 calibration scripts, shadow A/B infrastructure, and a V2 spec. The README is simply wrong.

---

## Findings

| ID | Category | File:Line | Severity | Effort | Description | Recommendation |
|----|----------|-----------|----------|--------|-------------|----------------|
| F001 | Architectural decay | `src/regime_detection/axis_series.py:134` | High | M | 11 single-method classes (`TrendDirectionSeriesClassifier`, `VolatilitySeriesClassifier`, …) in one 1,426-line file. Each class has exactly one method (`build`) with no shared state, making the class a function wrapper. The file also violates CLAUDE.md's 800-line limit by 78%. | Convert each classifier class to a `build_*_axis_series` free function (matching the existing `build_raw_outputs` pattern in each axis module). `axis_series.py` retains only the `AxisSeriesBundle`, `AxisSeriesResult`, and the two public entry points. |
| F002 | Architectural decay | `src/regime_detection/trend_direction.py:58`, `src/regime_detection/feature_store.py:292`, `src/regime_detection/trend_character.py:235` | High | S | `sma_50` is computed by three independent `.rolling(50).mean()` calls on the same SPY close series. `return_63d` is duplicated between `trend_direction.py:60` and `trend_character.py:238`. A TODO at `trend_direction_v2.py:32` acknowledges this. If one window is ever off-by-one the two axes silently diverge. | Add `sma_50`, `sma_200`, and `return_63d` to `_rolling_stats.py` and call them from `feature_store.build_feature_store`. Delete the inline computations in `trend_direction` and `trend_character`. |
| F003 | Type & contract debt | `src/regime_detection/models.py:28` | High | L | `evidence: dict[str, Any]` on every output model (11 instances: lines 31, 47, 70, 85, 117, 144, 170, 180, 209, 285, and more). The TODO at line 28 admits this is known. Callers can populate evidence with arbitrary keys; mismatches are caught only at read time. | Define per-axis evidence dataclasses or TypedDicts (e.g., `TrendDirectionEvidence`). Replace `dict[str, Any]` field-by-field as each axis is touched. The TODO already names the plan — execute it. |
| F004 | Type & contract debt | `src/regime_detection/transition_risk_series.py:265` | High | S | 14 `# type: ignore` comments in one 399-line file — 3.5% of lines are type suppressions. Root cause: `transition_score_inputs_by_date: dict[date, dict[str, float \| str]] \| None` uses string keys for mixed-type values; mypy can't infer `.get(key)` types and `.get()` result passes to `float()`. | Replace the loose dict with a `TransitionScoreInputs` dataclass (fields: `hmm_top_state_prob_now`, `hmm_top_state_prob_5d_ago`, `change_point_score`, etc.). All 14 suppressions disappear and the signature becomes self-documenting. |
| F005 | Test debt | `src/regime_detection/loaders.py` | High | M | 13% coverage (23/183 executable lines covered). `loaders.py` is the sole path for loading market data frames from disk; failures here are silent data shape issues downstream. | Add tests for the key loader paths with real fixture files. At minimum: happy path for each loader, empty-file guard, schema mismatch. |
| F006 | Test debt | `src/regime_detection/shadow_storage.py` | High | M | 0% coverage. 89 executable lines, no tests. `shadow_storage.py` manages durable shadow-run state; an untested storage layer means replay bugs go undetected until the 60-day shadow window. | Add unit tests covering write/read round-trips, idempotent writes, and error paths. |
| F007 | Test debt | `src/regime_detection/transition_score.py` | High | M | 22% coverage (14/65 lines covered). `transition_score.py` computes the V2 composite score that drives regime labeling. Low coverage on the scoring arithmetic means a miscalibrated weight goes undetected by tests. | Test each score component computation with known inputs and assert exact float output. |
| F008 | Architectural decay | `src/regime_data_fetch/aggregate_eps.py` | Medium | M | 1,102 lines. Contains download, parse-legacy-workbook, parse-wayback-workbook, append, and orchestration logic. Three separate parse functions (`_parse_cell_value`, `_parse_row_for_snapshot`, `_parse_row_legacy`) share no common base despite nearly identical structures (`run_aggregate_eps_fetch` vs `run_wayback_aggregate_eps_fetch`). | Split into: `aggregate_eps_fetch.py` (download + orchestration), `aggregate_eps_parse.py` (cell/row parsing shared by both paths), `aggregate_eps_wayback.py` (Wayback-specific logic). |
| F009 | Architectural decay | `src/regime_data_fetch/event_calendar.py` | Medium | M | 1,129 lines. Combines FOMC event fetching, BLS event fetching, Group A/B candidate building, text fetching, YAML serialization, label resolution, and window expansion. Each is a distinct concern. | Extract FOMC and BLS fetchers into separate files. Move label resolution and YAML writing into smaller helpers. |
| F010 | Architectural decay | `src/regime_detection/config.py` | Medium | S | 1,093 lines. A single file defines every config model for both V1 and V2 axes. The file has 43 top-level classes/functions. Adding a new V2 config model requires touching this 1,093-line file and invites accidental side-effects. | Split into `config_v1.py` (V1 frozen models), `config_v2.py` (V2 extension models), retaining `config.py` as a thin re-export shim. |
| F011 | Architectural decay | `src/regime_data_fetch/acquisition_consolidation.py` | Medium | M | 871 lines. Combines DB-merge logic, parquet reading, YAML events, params JSON handling, and summary generation. | Extract DB-merge helpers and YAML-event reading into separate modules. |
| F012 | Consistency rot | `src/regime_detection/central_bank_text.py:60`, `src/regime_detection/hmm_state.py:33`, `src/regime_detection/loaders.py:11` | Low | S | Logger variable naming uses three different conventions: `LOG`, `_LOGGER`, and `LOGGER`. AGENTS.md references `get_logger` but no such utility exists in the codebase. | Pick one convention (recommend `_LOG = logging.getLogger(__name__)` matching stdlib idiom). Apply consistently. No `get_logger` wrapper needed. |
| F013 | Error handling | `src/regime_data_fetch/event_sources/validators_hf_central_bank.py:54` | Medium | S | `except Exception:` returns `[_unknown(candidate) for candidate in central_bank_candidates]` with no log. Production HuggingFace fetch failures are invisible — the caller sees "unknown" labels and has no way to distinguish a network failure from a legitimate gap. | Add `LOGGER.warning("hf_central_bank parquet fetch failed", exc_info=True)` before the fallback return. |
| F014 | Error handling | `src/regime_data_fetch/acquisition_consolidation.py:829` | Low | S | `except Exception:` in `_augment_params_json` silently falls back to `{"raw_params_json": params_json}` with no log. The caller has no signal that JSON was malformed. | Add `LOG.warning("params_json unparseable, using raw fallback: %s", params_json[:200])` before the fallback. |
| F015 | Error handling | `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py:257` | Low | S | `except Exception:` falls back from Excel to CSV parsing. If the payload is neither Excel nor CSV, the CSV parse will also fail, propagating the original error — but the Excel error is already swallowed. | Catch `XLRDError` / `xlrd.XLRDError` specifically, or at least log at DEBUG level before the CSV fallback. |
| F016 | Dependency & config debt | CI: `.github/workflows/*.yml` | Medium | S | CI runs `pytest -q` but has no explicit `ruff check .` or `mypy` steps. The 50 existing ruff violations would never fail a merge. No coverage gate. No security scan (`bandit`). | Add three CI steps: `ruff check .`, `mypy src/ --ignore-missing-imports`, `pytest --cov=src --cov-fail-under=70`. |
| F017 | Architectural decay | `src/regime_detection/axis_series.py:1387` | Low | S | `input_by_date = [series for series in required_inputs]` is a no-op list copy. The variable name (`input_by_date`) also misrepresents what it holds (it's a list of series, not a dict keyed by date). | Replace with `required_inputs` directly, or `list(required_inputs)` if mutation isolation is needed. Rename if the semantics are different. |
| F018 | Architectural decay | `src/regime_detection/axis_series.py:165` | Low | S | `required_trading_days=200`, `252`, `63`, `50` are magic numbers scattered across classifier calls in `axis_series.py`. They correspond to ~1y, ~3y, ~1q, and ~2.5-month trading windows. | Define `_TREND_DIRECTION_LOOKBACK = 200`, `_VOLATILITY_LOOKBACK = 252`, etc. as module-level constants with comments citing the spec section that mandates each window. |
| F019 | Consistency rot | `scripts/` | Low | S | 36 E402 violations (module-level imports not at top) in 10+ script files. All follow the same sys.path-manipulation pattern before library imports. | Replace the inline `sys.path.insert` block with a shared `scripts/_bootstrap.py` (or move to a proper `__main__` entry point via `pyproject.toml [project.scripts]`) and fix the import order. |
| F020 | Consistency rot | `scripts/run_v2_calibration.py:98` | Low | S | 5 f-strings without placeholders (F541 ruff violations). Lines 98, 101, 102, 104, 111 use `f"..."` where `"..."` suffices. | Remove the `f` prefix from the 5 literal strings. |
| F021 | Documentation drift | `README.md` | Low | S | README says "V1 regime detection engine" and links only V1 docs. The codebase is on V2 with 32 merged PRs, V2 specs, shadow infrastructure, and calibration tooling. | Update README to reflect V2 state: list both V1 and V2 spec links, describe the dual-track architecture (V1 frozen + V2 extension), and note the shadow-runner workflow. |
| F022 | Test debt | `src/regime_detection/network_fragility_rules.py` | Medium | M | 27% coverage (46/173 lines). Network fragility is a V2 P1 axis with rules for sector dispersion and HY spread thresholds. Low coverage means rule-predicate regressions go undetected. | Add parameterized tests for each rule predicate with boundary values (at-threshold, above, below). |
| F023 | Test debt | `src/regime_detection/inflation_growth.py` | Medium | M | 37% coverage (82/222 lines). Inflation/growth is a V2 P1 axis. The 37% gap is in the rule-evaluation and input-building logic (lines 212–444). | Test `build_rule_inputs_for_date` and `evaluate_rules` with fixture inputs for each label (hot_inflation, stagflation, etc.). |
| F024 | Dependency & config debt | `src/regime_data_fetch/alpaca_daily.py:20` | Medium | S | `os.environ["ALPACA_API_KEY_ID"]` is read at function call time (good), but if the env var is missing, the KeyError message doesn't name the var in a user-friendly way. | Wrap in: `os.environ.get("ALPACA_API_KEY_ID") or _raise("ALPACA_API_KEY_ID env var required")` to emit a clean error. |
| F025 | Architectural decay | `src/regime_detection/breadth_state_v2.py:66` | Low | M | `from regime_data_fetch.pit_constituents import ...  # noqa: E402` inside `src/regime_detection/` (a core detection package) imports from `regime_data_fetch` (a data acquisition package). This inverts the layering: detection depends on acquisition. | Move the PIT constituent loading to a loader/context object so `breadth_state_v2` receives a DataFrame rather than importing from the sibling acquisition package. |
| F026 | Architectural decay | `src/regime_detection/credit_funding.py:247` | Low | S | `from regime_detection._rolling_stats import rolling_change_zscore as _change_zscore  # noqa: E402` is a mid-file import after function definitions. The `noqa` acknowledges the violation but doesn't fix the root cause (the import cannot be at top due to circular risk or positioning). | Audit whether the circular-import concern is real; if not, move the import to the top of the file and remove the `noqa`. |
| F027 | Test debt | `src/regime_detection/hmm_state.py` | Medium | M | 38% coverage (14/37 lines). The HMM state classifier is evidence for the V2 transition score. The untested 62% is the core model-fitting path (lines 83–145). | Add a test that fits an HMM on a synthetic 200-day return series and asserts that state counts are in [1, n_states] and probabilities sum to 1. |
| F028 | Documentation drift | `src/regime_detection/models.py:28` | Low | S | The TODO comment at line 28 (`# TODO(schema): Replace free-form evidence dicts axis-by-axis…`) has no assignee, no target slice, and no acceptance criteria. It is a wish, not a plan. | Convert to a Linear issue with a concrete acceptance test: "AxisOutput.evidence is a typed dataclass; mypy passes with --strict on models.py." |
| F029 | Performance & resource hygiene | `src/regime_detection/axis_series.py:1388` | Low | S | The `_build_axis_outputs` function iterates over sessions with `zip(..., strict=True)` which is correct, but it also builds `outputs_by_date`, `stable_by_date`, and `active_by_date` as three separate dicts in one loop, then constructs `AxisSeriesResult`. The three dicts are always in sync and could be a single pass. Minor but multiplied across 8+ axis classifiers per session run. | Not a blocking issue. Document if profiling shows significance. |
| F030 | Consistency rot | `src/regime_data_fetch/acquisition_consolidation.py:41` | Low | S | `consolidate_all_sources` returns `dict[str, object]` — a summary with no typed contract. Callers that unpack this summary (scripts that print or persist it) have no schema guarantees. | Define a `ConsolidationSummary` dataclass or TypedDict. |

---

## Top 5: If You Fix Nothing Else, Fix These

### 1. F001 — Decompose `axis_series.py`

**Why:** At 1,426 lines with 11 single-method classes, every new V2 axis slice adds ~100 lines here. By the time V2 is complete this file will exceed 2,000 lines. It already violates the CLAUDE.md limit.

**Concrete change:**
```python
# axis_series.py BEFORE: 11 classes each with def build()
class TrendDirectionSeriesClassifier:
    def build(self, context, feature_store) -> AxisSeriesResult: ...

# axis_series.py AFTER: 11 free functions, file shrinks to ~200 lines
def build_trend_direction_axis_series(context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
    ...

# build_axis_series_bundle calls them directly:
trend_direction = build_trend_direction_axis_series(context=context, feature_store=feature_store)
```
Each function can live in its axis module (`trend_direction.py` gains `build_axis_series`) or in a new thin `axis_classifiers/` sub-package.

---

### 2. F002 — Consolidate `sma_50` / `return_63d` into `_rolling_stats`

**Why:** Three independent `.rolling(50).mean()` computations on the same series guarantee divergence the moment one gets a `min_periods` adjustment or a shift that the others don't. The `_rolling_stats` module exists precisely for this.

**Concrete change:**
```python
# _rolling_stats.py — add:
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def period_return(series: pd.Series, window: int) -> pd.Series:
    return series / series.shift(window) - 1

# feature_store.py — replace:
# sma_50 = spy_close.rolling(50).mean()   ← DELETE
# Already computed in trend_direction.compute_features; pass through FeatureStore
```

---

### 3. F003 + F004 — Replace `evidence: dict[str, Any]` with typed evidences, starting with `TransitionScoreInputs`

**Why:** `transition_risk_series.py` has 14 `# type: ignore` comments because it works with a `dict[str, float | str]` that mypy can't reason about. Fixing the root type eliminates all 14 suppressions at once and makes the V2 scoring logic auditable.

**Concrete change:**
```python
# transition_score_inputs.py (new file):
@dataclass(frozen=True)
class TransitionScoreInputs:
    hmm_top_state_prob_now: float | None
    hmm_top_state_prob_5d_ago: float | None
    change_point_score: float | None
    realized_vol_short: float
    realized_vol_long: float
    pct_above_50dma: float
    drawdown_252d: float

# transition_risk_series.py: replace dict[date, dict[str, float | str]]
# with dict[date, TransitionScoreInputs] — all 14 type: ignore go away
```

---

### 4. F016 — Add `ruff`, `mypy`, and `--cov-fail-under` to CI

**Why:** 50 ruff violations exist today and pass CI. Without a gate, the number grows. Coverage is 55% with no floor.

**Concrete change in `.github/workflows/ci.yml`:**
```yaml
- name: Lint
  run: ruff check .

- name: Type check
  run: python -m mypy src/ --ignore-missing-imports

- name: Test with coverage
  run: pytest --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

### 5. F005 + F006 — Cover `loaders.py` (13%) and `shadow_storage.py` (0%)

**Why:** `loaders.py` is the sole path that reads market data from disk into DataFrames consumed by every classifier. `shadow_storage.py` manages durable V2 shadow-run state. Both are on critical paths with near-zero test coverage.

**For `loaders.py`:** Add fixtures for each loader format (parquet, SQLite OHLCV). Test schema validation failure raises with a clear message.

**For `shadow_storage.py`:** Add a test that writes a shadow run record, reads it back, and asserts field equality. Test idempotent write on duplicate session.

---

## Quick Wins

- [ ] **F020** — Remove `f` prefix from 5 literal strings in `scripts/run_v2_calibration.py` (5 minutes)
- [ ] **F013** — Add `LOGGER.warning(..., exc_info=True)` before the `except Exception` fallback in `validators_hf_central_bank.py:54` (5 minutes)
- [ ] **F014** — Add `LOG.warning(...)` before the `except Exception` fallback in `acquisition_consolidation.py:829` (5 minutes)
- [ ] **F017** — Replace `[series for series in required_inputs]` with `list(required_inputs)` in `axis_series.py:1387` (2 minutes)
- [ ] **F021** — Update README.md to reflect V2 state (15 minutes)
- [ ] **F012** — Standardize logger variable to `_LOG` across all six affected modules (10 minutes, no behavior change)
- [ ] **F018** — Extract `required_trading_days` magic numbers to named constants in `axis_series.py` (10 minutes)

---

## Things That Look Bad But Are Actually Fine

**`except Exception:` in `validators_gpr_gdelt.py:257` (GPR Excel→CSV fallback):** The pattern of falling back from Excel to CSV parsing on any error looks like silent failure, but the GPR source publishes the same data in both formats and the field has been stable for years. The fallback is defense against format drift, not error suppression. It would be better with a DEBUG log, but the logic is sound.

**`noqa: ARG001` on `network_fragility_rules.py:390` and `credit_funding.py:584`:** Both suppress "unused argument" warnings on `config` parameters. The pattern exists to maintain a uniform function signature across rule evaluators so they can be dispatched polymorphically. The `config` is intentionally ignored in these specific rule variants that have no thresholds. Not dead code — it's a deliberate interface uniformity choice.

**11 classifier classes in `axis_series.py` (F001) vs. functions:** While the classes are single-method and therefore functions in disguise, they do provide a future extension point if any classifier ever needs initialization state (e.g., a cached model). The refactor to functions is still correct (YAGNI — none currently need it), but the original choice isn't irrational.

**`breadth_state_v2.py` importing from `regime_data_fetch` (F025):** The import is guarded by an availability check and is used only for PIT constituent data that is genuinely a data-layer concern. The layering violation is real but the alternative (passing the DataFrame through MarketContext) would require threading PIT data through every caller. A seam in `FeatureStore` is the right fix, but the current approach is functional and isolated.

**`dict[str, object]` return types on several event-source functions:** Most of these are internal pipeline accumulator dicts that never cross a public API boundary. The loose typing is annoying but not dangerous — the callers are in the same file or test suite.

**`_build_axis_outputs` builds three dicts in one loop (F029):** This is correct and idiomatic Python. The three dicts serve different consumers (`outputs_by_date` for rich evidence access, `stable_labels_by_date`/`active_labels_by_date` for fast label lookup). The single-pass pattern avoids triple-iteration. Not a performance issue.

---

## Open Questions for the Maintainer

1. **`breadth_state` `deprecation` of `breadth_data` parameter (engine.py:48, 93 TODOs):** The V1 public API keeps `breadth_data` alive. Is there a deprecation timeline, or will it remain indefinitely? The answer determines whether the V2 path should add `breadth_data` handling or deprecate it.

2. **`scripts/verify_fixtures.py` at 1,055 lines:** Is this a standalone script or should it be migrated into the test suite? It appears to do fixture-level assertion work that belongs in pytest. If it's intentionally a CLI tool, it should be in `src/` with a proper entry point.

3. **Coverage gate:** The CI only runs `pytest -q` with no `--cov`. Is the 80% minimum from CLAUDE.md intended to be enforced by CI, or is it advisory? Current measured coverage is 55% on a representative subset.

4. **`pit_constituents.py:22` TODO** — "replace with a true point-in-time vendor feed (CRSP / Compustat…)": Is this a V2 slice item or deferred to V3? The breadth V2 results are sensitive to PIT constituent accuracy.

5. **`HMM / GMM cluster → label review` (AGENTS.md):** The review guideline says auto-labeling HMM clusters is P1 flag. Is there a documented review artifact for the current cluster→label mapping, or is it embedded in test fixtures only?
