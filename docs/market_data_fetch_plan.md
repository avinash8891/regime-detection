# Market Data Fetch Plan

This document now separates two workflows that were previously mixed together:

1. **Build / backfill fetches** used to develop and test V1/V2 locally
2. **Shadow-mode daily acquisition** used after V1 is frozen and running operationally

Those are not interchangeable. Historical backfills test engine logic. Forward shadow tests operational stability.

## 0. Artifact Storage and Materialization Contract

`data/raw/` is a local materialized cache, not the durable source of truth. It
stays gitignored so large, licensed, or frequently changing source artifacts do
not enter Git history. Any workflow that needs to run outside the original
machine must rebuild `data/raw/` from an explicit artifact manifest lockfile
tracked under `manifests/`.

The durable storage boundary is:

| Layer | Responsibility | Durable home |
|---|---|---|
| Raw capture | Exact bytes returned by a source or supplied by an operator: FRED JSON, Alpaca bars, Investing pages/CSVs, AAII files, manual workbooks | S3-compatible object storage under `raw_capture/` |
| Normalized | Source-specific cleaned tables that still preserve the source shape | S3-compatible object storage under `normalized/` |
| Canonical | Engine-ready parquet/SQLite/YAML contracts such as macro series, daily OHLCV, sentiment, event candidates, PIT constituents | S3-compatible object storage under `canonical/` |
| Run inputs | Frozen input bundle for one regime run or replay window | S3-compatible object storage under `run_inputs/` |
| Ledger | Fetch runs, source checkpoints, artifact URIs, hashes, row counts, date ranges, schema versions, lineage, and quarantine records | SQLite |
| Local cache | Files read by current scripts and loaders | `data/raw/` rebuilt from a manifest |
| Version control | Code, schemas, fetch contracts, implementation plans, and small manifest lockfiles under `manifests/` | Git |

SQLite is the artifact ledger, not the warehouse. Large source bodies and
processed tables live as files in object storage; SQLite records their URI,
hash, size, row count, date range, source metadata, producing code version, and
lineage. Small payloads may be embedded only when that improves operator
debugging without turning SQLite into the primary data store.

Every production fetch must finish in this order:

1. Fetch new source data plus a source-specific lookback window for revisions.
2. Store the exact raw bytes in object storage.
3. Normalize into source-shaped rows and store the normalized artifact.
4. Build or update the canonical engine artifact.
5. Validate schema, row count, date range, and hash.
6. Record artifact and lineage metadata in SQLite.
7. Advance the source checkpoint only after every required artifact has been
   written and validated.

Incremental fetches update the artifact lake. Regime-engine runs consume an
immutable manifest lockfile that pins exact canonical artifacts. A tracked
manifest under `manifests/runs/` is the only supported cross-environment
handoff: it states which S3 artifacts to materialize into which local
`data/raw/` paths and includes hashes that must verify before the engine
starts. `data/manifests/` is not a valid durable handoff location because
`data/` is intentionally gitignored.

Minimum manifest fields:

```yaml
artifact_set: regime_engine_YYYY-MM-DD
created_at_utc: "YYYY-MM-DDTHH:MM:SSZ"
storage_root: s3://regime-data
artifacts:
  - name: fred_macro_series
    stage: canonical
    uri: file:///absolute/artifact-root/canonical/macro/fred_macro_series/as_of=YYYY-MM-DD/fred_macro_series.parquet
    local_path: data/raw/macro/fred_macro_series.parquet
    sha256: "<hex>"
    schema_version: fred_macro_series.v1
    rows: 0
    min_date: YYYY-MM-DD
    max_date: YYYY-MM-DD
    required_for:
      - v2_calibration
      - profile_engine_30d
```

The implementation may keep existing local paths during migration, but the
contract is logical: `data/raw/` is replaceable, and the manifest plus artifact
store is what makes the data portable and replayable.

Default fetch behavior:

```bash
python3 scripts/fetch_regime_engine_v1_data.py \
  --fetch all \
  --artifact-store s3://regime-data \
  --emit-manifest
```

Operator credentials and source identities are resolved through a non-secret
pointer file, not by searching worktrees. Runners load the first available pointer from
`--operator-env-file`, `REGIME_OPERATOR_ENV_FILE`, repo-local
`.regime-operator.env`, or `~/.config/regime-detection/operator.env`. The
pointer may use `REGIME_ENV_FILES` for a path-separated list of secret env files
and/or provider-specific pointers such as `REGIME_TINYFISH_ENV`. The tracked
template `.regime-operator.env.example` lists every repo-known credential key
currently consumed by fetch paths. HDX HAPI is the exception: it does not use an
auth secret here; the code sends an `app_identifier` query parameter derived
from `HDX_HAPI_APP_IDENTIFIER` or `HDX_HAPI_APP_NAME` plus
`HDX_HAPI_APP_EMAIL`.

When `--emit-manifest` is supplied without a path, the fetch script writes an
immutable lockfile to `manifests/runs/regime_engine_<end-date>.yaml`. If a
stable alias is desired, update a small tracked alias such as
`manifests/latest.yaml` deliberately after validating the immutable lockfile.
Do not write durable manifests under `data/` or `.context/`.

### 0.1 Temporal Normalization Contract

All persistent canonical artifacts must use one temporal contract across data
sources. Source-specific time zones may exist only in raw captures or explicit
source-metadata columns; they must not leak into engine-facing parquet with
mixed semantics.

Canonical rules:

- **Intraday instants**: columns that represent a real instant in time must be
  named `*_timestamp_utc` or `*_time_utc`, stored in UTC, and round-trip as
  timezone-aware UTC timestamps or ISO-8601 strings with a `Z` suffix. Examples:
  Investing event occurrences, last-price timestamps, artifact creation times,
  and fetch-run start/finish times.
- **Source-local release timestamps**: if the source publishes a release time in
  local civil time, normalize the canonical instant to UTC and retain the source
  zone only as metadata, e.g. `release_timestamp_utc` plus
  `release_timezone="America/New_York"`. Existing `*_et` fields are tolerated
  during migration but are not the forward canonical storage shape.
- **Date-only observations**: columns that represent an observation date, NYSE
  session date, vintage date, constituent interval boundary, or monthly/weekly
  period must be stored as normalized `YYYY-MM-DD` dates, not midnight timestamps.
  Examples: FRED observation dates, Alpaca daily bars, AAII weekly dates,
  Cleveland Fed nowcast dates, S&P EPS observation dates, PIT start/end dates,
  and NYSE session dates.
- **Session alignment**: engine loaders may convert a UTC release timestamp to a
  NYSE session date at the loader boundary. That derived session date is a date,
  not a replacement for the stored UTC instant.
- **Raw preservation**: raw captures keep the source payload exactly as received,
  including source time zones, offsets, or strings. Normalized and canonical
  artifacts apply this contract.

Implementation status:

- Engine-facing loaders now route canonical date/date-index parsing through
  `regime_detection.temporal` so malformed dates fail with source-specific
  errors at the loader boundary.

Remaining TODO for the next data-contract slice:

1. Add writer-side helpers for UTC instants, source-local-to-UTC conversion,
   and artifact-level temporal contract assertions.
2. Route every canonical writer through those helpers before `to_parquet()`:
   FRED macro/vintages, Alpaca daily OHLCV, Investing economic/holiday/earnings,
   PMI, AAII, SF Fed news sentiment, Cleveland Fed nowcast, FOMC minutes, Powell
   speeches, event candidates/YAML export, EPS history, and PIT constituents.
3. Add schema tests that read each canonical parquet family and assert temporal
   column names, parseability, timezone handling, and date-only semantics. A
   source with `+05:30`, `ET`, naive timestamps, or mixed string/datetime values
   must fail before the artifact is recorded as valid.
4. Record the temporal schema version in SQLite artifact metadata and manifests
   so old mixed-semantics artifacts cannot be silently used in future runs.
5. Keep display-time conversion outside storage. Reports may render local time;
   stored canonical artifacts stay UTC instants or normalized dates only.

## 1. Validation Sequence

### 1.0 Current Status (mirrors `regime_engine_v1_final_spec.md` §14.1)

**V1 qualification: complete (operator-asserted per implementation plan).** V2 implementation is in progress under the unified V1+V2 design described in `regime_engine_v2_spec.md` §8. V2 slices 1.x, 2.x, and 4.1 (features-only) have shipped on top of V1; V1 byte-identity is enforced under `tests/test_v1_frozen_replay.py` and the `test_v1_contract_byte_identity_when_v2_features_absent` family of tests across each v2 slice. V2 spec ambiguities resolved during implementation are recorded in the V2 Implementation Ambiguity Log (entries #1 through #53 at time of writing).

The qualification gate prose below is retained as the canonical definition any future V1 revision branch must satisfy before authorizing parallel V2 work on that branch.

### 1.1 Qualification Gate (Retained)

V2 activation should rely on **both**:

1. **Historical walk-forward**
   - frozen V1 code/config
   - replay across historical out-of-sample dates
   - fast gate for engine correctness and label behavior
2. **Forward shadow run**
   - frozen V1 code/config
   - 252 consecutive NYSE trading sessions
   - slow gate for data-source reliability, scheduling, and real operational incidents

Config stays fluid through V1 implementation. It freezes only after historical walk-forward passes and before the forward shadow window starts.

## 2. Build / Backfill Fetches

Current repo entrypoint:

```text
scripts/fetch_regime_engine_v1_data.py
```

This is the **development/backfill** fetch path, not the future shadow runner.

### 2.0 Approved Source Decisions

These source choices are already approved and should be treated as explicit spec/document decisions, not silent fetch-layer substitutions:

- `DXY` is **not** fetched in V2 build mode. The spec-level field is `broad_usd_index`, sourced from FRED `DTWEXBGS`.
- PMI stays PMI. Do **not** substitute CFNAI or another macro proxy.
- PMI retrieval now uses live DBnomics / TradingEconomics by default. Repo-local manually supplied Investing release-history tables remain an explicit fallback via `--pmi-history-dir` for pinned historical backtests.
- `earnings_revision_breadth` is replaced by `aggregate_forward_eps_revision_direction`, sourced from S&P Global aggregate forward EPS data.
- **V2 §2B commodity returns**: `DBC` ETF (Invesco DB Commodity Index Tracking Fund) is the approved substitute for the Bloomberg Commodity Index (paid feed unavailable). Pinned in V2 §2B Ambiguity Log entry #48. Rows must carry a documented bias-warning analogous to the §1D PIT-source pattern; a future spec-amendment slice may replace DBC with a direct Bloomberg / Refinitiv commodity-index feed when vendor sourcing is approved.
- **V2 §2C HY/IG spread direction**: §2C carries two separate raw metrics plus an explicit effective downstream resolver. The authoritative real-OAS metric is sourced from **real ICE BofA Option-Adjusted Spread series, free on FRED** — `hy_oas = BAMLH0A0HYM2` (HY Master II OAS) and `ig_bbb_oas = BAMLC0A4CBBB` (BBB Corporate OAS), both in `V2_FRED_SERIES` (Ambiguity Log #49 closed; commits `814a8d5` + `9cad7e7`). Because FRED now exposes only 2023-05-15+ OAS history, ADR 0007 / Ambiguity Log #71 keeps the TLT-vs-HYG/LQD total-return-differential proxy as a **separate parallel metric** producing `credit_funding_state_proxy`. `credit_funding_effective_state` is the downstream field: OAS when it is the only classified signal, proxy when OAS is unavailable/stale/insufficient-history, and the higher-risk label when both directional labels diverge. Raw OAS/proxy series are never spliced.
  - **Coverage caveat (discovered 2026-05 re-fetch):** FRED now publishes only a **trailing ~3-year window** of these ICE BofA OAS series — `BAMLH0A0HYM2` and `BAMLC0A4CBBB` both start **2023-05-15** (confirmed against FRED's `/series` metadata: `observation_start = 2023-05-15`). ICE Data Indices tightened redistribution licensing; the series IDs are unchanged but pre-2023 history is no longer public on FRED. **Consequence:** §2C real-OAS backtests start around 2023-05. The proxy covers earlier history directionally from already-fetched `TLT`, `HYG`, and `LQD` closes and carries `credit_spread_proxy_total_return_differential` bias warnings. **Do not add `IEF` or `BIL`** for another proxy without a spec amendment.
- **V2 §2D event rows** (`budget_week`, `election_window`, `geopolitical_event`, `global_rate_decision`) are generated as candidate evidence first, then promoted under the Group B rules. `election_window`, `budget_week`, and global-rate decisions have deterministic/official-source paths. `geopolitical_event` now has live GPR + GDELT evidence generation plus implemented clients for ACLED, Uppsala/UCDP GED Candidate, and HDX HAPI conflict-event fetchers. **TODO:** ACLED and Uppsala/UCDP live pulls remain pending entitled API keys/account access; the current Gmail ACLED account can mint a token but the API endpoint denies raw data access. All geopolitical evidence still reaches YAML only through the human approval overlay; no geopolitical source auto-promotes.
- **V2 §1A `euphoria` label — resolved**: `sentiment_score = bull_bear_spread_8w_ma` from the AAII weekly survey; the AAII fetcher ships (`regime_data_fetch.aaii_sentiment`), and `euphoria` fires (Ambiguity Log #32 status update). Put-call ratio / Investors Intelligence remain *optional* alternative sources for a future calibration revision — they block no label and need no fetcher to ship V2.
- **V2 §1C `vol_crush` / `IV/RV spread` — resolved**: `implied_vol_30d = VIXCLS / 100` from FRED (the CBOE VIX IS the canonical model-free 30-day implied vol on SPX — no options-chain feed needed). `iv_rv_spread`, `implied_vol_30d`, `implied_vol_5d_change` ship on `VolatilityV2Features`; the `vol_crush` rule wires through (Ambiguity Log #19 / #20 status updates; ADR 0005).
- **V2 §2B `inflation_surprise_zscore` — resolved**: the analyst-survey `consensus_estimate` is substituted by the free Cleveland Fed inflation nowcast (ADR 0006). `regime_data_fetch.cleveland_fed_nowcast` downloads + parses the Cleveland Fed month-over-month nowcast webchart JSON directly (the full 2013-08→present archive, ~154 monthly vintages). The single-signal `inflation_shock` limb consumes `inflation_surprise_zscore` and is silent only during the 5y cold-start or when `cpi_nowcast` is unwired; the composite-shock limb is always active.
- **V2 §1D PIT constituent universe**: continues to use the GitHub `fja05680/sp500` `sp500_ticker_start_end.csv` as the V2 ship default (rows already carry `survivorship_biased_constituent_universe` warning). A `TODO` note is now in `src/regime_data_fetch/pit_constituents.py` recommending replacement with a true point-in-time vendor feed (CRSP / Compustat / FactSet / Norgate) when sourcing is approved. The expected vendor format matches the same ticker / start_date / end_date interval shape, so the parquet schema does not change on swap.

### 2.0A Data Inventory

Status meanings used below:

- `done-live-verified`: implemented and verified against the real source in this repo workflow
- `implemented-not-live-verified`: implemented in code but not live-verified in the current session
- `operator-assisted`: useful explicit tool, but not part of unattended `--fetch all` because it depends on a browser session, local file, archive, or historical-backfill decision
- `template-only`: only a placeholder/template exists
- `planned`: source/path identified, loader not implemented yet
- `hard-fail`: intentionally unsupported unless the spec/source decision changes

| Data | Source | Cadence | Output / Path | Status | Comment |
|---|---|---|---|---|---|
| V1/all stock universe symbol list | fixed regime universe artifact; PIT parquet only for explicit bootstrap/listing | ad hoc refresh | materialized as a JSON symbol list or `data/raw/daily_ohlcv_762/symbol=*/` tree by the manifest | done-live-verified | routine Alpaca constituent refreshes use the fixed 762-symbol universe (`--universe-json` or `--constituent-universe-dir`) and validate the expected count. PIT expansion is blocked unless `--allow-pit-constituent-universe` is explicit. The runner never uses Alpaca's full active asset catalog for regime-engine OHLCV |

> **Engine-facing ETF OHLCV — updated 2026-05.** `data/raw/daily_ohlcv/` (the partitioned parquet the engine's market-data loader actually reads) was re-fetched directly from Alpaca — SIP feed, `adjustment=split` — covering the full **26-symbol V2 ETF universe, 2016-01-04 → current, one consistent source** (26 symbols / 67,111 rows; `XLC` starts 2018-06-19, its genuine launch). This replaced a stale 2018-start parquet that was also missing `RSP`/`DBC` and disagreed with the SQLite store on prices. The SQLite `daily_ohlcv_rows` store described in the rows below is a **separate artifact** used only for the 762 PIT constituent stocks (V2 §1D PIT breadth) — it is not what the ETF axes load.

| 762-stock daily OHLCV backfill | Alpaca REST from the fixed 762-symbol regime universe; local parquet import remains an explicit backfill path | daily | `data/raw/daily_ohlcv_762/symbol=*/ohlcv.parquet` plus SQLite `daily_ohlcv_rows` | done-live-verified | routine future runs use `--fetch daily-ohlcv-constituents-alpaca`, or `--fetch all` only when the fixed universe artifact is already supplied via `--universe-json` or `--constituent-universe-dir`. The current fetch script still falls back to the PIT parquet when no fixed-universe input is provided, so `--fetch all` is not clean-bootstrap-safe yet. PIT constituent expansion is an explicit bootstrap mode only, not the default refresh universe. `--fetch daily-ohlcv-local-sqlite --daily-ohlcv-dir ...` stays available only for archived/local backfills |
| local 762-stock daily OHLCV import | existing partitioned parquet tree supplied by an operator | manual/replay | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | operator-assisted | explicit `--fetch daily-ohlcv-local-sqlite --daily-ohlcv-dir ...` materializes or replays an already-captured tree. It is excluded from unattended `--fetch all`; routine future refresh is Alpaca constituent OHLCV with the fixed 762-symbol artifact |
| `SPY` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V1 market anchor is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05`; Alpaca remains the documented refresh path if you later need a direct pull |
| `RSP` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V1 breadth proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05`; Alpaca remains the documented refresh path if you later need a direct pull |
| `VIX` daily proxy bars | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | the checked local market dataset covers the documented volatility proxy requirement operationally via the imported symbol set; if you later refresh from Alpaca directly, true `VIX` is preferred when the account supports it |
| `VIXY` daily proxy bars | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | documented operational volatility proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| `KRE` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V2 bank-stress proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| Sector ETF daily OHLCV: `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | all listed sector ETFs are present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| Cross-asset ETF daily OHLCV: `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | all listed cross-asset ETFs are present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| `DBC` daily OHLCV (V2 §2B commodity proxy) | direct Alpaca fetch (SIP, split-adjusted) | daily | `data/raw/daily_ohlcv/symbol=DBC/` | done-live-verified | approved substitute for Bloomberg Commodity Index per V2 §2B Ambiguity Log #48; present in the engine-facing `daily_ohlcv/` parquet, 2016-01-04 through 2026-05-13 in the current checkout. DBC is NOT in the §3.1 fragility universe — it is a V2 §2B-only feature input; rows must carry a documented bias-warning; vendor upgrade (Bloomberg / Refinitiv commodity index) noted as future spec-amendment path |
| Scheduled event rows: `FOMC` | generated repo-local YAML from Federal Reserve FOMC calendar pages | about 8 times per year | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse current `fomccalendars.htm`, add older meetings from `fomc_historical_year.htm` and yearly `fomchistoricalYYYY.htm` pages, dedupe by `meeting_end_date`, and store minutes release timestamps at `14:00 ET`; current live-verified coverage is `2007-10-31` through `2026-03-18` |
| Scheduled event rows: `CPI` | generated repo-local YAML from BLS yearly release-schedule pages, using the local yearly HTML archive when direct access is blocked | monthly | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `Consumer Price Index` / `Consumer Price Indexes` rows, and store release timestamps at `08:30 ET`; current checked YAML/report contains `131` CPI rows spanning `2016-01-20` through `2026-12-10` |
| Scheduled event rows: `NFP` | generated repo-local YAML from BLS yearly release-schedule pages, using the local yearly HTML archive when direct access is blocked | monthly | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `The Employment Situation` / `Employment Situation` rows, and store release timestamps at `08:30 ET`; current checked YAML/report contains `131` NFP rows spanning `2016-01-08` through `2026-12-04` |
| Rule-derived event window: `expiry_week` | computed from deterministic rules in config/runtime | monthly | no stored raw file | done-live-verified | compute the third Friday of each month, roll back to the previous NYSE trading day if that Friday is closed, then expand the configured NYSE trading-day window around the anchor; the runtime rule is now wired through `resolve_event_label()` and live-verified with NYSE holiday-sensitive months like `2019-04`, `2022-04`, and `2026-06` |
| Rule-derived event window: `earnings_season` | computed from deterministic rules in config/runtime | quarterly window | no stored raw file | done-live-verified | compute quarter windows starting on the second Monday of `Jan/Apr/Jul/Oct` and ending `+35` calendar days later; the runtime rule is now wired through `resolve_event_label()` and live-verified across `2015-01-01` through `2026-05-07` |
| V2 §2D event rows (`election_window`, `geopolitical_event`, `budget_week`, `global_rate_decision`) | official/deterministic adapters plus Group B candidate evidence | irregular | `data/raw/event_calendar/candidates/*.parquet`; promoted rows render into `configs/events/us_events.yaml` | implemented-live-source-paths | `election_window` is deterministic; `global_rate_decision` uses official BOE / ECB / BOJ pages; `budget_week` uses deterministic Sep-30 plus Treasury/GovInfo official budget discovery. `geopolitical_event` uses GPR daily-index spikes and GDELT daily Event export ZIPs for GPR spike windows. ACLED and Uppsala/UCDP client code is present, but live raw-event pulls are TODO pending entitled API keys/account access (`ACLED_API_TOKEN` must be API-authorized; `UCDP_ACCESS_TOKEN` required). HDX HAPI monthly/admin conflict evidence requires an app identity query parameter via `HDX_HAPI_APP_IDENTIFIER`, or both `HDX_HAPI_APP_NAME` and `HDX_HAPI_APP_EMAIL`; it is not an auth secret. Missing app identity is logged and skipped instead of using a fake default. Geopolitical rows remain overlay-only and are excluded from routine `macro_event_score`; V1 path remains unaffected. |
| V2 §2B `cpi_nowcast` (Cleveland Fed inflation nowcast) | Cleveland Fed month-over-month nowcast webchart JSON, free — `.../webcharts/inflationnowcasting/nowcast_month.json` (reachable over urllib) | intra-month; re-fetch monthly | `data/raw/cleveland_fed_nowcast/cpi_nowcast.parquet` | done | feeds `inflation_surprise_zscore` (V2 §2B single-signal `inflation_shock` limb); ADR 0006 substituted the free Cleveland Fed nowcast for the paid analyst `consensus_estimate`. `regime_data_fetch.cleveland_fed_nowcast` downloads + parses the JSON archive directly — 154 monthly vintages 2013-01→present in the current materialization, last non-empty CPI value per vintage keyed to its chart publication date. Manual-drop of the JSON is a fallback only |
| V2 §1A sentiment_score | AAII bull-bear weekly survey (sourced) | weekly | `data/raw/sentiment/aaii_sentiment.parquet` | done | `sentiment_score = bull_bear_spread_8w_ma`; `regime_data_fetch.aaii_sentiment` ships and the V2 §1A `euphoria` label fires (Ambiguity Log #32). Put-call ratio / Investors Intelligence are optional future calibration-revision sources only — they block no label |
| V2 §1C implied vol (`VIXCLS`) | FRED API (CBOE VIX — the model-free 30-day implied vol on SPX) | daily | `data/raw/macro/fred_macro_series.parquet` | done | `implied_vol_30d = VIXCLS / 100`; feeds V2 §1C `vol_crush` rule + `iv_rv_spread` feature (Ambiguity Log #19 / #20; ADR 0005). No options-chain feed needed — VIX is the canonical implied-vol series and is free on FRED |
| V2 §2C HY/IG OAS feeds | ICE BofA Option-Adjusted Spread series, free on FRED: `BAMLH0A0HYM2` (HY) + `BAMLC0A4CBBB` (BBB) | daily | `data/raw/macro/fred_macro_series.parquet` | done — **2023-05-15 → current only** | authoritative real-OAS metric sourced directly from FRED and in `V2_FRED_SERIES` (Ambiguity Log #49; commits `814a8d5` + `9cad7e7`). **Coverage limit:** FRED exposes only a trailing ~3-year window of these series (both start `2023-05-15` as of the 2026-05 re-fetch — ICE Data Indices licensing truncation, confirmed against FRED `/series` metadata); §2C real-OAS backtest depth is capped at ~2023. The separate `credit_funding_state_proxy` metric uses already-fetched `TLT`/`HYG`/`LQD` total-return differentials for longer directional history. `credit_funding_effective_state` is what downstream rules consume; do not add `IEF` / `BIL` without a spec amendment. |
| V2 §1D true PIT vendor data | CRSP / Compustat / FactSet / Norgate point-in-time S&P 500 membership (paid, not yet sourced) | event-driven membership changes | no output path | planned | V2 §1D currently uses `fja05680/sp500` GitHub CSV with documented `survivorship_biased_constituent_universe` warning (live-verified). True vendor PIT is the future upgrade path; TODO note added inline at `src/regime_data_fetch/pit_constituents.py`; the parquet schema (ticker / start_date / end_date intervals) is unchanged on swap. |
| `2y_yield` / `DGS2` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `10y_yield` / `DGS10` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `broad_usd_index` / `DTWEXBGS` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | explicit approved replacement for DXY; business-day macro release cadence; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `sofr` / `SOFR` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | overnight rate; next-business-day publication pattern; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `nfci` / `NFCI` | FRED API | weekly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Chicago Fed weekly update; do not assume fresh daily values; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items` / `CPIAUCSL` | FRED API | monthly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | monthly CPI level; available after BLS CPI release each month; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items_vintages` / `CPIAUCSL` realtime observations | FRED API with realtime params | monthly vintages | `data/raw/macro_vintages/cpi_all_items_vintages.parquet` | implemented-default-on | PIT/vintage view required by V2 §2A lines 2587-2593 for first-release historical replay (audit M2 — `docs/spec_code_data_audit_2026_05_15.md` §3.2). `--include-cpi-vintages` default is now **True** in `scripts/fetch_regime_engine_v1_data.py`; consumed by `loaders.load_cpi_vintages_first_release` and routed into `inflation_growth.compute_inflation_growth_features` when the `use_first_release_cpi_when_available` config flag is True. Emits `cpi_first_release_vintage_replay` provenance row on the feature-store bias-warnings frame |
| V2 §2A central-bank-text score | FOMC minutes + Powell speeches body_text (already fetched) | per release | derived in-engine via `regime_detection.central_bank_text` | done | deterministic-lexicon substitute for the spec's "LLM classifier" phrasing (V2 §2A lines 2578-2586). Required because V1 §2.2 stateless replay forbids LLM calls inside the engine — same precedent as DBC/AAII/Cleveland-Fed substitutes. Implemented in `src/regime_detection/central_bank_text.py`, loaded via `loaders.load_central_bank_text_score`, routed onto `MonetaryPressureV2Features.central_bank_text_score` as **evidence only — never a standalone label** per spec. Emits `central_bank_text_deterministic_lexicon_substitute` bias-warning row. See audit M1 — `docs/spec_code_data_audit_2026_05_15.md` §3.1 |
| V2 §1A SF Fed Daily News Sentiment Index | SF Fed XLSX (Shapiro/Sudhof/Wilson 2020) — `https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx` | weekly (workbook refresh) | `data/raw/news_sentiment/sf_fed_news_sentiment.parquet` | done | second sentiment voice alongside the AAII bull-bear 8w-MA `sentiment_score`. EVIDENCE ONLY — never read by the §1A `euphoria` rule predicate (spec line 164). Fetcher at `src/regime_data_fetch/sf_fed_news_sentiment.py`; loaded via `loaders.load_news_sentiment_series`; routed onto `TrendDirectionV2Features.news_sentiment_score` + derived `sentiment_concordance` (+1/0/-1/NaN). YAML block in `configs/core3-v2.0.0.yaml`. See V2 spec Ambiguity Log entry #74 + audit doc §4.1 |
| `iorb` / `IORB` | FRED API | business day | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | interest on reserve balances; use the published effective date from FRED; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| local `^NYICDX` dollar-index history | manual Yahoo Finance CSV export | daily | `data/raw/usd_index/nyicdx_daily.parquet` | operator-assisted | imported only by explicit `--fetch usd-index-local --usd-index-csv ...`; routine future USD ingestion for the regime engine is `broad_usd_index` from FRED `DTWEXBGS` through `--fetch macro`, so this local CSV path is intentionally excluded from unattended `--fetch all`; when `--acquisition-db` is supplied, the manual CSV file is recorded in SQLite before parquet/report output |
| PMI manufacturing headline values | live DBnomics primary, TradingEconomics backup; optional manual Investing TSV fallback | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | routine `--fetch pmi` / `--fetch all` uses live sources; live latest rows are merged into existing `us_ism_pmi_history.parquet` instead of replacing history, so a TradingEconomics latest-only fallback cannot collapse the historical PMI input. Use `--pmi-history-dir` only for pinned manual Investing histories |
| PMI services headline values | live DBnomics primary, TradingEconomics backup; optional manual Investing TSV fallback | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | routine `--fetch pmi` / `--fetch all` uses live sources; normalized output keeps monthly headline values and code-derived `10:00 ET` release timestamps |
| PMI release timestamps | code-derived ISM release calendar convention | monthly metadata | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | manufacturing = first business day 10:00 ET; services = third business day 10:00 ET |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | event-driven membership changes | `data/raw/pit_constituents/sp500_ticker_intervals.parquet` | done-live-verified | live fetch succeeded; rows carry `survivorship_biased_constituent_universe` warning and interval dates; when `--acquisition-db` is supplied, the raw GitHub CSV is recorded before parquet/report output |
| FOMC minutes raw text | Federal Reserve official pages: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm` + `https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm` + yearly `fomchistoricalYYYY.htm` pages + per-meeting `fomcminutesYYYYMMDD.htm` pages | about 8 times per year | `data/raw/fomc_minutes/fomc_minutes.parquet` | done-live-verified | fetched by walking the current calendar page for 2021+ meetings, walking the official historical year index for pre-2021 pages, then fetching each meeting HTML page and extracting title, meeting date text, body text, source URL, and PDF URL; current verified coverage reaches `2011-01-26` through `2026-03-18`; release timestamps encoded at `14:00 ET` on the Fed released date; when `--acquisition-db` is supplied, the raw listing pages, historical pages, and article pages are recorded before parquet/report output; pre-2011 not implemented yet |
| Powell speeches raw text | Federal Reserve official pages: `https://www.federalreserve.gov/newsevents/speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | irregular / event-driven | `data/raw/powell_speeches/powell_speeches.parquet` | done-live-verified | live fetch succeeded; current verified coverage reaches `2013-02-22` through `2026-03-21`; yearly archive pages are filtered to Powell-only entries and each speech page is fetched for title, speaker, location, and body text; Fed pages expose a date but no reliable publication time, so `publication_timestamp_precision=date_only` and timestamps are normalized to midnight Eastern; when `--acquisition-db` is supplied, the raw index page, yearly pages, and article pages are recorded before parquet/report output |
| aggregate forward EPS workbook snapshots | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx`, parsed from `ESTIMATES&PEs`; browser-assisted S&P auto path exists as explicit tool | manual snapshot / irregular | `data/raw/aggregate_forward_eps/sp500_eps_snapshots.parquet` | operator-assisted | fetched by explicit `--fetch eps --eps-workbook ...` or `--fetch eps-spglobal-auto`; both are excluded from unattended `--fetch all` because S&P access can require a local browser/session and the workbook itself says public files were discontinued. Parser extracts the workbook `as_of` date, historical quarterly rows, and the current forward estimate row; when `--acquisition-db` is supplied, the source workbook is recorded before output materialization |
| `aggregate_forward_eps_revision_direction_4w` | derived from the weekly EPS-snapshot accumulator | weekly | `data/raw/aggregate_forward_eps/sp500_eps_weekly_history.parquet` | operator-assisted | each explicit EPS snapshot run appends the workbook's current snapshot to `sp500_eps_weekly_history.parquet` (deduped by `observation_date`); `compute_eps_revision_direction_4w` derives the 4-week revision direction from that accumulator. The Wayback path (`--fetch eps-wayback`) is an explicit historical backfill, not a live unattended source |
| Investing.com economic calendar / holidays / earnings capture | Investing.com website via browser-backed capture, or archived captured pages | explicit operator-assisted capture/replay | live capture root at `data/raw/investing_live_archive/`; imported canonical outputs at `data/raw/investing/` plus SQLite artifact rows | operator-assisted | `--fetch investing-live` writes a dated archive under `investing_live_archive/`, then imports canonical `economic_events.parquet`, `holidays.parquet`, `earnings.parquet`, and `investing/raw_archive/`. `--fetch investing-archive-local` replays a supplied archive root into the same canonical outputs. Both are excluded from unattended `--fetch all` because Investing access can depend on local browser/session state and anti-bot behavior |
| Bloomberg / Refinitiv consensus surveys | paid vendor feeds | event-driven macro release cycle | no output path | hard-fail | unsupported unless spec explicitly adopts a paid source |
| I/B/E/S per-stock analyst revisions | paid vendor feeds | daily to weekly | no output path | hard-fail | unsupported in current V2 plan |
| ICE DXY history | licensed ICE feed | daily | no output path while spec stays on `broad_usd_index` | hard-fail | this row is only for the strict licensed ICE feed; do not point V2 at this row when using the available local Yahoo `^NYICDX` artifact instead |

### 2.1 V1 Build Scope

| Dataset | Symbols / Series | Source | Output |
|---|---|---|---|
| Daily OHLCV (market anchor) | `SPY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily OHLCV (breadth proxy) | `RSP` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily volatility proxy | `VIX` when Alpaca supports it, otherwise `VIXY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily OHLCV (stock universe) | 10B+ US stocks | Alpaca REST | `data/raw/daily_ohlcv/` |
| Scheduled event calendar | `FOMC`, `CPI`, `NFP` | generated YAML from Fed meeting pages + BLS release-schedule pages | `configs/events/us_events.yaml` |
| Rule-derived event windows | `expiry_week`, `earnings_season` | runtime rules in config/calendar logic | no stored raw file |

Constituent OHLCV universe source:

```text
data/raw/daily_ohlcv_762/symbol=*/ohlcv.parquet
```

The default operating model is a fixed 762-symbol regime universe materialized by the manifest. Operators pass either `--universe-json /path/to/symbols.json` or `--constituent-universe-dir /path/to/daily_ohlcv_762`. The PIT constituent parquet is available only behind `--allow-pit-constituent-universe` for bootstrap/backfill decisions, because using every historical PIT ticker expands beyond the intended 762-symbol engine universe.

When an operator already has a broader local OHLCV tree, materialize the runner-facing constituent tree explicitly instead of pointing runners at the broad source tree:

```bash
python3 scripts/materialize_constituent_ohlcv_tree.py \
  --source-tree data/raw/daily_ohlcv \
  --out-tree data/raw/daily_ohlcv_762 \
  --pit-parquet data/raw/pit_constituents/sp500_ticker_intervals.parquet \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD
```

The materializer accepts both `symbol=X/ohlcv.parquet` and partition-file source layouts, writes canonical `symbol=X/ohlcv.parquet` outputs, and fails loudly if any PIT-overlap constituent is missing unless `--allow-missing-symbols` is explicitly passed. It also writes `MANIFEST.sha256.json` under the output tree so a later run can verify exactly which symbol files were materialized.

### 2.2 V2 Build Scope

#### Market / Cross-Asset

| Dataset | Symbols / Series | Source | Output |
|---|---|---|---|
| Shared anchors | `SPY`, `RSP` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Bank stress proxy | `KRE` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Sector fragility universe | `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Cross-asset fragility universe | `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | Alpaca REST | `data/raw/daily_ohlcv/` |
| V2 §2B commodity-index proxy | `DBC` (Invesco DB Commodity Index Tracking Fund) | Alpaca REST | `data/raw/daily_ohlcv/` |
| Volatility proxy | `VIX` when available, otherwise `VIXY` | Alpaca REST | `data/raw/daily_ohlcv/` |

#### Macro

| Logical field | Series | Source | Output |
|---|---|---|---|
| `2y_yield` | `DGS2` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `10y_yield` | `DGS10` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `broad_usd_index` | `DTWEXBGS` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `sofr` | `SOFR` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `nfci` | `NFCI` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `cpi_all_items` | `CPIAUCSL` | FRED | `data/raw/macro/fred_macro_series.parquet` |
| `cpi_all_items_vintages` | `CPIAUCSL` with realtime params | FRED / ALFRED-style observations | `data/raw/macro_vintages/cpi_all_items_vintages.parquet` |
| `iorb` | `IORB` | FRED | `data/raw/macro/fred_macro_series.parquet` |

Runtime alignment note: V2 macro feature math reads FRED observations with
latest-known-as-of semantics on the NYSE session calendar. DGS2, DGS10,
DTWEXBGS, SOFR, IORB, and OAS are forward-filled for rolling computations,
while classifiers enforce freshness and staleness budgets. Do not treat a
one-session publication lag as missing; do treat values older than the
configured budget as stale/`unknown`.

#### Higher-Maintenance Inputs

These are part of the V2 data plan but are not all implemented yet:

| Dataset | Intended source | Notes |
|---|---|---|
| PMI manufacturing/services | repo-local manually supplied Investing release-history tables | use real PMI, not CFNAI substitution; current canonical backtest source is the repo-local manufacturing/services history tables aligned to the actual OHLCV coverage window; release timestamps remain locked to the ISM calendar convention |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | bias warning must be carried in output/report; current ingest stores ticker start/end intervals |
| FOMC minutes | Federal Reserve `fomccalendars.htm` + `fomc_historical_year.htm` + `fomchistoricalYYYY.htm` + minutes HTML pages | release timestamps required; current fetcher gets 2021+ meetings from the live calendar page, gets pre-2021 year pages from the official historical index, dedupes by `meeting_end_date`, and stores title, meeting date text, release timestamp, body text, source URL, and PDF URL; current verified lower bound is `2011-01-26` |
| Powell speeches | Federal Reserve `speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | current fetcher walks the Fed speeches index to yearly archives, filters archive rows to Powell-only entries, then fetches each Powell speech page and stores speech date, normalized publication timestamp, timestamp precision, title, speaker, location, body text, and source URL |
| Aggregate forward EPS revision direction | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx` parsed from `ESTIMATES&PEs` | the loader stores workbook-date observations and the current forward-estimate row, **and** accumulates one current-snapshot row per weekly fetch into `sp500_eps_weekly_history.parquet` (deduped by `observation_date`). `compute_eps_revision_direction_4w` reads that accumulator to produce `aggregate_forward_eps_revision_direction_4w`; the series is forward-filled onto the SPY session index and consumed by the V2 §2B `earnings_expansion` / `earnings_contraction` rules (Ambiguity Log #48 closed). All-NaN during the >4-week accumulator cold-start, so the two labels stay silent until enough weekly fetches accumulate. No paid feed — the weekly series is built by accumulation, not by a new source |
| Event calendar extension | generated `FOMC` / `CPI` / `NFP` YAML plus runtime `expiry_week` / `earnings_season` rules; V2 §2D adds official/deterministic/candidate-generated `budget_week` / `election_window` / `geopolitical_event` / `global_rate_decision` evidence | V1 auto-generates the V1 scheduled events and computes rule-derived windows at runtime. V2 §2D candidate generation writes parquet evidence first; only approved/promoted rows enter YAML. GPR/GDELT geopolitical candidates are never auto-promoted. `macro_event_score` in §4.2 includes `budget_week` / `election_window` / `global_rate_decision` in addition to `fed_week` / `cpi_week` / `nfp_week`; `geopolitical_event` is separate high-impact evidence. |
| V2 §2B commodity-index proxy (DBC) | Alpaca REST daily OHLCV | substitute for Bloomberg Commodity Index per Ambiguity Log #48; bias-warning row must be carried in feature-store output; not a §3.1 fragility universe member — V2 §2B-only feature input |
| V2 §2C HY/IG OAS + proxy | FRED `BAMLH0A0HYM2` (HY Master II OAS) + `BAMLC0A4CBBB` (BBB Corporate OAS), plus already-fetched `TLT`/`HYG`/`LQD` closes | real bps-level OAS produces `credit_funding_state` for 2023-05-15+; TLT-vs-HYG/LQD total-return differentials produce separate `credit_funding_state_proxy` for longer directional history (Ambiguity Log #49 + #71, ADR 0007). `credit_funding_effective_state` resolves the two classified labels for downstream rules without splicing raw series. **Do not add IEF or BIL to the universe** for a different proxy without a spec amendment. |

Implemented scheduled-event logic:

- `FOMC`: parse the current Fed meeting calendar page for recent years, then parse the Fed historical year index and yearly `fomchistoricalYYYY.htm` pages for older years. Emit one row per meeting using `meeting_end_date` as the event `date`. The associated minutes release date comes from the same Fed meeting listings, and the stored `release_timestamp_et` is encoded at `14:00 ET`.
- `CPI`: parse the official BLS yearly release schedules and keep only rows whose release title is `Consumer Price Index` or `Consumer Price Indexes`. The canonical source pages are the yearly BLS schedule pages under `/schedule/YYYY/` and `/schedule/YYYY/home.htm`; when direct BLS access is blocked, the repo can use the saved yearly HTML archive instead. Emit one row per monthly CPI release and store `release_timestamp_et` at the scheduled `08:30 ET` release time.
- `NFP`: parse the same BLS yearly release schedules and keep only rows whose release title is `The Employment Situation` or `Employment Situation`. The same local yearly HTML archive can stand in for the live site when direct BLS access is blocked. Emit one row per monthly NFP release and store `release_timestamp_et` at `08:30 ET`.
- Scheduled YAML is consumed through `load_scheduled_events_yaml()`, and `resolve_event_label()` expands scheduled NYSE trading-day windows with precedence `fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown`.
- `expiry_week`: compute the monthly options-expiry anchor as the third Friday, roll back to the previous NYSE trading day if that Friday is closed, then expand the runtime window `[-2, 0]` trading days around that anchor.
- `earnings_season`: compute quarterly windows anchored on the second Monday of `Jan/Apr/Jul/Oct`, ending `+35` calendar days later, and apply them at runtime rather than storing historical rows.
- Output rows are sorted by `release_timestamp_et` and written to generated YAML; the file is generated, not hand-edited.
- Current truth: the checked generated YAML and report close the V1 fixture-window CPI/NFP gap in this repo. The current artifact contains `454` total events: `147` FOMC rows (`2007-10-31` through `2026-03-18`), `131` CPI rows (`2016-01-20` through `2026-12-10`), and `131` NFP rows (`2016-01-08` through `2026-12-04`), with additional approved V2 rows for budget/election/global-rate decisions. The source is the BLS yearly schedule structure via the repo-local archive-backed fetch path.

### 2.3 Development Date Ranges

- Default V1-friendly range:
  - `2016-01-04` through today
- Recommended V2 backfill range:
  - `2016-01-04` through today, matching the checked OHLCV lower bound currently available in SQLite

The wider V2 range covers:
- 2010-05-06 and later V2 golden dates
- 504-day percentiles
- 5-year z-score baselines
- 250/252-day long lookbacks

## 3. Shadow-Mode Daily Acquisition

This section is the **operational plan** for the future V1 shadow runner. It is intentionally different from the development fetch path above.

### 3.1 Authoritative Daily Source for Shadow

Use:

- **Stooq** for `SPY`, `RSP`, and `VIX` daily data during shadow mode
- **FRED** for macro series used by V2-style evidence layers

Rationale:

- shadow replay must be reproducible from archived daily inputs
- daily source bytes must be frozen before classification
- Stooq is acceptable as the free, no-auth daily source for the shadow window
- if Stooq has a quality incident during shadow, upgrade the shadow source to Tiingo and restart or document according to incident policy

### 3.2 Shadow Storage

Primary store:

- local VPS `SQLite` ledger

Canonical artifacts:

- one JSON output per trading session
- one parquet input archive per trading session

Recommended shape:

```text
shadow_run/
├── regime_shadow.db
├── outputs/
│   └── YYYY-MM-DD.json
└── input_archives/
    └── YYYY-MM-DD/
        ├── market_data.parquet
        ├── events.yaml
        └── checksums.json
```

### 3.3 Shadow Rules

- archive inputs **before** calling `classify`
- replay historical dates only from archived inputs, never by re-fetching
- keep one immutable row per `(as_of_date, engine_version, config_version)`
- add a dead-man's-switch monitor so missed trading days alert within 24 hours
- cosmetic bugs do not restart the shadow year
- classifying bugs do restart the shadow year

### 3.4 Shadow Is Not Yet the Current Script

The current repo fetch script is a development/backfill tool. It is **not** the shadow runner and should not be treated as satisfying the shadow-mode operational spec by itself.

## 4. Current Gaps

The hard data blockers from earlier V2 slices are now closed (see "Recently closed" below) — the FRED-availability audit, the weekly-snapshot accumulator pattern, and the ADR-directed source substitutions resolved them with no paid feed. What remains is a short list of non-blocking gaps:

- **V2 §1D true PIT vendor data** — V2 §1D currently ships on `fja05680/sp500` GitHub approximation with documented `survivorship_biased_constituent_universe` warning; CRSP / Compustat / FactSet / Norgate is the future vendor upgrade (an upgrade path, not a blocker — the feature ships today)
- **CPI vintage artifact** — `cpi_all_items_vintages.parquet` is now **default-on** in `scripts/fetch_regime_engine_v1_data.py` and consumed by `inflation_growth.compute_inflation_growth_features` via the `cpi_first_release` seam (audit M2). After the next fetch run with default flags the parquet materializes and replay becomes PIT-accurate. Use `--no-include-cpi-vintages` to skip if explicitly intended (e.g., shadow-mode pinned to revised CPI).
- **Put-call ratio / Investors Intelligence sentiment** — optional alternative `sentiment_score` sources for a future calibration revision; no fetcher built, but they block no label (`euphoria` fires via AAII)
- **Dedicated shadow runner** with SQLite ledger and archived daily input snapshots — operational gap, not a data-source gap

### Recently closed (no longer degraded)

- **V2 §2A central-bank-text → hawkish/dovish evidence** — closed by audit M1 (`docs/spec_code_data_audit_2026_05_15.md` §3.1). FOMC minutes + Powell speeches were already fetched and stored; the engine now scores them deterministically via `regime_detection.central_bank_text` (lexicon-based, replay-safe) and surfaces the score on `MonetaryPressureV2Features.central_bank_text_score` as evidence — never a standalone label per spec. Bias warning `central_bank_text_deterministic_lexicon_substitute` is emitted. The lexicon substitution mirrors the existing precedent (DBC ← BCOM, AAII ← Bloomberg consensus, Cleveland Fed ← analyst consensus, VIXCLS ← options-IV, fja05680 ← vendor PIT).
- **V2 §2A first-release CPI for historical replay** — closed by audit M2 (`docs/spec_code_data_audit_2026_05_15.md` §3.2). `--include-cpi-vintages` default flipped to True; `loaders.load_cpi_vintages_first_release` picks the earliest `realtime_start` per reference date; `inflation_growth.compute_inflation_growth_features` accepts a `cpi_first_release` series and substitutes it for revised CPIAUCSL when the `use_first_release_cpi_when_available` flag is True (default). Bias warning `cpi_first_release_vintage_replay` is emitted on the three CPI-derived features.

- **Weekly `aggregate_forward_eps_revision_direction_4w`** — closed by the `aggregate_eps` weekly-snapshot accumulator + end-to-end engine wiring (Ambiguity Log #48). The weekly series is built by accumulating one current-snapshot row per weekly fetch. The >4-week cold-start is collapsed in one shot by `run_wayback_aggregate_eps_fetch`: the Wayback backfill now automatically seeds `data/raw/aggregate_forward_eps/sp500_eps_weekly_history.parquet` from the historical timeline, with existing live rows winning on date collision. `earnings_expansion` / `earnings_contraction` then have real evidence immediately. No paid feed.
- **V2 §2B `inflation_surprise_zscore`** — closed by ADR 0006: the free Cleveland Fed inflation nowcast substitutes for the paid analyst `consensus_estimate`. `regime_data_fetch.cleveland_fed_nowcast` downloads + parses the Cleveland Fed month-over-month nowcast webchart JSON directly (verified reachable over urllib) — 154 monthly vintages 2013-01→present in the current materialization, no operator action and no manual drop required. Category labels in the feed can be `MM/DD/YYYY`, `YYYY-MM-DD`, or no-year `MM/DD`; the parser keys each nowcast to the last non-empty point's publication date without leaking future values.
- **V2 §1A `sentiment_score`** — closed: AAII fetcher ships, `sentiment_score = bull_bear_spread_8w_ma`, `euphoria` fires (Ambiguity Log #32).
- **V2 §1C implied vol** — closed: `implied_vol_30d = VIXCLS / 100` from FRED; `vol_crush` + `iv_rv_spread` ship (Ambiguity Log #19 / #20; ADR 0005).
- **V2 §2C HY/IG OAS + parallel proxy** — closed: real ICE BofA OAS is sourced free from FRED (`BAMLH0A0HYM2` / `BAMLC0A4CBBB`) for 2023-05-15+; the TLT-vs-HYG/LQD total-return proxy remains as a separate `credit_funding_state_proxy` metric for longer directional history, and `credit_funding_effective_state` resolves the two labels for downstream rules (Ambiguity Log #49 + #71, ADR 0007).
- **30-session profile evidence completeness** — closed by ADR 0008. Macro feature math now uses latest-known-as-of FRED alignment with freshness gates, HMM/GMM evidence is emitted point-in-time per warmed session, and profile/timeline materialization keeps five extra warmed sessions for `hmm_probability_shift[t-5]`.
## 5. Explicit Hard Failures

The fetch layer should fail loudly, not substitute silently, for these unsupported inputs:

- Bloomberg / Refinitiv consensus-survey feeds (unless V2 spec explicitly adopts the paid source)
- Licensed ICE DXY (the spec remains on `broad_usd_index` from FRED `DTWEXBGS`)
- I/B/E/S per-stock analyst revision feeds — V2 §2B uses the *aggregate* S&P Global forward-EPS series, not per-stock revisions.
- `IEF`, `BIL` — these ETFs are **not** in the V2 cross-asset fragility universe by design. V2 §2C consumes the real ICE BofA OAS series from FRED directly and computes its approved parallel proxy from already-fetched `TLT`/`HYG`/`LQD`; **do not extend the universe** with IEF/BIL for another proxy without a spec amendment.

Note: ICE BofA OAS, options-implied vol, and AAII sentiment were previously listed here as hard failures ("currently absent"). They are now **sourced free** — ICE BofA OAS via FRED `BAMLH0A0HYM2` / `BAMLC0A4CBBB`, implied vol via FRED `VIXCLS`, AAII via the `aaii_sentiment` fetcher — and are no longer hard-fail inputs.

Documented substitute policies (each pinned in the V2 Implementation Ambiguity Log):

- **CPI surprise**: substituted by the free Cleveland Fed inflation nowcast for the analyst `consensus_estimate` (ADR 0006 / Ambiguity Log #48); the feature carries a model-relative bias-warning row. Fetch path: `regime_data_fetch.cleveland_fed_nowcast`.
- **`broad_usd_index`**: approved field name for the free FRED route (FRED `DTWEXBGS`); do not back-door ICE DXY semantics into it.
- **Bloomberg Commodity Index**: substituted by `DBC` ETF per V2 §2B / Ambiguity Log #48; bias-warning row required in feature-store output.
- **HY/IG OAS**: sourced directly as real ICE BofA OAS from FRED (`BAMLH0A0HYM2` / `BAMLC0A4CBBB`) for `credit_funding_state`; the TLT-vs-HYG/LQD total-return proxy is separate `credit_funding_state_proxy`; downstream uses `credit_funding_effective_state` with source/agreement evidence (Ambiguity Log #49 + #71, ADR 0007).
- **PIT S&P 500 membership**: currently using `fja05680/sp500` GitHub CSV with documented `survivorship_biased_constituent_universe` warning; vendor upgrade noted in `pit_constituents.py` TODO.

## 6. Source Rules

General discipline:

- Do not silently substitute a different economic concept because it is cheaper.
- Every approved substitute is recorded in the V2 Implementation Ambiguity Log with a bias-warning requirement on the feature-store output; introducing a new substitute requires a spec amendment, not a fetch-layer choice.
- For forward shadow, archive exact inputs used each day before classification.

V1 / V2 source pins (each enforces a spec field):

- If the spec says **PMI**, fetch PMI (ISM Manufacturing / Services). Do not substitute CFNAI or another macro proxy.
- If the spec says **`broad_usd_index`**, fetch FRED `DTWEXBGS`. Do not back-door ICE DXY semantics into this field.
- If the spec says **`aggregate_forward_eps_revision_direction`**, fetch the aggregate S&P Global series; do not substitute a per-stock breadth proxy.
- For development/backfill, prefer Alpaca `VIX`; when unavailable, use `VIXY` as the documented operational proxy.
- For V1 event calendar work, generate scheduled `FOMC` rows from official Fed meeting pages, generate scheduled `CPI` / `NFP` rows from official BLS release schedules, compute `expiry_week` / `earnings_season` from deterministic rules.

V2 spec-amendment pins (this session, Ambiguity Log #46–#53):

- If V2 §2B says **commodity returns**, fetch `DBC` ETF as the approved Bloomberg Commodity Index substitute (Ambiguity Log #48). Emit a bias-warning row in the V2 §2B feature-store output.
- If V2 §2C says **HY/IG credit spread**, read the real ICE BofA OAS series from FRED — `BAMLH0A0HYM2` (HY) and `BAMLC0A4CBBB` (BBB), both in `V2_FRED_SERIES` (Ambiguity Log #49). Keep the approved TLT-vs-HYG/LQD proxy as the separate `credit_funding_state_proxy` metric for longer directional history and route downstream rules through `credit_funding_effective_state` (Ambiguity Log #71 / ADR 0007). Do not add IEF/BIL or introduce another ETF-pair proxy without a spec amendment.
- If V2 §2D says **`budget_week` / `election_window` / `geopolitical_event` / `global_rate_decision`**, use the event-source pipeline: deterministic/official adapters for scheduled/official rows and Group B candidate parquet for approval-gated rows. `geopolitical_event` may be generated from GPR, GDELT, HDX HAPI, and, after TODO API-key entitlement is resolved, ACLED plus Uppsala/UCDP evidence. It must stay overlay-only; do not auto-promote it from external news APIs, humanitarian aggregates, or LLM extraction.
- If V2 §1A says **`euphoria`**, read `sentiment_score = bull_bear_spread_8w_ma` from the AAII fetcher (`regime_data_fetch.aaii_sentiment`); the label fires (Ambiguity Log #32). Do not substitute a different sentiment proxy; put-call / Investors Intelligence are optional future calibration sources only.
- If V2 §1C says **`vol_crush`** or **IV/RV spread**, read `implied_vol_30d = VIXCLS / 100` from FRED (Ambiguity Log #19 / #20; ADR 0005). Do not synthesise IV from realized-vol — VIX is the canonical implied-vol series.
- If V2 §2B says **`inflation_surprise_zscore`**, read `cpi_nowcast` from the Cleveland Fed inflation nowcast (`regime_data_fetch.cleveland_fed_nowcast`) as the `consensus_estimate` substitute (ADR 0006). Do not source a paid analyst-survey feed.
- If V2 §1D says **`pct_above_50dma`** or other PIT-constituent breadth features, use the `fja05680/sp500` PIT intervals (current default) combined with the 762-stock daily OHLCV (already in SQLite). Emit a `survivorship_biased_constituent_universe` warning row and keep the TODO note in `pit_constituents.py` pointing to the future vendor PIT upgrade.
