# Market Data Fetch Plan

This document now separates two workflows that were previously mixed together:

1. **Build / backfill fetches** used to develop and test V1/V2 locally
2. **Shadow-mode daily acquisition** used after V1 is frozen and running operationally

Those are not interchangeable. Historical backfills test engine logic. Forward shadow tests operational stability.

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
- PMI retrieval for backtesting now uses repo-local manually supplied Investing release-history tables for manufacturing and services, aligned to the live OHLCV lower bound. The older DBnomics / TradingEconomics fetch path is no longer the canonical backtest source in this repo.
- `earnings_revision_breadth` is replaced by `aggregate_forward_eps_revision_direction`, sourced from S&P Global aggregate forward EPS data.
- **V2 §2B commodity returns**: `DBC` ETF (Invesco DB Commodity Index Tracking Fund) is the approved substitute for the Bloomberg Commodity Index (paid feed unavailable). Pinned in V2 §2B Ambiguity Log entry #48. Rows must carry a documented bias-warning analogous to the §1D PIT-source pattern; a future spec-amendment slice may replace DBC with a direct Bloomberg / Refinitiv commodity-index feed when vendor sourcing is approved.
- **V2 §2C HY/IG spread direction**: now sourced from **real ICE BofA Option-Adjusted Spread series, free on FRED** — `hy_oas = BAMLH0A0HYM2` (HY Master II OAS) and `ig_bbb_oas = BAMLC0A4CBBB` (BBB Corporate OAS), both in `V2_FRED_SERIES` (Ambiguity Log #49 closed; commits `814a8d5` + `9cad7e7`). The earlier TLT-vs-HYG/LQD return-differential proxy and its dual-source fallback path were **deleted** — `credit_funding` single-sources on the FRED OAS feeds and lists them in `REQUIRED_MACRO_KEYS`. **Do not add `IEF` or `BIL` to the universe** for a credit-spread proxy — the §2C seam consumes the FRED OAS series directly; no proxy path exists.
- **V2 §2D event rows** (`budget_week`, `election_window`, `geopolitical_event`, `global_rate_decision`) are now **supported as manual operator-curated YAML** at the spec level (V2 §2D + §4.2 + Ambiguity Log #50). They were previously documented as `hard-fail` in this plan; the spec-amendment cycle this session promoted them to manual-YAML-supported (no auto-fetcher; rows are manually authored by the operator following the §2D YAML schema). `election_window` default window is `[-5, +10]` trading days. `global_rate_decision` covers BOE / ECB / BOJ scheduled meetings (analogous to V1 FOMC pre-2021 manual path). `geopolitical_event` stays a manual flag for war / sanctions / terrorism. `macro_event_score` in §4.2 now scores 1.0 on these labels in addition to `fed_week` / `cpi_week` / `nfp_week`.
- **V2 §1A `euphoria` label — resolved**: `sentiment_score = bull_bear_spread_8w_ma` from the AAII weekly survey; the AAII fetcher ships (`regime_data_fetch.aaii_sentiment`), and `euphoria` fires (Ambiguity Log #32 status update). Put-call ratio / Investors Intelligence remain *optional* alternative sources for a future calibration revision — they block no label and need no fetcher to ship V2.
- **V2 §1C `vol_crush` / `IV/RV spread` — resolved**: `implied_vol_30d = VIXCLS / 100` from FRED (the CBOE VIX IS the canonical model-free 30-day implied vol on SPX — no options-chain feed needed). `iv_rv_spread`, `implied_vol_30d`, `implied_vol_5d_change` ship on `VolatilityV2Features`; the `vol_crush` rule wires through (Ambiguity Log #19 / #20 status updates; ADR 0005).
- **V2 §2B `inflation_surprise_zscore` — resolved**: the analyst-survey `consensus_estimate` is substituted by the free Cleveland Fed inflation nowcast (ADR 0006). The dedicated `regime_data_fetch.cleveland_fed_nowcast` fetch path is built (manual-drop architecture; CSV schema is operator-verified on first run). The single-signal `inflation_shock` limb consumes `inflation_surprise_zscore` and is silent only during the 5y cold-start or when `cpi_nowcast` is unwired; the composite-shock limb is always active.
- **V2 §1D PIT constituent universe**: continues to use the GitHub `fja05680/sp500` `sp500_ticker_start_end.csv` as the V2 ship default (rows already carry `survivorship_biased_constituent_universe` warning). A `TODO` note is now in `src/regime_data_fetch/pit_constituents.py` recommending replacement with a true point-in-time vendor feed (CRSP / Compustat / FactSet / Norgate) when sourcing is approved. The expected vendor format matches the same ticker / start_date / end_date interval shape, so the parquet schema does not change on swap.

### 2.0A Data Inventory

Status meanings used below:

- `done-live-verified`: implemented and verified against the real source in this repo workflow
- `implemented-not-live-verified`: implemented in code but not live-verified in the current session
- `template-only`: only a placeholder/template exists
- `planned`: source/path identified, loader not implemented yet
- `hard-fail`: intentionally unsupported unless the spec/source decision changes

| Data | Source | Cadence | Output / Path | Status | Comment |
|---|---|---|---|---|---|
| US universe cache JSON | manually supplied external artifact | ad hoc refresh | `data/raw/universe/us_universe_cache.json` if you choose to keep a local copy | template-only | this repo no longer builds or refreshes the universe cache from yfinance; if you want to preserve a cache artifact, manage it outside this repo and treat it as an external input |
| 10B+ US stock universe symbol list | explicit `--universe-json` symbol file | ad hoc refresh | loaded in-memory from `--universe-json` | done-live-verified | V1/all stock-universe fetches now require an explicit JSON list of symbols; this repo does not attempt to derive or refresh the universe membership itself |
| 762-stock daily OHLCV backfill | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | live-verified local import from `/Users/avinashvankadaru/.superset/worktrees/regime-detection/v1-of-regime-detection/data/raw/daily_ohlcv` now exists in SQLite with `763` symbols, `1,686,570` rows, and checked coverage `2016-01-04` through `2026-05-05`; the direct Alpaca fetch path is still available but is no longer the only grounded source of truth here |
| `SPY` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V1 market anchor is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05`; Alpaca remains the documented refresh path if you later need a direct pull |
| `RSP` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V1 breadth proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05`; Alpaca remains the documented refresh path if you later need a direct pull |
| `VIX` daily proxy bars | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | the checked local market dataset covers the documented volatility proxy requirement operationally via the imported symbol set; if you later refresh from Alpaca directly, true `VIX` is preferred when the account supports it |
| `VIXY` daily proxy bars | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | documented operational volatility proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| `KRE` daily OHLCV | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | V2 bank-stress proxy is present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| Sector ETF daily OHLCV: `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | all listed sector ETFs are present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| Cross-asset ETF daily OHLCV: `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | local imported parquet dataset; Alpaca REST remains an optional refresh path | daily | SQLite `daily_ohlcv_rows` plus source parquet artifacts in the acquisition DB | done-live-verified | all listed cross-asset ETFs are present in the live-verified local OHLCV import covering `2016-01-04` through `2026-05-05` |
| `DBC` daily OHLCV (V2 §2B commodity proxy) | local imported parquet dataset; Alpaca REST as refresh path | daily | SQLite `daily_ohlcv_rows` (extension row) | planned | approved substitute for Bloomberg Commodity Index per V2 §2B Ambiguity Log #48; DBC is NOT in the §3.1 fragility universe — it is a V2 §2B-only feature input; rows must carry a documented bias-warning; vendor upgrade (Bloomberg / Refinitiv commodity index) noted as future spec-amendment path |
| Scheduled event rows: `FOMC` | generated repo-local YAML from Federal Reserve FOMC calendar pages | about 8 times per year | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse current `fomccalendars.htm`, add older meetings from `fomc_historical_year.htm` and yearly `fomchistoricalYYYY.htm` pages, dedupe by `meeting_end_date`, and store minutes release timestamps at `14:00 ET`; current live-verified coverage is `2007-10-31` through `2026-03-18` |
| Scheduled event rows: `CPI` | generated repo-local YAML from BLS yearly release-schedule pages, using the local yearly HTML archive when direct access is blocked | monthly | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `Consumer Price Index` / `Consumer Price Indexes` rows, and store release timestamps at `08:30 ET`; the checked generated YAML currently contains `311` CPI rows spanning `2000-01-14` through `2025-12-18` |
| Scheduled event rows: `NFP` | generated repo-local YAML from BLS yearly release-schedule pages, using the local yearly HTML archive when direct access is blocked | monthly | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `The Employment Situation` / `Employment Situation` rows, and store release timestamps at `08:30 ET`; the checked generated YAML currently contains `311` NFP rows spanning `2000-01-07` through `2025-12-16` |
| Rule-derived event window: `expiry_week` | computed from deterministic rules in config/runtime | monthly | no stored raw file | done-live-verified | compute the third Friday of each month, roll back to the previous NYSE trading day if that Friday is closed, then expand the configured NYSE trading-day window around the anchor; the runtime rule is now wired through `resolve_event_label()` and live-verified with NYSE holiday-sensitive months like `2019-04`, `2022-04`, and `2026-06` |
| Rule-derived event window: `earnings_season` | computed from deterministic rules in config/runtime | quarterly window | no stored raw file | done-live-verified | compute quarter windows starting on the second Monday of `Jan/Apr/Jul/Oct` and ending `+35` calendar days later; the runtime rule is now wired through `resolve_event_label()` and live-verified across `2015-01-01` through `2026-05-07` |
| V2 §2D event rows (`election_window`, `geopolitical_event`, `budget_week`, `global_rate_decision`) | manual operator-curated YAML following the V2 §2D schema | irregular | extends `configs/events/us_events.yaml` (or a sibling `configs/events/us_events_v2.yaml`) | manual-YAML-supported | promoted from `hard-fail` to supported in this V2 spec-amendment cycle (Ambiguity Log #50). `election_window` default window is `[-5, +10]` trading days. `global_rate_decision` covers BOE / ECB / BOJ scheduled meetings; operator maintains the YAML by hand (no auto-fetcher). `geopolitical_event` stays manual flag for war/sanctions/terrorism. `macro_event_score` in §4.2 now includes these labels. V1 path remains unaffected (V1 only emits `fed_week` / `cpi_week` / `nfp_week` / `expiry_week` / `earnings_season`). |
| V2 §2B `cpi_nowcast` (Cleveland Fed inflation nowcast) | Cleveland Fed "Inflation Nowcasting" CSV export (free) | intra-month, re-fetch weekly/monthly | `data/raw/cleveland_fed_nowcast/cpi_nowcast.parquet` | fetch-path-built | feeds `inflation_surprise_zscore` (V2 §2B single-signal `inflation_shock` limb); ADR 0006 substituted the free Cleveland Fed nowcast for the paid analyst `consensus_estimate`. `regime_data_fetch.cleveland_fed_nowcast` ships the manual-drop fetcher; operator drops the CSV and verifies the column mapping (`date_column` / `value_column` / `value_scale`) on first run |
| V2 §1A sentiment_score | AAII bull-bear weekly survey (sourced) | weekly | `data/raw/sentiment/aaii_sentiment.parquet` | done | `sentiment_score = bull_bear_spread_8w_ma`; `regime_data_fetch.aaii_sentiment` ships and the V2 §1A `euphoria` label fires (Ambiguity Log #32). Put-call ratio / Investors Intelligence are optional future calibration-revision sources only — they block no label |
| V2 §1C implied vol (`VIXCLS`) | FRED API (CBOE VIX — the model-free 30-day implied vol on SPX) | daily | `data/raw/macro/fred_macro_series.parquet` | done | `implied_vol_30d = VIXCLS / 100`; feeds V2 §1C `vol_crush` rule + `iv_rv_spread` feature (Ambiguity Log #19 / #20; ADR 0005). No options-chain feed needed — VIX is the canonical implied-vol series and is free on FRED |
| V2 §2C HY/IG OAS feeds | ICE BofA Option-Adjusted Spread series, free on FRED: `BAMLH0A0HYM2` (HY) + `BAMLC0A4CBBB` (BBB) | daily | `data/raw/macro/fred_macro_series.parquet` | done | sourced directly from FRED and in `V2_FRED_SERIES` (Ambiguity Log #49; commits `814a8d5` + `9cad7e7`). The earlier TLT-vs-HYG/LQD proxy and dual-source fallback were deleted — `credit_funding` single-sources on these and lists them in `REQUIRED_MACRO_KEYS`. Do not add `IEF` / `BIL` for a different proxy |
| V2 §1D true PIT vendor data | CRSP / Compustat / FactSet / Norgate point-in-time S&P 500 membership (paid, not yet sourced) | event-driven membership changes | no output path | planned | V2 §1D currently uses `fja05680/sp500` GitHub CSV with documented `survivorship_biased_constituent_universe` warning (live-verified). True vendor PIT is the future upgrade path; TODO note added inline at `src/regime_data_fetch/pit_constituents.py`; the parquet schema (ticker / start_date / end_date intervals) is unchanged on swap. |
| `2y_yield` / `DGS2` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `10y_yield` / `DGS10` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `broad_usd_index` / `DTWEXBGS` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | explicit approved replacement for DXY; business-day macro release cadence; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `sofr` / `SOFR` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | overnight rate; next-business-day publication pattern; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `nfci` / `NFCI` | FRED API | weekly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Chicago Fed weekly update; do not assume fresh daily values; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items` / `CPIAUCSL` | FRED API | monthly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | monthly CPI level; available after BLS CPI release each month; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items_vintages` / `CPIAUCSL` realtime observations | FRED API with realtime params | monthly vintages | `data/raw/macro_vintages/cpi_all_items_vintages.parquet` | done-live-verified | PIT/vintage view; new vintage appears on CPI release cycle; when `--acquisition-db` is supplied, the raw realtime-observations JSON response is recorded before parquet/report output |
| `iorb` / `IORB` | FRED API | business day | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | interest on reserve balances; use the published effective date from FRED; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| local `^NYICDX` dollar-index history | manual Yahoo Finance CSV export | daily | `data/raw/usd_index/nyicdx_daily.parquet` | done-live-verified | imported via `--fetch usd-index-local --usd-index-csv /Users/avinashvankadaru/Desktop/NYICDX_history.csv`; keep this row explicit for V2 so any DXY-style downstream use points to the Yahoo `^NYICDX` artifact and not to the separate unsupported licensed ICE feed row; when `--acquisition-db` is supplied, the manual CSV file is recorded in SQLite before parquet/report output |
| PMI manufacturing headline values | repo-local manually supplied Investing release-history table | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | current canonical backtest source is the repo-local manufacturing PMI history table derived from the Investing manufacturing PMI calendar page and aligned to the OHLCV lower bound `2016-01-04`; normalized output keeps monthly headline values and code-derived `10:00 ET` release timestamps |
| PMI services headline values | repo-local manually supplied Investing release-history table | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | current canonical backtest source is the repo-local services PMI history table derived from the Investing services PMI calendar page and aligned to the OHLCV lower bound `2016-01-04`; normalized output keeps monthly headline values and code-derived `10:00 ET` release timestamps |
| PMI release timestamps | code-derived ISM release calendar convention | monthly metadata | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | manufacturing = first business day 10:00 ET; services = third business day 10:00 ET |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | event-driven membership changes | `data/raw/pit_constituents/sp500_ticker_intervals.parquet` | done-live-verified | live fetch succeeded; rows carry `survivorship_biased_constituent_universe` warning and interval dates; when `--acquisition-db` is supplied, the raw GitHub CSV is recorded before parquet/report output |
| FOMC minutes raw text | Federal Reserve official pages: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm` + `https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm` + yearly `fomchistoricalYYYY.htm` pages + per-meeting `fomcminutesYYYYMMDD.htm` pages | about 8 times per year | `data/raw/fomc_minutes/fomc_minutes.parquet` | done-live-verified | fetched by walking the current calendar page for 2021+ meetings, walking the official historical year index for pre-2021 pages, then fetching each meeting HTML page and extracting title, meeting date text, body text, source URL, and PDF URL; current verified coverage reaches `2011-01-26` through `2026-03-18`; release timestamps encoded at `14:00 ET` on the Fed released date; when `--acquisition-db` is supplied, the raw listing pages, historical pages, and article pages are recorded before parquet/report output; pre-2011 not implemented yet |
| Powell speeches raw text | Federal Reserve official pages: `https://www.federalreserve.gov/newsevents/speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | irregular / event-driven | `data/raw/powell_speeches/powell_speeches.parquet` | done-live-verified | live fetch succeeded; current verified coverage reaches `2013-02-22` through `2026-03-21`; yearly archive pages are filtered to Powell-only entries and each speech page is fetched for title, speaker, location, and body text; Fed pages expose a date but no reliable publication time, so `publication_timestamp_precision=date_only` and timestamps are normalized to midnight Eastern; when `--acquisition-db` is supplied, the raw index page, yearly pages, and article pages are recorded before parquet/report output |
| aggregate forward EPS workbook snapshots | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx`, parsed from `ESTIMATES&PEs` | manual snapshot / irregular | `data/raw/aggregate_forward_eps/sp500_eps_snapshots.parquet` | done-live-verified | fetched by pointing `--fetch eps --eps-workbook /Users/avinashvankadaru/Desktop/sp-500-eps-est.xlsx` at the saved workbook; parser extracts the workbook `as_of` date, historical quarterly observation rows, and the current forward estimate row into parquet plus `aggregate_eps_fetch_report.json`; when `--acquisition-db` is supplied, the manual workbook file is recorded in the shared SQLite acquisition store before output materialization; live-verified workbook date is `2026-01-30`; the workbook itself says the public files were discontinued |
| `aggregate_forward_eps_revision_direction_4w` | derived from the weekly EPS-snapshot accumulator | weekly | `data/raw/aggregate_forward_eps/sp500_eps_weekly_history.parquet` | done | each weekly `--fetch eps` run appends the workbook's current snapshot to `sp500_eps_weekly_history.parquet` (deduped by `observation_date`); `compute_eps_revision_direction_4w` derives the 4-week revision direction from that accumulator (Ambiguity Log #48 closed). The single workbook only exposes quarterly history + one current point — weekly granularity is built by accumulation. All-NaN during the >4-week cold-start. The Wayback backfill path additionally records raw CDX listings, archived workbook files, and derived timeline artifacts in SQLite when `--acquisition-db` is supplied |
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

Universe source:

```text
--universe-json /path/to/symbols.json
```

provided explicitly by the operator.

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

#### Higher-Maintenance Inputs

These are part of the V2 data plan but are not all implemented yet:

| Dataset | Intended source | Notes |
|---|---|---|
| PMI manufacturing/services | repo-local manually supplied Investing release-history tables | use real PMI, not CFNAI substitution; current canonical backtest source is the repo-local manufacturing/services history tables aligned to the actual OHLCV coverage window; release timestamps remain locked to the ISM calendar convention |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | bias warning must be carried in output/report; current ingest stores ticker start/end intervals |
| FOMC minutes | Federal Reserve `fomccalendars.htm` + `fomc_historical_year.htm` + `fomchistoricalYYYY.htm` + minutes HTML pages | release timestamps required; current fetcher gets 2021+ meetings from the live calendar page, gets pre-2021 year pages from the official historical index, dedupes by `meeting_end_date`, and stores title, meeting date text, release timestamp, body text, source URL, and PDF URL; current verified lower bound is `2011-01-26` |
| Powell speeches | Federal Reserve `speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | current fetcher walks the Fed speeches index to yearly archives, filters archive rows to Powell-only entries, then fetches each Powell speech page and stores speech date, normalized publication timestamp, timestamp precision, title, speaker, location, body text, and source URL |
| Aggregate forward EPS revision direction | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx` parsed from `ESTIMATES&PEs` | the loader stores workbook-date observations and the current forward-estimate row, **and** accumulates one current-snapshot row per weekly fetch into `sp500_eps_weekly_history.parquet` (deduped by `observation_date`). `compute_eps_revision_direction_4w` reads that accumulator to produce `aggregate_forward_eps_revision_direction_4w`; the series is forward-filled onto the SPY session index and consumed by the V2 §2B `earnings_expansion` / `earnings_contraction` rules (Ambiguity Log #48 closed). All-NaN during the >4-week accumulator cold-start, so the two labels stay silent until enough weekly fetches accumulate. No paid feed — the weekly series is built by accumulation, not by a new source |
| Event calendar extension | generated `FOMC` / `CPI` / `NFP` YAML plus runtime `expiry_week` / `earnings_season` rules; V2 §2D adds manual-YAML `budget_week` / `election_window` / `geopolitical_event` / `global_rate_decision` | V1 auto-generates the V1 scheduled events and computes rule-derived windows at runtime; V2 §2D adds operator-curated YAML rows for the 4 new event labels (no auto-fetcher, no auto-derivation from external news APIs). `macro_event_score` in §4.2 now includes the V2 §2D labels in addition to `fed_week` / `cpi_week` / `nfp_week`. |
| V2 §2B commodity-index proxy (DBC) | Alpaca REST daily OHLCV | substitute for Bloomberg Commodity Index per Ambiguity Log #48; bias-warning row must be carried in feature-store output; not a §3.1 fragility universe member — V2 §2B-only feature input |
| V2 §2C HY/IG OAS (ICE BofA, free on FRED) | FRED `BAMLH0A0HYM2` (HY Master II OAS) + `BAMLC0A4CBBB` (BBB Corporate OAS), in `V2_FRED_SERIES` | real bps-level OAS — `credit_funding` single-sources on these (Ambiguity Log #49; commits `814a8d5` + `9cad7e7`). The earlier TLT-vs-HYG/LQD return-differential proxy and its dual-source fallback path were **deleted**; the FRED OAS series are in `REQUIRED_MACRO_KEYS` so the §2C seam does not build without them. **Do not add IEF or BIL to the universe** for a different proxy — no proxy path exists. |

Implemented scheduled-event logic:

- `FOMC`: parse the current Fed meeting calendar page for recent years, then parse the Fed historical year index and yearly `fomchistoricalYYYY.htm` pages for older years. Emit one row per meeting using `meeting_end_date` as the event `date`. The associated minutes release date comes from the same Fed meeting listings, and the stored `release_timestamp_et` is encoded at `14:00 ET`.
- `CPI`: parse the official BLS yearly release schedules and keep only rows whose release title is `Consumer Price Index` or `Consumer Price Indexes`. The canonical source pages are the yearly BLS schedule pages under `/schedule/YYYY/` and `/schedule/YYYY/home.htm`; when direct BLS access is blocked, the repo can use the saved yearly HTML archive instead. Emit one row per monthly CPI release and store `release_timestamp_et` at the scheduled `08:30 ET` release time.
- `NFP`: parse the same BLS yearly release schedules and keep only rows whose release title is `The Employment Situation` or `Employment Situation`. The same local yearly HTML archive can stand in for the live site when direct BLS access is blocked. Emit one row per monthly NFP release and store `release_timestamp_et` at `08:30 ET`.
- Scheduled YAML is consumed through `load_scheduled_events_yaml()`, and `resolve_event_label()` expands scheduled NYSE trading-day windows with precedence `fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown`.
- `expiry_week`: compute the monthly options-expiry anchor as the third Friday, roll back to the previous NYSE trading day if that Friday is closed, then expand the runtime window `[-2, 0]` trading days around that anchor.
- `earnings_season`: compute quarterly windows anchored on the second Monday of `Jan/Apr/Jul/Oct`, ending `+35` calendar days later, and apply them at runtime rather than storing historical rows.
- Output rows are sorted by `release_timestamp_et` and written to generated YAML; the file is generated, not hand-edited.
- Current truth: the checked generated YAML and report close the historical CPI/NFP gap in this repo. The current artifact contains `311` CPI rows (`2000-01-14` through `2025-12-18`) and `311` NFP rows (`2000-01-07` through `2025-12-16`), sourced from the BLS yearly schedule structure via the repo-local archive-backed fetch path.

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

- **`DBC` daily OHLCV** — approved V2 §2B substitute for Bloomberg Commodity Index but not yet pulled into the local SQLite import (planned row in §2.0A inventory)
- **V2 §1D true PIT vendor data** — V2 §1D currently ships on `fja05680/sp500` GitHub approximation with documented `survivorship_biased_constituent_universe` warning; CRSP / Compustat / FactSet / Norgate is the future vendor upgrade (an upgrade path, not a blocker — the feature ships today)
- **`cpi_nowcast` data flow** — the `cleveland_fed_nowcast` fetch path is built, but the operator must drop the Cleveland Fed CSV and verify its column mapping before `inflation_surprise_zscore` is non-NaN (operator action, not an engineering gap)
- **Put-call ratio / Investors Intelligence sentiment** — optional alternative `sentiment_score` sources for a future calibration revision; no fetcher built, but they block no label (`euphoria` fires via AAII)
- **Dedicated shadow runner** with SQLite ledger and archived daily input snapshots — operational gap, not a data-source gap

### Recently closed (no longer degraded)

- **Weekly `aggregate_forward_eps_revision_direction_4w`** — closed by the `aggregate_eps` weekly-snapshot accumulator + end-to-end engine wiring (Ambiguity Log #48). The weekly series is built by accumulating one current-snapshot row per weekly fetch; `earnings_expansion` / `earnings_contraction` fire once the >4-week cold-start fills. No paid feed.
- **V2 §2B `inflation_surprise_zscore`** — closed by ADR 0006: the free Cleveland Fed inflation nowcast substitutes for the paid analyst `consensus_estimate`. The `regime_data_fetch.cleveland_fed_nowcast` fetch path is built (manual-drop; operator-verified CSV schema).
- **V2 §1A `sentiment_score`** — closed: AAII fetcher ships, `sentiment_score = bull_bear_spread_8w_ma`, `euphoria` fires (Ambiguity Log #32).
- **V2 §1C implied vol** — closed: `implied_vol_30d = VIXCLS / 100` from FRED; `vol_crush` + `iv_rv_spread` ship (Ambiguity Log #19 / #20; ADR 0005).
- **V2 §2C HY/IG OAS** — closed: real ICE BofA OAS sourced free from FRED (`BAMLH0A0HYM2` / `BAMLC0A4CBBB`); the TLT-vs-HYG/LQD proxy path was deleted (Ambiguity Log #49).

## 5. Explicit Hard Failures

The fetch layer should fail loudly, not substitute silently, for these unsupported inputs:

- Bloomberg / Refinitiv consensus-survey feeds (unless V2 spec explicitly adopts the paid source)
- Licensed ICE DXY (the spec remains on `broad_usd_index` from FRED `DTWEXBGS`)
- I/B/E/S per-stock analyst revision feeds — V2 §2B uses the *aggregate* S&P Global forward-EPS series, not per-stock revisions.
- `IEF`, `BIL` — these ETFs are **not** in the V2 cross-asset fragility universe by design. V2 §2C consumes the real ICE BofA OAS series from FRED directly; **do not extend the universe** with IEF/BIL to compute an alternative spread proxy — no proxy path exists and a spec amendment would be required to add one.

Note: ICE BofA OAS, options-implied vol, and AAII sentiment were previously listed here as hard failures ("currently absent"). They are now **sourced free** — ICE BofA OAS via FRED `BAMLH0A0HYM2` / `BAMLC0A4CBBB`, implied vol via FRED `VIXCLS`, AAII via the `aaii_sentiment` fetcher — and are no longer hard-fail inputs.

Documented substitute policies (each pinned in the V2 Implementation Ambiguity Log):

- **CPI surprise**: substituted by the free Cleveland Fed inflation nowcast for the analyst `consensus_estimate` (ADR 0006 / Ambiguity Log #48); the feature carries a model-relative bias-warning row. Fetch path: `regime_data_fetch.cleveland_fed_nowcast`.
- **`broad_usd_index`**: approved field name for the free FRED route (FRED `DTWEXBGS`); do not back-door ICE DXY semantics into it.
- **Bloomberg Commodity Index**: substituted by `DBC` ETF per V2 §2B / Ambiguity Log #48; bias-warning row required in feature-store output.
- **HY/IG OAS**: sourced directly as real ICE BofA OAS from FRED (`BAMLH0A0HYM2` / `BAMLC0A4CBBB`) — no longer a substitute (Ambiguity Log #49 closed; the TLT-vs-HYG/LQD proxy path was deleted).
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
- If V2 §2C says **HY/IG credit spread**, read the real ICE BofA OAS series from FRED — `BAMLH0A0HYM2` (HY) and `BAMLC0A4CBBB` (BBB), both in `V2_FRED_SERIES` (Ambiguity Log #49). Do not add IEF/BIL or build an ETF-pair proxy; the TLT-vs-HYG/LQD proxy path was deleted.
- If V2 §2D says **`budget_week` / `election_window` / `geopolitical_event` / `global_rate_decision`**, expect operator-curated YAML rows following the §2D schema (Ambiguity Log #50). Do not silently auto-derive these from external news APIs or LLM extraction.
- If V2 §1A says **`euphoria`**, read `sentiment_score = bull_bear_spread_8w_ma` from the AAII fetcher (`regime_data_fetch.aaii_sentiment`); the label fires (Ambiguity Log #32). Do not substitute a different sentiment proxy; put-call / Investors Intelligence are optional future calibration sources only.
- If V2 §1C says **`vol_crush`** or **IV/RV spread**, read `implied_vol_30d = VIXCLS / 100` from FRED (Ambiguity Log #19 / #20; ADR 0005). Do not synthesise IV from realized-vol — VIX is the canonical implied-vol series.
- If V2 §2B says **`inflation_surprise_zscore`**, read `cpi_nowcast` from the Cleveland Fed inflation nowcast (`regime_data_fetch.cleveland_fed_nowcast`) as the `consensus_estimate` substitute (ADR 0006). Do not source a paid analyst-survey feed.
- If V2 §1D says **`pct_above_50dma`** or other PIT-constituent breadth features, use the `fja05680/sp500` PIT intervals (current default) combined with the 762-stock daily OHLCV (already in SQLite). Emit a `survivorship_biased_constituent_universe` warning row and keep the TODO note in `pit_constituents.py` pointing to the future vendor PIT upgrade.
