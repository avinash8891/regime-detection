# Market Data Fetch Plan

This document now separates two workflows that were previously mixed together:

1. **Build / backfill fetches** used to develop and test V1/V2 locally
2. **Shadow-mode daily acquisition** used after V1 is frozen and running operationally

Those are not interchangeable. Historical backfills test engine logic. Forward shadow tests operational stability.

## 1. Validation Sequence

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
- PMI retrieval uses alternate redistribution sources, not direct ISM scraping in this repo path:
  - primary: DBnomics
  - backup: TradingEconomics
  - stale primary data must fail loudly and fall through explicitly
- `earnings_revision_breadth` is replaced by `aggregate_forward_eps_revision_direction`, sourced from S&P Global aggregate forward EPS data.

### 2.0A Data Inventory

Status meanings used below:

- `done-live-verified`: implemented and verified against the real source in this repo workflow
- `implemented-not-live-verified`: implemented in code but not live-verified in the current session
- `template-only`: only a placeholder/template exists
- `planned`: source/path identified, loader not implemented yet
- `hard-fail`: intentionally unsupported unless the spec/source decision changes

| Data | Source | Cadence | Output / Path | Status | Comment |
|---|---|---|---|---|---|
| US universe cache JSON | `market-data-hub` seed list + yfinance market-cap refresh | ad hoc refresh | `data/raw/universe/us_universe_cache.json` | implemented-not-live-verified | built by `build_or_load_us_universe_10b_cache()`; refresh when the stock universe is rebuilt |
| 10B+ US stock universe symbol list | universe cache JSON above | ad hoc refresh | loaded in-memory from `data/raw/universe/us_universe_cache.json` or `--universe-json` | implemented-not-live-verified | available immediately after universe-cache build; used for V1/all stock-universe fetches |
| 762-stock daily OHLCV backfill | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | available after market close for each trading session; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| `SPY` daily OHLCV | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V1 market anchor; fetch after NYSE close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| `RSP` daily OHLCV | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V1 breadth proxy; fetch after NYSE close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| `VIX` daily proxy bars | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | only if Alpaca account returns true `VIX`; fetch after market close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| `VIXY` daily proxy bars | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | documented operational proxy when true `VIX` is unavailable; fetch after market close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| `KRE` daily OHLCV | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 bank-stress proxy; fetch after market close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| Sector ETF daily OHLCV: `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 fragility / sector breadth universe; fetch after market close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| Cross-asset ETF daily OHLCV: `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | Alpaca REST | daily | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 cross-asset fragility universe; fetch after market close; when `--acquisition-db` is supplied, the normalized Alpaca fetch-boundary payload is recorded in SQLite before parquet/report output |
| Scheduled event rows: `FOMC` | generated repo-local YAML from Federal Reserve FOMC calendar pages | about 8 times per year | `configs/events/us_events.yaml` | done-live-verified | generated by `--fetch events`; parse current `fomccalendars.htm`, add older meetings from `fomc_historical_year.htm` and yearly `fomchistoricalYYYY.htm` pages, dedupe by `meeting_end_date`, and store minutes release timestamps at `14:00 ET`; current live-verified coverage is `2007-10-31` through `2026-03-18` |
| Scheduled event rows: `CPI` | generated repo-local YAML from BLS yearly release-schedule pages | monthly | `configs/events/us_events.yaml` | implemented-not-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `Consumer Price Index` / `Consumer Price Indexes` rows, and store release timestamps at `08:30 ET`; parser/tests and event-label wiring are green, but full historical live fetch from this environment is still blocked by BLS `HTTP 403` |
| Scheduled event rows: `NFP` | generated repo-local YAML from BLS yearly release-schedule pages | monthly | `configs/events/us_events.yaml` | implemented-not-live-verified | generated by `--fetch events`; parse BLS yearly schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`, keep only `The Employment Situation` / `Employment Situation` rows, and store release timestamps at `08:30 ET`; parser/tests and event-label wiring are green, but full historical live fetch from this environment is still blocked by BLS `HTTP 403` |
| Rule-derived event window: `expiry_week` | computed from deterministic rules in config/runtime | monthly | no stored raw file | done-live-verified | compute the third Friday of each month, roll back to the previous NYSE trading day if that Friday is closed, then expand the configured NYSE trading-day window around the anchor; the runtime rule is now wired through `resolve_event_label()` and live-verified with NYSE holiday-sensitive months like `2019-04`, `2022-04`, and `2026-06` |
| Rule-derived event window: `earnings_season` | computed from deterministic rules in config/runtime | quarterly window | no stored raw file | done-live-verified | compute quarter windows starting on the second Monday of `Jan/Apr/Jul/Oct` and ending `+35` calendar days later; the runtime rule is now wired through `resolve_event_label()` and live-verified across `2015-01-01` through `2026-05-07` |
| Ad-hoc V2 event rows (`election_window`, `geopolitical_event`, `budget_week`) | manual curated YAML if ever adopted | irregular | no output path in V1 | hard-fail | explicitly skipped in V1; no fetcher or generated dataset should pretend these exist |
| `2y_yield` / `DGS2` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `10y_yield` / `DGS10` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Treasury daily constant-maturity yield; typically published on business days after market hours; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `broad_usd_index` / `DTWEXBGS` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | explicit approved replacement for DXY; business-day macro release cadence; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `sofr` / `SOFR` | FRED API | daily | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | overnight rate; next-business-day publication pattern; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `nfci` / `NFCI` | FRED API | weekly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | Chicago Fed weekly update; do not assume fresh daily values; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items` / `CPIAUCSL` | FRED API | monthly | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | monthly CPI level; available after BLS CPI release each month; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| `cpi_all_items_vintages` / `CPIAUCSL` realtime observations | FRED API with realtime params | monthly vintages | `data/raw/macro_vintages/cpi_all_items_vintages.parquet` | done-live-verified | PIT/vintage view; new vintage appears on CPI release cycle; when `--acquisition-db` is supplied, the raw realtime-observations JSON response is recorded before parquet/report output |
| `iorb` / `IORB` | FRED API | business day | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | interest on reserve balances; use the published effective date from FRED; when `--acquisition-db` is supplied, the raw FRED JSON response is recorded before parquet/report output |
| PMI manufacturing headline values | DBnomics primary, TradingEconomics backup | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | available first business day of the following month at 10:00 ET; live run rejected stale DBnomics data and selected TradingEconomics for `2026-04` |
| PMI services headline values | DBnomics primary, TradingEconomics backup | monthly | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | available third business day of the following month at 10:00 ET; live run selected TradingEconomics `2026-04` after stale-primary rejection |
| PMI release timestamps | code-derived ISM release calendar convention | monthly metadata | `data/raw/pmi/us_ism_pmi.parquet` | done-live-verified | manufacturing = first business day 10:00 ET; services = third business day 10:00 ET |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | event-driven membership changes | `data/raw/pit_constituents/sp500_ticker_intervals.parquet` | done-live-verified | live fetch succeeded; rows carry `survivorship_biased_constituent_universe` warning and interval dates |
| FOMC minutes raw text | Federal Reserve official pages: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm` + `https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm` + yearly `fomchistoricalYYYY.htm` pages + per-meeting `fomcminutesYYYYMMDD.htm` pages | about 8 times per year | `data/raw/fomc_minutes/fomc_minutes.parquet` | done-live-verified | fetched by walking the current calendar page for 2021+ meetings, walking the official historical year index for pre-2021 pages, then fetching each meeting HTML page and extracting title, meeting date text, body text, source URL, and PDF URL; current verified coverage reaches `2011-01-26` through `2026-03-18`; release timestamps encoded at `14:00 ET` on the Fed released date; pre-2011 not implemented yet |
| Powell speeches raw text | Federal Reserve official pages: `https://www.federalreserve.gov/newsevents/speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | irregular / event-driven | `data/raw/powell_speeches/powell_speeches.parquet` | done-live-verified | live fetch succeeded; current verified coverage reaches `2013-02-22` through `2026-03-21`; yearly archive pages are filtered to Powell-only entries and each speech page is fetched for title, speaker, location, and body text; Fed pages expose a date but no reliable publication time, so `publication_timestamp_precision=date_only` and timestamps are normalized to midnight Eastern |
| aggregate forward EPS workbook snapshots | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx`, parsed from `ESTIMATES&PEs` | manual snapshot / irregular | `data/raw/aggregate_forward_eps/sp500_eps_snapshots.parquet` | done-live-verified | fetched by pointing `--fetch eps --eps-workbook /Users/avinashvankadaru/Desktop/sp-500-eps-est.xlsx` at the saved workbook; parser extracts the workbook `as_of` date, historical quarterly observation rows, and the current forward estimate row into parquet plus `aggregate_eps_fetch_report.json`; when `--acquisition-db` is supplied, the manual workbook file is recorded in the shared SQLite acquisition store before output materialization; live-verified workbook date is `2026-01-30`; the workbook itself says the public files were discontinued |
| `aggregate_forward_eps_revision_direction_4w` | derived from weekly aggregate forward EPS history | weekly | no output path yet | planned | not derivable from the current captured workbook alone because it exposes quarterly historical observations plus one current snapshot, not a weekly revision time series |
| Bloomberg / Refinitiv consensus surveys | paid vendor feeds | event-driven macro release cycle | no output path | hard-fail | unsupported unless spec explicitly adopts a paid source |
| I/B/E/S per-stock analyst revisions | paid vendor feeds | daily to weekly | no output path | hard-fail | unsupported in current V2 plan |
| ICE DXY history | licensed ICE feed | daily | no output path while spec stays on `broad_usd_index` | hard-fail | only relevant if spec changes back from `broad_usd_index` |

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
data/raw/universe/us_universe_cache.json
```

built from the `market-data-hub` seed list.

### 2.2 V2 Build Scope

#### Market / Cross-Asset

| Dataset | Symbols / Series | Source | Output |
|---|---|---|---|
| Shared anchors | `SPY`, `RSP` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Bank stress proxy | `KRE` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Sector fragility universe | `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Cross-asset fragility universe | `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | Alpaca REST | `data/raw/daily_ohlcv/` |
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
| PMI manufacturing/services | DBnomics primary, TradingEconomics backup | use real PMI, not CFNAI substitution; reject stale primary data loudly; release timestamps locked to ISM calendar convention |
| PIT S&P 500 constituents | `fja05680/sp500` `sp500_ticker_start_end.csv` | bias warning must be carried in output/report; current ingest stores ticker start/end intervals |
| FOMC minutes | Federal Reserve `fomccalendars.htm` + `fomc_historical_year.htm` + `fomchistoricalYYYY.htm` + minutes HTML pages | release timestamps required; current fetcher gets 2021+ meetings from the live calendar page, gets pre-2021 year pages from the official historical index, dedupes by `meeting_end_date`, and stores title, meeting date text, release timestamp, body text, source URL, and PDF URL; current verified lower bound is `2011-01-26` |
| Powell speeches | Federal Reserve `speeches.htm?speaker=Jerome+H.+Powell` + yearly `YYYY-speeches.htm` archives + per-speech `powellYYYYMMDDx.htm` pages | current fetcher walks the Fed speeches index to yearly archives, filters archive rows to Powell-only entries, then fetches each Powell speech page and stores speech date, normalized publication timestamp, timestamp precision, title, speaker, location, body text, and source URL |
| Aggregate forward EPS revision direction | manually downloaded S&P Global workbook `sp-500-eps-est.xlsx` parsed from `ESTIMATES&PEs` | current implemented loader is a real snapshot ingest: it stores workbook-date observations and the current forward estimate row from the saved local workbook. It does not yet produce `aggregate_forward_eps_revision_direction_4w` because the captured public workbook does not expose weekly revision history |
| Event calendar extension | generated `FOMC` / `CPI` / `NFP` YAML plus runtime `expiry_week` / `earnings_season` rules | V1 should auto-generate scheduled events and compute rule-derived windows at runtime; ad-hoc V2 events stay out of scope |

Implemented scheduled-event logic:

- `FOMC`: parse the current Fed meeting calendar page for recent years, then parse the Fed historical year index and yearly `fomchistoricalYYYY.htm` pages for older years. Emit one row per meeting using `meeting_end_date` as the event `date`. The associated minutes release date comes from the same Fed meeting listings, and the stored `release_timestamp_et` is encoded at `14:00 ET`.
- `CPI`: parse the official BLS yearly release schedules and keep only rows whose release title is `Consumer Price Index` or `Consumer Price Indexes`. The canonical source pages are the yearly BLS schedule pages under `/schedule/YYYY/` and `/schedule/YYYY/home.htm`. Emit one row per monthly CPI release and store `release_timestamp_et` at the scheduled `08:30 ET` release time.
- `NFP`: parse the same BLS yearly release schedules and keep only rows whose release title is `The Employment Situation` or `Employment Situation`. Emit one row per monthly NFP release and store `release_timestamp_et` at `08:30 ET`.
- Scheduled YAML is consumed through `load_scheduled_events_yaml()`, and `resolve_event_label()` expands scheduled NYSE trading-day windows with precedence `fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown`.
- `expiry_week`: compute the monthly options-expiry anchor as the third Friday, roll back to the previous NYSE trading day if that Friday is closed, then expand the runtime window `[-2, 0]` trading days around that anchor.
- `earnings_season`: compute quarterly windows anchored on the second Monday of `Jan/Apr/Jul/Oct`, ending `+35` calendar days later, and apply them at runtime rather than storing historical rows.
- Output rows are sorted by `release_timestamp_et` and written to generated YAML; the file is generated, not hand-edited.
- Current truth: the parser and generator for the BLS side are implemented and tested, but live historical BLS fetches from this environment are currently blocked by `HTTP 403`, so CPI/NFP source semantics are correct in code even though the full live backfill is not yet verified here.

### 2.3 Development Date Ranges

- Default V1-friendly range:
  - `2015-01-01` through today
- Recommended V2 backfill range:
  - `2004-01-01` through today

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

Still unresolved or not fully implemented:

- full historical live verification for BLS-backed `CPI` / `NFP` generation, or wiring the repo fetch path to a local BLS HTML archive when direct BLS access is blocked
- weekly `aggregate_forward_eps_revision_direction_4w` derivation from a true weekly revision history
- dedicated shadow runner with SQLite ledger and archived daily input snapshots

## 5. Explicit Hard Failures

The fetch layer should fail loudly, not substitute silently, for these unsupported inputs:

- Bloomberg / Refinitiv consensus-survey feeds
- I/B/E/S per-stock analyst revision feeds
- licensed ICE DXY, if the spec remains on `broad_usd_index`

Documented substitute policies:

- CPI surprise work may use a documented nowcast/expectation substitute only when the spec names that methodology explicitly.
- `broad_usd_index` is the approved field name for the free FRED route; do not back-door ICE DXY semantics into it.

## 6. Source Rules

- Do not silently substitute a different economic concept because it is cheaper.
- If the spec says PMI, fetch PMI.
- If the spec says `broad_usd_index`, fetch `DTWEXBGS`.
- If the spec says `aggregate_forward_eps_revision_direction`, fetch the aggregate S&P Global series, not a per-stock breadth proxy.
- For development/backfill, prefer Alpaca `VIX`; when unavailable, use `VIXY` as the documented operational proxy.
- For forward shadow, archive exact inputs used each day before classification.
- For V1 event calendar work, generate scheduled `FOMC` rows from official Fed meeting pages, generate scheduled `CPI` / `NFP` rows from official BLS release schedules, compute `expiry_week` / `earnings_season` from deterministic rules, and skip ad-hoc events.
