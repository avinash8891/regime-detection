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
- `earnings_revision_breadth` is replaced by `aggregate_forward_eps_revision_direction`, sourced from S&P Global aggregate forward EPS data.

### 2.0A Data Inventory

Status meanings used below:

- `done-live-verified`: implemented and verified against the real source in this repo workflow
- `implemented-not-live-verified`: implemented in code but not live-verified in the current session
- `template-only`: only a placeholder/template exists
- `planned`: source/path identified, loader not implemented yet
- `hard-fail`: intentionally unsupported unless the spec/source decision changes

| Data | Source | Output / Path | Status | Notes |
|---|---|---|---|---|
| US universe cache JSON | `market-data-hub` seed list + yfinance market-cap refresh | `data/raw/universe/us_universe_cache.json` | implemented-not-live-verified | built by `build_or_load_us_universe_10b_cache()` |
| 10B+ US stock universe symbol list | universe cache JSON above | loaded in-memory from `data/raw/universe/us_universe_cache.json` or `--universe-json` | implemented-not-live-verified | used for V1/all stock-universe fetches |
| 762-stock daily OHLCV backfill | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | blocked on missing local Alpaca creds during this session |
| `SPY` daily OHLCV | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V1 market anchor |
| `RSP` daily OHLCV | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V1 breadth proxy |
| `VIX` daily proxy bars | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | only if Alpaca account returns true `VIX` |
| `VIXY` daily proxy bars | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | documented operational proxy when true `VIX` is unavailable |
| `KRE` daily OHLCV | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 bank-stress proxy |
| Sector ETF daily OHLCV: `XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY` | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 fragility / sector breadth universe |
| Cross-asset ETF daily OHLCV: `QQQ,IWM,EFA,EEM,TLT,HYG,LQD,GLD,USO,UUP` | Alpaca REST | `data/raw/daily_ohlcv/` | implemented-not-live-verified | V2 cross-asset fragility universe |
| Event calendar template (V1 + V2 sample rows) | repo-local generated YAML | `data/raw/event_calendar/events.template.yaml` | template-only | not real historical event data |
| `2y_yield` / `DGS2` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | live fetch succeeded |
| `10y_yield` / `DGS10` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | live fetch succeeded |
| `broad_usd_index` / `DTWEXBGS` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | explicit approved replacement for DXY |
| `sofr` / `SOFR` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | live fetch succeeded |
| `nfci` / `NFCI` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | weekly series; live fetch succeeded |
| `cpi_all_items` / `CPIAUCSL` | FRED API | `data/raw/macro/fred_macro_series.parquet` | done-live-verified | monthly series; live fetch succeeded |
| `cpi_all_items_vintages` / `CPIAUCSL` realtime observations | FRED API with realtime params | `data/raw/macro_vintages/cpi_all_items_vintages.parquet` | done-live-verified | live fetch succeeded |
| `iorb` | Federal Reserve Board H.15 / IORB release | dedicated raw dataset not yet created | planned | source chosen, loader not implemented |
| PMI manufacturing headline values | ISM pages / PDFs | dedicated PMI dataset not yet created | planned | keep PMI as PMI; no CFNAI substitution |
| PMI services headline values | ISM pages / PDFs | dedicated PMI dataset not yet created | planned | release timestamp must follow ISM calendar |
| PMI release timestamps | ISM calendar convention | derived in code, dedicated dataset not yet created | planned | manufacturing = first business day 10:00 ET; services = third business day 10:00 ET |
| PIT S&P 500 constituents | `fja05680/sp500` initial source | dedicated PIT dataset not yet created | planned | output must carry bias warning |
| FOMC minutes raw text | Federal Reserve site | dedicated raw text dataset not yet created | planned | release timestamps required |
| Powell speeches raw text | Federal Reserve site | dedicated raw text dataset not yet created | planned | release timestamps required |
| `aggregate_forward_eps_revision_direction` | S&P Global aggregate forward EPS sheet | dedicated weekly dataset not yet created | planned | renamed replacement for earnings revision breadth |
| Bloomberg / Refinitiv consensus surveys | paid vendor feeds | no output path | hard-fail | unsupported unless spec explicitly adopts a paid source |
| I/B/E/S per-stock analyst revisions | paid vendor feeds | no output path | hard-fail | unsupported in current V2 plan |
| ICE DXY history | licensed ICE feed | no output path while spec stays on `broad_usd_index` | hard-fail | only relevant if spec changes back from `broad_usd_index` |

### 2.1 V1 Build Scope

| Dataset | Symbols / Series | Source | Output |
|---|---|---|---|
| Daily OHLCV (market anchor) | `SPY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily OHLCV (breadth proxy) | `RSP` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily volatility proxy | `VIX` when Alpaca supports it, otherwise `VIXY` | Alpaca REST | `data/raw/daily_ohlcv/` |
| Daily OHLCV (stock universe) | 10B+ US stocks | Alpaca REST | `data/raw/daily_ohlcv/` |
| Event calendar placeholder | V1 event types | repo-local YAML template | `data/raw/event_calendar/events.template.yaml` |

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
| `iorb` | Fed Board H.15 / IORB release | Federal Reserve Board | dedicated IORB raw dataset when loader lands |

#### Higher-Maintenance Inputs

These are part of the V2 data plan but are not all implemented yet:

| Dataset | Intended source | Notes |
|---|---|---|
| PMI manufacturing/services | ISM pages / PDFs | use real PMI, not CFNAI substitution; release timestamps locked to ISM calendar; scraper failures must be loud, not silent |
| PIT S&P 500 constituents | `fja05680/sp500` initial source | bias must be documented explicitly in output/report |
| FOMC minutes / Powell speeches | Federal Reserve site | release timestamps required |
| Aggregate forward EPS revision direction | S&P Global aggregate forward EPS sheet | renamed replacement for earnings revision breadth |
| Event calendar extension | repo-local manual YAML | V2 event types and windows |

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

- real historical event-calendar ingestion
- real PMI backfill loader
- real PIT constituent loader
- real Fed text loader
- real aggregate forward EPS loader
- real IORB loader
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
