# V2 Raw Fixture Provenance

## Daily OHLCV

Source: repo-local raw parquet files under `data/raw/daily_ohlcv/symbol=*/aec4c8f262e34dfc8f5e81ae57a54e66-0.parquet`.

Derived file: `tests/fixtures/raw/v2/daily_ohlcv.csv`.

Date span: 2009-01-02 through 2026-05-13.

### Historical extension (2009-01-02 -> 2018-12-31)

The pre-2019 span was backfilled so the four pre-2019 §9.4 golden dates
(Flash Crash 2010-05-06, US downgrade 2011-08-08, China devaluation
2015-08-24, Q4-2018 stress 2018-10-10) classify live. Source: Yahoo Finance
`/v8/finance/chart` daily bars, fetched per-symbol in yearly chunks on a
local residential host (Alpaca has no pre-2016 data; Yahoo's chart API
rate-limits datacenter IPs). The real `^VIX` index is mapped to `VIX`
(2010-05-06 close 32.80 — the genuine index, not the `VIXY` ETF proxy).
Partial coverage by inception: `VIXY` 2011-01-04, `XLRE` 2015-10-08, `XLC`
2018-06-19. The 2019-01-02+ rows are byte-identical to the prior fixture.

Included symbols: `SPY`, `RSP`, `VIXY`, all V2 network-fragility sector ETFs (`XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, `XLY`), all V2 network-fragility cross-asset ETFs (`QQQ`, `IWM`, `EFA`, `EEM`, `TLT`, `IEF`, `GLD`, `HYG`, `LQD`, `USO`, `DBC`, `UUP`), and `KRE` for the V2 credit/funding axis.

Columns retained: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.

## FRED Macro

Source: repo-local raw parquet file `data/raw/macro/fred_macro_series.parquet`.

Derived file: `tests/fixtures/raw/v2/fred_macro_series.csv`.

Rows: 8,695. Series: `SOFR` (`sofr`), `IORB` (`iorb`), `NFCI` (`nfci`), `DTWEXBGS` (`broad_usd_index`), `BAMLH0A0HYM2` (`hy_oas`), `BAMLC0A4CBBB` (`ig_bbb_oas`). Date span: 2016-01-01 through 2026-05-14.

Columns retained: `date`, `series_id`, `logical_name`, `value`.
