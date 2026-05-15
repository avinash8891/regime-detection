# V2 Raw Fixture Provenance

## Daily OHLCV

Source: repo-local raw parquet files under `data/raw/daily_ohlcv/symbol=*/aec4c8f262e34dfc8f5e81ae57a54e66-0.parquet`.

Derived file: `tests/fixtures/raw/v2/daily_ohlcv.csv`.

Rows: 46,275. Symbols: 25. Date span: 2019-01-02 through 2026-05-13.

Included symbols: `SPY`, `RSP`, `VIXY`, all V2 network-fragility sector ETFs (`XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, `XLY`), all V2 network-fragility cross-asset ETFs (`QQQ`, `IWM`, `EFA`, `EEM`, `TLT`, `GLD`, `HYG`, `LQD`, `USO`, `UUP`), and `KRE` for the V2 credit/funding axis.

Columns retained: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.

## FRED Macro

Source: repo-local raw parquet file `data/raw/macro/fred_macro_series.parquet`.

Derived file: `tests/fixtures/raw/v2/fred_macro_series.csv`.

Rows: 8,695. Series: `SOFR` (`sofr`), `IORB` (`iorb`), `NFCI` (`nfci`), `DTWEXBGS` (`broad_usd_index`), `BAMLH0A0HYM2` (`hy_oas`), `BAMLC0A4CBBB` (`ig_bbb_oas`). Date span: 2016-01-01 through 2026-05-14.

Columns retained: `date`, `series_id`, `logical_name`, `value`.
