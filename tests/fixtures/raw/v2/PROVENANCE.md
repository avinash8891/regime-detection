# V2 Daily OHLCV Fixture Provenance

Source: repo-local raw parquet files under `data/raw/daily_ohlcv/symbol=*/aec4c8f262e34dfc8f5e81ae57a54e66-0.parquet`.

Derived file: `tests/fixtures/raw/v2/daily_ohlcv.csv`.

Rows: 44,424. Symbols: 24. Date span: 2019-01-02 through 2026-05-13.

Included symbols: `SPY`, `RSP`, `VIXY`, all V2 network-fragility sector ETFs (`XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, `XLY`), and all V2 network-fragility cross-asset ETFs (`QQQ`, `IWM`, `EFA`, `EEM`, `TLT`, `GLD`, `HYG`, `LQD`, `USO`, `UUP`).

Columns retained: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
