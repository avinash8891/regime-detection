from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_data_fetch.universe import load_symbols_from_pit_constituents_parquet


def test_load_symbols_from_pit_constituents_parquet(tmp_path: Path) -> None:
    parquet = tmp_path / "sp500_ticker_intervals.parquet"
    pd.DataFrame(
        [
            {"ticker": "MSFT", "start_date": "2015-01-01", "end_date": None},
            {"ticker": "AAPL", "start_date": "2015-01-01", "end_date": "2026-12-31"},
            {"ticker": "AAPL", "start_date": "2010-01-01", "end_date": "2014-12-31"},
        ]
    ).to_parquet(parquet, index=False)

    assert load_symbols_from_pit_constituents_parquet(parquet) == ["AAPL", "MSFT"]
