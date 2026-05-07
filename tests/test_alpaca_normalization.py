from __future__ import annotations

import datetime as dt

import pandas as pd

from regime_data_fetch.alpaca_daily import verify_min_start_date


def test_verify_min_start_date_ok() -> None:
    df = pd.DataFrame(
        [
            {"date": dt.date(2015, 1, 2), "symbol": "SPY"},
            {"date": dt.date(2015, 1, 5), "symbol": "SPY"},
            {"date": dt.date(2016, 1, 4), "symbol": "RSP"},
        ]
    )
    min_date, ok = verify_min_start_date(df, symbol="SPY", required_start=dt.date(2015, 1, 1))
    assert min_date == dt.date(2015, 1, 2)
    assert ok is True


def test_verify_min_start_date_missing() -> None:
    df = pd.DataFrame([{"date": dt.date(2016, 1, 4), "symbol": "RSP"}])
    min_date, ok = verify_min_start_date(df, symbol="SPY", required_start=dt.date(2015, 1, 1))
    assert min_date is None
    assert ok is False

