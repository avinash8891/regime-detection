from __future__ import annotations

import datetime as dt
import logging
import sys
import types

import pandas as pd
import pytest

from regime_data_fetch.alpaca_daily import fetch_daily_bars_alpaca, verify_min_start_date


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


def test_alpaca_client_missing_env_raises_project_scoped_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from regime_data_fetch import alpaca_daily

    monkeypatch.setenv("ALPACA_API_KEY_ID", "configured-key-value")
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        alpaca_daily._get_alpaca_client()

    message = str(exc_info.value)
    assert "regime_data_fetch Alpaca credentials are not configured" in message
    assert "ALPACA_API_SECRET_KEY" in message
    assert "ALPACA_API_KEY_ID" not in message
    assert "configured-key-value" not in message


def test_fetch_daily_bars_verbose_logs_structured_progress(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeAdjustment:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeDataFeed:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeTimeFrame:
        Day = "day"

    class FakeStockBarsRequest:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeBar:
        timestamp = dt.datetime(2026, 5, 15, tzinfo=dt.timezone.utc)
        open = 100.0
        high = 101.0
        low = 99.0
        close = 100.5
        volume = 1000

    class FakeResponse:
        data = {
            "BRK.B": [FakeBar()],
            "SPY": [FakeBar()],
        }

    class FakeClient:
        def get_stock_bars(self, request: FakeStockBarsRequest) -> FakeResponse:
            assert request.symbol_or_symbols == ["BRK.B", "SPY"]
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "alpaca.data.requests",
        types.SimpleNamespace(StockBarsRequest=FakeStockBarsRequest),
    )
    monkeypatch.setitem(
        sys.modules,
        "alpaca.data.timeframe",
        types.SimpleNamespace(TimeFrame=FakeTimeFrame),
    )
    monkeypatch.setitem(
        sys.modules,
        "alpaca.data.enums",
        types.SimpleNamespace(Adjustment=FakeAdjustment, DataFeed=FakeDataFeed),
    )
    monkeypatch.setattr("regime_data_fetch.alpaca_daily._get_alpaca_client", lambda: FakeClient())

    caplog.set_level(logging.INFO, logger="regime_data_fetch.alpaca_daily")
    result = fetch_daily_bars_alpaca(
        symbols=["BRK-B", "SPY"],
        start_date=dt.date(2026, 5, 15),
        end_date=dt.date(2026, 5, 15),
        batch_size=2,
        verbose=True,
    )

    assert result.missing_symbols == []
    assert result.df["symbol"].tolist() == ["BRK-B", "SPY"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "alpaca daily bars batch request",
        "alpaca daily bars batch complete",
    ]
    request_record, complete_record = caplog.records
    assert request_record.data_source == "alpaca_daily_ohlcv"
    assert request_record.batch_index == 1
    assert request_record.batch_count == 1
    assert request_record.requested_symbol_count == 2
    assert request_record.total_symbol_count == 2
    assert complete_record.returned_symbol_count == 2
    assert complete_record.cumulative_frame_count == 2
