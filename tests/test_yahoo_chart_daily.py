from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
import urllib.parse

import pytest

from regime_data_fetch.yahoo_chart_daily import DAILY_OHLCV_COLUMNS
from regime_data_fetch.yahoo_chart_daily import fetch_daily_bars_yahoo_chart


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_fetch_daily_bars_yahoo_chart_uses_chart_endpoint_and_raw_close_schema(
    tmp_path: Path,
) -> None:
    payload = Path(
        "tests/fixtures/yahoo_chart_spy_2005_03_07_2005_03_08.json"
    ).read_bytes()
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout: float):
        del timeout
        requested_urls.append(request.full_url)
        return _FakeResponse(payload)

    result = fetch_daily_bars_yahoo_chart(
        symbols=["SPY"],
        start_date=dt.date(2005, 3, 7),
        end_date=dt.date(2005, 3, 8),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == []
    assert len(requested_urls) == 1
    assert requested_urls[0].startswith(
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY?"
    )
    assert "interval=1d" in requested_urls[0]
    assert "events=history" in requested_urls[0]
    assert result.df.to_dict(orient="records") == [
        {
            "date": dt.date(2005, 3, 7),
            "symbol": "SPY",
            "open": 121.1299972534,
            "high": 122.1600036621,
            "low": 120.9800033569,
            "close": 121.7200012207,
            "volume": 55748000,
            "adjusted_close": 121.7200012207,
        },
        {
            "date": dt.date(2005, 3, 8),
            "symbol": "SPY",
            "open": 121.0899963379,
            "high": 122.1299972534,
            "low": 120.8799972534,
            "close": 121.0800018311,
            "volume": 45771000,
            "adjusted_close": 121.0800018311,
        },
    ]


def test_fetch_daily_bars_yahoo_chart_sends_browser_headers_and_timeout() -> None:
    captured = {}
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1_746_437_400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [560.0],
                                "high": [562.5],
                                "low": [559.5],
                                "close": [561.25],
                                "volume": [62_000_000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = request.headers["User-agent"]
        captured["accept"] = request.headers["Accept"]
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["BRK.B"],
        start_date=dt.date(2025, 5, 5),
        end_date=dt.date(2025, 5, 5),
        timeout_sec=17.5,
        urlopen=fake_urlopen,
    )

    parsed = urllib.parse.urlparse(captured["url"])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/v8/finance/chart/BRK.B"
    assert query["interval"] == ["1d"]
    assert query["events"] == ["history"]
    assert query["includeAdjustedClose"] == ["true"]
    assert captured["timeout"] == 17.5
    assert "Chrome/126.0.0.0" in captured["user_agent"]
    assert captured["accept"] == "application/json"
    assert result.df.to_dict(orient="records") == [
        {
            "date": dt.date(2025, 5, 5),
            "symbol": "BRK.B",
            "open": 560.0,
            "high": 562.5,
            "low": 559.5,
            "close": 561.25,
            "volume": 62_000_000,
            "adjusted_close": 561.25,
        }
    ]


def test_fetch_daily_bars_yahoo_chart_rejects_invalid_utf8_json() -> None:
    def fake_urlopen(_request, timeout: float):
        del timeout
        return _FakeResponse(b'{"chart": "\xff"}')

    with pytest.raises(UnicodeDecodeError):
        fetch_daily_bars_yahoo_chart(
            symbols=["SPY"],
            start_date=dt.date(2025, 5, 5),
            end_date=dt.date(2025, 5, 5),
            urlopen=fake_urlopen,
        )


def test_fetch_daily_bars_yahoo_chart_marks_empty_chart_result_as_missing_symbol() -> (
    None
):
    payload = {"chart": {"result": [], "error": None}}

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["NO_DATA"],
        start_date=dt.date(2026, 1, 2),
        end_date=dt.date(2026, 1, 5),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == ["NO_DATA"]
    assert list(result.df.columns) == DAILY_OHLCV_COLUMNS
    assert result.df.empty


def test_fetch_daily_bars_yahoo_chart_marks_missing_quotes_as_missing_symbol() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1_767_153_600],
                    "indicators": {"quote": []},
                }
            ],
            "error": None,
        }
    }

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["NO_QUOTES"],
        start_date=dt.date(2026, 1, 1),
        end_date=dt.date(2026, 1, 2),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == ["NO_QUOTES"]
    assert result.df.empty


def test_fetch_daily_bars_yahoo_chart_marks_all_invalid_rows_as_missing_symbol() -> (
    None
):
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [
                        int(
                            dt.datetime(
                                2026, 1, 2, 14, tzinfo=dt.timezone.utc
                            ).timestamp()
                        )
                    ],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [101.0],
                                "low": [99.0],
                                "close": [100.5],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["NO_VOLUME"],
        start_date=dt.date(2026, 1, 2),
        end_date=dt.date(2026, 1, 2),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == ["NO_VOLUME"]
    assert result.df.empty


def test_fetch_daily_bars_yahoo_chart_uses_utc_when_exchange_timezone_is_missing_or_unknown() -> (
    None
):
    timestamps = [
        int(dt.datetime(2026, 1, 2, 0, 30, tzinfo=dt.timezone.utc).timestamp())
    ]
    base_result = {
        "timestamp": timestamps,
        "indicators": {
            "quote": [
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000],
                }
            ]
        },
    }
    payloads = [
        {"chart": {"result": [{**base_result, "meta": {}}], "error": None}},
        {
            "chart": {
                "result": [
                    {
                        **base_result,
                        "meta": {"exchangeTimezoneName": "Not/A_Timezone"},
                    }
                ],
                "error": None,
            }
        },
    ]
    payload_iter = iter(payloads)

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(next(payload_iter)).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["MISSING_TZ", "BAD_TZ"],
        start_date=dt.date(2026, 1, 2),
        end_date=dt.date(2026, 1, 2),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == []
    assert result.df[["symbol", "date"]].to_dict(orient="records") == [
        {"symbol": "BAD_TZ", "date": dt.date(2026, 1, 2)},
        {"symbol": "MISSING_TZ", "date": dt.date(2026, 1, 2)},
    ]


def test_fetch_daily_bars_yahoo_chart_logs_symbol_progress_when_verbose(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {"chart": {"result": [], "error": None}}

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    with caplog.at_level(logging.INFO, logger="regime_data_fetch.yahoo_chart_daily"):
        fetch_daily_bars_yahoo_chart(
            symbols=["SPY"],
            start_date=dt.date(2026, 1, 2),
            end_date=dt.date(2026, 1, 5),
            verbose=True,
            urlopen=fake_urlopen,
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.data_source == "yahoo_chart_daily_ohlcv"
    assert record.symbol == "SPY"
    assert record.symbol_index == 1
    assert record.symbol_count == 1


def test_fetch_daily_bars_yahoo_chart_raises_when_chart_object_is_missing() -> None:
    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(b'{"finance": {"result": []}}')

    with pytest.raises(RuntimeError, match="missing chart object for SPY"):
        fetch_daily_bars_yahoo_chart(
            symbols=["SPY"],
            start_date=dt.date(2026, 1, 2),
            end_date=dt.date(2026, 1, 5),
            urlopen=fake_urlopen,
        )


def test_fetch_daily_bars_yahoo_chart_raises_on_yahoo_error_payload() -> None:
    payload = {
        "chart": {
            "result": None,
            "error": {"code": "Not Found", "description": "No data found"},
        }
    }

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    with pytest.raises(RuntimeError, match="Yahoo chart error for MISSING"):
        fetch_daily_bars_yahoo_chart(
            symbols=["MISSING"],
            start_date=dt.date(2026, 1, 2),
            end_date=dt.date(2026, 1, 5),
            urlopen=fake_urlopen,
        )


def test_fetch_daily_bars_yahoo_chart_skips_null_rows_and_dates_outside_request() -> (
    None
):
    timestamps = [
        int(dt.datetime(2026, 1, 1, 14, tzinfo=dt.timezone.utc).timestamp()),
        int(dt.datetime(2026, 1, 2, 14, tzinfo=dt.timezone.utc).timestamp()),
        int(dt.datetime(2026, 1, 3, 14, tzinfo=dt.timezone.utc).timestamp()),
    ]
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None, 102.0],
                                "high": [101.0, 102.0, 103.0],
                                "low": [99.0, 100.0, 101.0],
                                "close": [100.5, 101.5, 102.5],
                                "volume": [1000, 2000, 3000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    def fake_urlopen(_request, timeout: float):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    result = fetch_daily_bars_yahoo_chart(
        symbols=["SPY"],
        start_date=dt.date(2026, 1, 1),
        end_date=dt.date(2026, 1, 2),
        urlopen=fake_urlopen,
    )

    assert result.missing_symbols == []
    assert result.df.to_dict(orient="records") == [
        {
            "date": dt.date(2026, 1, 1),
            "symbol": "SPY",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "adjusted_close": 100.5,
        }
    ]


def test_fetch_daily_bars_yahoo_chart_rejects_unsupported_adjustment_and_bad_date_range() -> (
    None
):
    with pytest.raises(ValueError, match="supports only adjustment='raw'"):
        fetch_daily_bars_yahoo_chart(
            symbols=["SPY"],
            start_date=dt.date(2026, 1, 2),
            end_date=dt.date(2026, 1, 5),
            adjustment="split",
        )

    with pytest.raises(ValueError, match="end_date must be >= start_date"):
        fetch_daily_bars_yahoo_chart(
            symbols=["SPY"],
            start_date=dt.date(2026, 1, 5),
            end_date=dt.date(2026, 1, 2),
        )
