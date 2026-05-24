from __future__ import annotations

import datetime as dt
from pathlib import Path

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
