from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
import urllib.error
import urllib.parse

import pandas as pd
import pytest

from regime_data_fetch import fred


_FRED_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "raw" / "fred"


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_BytesResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=code,
        msg=f"HTTP {code}",
        hdrs={},
        fp=io.BytesIO(b"{}"),
    )


def test_parse_fred_series_json_preserves_realtime_dates_and_null_observations() -> None:
    payload = (_FRED_FIXTURE_DIR / "dgs10_observations.json").read_text()

    frame = fred.parse_fred_series_json(payload, series_id="DGS10")

    assert frame.drop(columns=["value"]).to_dict(orient="records") == [
        {
            "date": dt.date(2026, 5, 18),
            "series_id": "DGS10",
            "realtime_start": "2026-05-20",
            "realtime_end": "2026-05-20",
        },
        {
            "date": dt.date(2026, 5, 19),
            "series_id": "DGS10",
            "realtime_start": "2026-05-20",
            "realtime_end": "2026-05-20",
        },
        {
            "date": dt.date(2026, 5, 20),
            "series_id": "DGS10",
            "realtime_start": "2026-05-20",
            "realtime_end": "2026-05-20",
        },
    ]
    assert frame["value"].iloc[0] == 4.45
    assert pd.isna(frame["value"].iloc[1])
    assert frame["value"].iloc[2] == 4.49


def test_fetch_fred_series_builds_observation_url_with_realtime_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (_FRED_FIXTURE_DIR / "dgs10_observations.json").read_bytes()
    captured_urls: list[str] = []

    def fake_urlopen(url: str):
        captured_urls.append(url)
        return _BytesResponse(payload)

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)

    frame = fred.fetch_fred_series(
        series_id="DGS10",
        start_date=dt.date(2026, 5, 18),
        end_date=dt.date(2026, 5, 20),
        api_key="test-key",
        realtime_start="2026-05-20",
        realtime_end="2026-05-20",
        max_retries=1,
        base_sleep_sec=0.0,
    )

    assert frame["value"].iloc[0] == 4.45
    assert pd.isna(frame["value"].iloc[1])
    assert frame["value"].iloc[2] == 4.49
    parsed = urllib.parse.urlparse(captured_urls[0])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.stlouisfed.org"
    assert parsed.path == "/fred/series/observations"
    assert query == {
        "series_id": ["DGS10"],
        "observation_start": ["2026-05-18"],
        "observation_end": ["2026-05-20"],
        "file_type": ["json"],
        "api_key": ["test-key"],
        "realtime_start": ["2026-05-20"],
        "realtime_end": ["2026-05-20"],
    }


def test_fetch_fred_vintage_dates_builds_vintage_endpoint_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (_FRED_FIXTURE_DIR / "dgs10_vintagedates.json").read_bytes()
    captured_urls: list[str] = []

    def fake_urlopen(url: str):
        captured_urls.append(url)
        return _BytesResponse(payload)

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)

    body = fred.fetch_fred_vintage_dates(
        series_id="DGS10",
        api_key="test-key",
        max_retries=1,
        base_sleep_sec=0.0,
    )

    assert body == payload.decode("utf-8")
    parsed = urllib.parse.urlparse(captured_urls[0])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/fred/series/vintagedates"
    assert query == {
        "series_id": ["DGS10"],
        "file_type": ["json"],
        "api_key": ["test-key"],
    }


def test_fetch_url_text_retries_transient_urlerror_then_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(url: str):
        assert url == "https://example.test/fred"
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary DNS failure")
        return _BytesResponse(b'{"ok": true}')

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fred.random, "uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr(fred.time, "sleep", sleeps.append)

    body = fred._fetch_url_text_with_retries(
        url="https://example.test/fred",
        max_retries=2,
        base_sleep_sec=0.25,
        error_prefix="FRED fetch failed for DGS10",
    )

    assert body == '{"ok": true}'
    assert attempts["count"] == 2
    assert sleeps == [0.25]


def test_fetch_url_text_does_not_retry_non_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}

    def fake_urlopen(url: str):
        attempts["count"] += 1
        raise _http_error(url, 400)

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        fred._fetch_url_text_with_retries(
            url="https://example.test/fred?series_id=DGS10",
            max_retries=3,
            base_sleep_sec=0.0,
            error_prefix="FRED fetch failed for DGS10",
        )

    assert excinfo.value.code == 400
    assert attempts["count"] == 1


def test_fetch_url_text_retries_transient_http_error_until_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(url: str):
        attempts["count"] += 1
        raise _http_error(url, 503)

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fred.random, "uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr(fred.time, "sleep", sleeps.append)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        fred._fetch_url_text_with_retries(
            url="https://example.test/fred?series_id=DGS10",
            max_retries=3,
            base_sleep_sec=0.5,
            error_prefix="FRED fetch failed for DGS10",
        )

    assert excinfo.value.code == 503
    assert attempts["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_parse_fred_series_json_returns_empty_frame_for_empty_observations() -> None:
    frame = fred.parse_fred_series_json('{"observations": []}', series_id="DGS10")

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
