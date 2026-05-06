from __future__ import annotations

import datetime as dt
import json
import random
import time
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
ALFRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGEDATES_API = "https://api.stlouisfed.org/fred/series/vintagedates"


def fetch_fred_series(
    *,
    series_id: str,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str | None = None,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    max_retries: int = 4,
    base_sleep_sec: float = 2.0,
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if realtime_start:
        params["realtime_start"] = realtime_start
    if realtime_end:
        params["realtime_end"] = realtime_end

    payload_text = fetch_fred_series_json(
        series_id=series_id,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        realtime_start=realtime_start,
        realtime_end=realtime_end,
        max_retries=max_retries,
        base_sleep_sec=base_sleep_sec,
    )
    return parse_fred_series_json(payload_text, series_id=series_id)


def fetch_fred_series_json(
    *,
    series_id: str,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str | None = None,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    max_retries: int = 4,
    base_sleep_sec: float = 2.0,
) -> str:
    params = {
        "series_id": series_id,
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if realtime_start:
        params["realtime_start"] = realtime_start
    if realtime_end:
        params["realtime_end"] = realtime_end

    url = f"{FRED_API}?{urllib.parse.urlencode(params)}"
    return _fetch_url_text_with_retries(
        url=url,
        max_retries=max_retries,
        base_sleep_sec=base_sleep_sec,
        error_prefix=f"FRED fetch failed for {series_id}",
    )


def parse_fred_series_json(payload_text: str, *, series_id: str) -> pd.DataFrame:
    payload = json.loads(payload_text)
    rows: list[dict[str, object]] = []
    for obs in payload.get("observations", []):
        value = obs.get("value")
        if value in {None, "."}:
            numeric = None
        else:
            numeric = float(value)
        rows.append(
            {
                "date": dt.date.fromisoformat(obs["date"]),
                "series_id": series_id,
                "value": numeric,
                "realtime_start": obs.get("realtime_start"),
                "realtime_end": obs.get("realtime_end"),
            }
        )
    return pd.DataFrame.from_records(rows)


def fetch_fred_vintage_dates(
    *,
    series_id: str,
    api_key: str | None = None,
    max_retries: int = 4,
    base_sleep_sec: float = 2.0,
) -> str:
    params = {
        "series_id": series_id,
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{FRED_VINTAGEDATES_API}?{urllib.parse.urlencode(params)}"

    return _fetch_url_text_with_retries(
        url=url,
        max_retries=max_retries,
        base_sleep_sec=base_sleep_sec,
        error_prefix=f"FRED vintagedates fetch failed for {series_id}",
    )


def _fetch_url_text_with_retries(
    *,
    url: str,
    max_retries: int,
    base_sleep_sec: float,
    error_prefix: str,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code < 500 or attempt >= max_retries:
                raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise

        sleep = base_sleep_sec * (2 ** (attempt - 1))
        sleep = sleep + random.uniform(0.0, min(1.0, sleep * 0.1))
        time.sleep(sleep)
    raise RuntimeError(f"{error_prefix}: {last_exc}") from last_exc
