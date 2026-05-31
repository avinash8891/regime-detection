from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import urlopen as stdlib_urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from regime_data_fetch._http import fetch_text
from regime_data_fetch.alpaca_daily import DailyBarsFetchResult

logger = logging.getLogger(__name__)

YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
DAILY_OHLCV_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
]


def fetch_daily_bars_yahoo_chart(
    *,
    symbols: list[str],
    start_date: dt.date,
    end_date: dt.date,
    adjustment: str = "raw",
    feed: str | None = None,
    batch_size: int = 100,
    verbose: bool = False,
    timeout_sec: float = 30.0,
    urlopen: Callable[..., Any] = stdlib_urlopen,
) -> DailyBarsFetchResult:
    """Fetch daily OHLCV bars from Yahoo Finance's chart endpoint.

    Returns the same long DataFrame contract as fetch_daily_bars_alpaca:
        date, symbol, open, high, low, close, volume, adjusted_close

    The project stores raw OHLCV with adjusted_close = close for Alpaca parity.
    """
    del feed
    del batch_size
    if adjustment != "raw":
        raise ValueError(
            "Yahoo chart daily fetch supports only adjustment='raw' for this schema"
        )
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        if verbose:
            logger.info(
                "yahoo chart daily bars request",
                extra={
                    "data_source": "yahoo_chart_daily_ohlcv",
                    "symbol": symbol,
                    "symbol_index": index,
                    "symbol_count": len(symbols),
                },
            )
        payload = _fetch_yahoo_chart_json(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeout_sec=timeout_sec,
            urlopen=urlopen,
        )
        frame = _parse_yahoo_chart_payload(
            payload=payload,
            requested_symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if frame.empty:
            missing.append(symbol)
            continue
        frames.append(frame)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=DAILY_OHLCV_COLUMNS)

    return DailyBarsFetchResult(df=out, missing_symbols=missing)


def _fetch_yahoo_chart_json(
    *,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    timeout_sec: float,
    urlopen: Callable[..., Any],
) -> dict[str, Any]:
    payload = fetch_text(
        _build_yahoo_chart_url(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        ),
        headers={"Accept": "application/json"},
        timeout=timeout_sec,
        errors="strict",
        urlopen=urlopen,
    )
    return json.loads(payload)


def _build_yahoo_chart_url(
    *,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    period1 = int(
        dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc).timestamp()
    )
    period2 = int(
        dt.datetime.combine(
            end_date + dt.timedelta(days=1),
            dt.time.min,
            tzinfo=dt.timezone.utc,
        ).timestamp()
    )
    params = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    return f"{YAHOO_CHART_BASE_URL}/{quote(symbol, safe='')}?{params}"


def _parse_yahoo_chart_payload(
    *,
    payload: dict[str, Any],
    requested_symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise RuntimeError(
            f"Yahoo chart response missing chart object for {requested_symbol}"
        )
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {requested_symbol}: {error}")

    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=DAILY_OHLCV_COLUMNS)
    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        return pd.DataFrame(columns=DAILY_OHLCV_COLUMNS)

    quote_data = quotes[0]
    adjusted_close_data = _adjusted_close_data(indicators)
    timezone_name = (result.get("meta") or {}).get("exchangeTimezoneName")
    timezone = _exchange_timezone(timezone_name)
    rows: list[dict[str, object]] = []
    for row_index, epoch in enumerate(timestamps):
        row_date = (
            dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
            .astimezone(timezone)
            .date()
        )
        if row_date < start_date or row_date > end_date:
            continue
        open_value = _value_at(quote_data, "open", row_index)
        high_value = _value_at(quote_data, "high", row_index)
        low_value = _value_at(quote_data, "low", row_index)
        close_value = _value_at(quote_data, "close", row_index)
        volume_value = _value_at(quote_data, "volume", row_index)
        adjusted_close_value = _value_at(adjusted_close_data, "adjclose", row_index)
        if None in {open_value, high_value, low_value, close_value, volume_value}:
            continue
        rows.append(
            {
                "date": row_date,
                "symbol": requested_symbol,
                "open": float(open_value),
                "high": float(high_value),
                "low": float(low_value),
                "close": float(close_value),
                "volume": int(volume_value),
                "adjusted_close": float(
                    close_value
                    if adjusted_close_value is None
                    else adjusted_close_value
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=DAILY_OHLCV_COLUMNS)
    return pd.DataFrame.from_records(rows, columns=DAILY_OHLCV_COLUMNS)


def _exchange_timezone(timezone_name: object) -> dt.tzinfo:
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        return dt.timezone.utc
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return dt.timezone.utc


def _value_at(data: dict[str, Any], key: str, index: int) -> object | None:
    values = data.get(key)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _adjusted_close_data(indicators: dict[str, Any]) -> dict[str, Any]:
    adjusted_close_entries = indicators.get("adjclose") or []
    if not adjusted_close_entries:
        return {}
    adjusted_close_data = adjusted_close_entries[0]
    if not isinstance(adjusted_close_data, dict):
        return {}
    return adjusted_close_data
