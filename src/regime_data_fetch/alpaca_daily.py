from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DailyBarsFetchResult:
    df: pd.DataFrame
    missing_symbols: list[str]


def _get_alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY_ID"],
        secret_key=os.environ["ALPACA_API_SECRET_KEY"],
    )


def _sanitize_alpaca_symbol(sym: str) -> str:
    # Alpaca uses dots not hyphens for class shares
    return sym.replace("-", ".")


def fetch_daily_bars_alpaca(
    *,
    symbols: list[str],
    start_date: _dt.date,
    end_date: _dt.date,
    adjustment: str = "raw",
    feed: str | None = None,
    batch_size: int = 100,
    verbose: bool = False,
) -> DailyBarsFetchResult:
    """Fetch daily OHLCV bars from Alpaca for symbols in [start_date, end_date].

    Returns a long DataFrame with columns:
        date, symbol, open, high, low, close, volume, adjusted_close

    Note: Alpaca does not provide a separate adjusted_close column. We always
    include adjusted_close = close so downstream schema stays stable.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    from alpaca.data.enums import DataFeed

    if adjustment not in {"raw", "split", "dividend", "all"}:
        raise ValueError(f"Unknown adjustment: {adjustment!r}")
    if feed is not None and feed not in {"sip", "iex", "otc"}:
        raise ValueError(f"Unknown Alpaca feed: {feed!r} (expected sip|iex|otc)")

    start_dt = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=_dt.timezone.utc)
    # inclusive end date: request through end-of-day UTC
    end_dt = _dt.datetime.combine(end_date, _dt.time.max, tzinfo=_dt.timezone.utc)

    client = _get_alpaca_client()

    out_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    # Keep original symbols for output; sanitize only for request.
    sym_map = {_sanitize_alpaca_symbol(s): s for s in symbols}
    req_syms = list(sym_map.keys())

    for i in range(0, len(req_syms), batch_size):
        batch = req_syms[i : i + batch_size]
        if verbose:
            print(f"[alpaca] daily batch {i//batch_size + 1}/{(len(req_syms)+batch_size-1)//batch_size}: requesting {len(batch)} symbols", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            adjustment=Adjustment(adjustment),
            feed=(DataFeed(feed) if feed else None),
        )
        resp = client.get_stock_bars(req)
        bar_data = resp.data if hasattr(resp, "data") else {}

        for req_sym in batch:
            canonical = sym_map[req_sym]
            bars = bar_data.get(req_sym)
            if not bars:
                missing.append(canonical)
                continue
            rows = []
            for b in bars:
                ts = b.timestamp
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.replace(tzinfo=_dt.timezone.utc)
                # For daily bars, date is what we care about.
                d = ts.date()
                rows.append(
                    {
                        "date": d,
                        "symbol": canonical,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": int(b.volume),
                    }
                )
            df = pd.DataFrame.from_records(rows)
            if df.empty:
                missing.append(canonical)
                continue
            df["adjusted_close"] = df["close"]
            out_frames.append(df)
        if verbose:
            got = sum(1 for s in batch if s in bar_data and bar_data.get(s))
            print(f"[alpaca] batch done: got_bars_for={got}/{len(batch)} (cumulative_frames={len(out_frames)})", flush=True)

    if out_frames:
        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    else:
        out = pd.DataFrame(
            columns=["date", "symbol", "open", "high", "low", "close", "volume", "adjusted_close"]
        )

    # De-dup missing (can happen if empty per-batch) and preserve first-seen order.
    seen: set[str] = set()
    missing_unique: list[str] = []
    for s in missing:
        if s not in seen:
            seen.add(s)
            missing_unique.append(s)

    return DailyBarsFetchResult(df=out, missing_symbols=missing_unique)


def verify_min_start_date(
    df: pd.DataFrame, *, symbol: str, required_start: _dt.date
) -> tuple[_dt.date | None, bool]:
    """Return (min_date, ok) for a single symbol inside a long daily bars df."""
    sdf = df[df["symbol"] == symbol]
    if sdf.empty:
        return None, False
    min_date = sdf["date"].min()
    # pandas may return Timestamp/NaT-like; normalize to python date
    if hasattr(min_date, "date"):
        min_date = min_date.date()
    # Required start dates are specified as calendar dates in the spec, but the
    # raw market data is trading days only. We consider it "ok" if the first
    # returned trading day is within a small tolerance window after required_start.
    #
    # We still report the concrete min_date so callers can validate precisely.
    tolerance_days = 7
    ok = bool(min_date and min_date <= (required_start + _dt.timedelta(days=tolerance_days)))
    return min_date, ok
