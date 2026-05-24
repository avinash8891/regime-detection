from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from regime_data_fetch.alpaca_daily import DailyBarsFetchResult
from regime_data_fetch.daily_bars_provider import fetch_daily_bars_with_provider


def _frame(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": dt.date(2026, 5, 15),
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "adjusted_close": 100.5,
            }
            for symbol in symbols
        ]
    )


def test_alpaca_yahoo_fallback_fetches_missing_symbols_from_yahoo_only() -> None:
    calls: list[tuple[str, list[str]]] = []

    def alpaca_fetcher(**kwargs) -> DailyBarsFetchResult:
        calls.append(("alpaca", kwargs["symbols"]))
        return DailyBarsFetchResult(df=_frame(["AAPL"]), missing_symbols=["SPY"])

    def yahoo_fetcher(**kwargs) -> DailyBarsFetchResult:
        calls.append(("yahoo", kwargs["symbols"]))
        return DailyBarsFetchResult(df=_frame(["SPY"]), missing_symbols=[])

    result = fetch_daily_bars_with_provider(
        provider="alpaca-yahoo-fallback",
        symbols=["AAPL", "SPY"],
        start_date=dt.date(2026, 5, 15),
        end_date=dt.date(2026, 5, 15),
        adjustment="raw",
        feed=None,
        verbose=False,
        alpaca_fetcher=alpaca_fetcher,
        yahoo_fetcher=yahoo_fetcher,
    )

    assert calls == [("alpaca", ["AAPL", "SPY"]), ("yahoo", ["SPY"])]
    assert result.missing_symbols == []
    assert result.df["symbol"].tolist() == ["AAPL", "SPY"]


def test_alpaca_yahoo_fallback_fetches_all_symbols_from_yahoo_when_alpaca_fails() -> None:
    calls: list[tuple[str, list[str]]] = []

    def alpaca_fetcher(**kwargs) -> DailyBarsFetchResult:
        calls.append(("alpaca", kwargs["symbols"]))
        raise RuntimeError("Alpaca rate limited")

    def yahoo_fetcher(**kwargs) -> DailyBarsFetchResult:
        calls.append(("yahoo", kwargs["symbols"]))
        return DailyBarsFetchResult(df=_frame(kwargs["symbols"]), missing_symbols=[])

    result = fetch_daily_bars_with_provider(
        provider="alpaca-yahoo-fallback",
        symbols=["AAPL", "SPY"],
        start_date=dt.date(2026, 5, 15),
        end_date=dt.date(2026, 5, 15),
        adjustment="raw",
        feed=None,
        verbose=False,
        alpaca_fetcher=alpaca_fetcher,
        yahoo_fetcher=yahoo_fetcher,
    )

    assert calls == [("alpaca", ["AAPL", "SPY"]), ("yahoo", ["AAPL", "SPY"])]
    assert result.missing_symbols == []
    assert result.df["symbol"].tolist() == ["AAPL", "SPY"]


def test_unknown_daily_bars_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown daily bars provider"):
        fetch_daily_bars_with_provider(
            provider="bad-provider",
            symbols=["SPY"],
            start_date=dt.date(2026, 5, 15),
            end_date=dt.date(2026, 5, 15),
            adjustment="raw",
            feed=None,
            verbose=False,
            alpaca_fetcher=lambda **kwargs: DailyBarsFetchResult(
                df=_frame(["SPY"]), missing_symbols=[]
            ),
            yahoo_fetcher=lambda **kwargs: DailyBarsFetchResult(
                df=_frame(["SPY"]), missing_symbols=[]
            ),
        )
