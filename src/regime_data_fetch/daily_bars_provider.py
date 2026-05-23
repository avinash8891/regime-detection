from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable

import pandas as pd

from regime_data_fetch.alpaca_daily import DailyBarsFetchResult

DAILY_BARS_PROVIDERS = ("alpaca", "yahoo-chart", "alpaca-yahoo-fallback")

logger = logging.getLogger(__name__)


def fetch_daily_bars_with_provider(
    *,
    provider: str,
    symbols: list[str],
    start_date: dt.date,
    end_date: dt.date,
    adjustment: str,
    feed: str | None,
    verbose: bool,
    alpaca_fetcher: Callable[..., DailyBarsFetchResult],
    yahoo_fetcher: Callable[..., DailyBarsFetchResult],
) -> DailyBarsFetchResult:
    if provider == "alpaca":
        return alpaca_fetcher(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjustment=adjustment,
            feed=feed,
            verbose=verbose,
        )
    if provider == "yahoo-chart":
        return yahoo_fetcher(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjustment=adjustment,
            feed=feed,
            verbose=verbose,
        )
    if provider != "alpaca-yahoo-fallback":
        raise ValueError(f"Unknown daily bars provider: {provider!r}")

    try:
        alpaca_result = alpaca_fetcher(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjustment=adjustment,
            feed=feed,
            verbose=verbose,
        )
    except Exception as exc:
        logger.warning(
            "alpaca daily bars failed; falling back to yahoo chart",
            extra={
                "data_source": "daily_bars_provider",
                "fallback_source": "yahoo-chart",
                "symbol_count": len(symbols),
                "error_type": type(exc).__name__,
            },
        )
        return yahoo_fetcher(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjustment=adjustment,
            feed=feed,
            verbose=verbose,
        )

    missing = alpaca_result.missing_symbols
    if not missing:
        return alpaca_result

    yahoo_result = yahoo_fetcher(
        symbols=missing,
        start_date=start_date,
        end_date=end_date,
        adjustment=adjustment,
        feed=feed,
        verbose=verbose,
    )
    frames = [
        frame
        for frame in (alpaca_result.df, yahoo_result.df)
        if not frame.empty
    ]
    if frames:
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"], kind="stable")
            .reset_index(drop=True)
        )
    else:
        combined = alpaca_result.df
    return DailyBarsFetchResult(
        df=combined,
        missing_symbols=yahoo_result.missing_symbols,
    )
