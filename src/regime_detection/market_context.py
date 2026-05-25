from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation
from typing import Annotated

from regime_detection.calendar import (
    as_date,
    nyse_sessions_between,
    require_nyse_trading_day,
)
from regime_detection.config import RegimeConfig
from regime_detection.loaders import load_event_calendar


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    end_date: date
    config: RegimeConfig
    sessions: tuple[date, ...]
    spy_ohlcv: pd.DataFrame
    rsp_close: pd.Series
    vix_proxy_close: pd.Series | None
    normalized_event_calendar: pd.DataFrame | None = None
    sector_etf_closes: dict[str, pd.Series] | None = None  # v2 §3.1
    cross_asset_closes: dict[str, pd.Series] | None = None  # v2 §3.1
    macro_series: dict[str, pd.Series] | None = None  # v2 §2A/§2B/§2C FRED series
    pit_constituent_intervals: pd.DataFrame | None = None  # v2 §1D PIT breadth seam
    constituent_ohlcv: Annotated[dict[str, pd.DataFrame] | None, SkipValidation] = (
        None  # v2 §1D PIT breadth seam
    )
    aaii_sentiment: pd.DataFrame | None = (
        None  # v2 §1A euphoria sentiment seam (ADR 0004)
    )
    implied_vol_30d: pd.Series | None = (
        None  # v2 §1C vol_crush seam — FRED VIXCLS/100 (ADR 0005)
    )
    # v2 §2A central-bank-text evidence seam — deterministic-lexicon score
    # over FOMC minutes + Powell speech body_text per release. See
    # implementation decision and verification notes §3.1
    # (M1). Always evidence-only — never consumed by §2A rule predicates.
    central_bank_text_releases: pd.DataFrame | None = None
    # v2 §2A first-release CPI seam for historical replay (Ambiguity
    # decision note / spec lines 2956-2957). Series keyed by RELEASE DATE (not
    # reference date) of the value-as-of-release. See
    # verification notes §3.2 (M2). When None, the
    # existing latest-revision CPIAUCSL path is preserved unchanged.
    cpi_first_release: pd.Series | None = None
    # v2 §1A SF Fed Daily News Sentiment Index — evidence-only second
    # sentiment voice alongside the AAII bull-bear 8w-MA `sentiment_score`.
    # NOT consumed by the `euphoria` rule predicate; surfaces on
    # TrendDirectionV2Features.news_sentiment_score for evidence dicts and
    # the calibration summary. See implementation decision and
    # verification notes §4.1 (Post-M1/M2 follow-up).
    news_sentiment: pd.Series | None = None


def build_market_context(
    *,
    end_date: date,
    market_data: pd.DataFrame,
    config: RegimeConfig,
    vix_data: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
    sector_etf_closes: dict[str, pd.Series] | None = None,
    cross_asset_closes: dict[str, pd.Series] | None = None,
    macro_series: dict[str, pd.Series] | None = None,
    pit_constituent_intervals: pd.DataFrame | None = None,
    constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
    aaii_sentiment: pd.DataFrame | None = None,
    implied_vol_30d: pd.Series | None = None,
    central_bank_text_releases: pd.DataFrame | None = None,
    cpi_first_release: pd.Series | None = None,
    news_sentiment: pd.Series | None = None,
) -> MarketContext:
    end_date = as_date(end_date)
    require_nyse_trading_day(end_date)
    normalized_market_data = _normalize_market_data_for_runtime(market_data)
    cap_weight_symbol = config.etf_proxy.cap_weight_index
    equal_weight_symbol = config.etf_proxy.equal_weight_proxy
    _require_market_data_contract(
        normalized_market_data, as_of_date=end_date, cap_weight_symbol=cap_weight_symbol
    )
    _require_constituent_ohlcv_contract(constituent_ohlcv)

    spy_ohlcv = _spy_ohlcv_frame(
        normalized_market_data, as_of_date=end_date, symbol=cap_weight_symbol
    )
    rsp_close = _symbol_close_series(
        normalized_market_data, symbol=equal_weight_symbol, as_of_date=end_date
    )
    vix_proxy_close = _resolve_vix_proxy_close(
        market_data=normalized_market_data,
        vix_data=vix_data,
        as_of_date=end_date,
    )
    normalized_event_calendar = (
        None
        if event_calendar is None
        else load_event_calendar(event_calendar, market=config.event_calendar.market)
    )
    reindexed_sector_etf_closes = _reindex_optional_close_dict(
        sector_etf_closes, spy_ohlcv.index
    )
    reindexed_cross_asset_closes = _reindex_optional_close_dict(
        cross_asset_closes, spy_ohlcv.index
    )
    reindexed_implied_vol_30d = (
        None if implied_vol_30d is None else implied_vol_30d.reindex(spy_ohlcv.index)
    )
    return MarketContext(
        end_date=end_date,
        config=config,
        sessions=tuple(spy_ohlcv.index.date),
        spy_ohlcv=spy_ohlcv,
        rsp_close=rsp_close,
        vix_proxy_close=vix_proxy_close,
        normalized_event_calendar=normalized_event_calendar,
        sector_etf_closes=reindexed_sector_etf_closes,
        cross_asset_closes=reindexed_cross_asset_closes,
        macro_series=_reindex_macro_to_sessions(macro_series, spy_ohlcv.index),
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
        aaii_sentiment=aaii_sentiment,
        implied_vol_30d=reindexed_implied_vol_30d,
        central_bank_text_releases=central_bank_text_releases,
        cpi_first_release=cpi_first_release,
        news_sentiment=news_sentiment,
    )


def _reindex_macro_to_sessions(
    macro_series: dict[str, pd.Series] | None,
    spy_index: pd.DatetimeIndex,
) -> dict[str, pd.Series] | None:
    """Extend macro series index to include SPY sessions without clipping.

    Adds NYSE-only dates (e.g., Columbus Day) to each macro series' index
    so downstream lookups via `.get(dt)` find the date in the index (even
    if the value is NaN). Does NOT forward-fill — staleness detection
    relies on NaN gaps to identify stale/truncated sources.
    """
    if macro_series is None:
        return None
    out: dict[str, pd.Series] = {}
    for key, series in macro_series.items():
        combined_index = series.index.union(spy_index).sort_values()
        out[key] = series.reindex(combined_index)
    return out


def _reindex_optional_close_dict(
    series_dict: dict[str, pd.Series] | None,
    target_index: pd.Index,
) -> dict[str, pd.Series] | None:
    if series_dict is None:
        return None
    out: dict[str, pd.Series] = {}
    for key, series in series_dict.items():
        out[key] = series.reindex(target_index)
    return out


def slice_context_to_recent_sessions(
    *, context: MarketContext, required_sessions: int
) -> MarketContext:
    if required_sessions >= len(context.sessions):
        return context
    keep_sessions = list(context.sessions[-required_sessions:])
    start_ts = pd.Timestamp(keep_sessions[0])
    spy_ohlcv = context.spy_ohlcv.loc[start_ts:]
    return _with_sliced_session_data(
        context=context,
        end_date=context.end_date,
        spy_ohlcv=spy_ohlcv,
    )


def slice_context_to_end_date(
    *, context: MarketContext, end_date: date
) -> MarketContext:
    end_date = as_date(end_date)
    require_nyse_trading_day(end_date)
    if end_date > context.end_date:
        raise ValueError(
            "Provided MarketContext does not cover requested end_date. "
            f"context.end_date={context.end_date.isoformat()} requested={end_date.isoformat()}"
        )
    if end_date not in context.sessions:
        raise ValueError(
            "Provided MarketContext does not contain requested NYSE session. "
            f"requested={end_date.isoformat()}"
        )
    if end_date == context.end_date:
        return context

    end_ts = pd.Timestamp(end_date)
    spy_ohlcv = context.spy_ohlcv.loc[:end_ts]
    return _with_sliced_session_data(
        context=context,
        end_date=end_date,
        spy_ohlcv=spy_ohlcv,
    )


def _with_sliced_session_data(
    *,
    context: MarketContext,
    end_date: date,
    spy_ohlcv: pd.DataFrame,
) -> MarketContext:
    rsp_close = context.rsp_close.reindex(spy_ohlcv.index)
    vix_proxy_close = None
    if context.vix_proxy_close is not None:
        vix_proxy_close = context.vix_proxy_close.reindex(spy_ohlcv.index)
    return MarketContext(
        end_date=end_date,
        config=context.config,
        sessions=tuple(spy_ohlcv.index.date),
        spy_ohlcv=spy_ohlcv,
        rsp_close=rsp_close,
        vix_proxy_close=vix_proxy_close,
        normalized_event_calendar=context.normalized_event_calendar,
        sector_etf_closes=_reindex_optional_close_dict(
            context.sector_etf_closes, spy_ohlcv.index
        ),
        cross_asset_closes=_reindex_optional_close_dict(
            context.cross_asset_closes, spy_ohlcv.index
        ),
        macro_series=context.macro_series,
        pit_constituent_intervals=context.pit_constituent_intervals,
        constituent_ohlcv=context.constituent_ohlcv,
        aaii_sentiment=context.aaii_sentiment,
        implied_vol_30d=(
            None
            if context.implied_vol_30d is None
            else context.implied_vol_30d.reindex(spy_ohlcv.index)
        ),
        central_bank_text_releases=context.central_bank_text_releases,
        cpi_first_release=context.cpi_first_release,
        news_sentiment=context.news_sentiment,
    )


def _normalize_market_data_for_runtime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        return df
    out = df.copy()
    # errors="raise" (default) — bad/malformed date strings must fail loud
    # at the ingestion boundary. The previous errors="coerce" silently
    # produced NaT, which the downstream dropna() in
    # _require_market_data_contract then dropped, allowing bad-date rows
    # to bypass NYSE-session validation. Wrap to surface a project-scoped
    # error message instead of the raw pandas exception.
    try:
        out["date"] = pd.to_datetime(out["date"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"market_data contains malformed date values: {exc}") from exc
    return out


def _require_market_data_contract(
    df: pd.DataFrame, *, as_of_date: date, cap_weight_symbol: str = "SPY"
) -> None:
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    if df.empty:
        raise ValueError("market_data must not be empty")
    if (df["symbol"] == cap_weight_symbol).sum() == 0:
        raise ValueError(f"market_data must contain {cap_weight_symbol} rows")
    if df["date"].isna().any():
        # Belt-and-braces: even though _normalize_market_data_for_runtime raises
        # on coercion errors, defend against callers that bypass the normalizer
        # and pass NaT-containing frames in directly.
        raise ValueError(
            "market_data contains null date values; reject at the ingestion boundary"
        )
    dates = df["date"].dt.date
    has_cap_weight_asof = (
        (df["symbol"] == cap_weight_symbol) & (dates == as_of_date)
    ).any()
    if not bool(has_cap_weight_asof):
        raise ValueError(
            f"market_data must include {cap_weight_symbol} row for as_of_date={as_of_date.isoformat()}"
        )
    uniq_dates = sorted({d for d in dates.dropna().unique()})
    if uniq_dates:
        start = min(uniq_dates)
        end = max(uniq_dates)
        sessions = nyse_sessions_between(start, end)
        session_set = set(sessions)
        bad_dates = [d for d in uniq_dates if d not in session_set]
        if bad_dates:
            raise ValueError(
                "market_data contains non-NYSE session dates (forbidden in V1). "
                f"Examples: {bad_dates[:5]}"
            )


def _require_constituent_ohlcv_contract(
    constituent_ohlcv: dict[str, pd.DataFrame] | None,
) -> None:
    if constituent_ohlcv is None:
        return
    required_cols = {"open", "high", "low", "close", "volume", "adjusted_close"}
    for ticker, frame in constituent_ohlcv.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(
                "constituent_ohlcv frame must be a pandas DataFrame. "
                f"ticker={ticker!r} actual_type={type(frame).__name__}"
            )
        missing = sorted(required_cols - set(frame.columns))
        if missing:
            raise ValueError(
                "constituent_ohlcv frame missing required columns. "
                f"ticker={ticker!r} missing={missing}"
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(
                "constituent_ohlcv frame must use a DatetimeIndex date index. "
                f"ticker={ticker!r} actual_index_type={type(frame.index).__name__}"
            )
        if frame.index.hasnans:
            raise ValueError(
                "constituent_ohlcv frame contains null dates in DatetimeIndex. "
                f"ticker={ticker!r}"
            )


def _spy_ohlcv_frame(
    df: pd.DataFrame, *, as_of_date: date, symbol: str = "SPY"
) -> pd.DataFrame:
    s = df[df["symbol"] == symbol].copy()
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    s = s.set_index("date")
    return s[["open", "high", "low", "close", "volume"]]


def _symbol_close_series(
    df: pd.DataFrame, *, symbol: str, as_of_date: date
) -> pd.Series:
    s = df[df["symbol"] == symbol].copy()
    if s.empty:
        raise ValueError(f"market_data missing required symbol for V1: {symbol}")
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    out = pd.Series(s["close"].to_numpy(), index=pd.to_datetime(s["date"]))
    out.name = "close"
    return out


def _resolve_vix_proxy_close(
    *,
    market_data: pd.DataFrame,
    vix_data: pd.DataFrame | None,
    as_of_date: date,
) -> pd.Series:
    if vix_data is not None:
        if "date" not in vix_data.columns or "close" not in vix_data.columns:
            raise ValueError("vix_data must contain date and close columns")
        s = vix_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(s["date"]):
            s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values("date")
        s = s[s["date"].dt.date <= as_of_date]
        out = pd.Series(s["close"].to_numpy(), index=pd.to_datetime(s["date"]))
        out.name = "close"
        return out
    return _symbol_close_series(market_data, symbol="VIX", as_of_date=as_of_date)
