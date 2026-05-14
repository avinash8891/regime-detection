from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation
from typing import Annotated

from regime_detection.calendar import as_date, nyse_sessions_between, require_nyse_trading_day
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
    constituent_ohlcv: Annotated[dict[str, pd.DataFrame] | None, SkipValidation] = None  # v2 §1D PIT breadth seam
    aaii_sentiment: pd.DataFrame | None = None  # v2 §1A euphoria sentiment seam (Log #32)


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
) -> MarketContext:
    end_date = as_date(end_date)
    require_nyse_trading_day(end_date)
    normalized_market_data = _normalize_market_data_for_runtime(market_data)
    _require_market_data_contract(normalized_market_data, as_of_date=end_date)

    spy_ohlcv = _spy_ohlcv_frame(normalized_market_data, as_of_date=end_date)
    rsp_close = _symbol_close_series(normalized_market_data, symbol="RSP", as_of_date=end_date)
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
    reindexed_sector_etf_closes = _reindex_optional_close_dict(sector_etf_closes, spy_ohlcv.index)
    reindexed_cross_asset_closes = _reindex_optional_close_dict(cross_asset_closes, spy_ohlcv.index)
    reindexed_macro_series = _reindex_optional_close_dict(macro_series, spy_ohlcv.index)
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
        macro_series=reindexed_macro_series,
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
        aaii_sentiment=aaii_sentiment,
    )


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


def slice_context_to_recent_sessions(*, context: MarketContext, required_sessions: int) -> MarketContext:
    if required_sessions >= len(context.sessions):
        return context
    keep_sessions = list(context.sessions[-required_sessions:])
    start_ts = pd.Timestamp(keep_sessions[0])
    spy_ohlcv = context.spy_ohlcv.loc[start_ts:]
    rsp_close = context.rsp_close.reindex(spy_ohlcv.index)
    vix_proxy_close = None
    if context.vix_proxy_close is not None:
        vix_proxy_close = context.vix_proxy_close.reindex(spy_ohlcv.index)
    return MarketContext(
        end_date=context.end_date,
        config=context.config,
        sessions=tuple(spy_ohlcv.index.date),
        spy_ohlcv=spy_ohlcv,
        rsp_close=rsp_close,
        vix_proxy_close=vix_proxy_close,
        normalized_event_calendar=context.normalized_event_calendar,
        sector_etf_closes=_reindex_optional_close_dict(context.sector_etf_closes, spy_ohlcv.index),
        cross_asset_closes=_reindex_optional_close_dict(context.cross_asset_closes, spy_ohlcv.index),
        macro_series=_reindex_optional_close_dict(context.macro_series, spy_ohlcv.index),
        # PIT seams: pass through as-is. Intervals carry their own start/end
        # dates and constituent_ohlcv frames carry per-ticker date columns;
        # downstream readers handle date-bounded queries themselves. Dropping
        # them here would silently disable §1D PIT breadth feature compute.
        pit_constituent_intervals=context.pit_constituent_intervals,
        constituent_ohlcv=context.constituent_ohlcv,
        # §1A euphoria sentiment seam: pass through as-is. AAII rows carry
        # their own `date` / `publication_date` columns; the feature_store
        # forward-fills onto the spy session index downstream. Dropping
        # here would silently disable the euphoria predicate.
        aaii_sentiment=context.aaii_sentiment,
    )


def slice_context_to_end_date(*, context: MarketContext, end_date: date) -> MarketContext:
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
        sector_etf_closes=_reindex_optional_close_dict(context.sector_etf_closes, spy_ohlcv.index),
        cross_asset_closes=_reindex_optional_close_dict(context.cross_asset_closes, spy_ohlcv.index),
        macro_series=_reindex_optional_close_dict(context.macro_series, spy_ohlcv.index),
        # PIT seams: pass through as-is. See slice_context_to_recent_sessions
        # for rationale — dropping them here silently disables §1D PIT breadth.
        pit_constituent_intervals=context.pit_constituent_intervals,
        constituent_ohlcv=context.constituent_ohlcv,
        # §1A euphoria sentiment seam: pass through as-is (see
        # slice_context_to_recent_sessions for rationale).
        aaii_sentiment=context.aaii_sentiment,
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
        raise ValueError(
            f"market_data contains malformed date values: {exc}"
        ) from exc
    return out


def _require_market_data_contract(df: pd.DataFrame, *, as_of_date: date) -> None:
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    if df.empty:
        raise ValueError("market_data must not be empty")
    if (df["symbol"] == "SPY").sum() == 0:
        raise ValueError("market_data must contain SPY rows for V1")
    if df["date"].isna().any():
        # Belt-and-braces: even though _normalize_market_data_for_runtime raises
        # on coercion errors, defend against callers that bypass the normalizer
        # and pass NaT-containing frames in directly.
        raise ValueError("market_data contains null date values; reject at the ingestion boundary")
    dates = df["date"].dt.date
    has_spy_asof = ((df["symbol"] == "SPY") & (dates == as_of_date)).any()
    if not bool(has_spy_asof):
        raise ValueError(f"market_data must include SPY row for as_of_date={as_of_date.isoformat()}")
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


def _spy_ohlcv_frame(df: pd.DataFrame, *, as_of_date: date) -> pd.DataFrame:
    s = df[df["symbol"] == "SPY"].copy()
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    s = s.set_index("date")
    return s[["open", "high", "low", "close", "volume"]]


def _symbol_close_series(df: pd.DataFrame, *, symbol: str, as_of_date: date) -> pd.Series:
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
) -> pd.Series | None:
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
    for symbol in ["VIXY", "VIX", "^VIX"]:
        try:
            return _symbol_close_series(market_data, symbol=symbol, as_of_date=as_of_date)
        except ValueError:
            continue
    return None
