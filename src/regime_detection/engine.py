from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from regime_detection.calendar import as_date, require_nyse_trading_day
from regime_detection.config import RegimeConfig, default_config_path, load_regime_config
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    MonetaryPressureOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskOutput,
)
from regime_detection.versioning import engine_version
from regime_detection.trend_direction import classify_series as classify_trend_direction
from regime_detection.trend_character import classify_series as classify_trend_character
from regime_detection.volatility_state import classify_series as classify_volatility_state
from regime_detection.breadth_state import classify_series as classify_breadth_state


class RegimeEngine:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else default_config_path()
        self._config = load_regime_config(path)

    @property
    def config(self) -> RegimeConfig:
        return self._config

    def classify(
        self,
        *,
        as_of_date: date,
        market_data: pd.DataFrame,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
    ) -> RegimeOutput:
        # Normalize as_of_date early. Callers often pass datetime/pandas Timestamp.
        # We need `date` for both calendar enforcement and market_data contract checks.
        as_of_date = as_date(as_of_date)
        cfg = config if config is not None else self._config
        if cfg.trading_calendar != "NYSE":
            raise ValueError(f"V1 supports only NYSE trading calendar. Got: {cfg.trading_calendar}")

        require_nyse_trading_day(as_of_date)

        _require_market_data_contract(market_data, as_of_date=as_of_date)
        if event_calendar is not None:
            # Slice 7 implements event calendar; until then, fail loudly rather than ignore.
            raise ValueError("event_calendar is not implemented yet (Slice 7). Pass event_calendar=None for V1.")

        # Slice 3: Trend Direction implemented; remaining axes stay unknown until their slices land.
        spy_ohlcv = _spy_ohlcv_frame(market_data, as_of_date=as_of_date)
        spy_close = spy_ohlcv["close"]
        spy_high = spy_ohlcv["high"]
        spy_low = spy_ohlcv["low"]
        vixy_close = _resolve_symbol_close(
            market_data,
            symbol="VIXY",
            as_of_date=as_of_date,
            override=vix_data,
            override_name="vix_data",
        )
        rsp_close = _resolve_symbol_close(
            market_data,
            symbol="RSP",
            as_of_date=as_of_date,
            override=breadth_data,
            override_name="breadth_data",
        )
        trend_direction = classify_trend_direction(
            close=spy_close,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.trend_direction_deescalation_days,
        )
        trend_character = classify_trend_character(
            close=spy_close,
            high=spy_high,
            low=spy_low,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.trend_character_deescalation_days,
        )
        volatility_state = classify_volatility_state(
            close=spy_close,
            vix_proxy_close=vixy_close,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.volatility_deescalation_days,
        )
        breadth_state = classify_breadth_state(
            spy_close=spy_close,
            rsp_close=rsp_close,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.breadth_deescalation_days,
        )

        unknown_axis = _unknown_axis_output()
        unknown_breadth = _unknown_breadth_output()

        structural = StructuralCausalState(
            event_calendar=_unknown_event_calendar_output(),
            monetary_pressure=MonetaryPressureOutput(
                label="unknown",
                reason="not_implemented_v1",
            ),
        )

        return RegimeOutput(
            engine_version=engine_version(),
            config_version=cfg.config_version,
            as_of_date=as_of_date,
            market="SPY",
            trend_direction=trend_direction,
            trend_character=trend_character,
            volatility_state=volatility_state,
            breadth_state=breadth_state,
            structural_causal_state=structural,
            network_fragility=NetworkFragilityOutput(
                label="not_implemented_v1",
                reason="breadth_state_used_as_v1_fragility_proxy",
            ),
            transition_risk=TransitionRiskOutput(
                label="unknown",
                evidence={"reason": "not_implemented_v1"},
            ),
            strategy_response=_unknown_strategy_response(),
        )


def _require_market_data_contract(df: pd.DataFrame, *, as_of_date: date) -> None:
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    if df.empty:
        raise ValueError("market_data must not be empty")
    if (df["symbol"] == "SPY").sum() == 0:
        raise ValueError("market_data must contain SPY rows for V1")
    dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    if dates.isna().any():
        raise ValueError("market_data contains unparseable date values")
    has_spy_asof = ((df["symbol"] == "SPY") & (dates == as_of_date)).any()
    if not bool(has_spy_asof):
        raise ValueError(f"market_data must include SPY row for as_of_date={as_of_date.isoformat()}")

    # Contract: at most one row per (date, symbol). Duplicates can break rule evaluation.
    key = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d") + "|" + df["symbol"].astype(str)
    if key.duplicated().any():
        raise ValueError("market_data contains duplicate (date, symbol) rows")

    for col in ["open", "high", "low", "close", "volume"]:
        if df[col].isna().any():
            raise ValueError(f"market_data contains nulls in {col}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"market_data column {col} must be numeric")


def _spy_ohlcv_frame(df: pd.DataFrame, *, as_of_date: date) -> pd.DataFrame:
    s = df[df["symbol"] == "SPY"].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    s = s.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    return s[["open", "high", "low", "close", "volume"]]


def _resolve_symbol_close(
    market_data: pd.DataFrame,
    *,
    symbol: str,
    as_of_date: date,
    override: pd.DataFrame | None,
    override_name: str,
) -> pd.Series:
    """
    V1 supports providing some single-series inputs either via `market_data` rows
    or via the dedicated parameter. Never silently ignore the dedicated parameter.
    """
    if override is None:
        return _symbol_close_series(market_data, symbol=symbol, as_of_date=as_of_date)
    return _override_close_series(override, symbol=symbol, as_of_date=as_of_date, override_name=override_name)


def _symbol_close_series(df: pd.DataFrame, *, symbol: str, as_of_date: date) -> pd.Series:
    s = df[df["symbol"] == symbol].copy()
    if s.empty:
        raise ValueError(
            f"market_data must contain {symbol} rows (or pass {symbol} via the dedicated input parameter)"
        )
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    s = s.drop_duplicates(subset=["date"], keep="last")
    out = pd.Series(s["close"].to_numpy(), index=pd.to_datetime(s["date"]))
    out = out.sort_index()
    out.name = "close"
    return out


def _override_close_series(
    override: pd.DataFrame,
    *,
    symbol: str,
    as_of_date: date,
    override_name: str,
) -> pd.Series:
    cols = set(override.columns)
    if {"date", "close"}.issubset(cols) and "symbol" not in cols:
        s = override.copy()
    elif {"date", "symbol", "close"}.issubset(cols):
        s = override[override["symbol"] == symbol].copy()
        if s.empty:
            raise ValueError(f"{override_name} must contain rows for symbol={symbol!r}")
    else:
        raise ValueError(
            f"{override_name} must contain columns ['date','close'] or ['date','symbol','close']; got columns={sorted(cols)}"
        )
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    if s["date"].isna().any():
        raise ValueError(f"{override_name} contains unparseable date values")
    s = s.drop_duplicates(subset=["date"], keep="last")
    out = pd.Series(s["close"].to_numpy(), index=pd.to_datetime(s["date"]))
    out = out.sort_index()
    out.name = "close"
    return out


def _unknown_data_quality(*, reason: str, status: str) -> DataQuality:
    return DataQuality(
        status=status, freshness_days=None, completeness=None, reason=reason
    )


def _unknown_axis_output() -> AxisOutput:
    return AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "not_implemented_v1"},
        data_quality=_unknown_data_quality(reason="not_implemented_v1", status="insufficient_data"),
    )


def _unknown_breadth_output() -> BreadthStateOutput:
    return BreadthStateOutput(
        mode="etf_proxy",
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "not_implemented_v1", "proxy": "RSP/SPY"},
        data_quality=_unknown_data_quality(reason="not_implemented_v1", status="insufficient_data"),
    )


def _unknown_event_calendar_output() -> EventCalendarOutput:
    return EventCalendarOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"all_matching_events": [], "selected_via_precedence": "unknown"},
    )


def _default_strategy_response() -> StrategyResponse:
    # Slice 1 placeholder: strategy_response is fully defined in Slice 9.
    return StrategyResponse(
        position_size_multiplier=1.0,
        allow_trend_following=True,
        allow_mean_reversion=True,
        leverage_allowed=True,
        allow_buy_dip=True,
        allow_breakout=True,
        allow_shorts=True,
        require_confirmation_for_new_longs=False,
        require_confirmation_for_shorts=False,
        log_for_review=True,
        modifiers_applied=["not_implemented_v1"],
    )


def _unknown_strategy_response() -> StrategyResponse:
    # Until Slice 9, avoid implying that an unknown regime is safe/tradable.
    return StrategyResponse(
        position_size_multiplier=0.0,
        allow_trend_following=False,
        allow_mean_reversion=False,
        leverage_allowed=False,
        allow_buy_dip=False,
        allow_breakout=False,
        allow_shorts=False,
        require_confirmation_for_new_longs=True,
        require_confirmation_for_shorts=True,
        log_for_review=True,
        modifiers_applied=["unknown_regime_not_implemented_v1"],
    )
