from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from regime_detection.calendar import as_date, require_nyse_trading_day
from regime_detection.config import RegimeConfig, load_default_regime_config, load_regime_config
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.loaders import load_event_calendar
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    LabelReasonOutput,
    RegimeOutput,
    RegimeTimeline,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskOutput,
)
from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.strategy_response import build_strategy_response
from regime_detection.trend_direction import (
    apply_hysteresis as apply_trend_direction_hysteresis,
    classify_series as classify_trend_direction,
    compute_features as compute_trend_direction_features,
    raw_label_for_day as trend_direction_raw_label_for_day,
)
from regime_detection.trend_character import (
    _RISK_RANK as TREND_CHARACTER_RISK_RANK,
    classify_series as classify_trend_character,
    compute_features as compute_trend_character_features,
    raw_label_for_day as trend_character_raw_label_for_day,
)
from regime_detection.transition_risk import classify_transition_risk
from regime_detection.volatility_state import (
    _RISK_RANK as VOLATILITY_RISK_RANK,
    classify_series as classify_volatility_state,
    compute_features as compute_volatility_features,
    raw_label_for_day as volatility_raw_label_for_day,
)
from regime_detection.breadth_state import (
    _RISK_RANK as BREADTH_RISK_RANK,
    classify_series as classify_breadth_state,
    compute_features as compute_breadth_features,
    raw_label_for_day as breadth_raw_label_for_day,
)
from regime_detection.versioning import engine_version


class RegimeEngine:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        if config_path is None:
            self._config = load_default_regime_config()
        else:
            self._config = load_regime_config(Path(config_path))

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

        spy_ohlcv = _spy_ohlcv_frame(market_data, as_of_date=as_of_date)
        spy_close = spy_ohlcv["close"]
        spy_high = spy_ohlcv["high"]
        spy_low = spy_ohlcv["low"]
        vix_proxy_close = _resolve_vix_proxy_close(
            market_data=market_data,
            vix_data=vix_data,
            as_of_date=as_of_date,
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
            vix_proxy_close=vix_proxy_close,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.volatility_deescalation_days,
        )
        rsp_close = _symbol_close_series(market_data, symbol="RSP", as_of_date=as_of_date)
        breadth_state = classify_breadth_state(
            spy_close=spy_close,
            rsp_close=rsp_close,
            as_of_date=as_of_date,
            deescalation_days=cfg.hysteresis.breadth_deescalation_days,
        )
        normalized_event_calendar = None if event_calendar is None else load_event_calendar(event_calendar, market=cfg.event_calendar.market)
        event_calendar_output = classify_event_calendar(
            as_of_date=as_of_date,
            event_calendar=normalized_event_calendar,
            config=cfg,
        )
        history = _axis_history(
            as_of_date=as_of_date,
            spy_ohlcv=spy_ohlcv,
            rsp_close=rsp_close,
            vix_proxy_close=vix_proxy_close,
            event_calendar=normalized_event_calendar,
            cfg=cfg,
        )
        stable_changed_today, days_since_axis_switch, trend_direction_stable_history = _history_metrics(history)
        sma_50 = spy_close.rolling(50).mean().loc[pd.Timestamp(as_of_date)]
        transition_risk = classify_transition_risk(
            as_of_date=as_of_date,
            trend_direction_active=trend_direction.active_label,
            trend_direction_stable_history=trend_direction_stable_history,
            trend_character_active=trend_character.active_label,
            volatility_state_active=volatility_state.active_label,
            breadth_state_active=breadth_state.active_label,
            stable_changed_today=stable_changed_today,
            days_since_axis_switch=days_since_axis_switch,
            close=float(spy_close.loc[pd.Timestamp(as_of_date)]),
            sma_50=None if pd.isna(sma_50) else float(sma_50),
        )
        strategy_response = build_strategy_response(
            trend_direction_active=trend_direction.active_label,
            trend_character_active=trend_character.active_label,
            volatility_state_active=volatility_state.active_label,
            breadth_state_active=breadth_state.active_label,
            transition_risk_label=transition_risk.label,
            event_calendar_active=event_calendar_output.active_label,
        )

        structural = StructuralCausalState(
            event_calendar=event_calendar_output,
            monetary_pressure=LabelReasonOutput(
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
            network_fragility=LabelReasonOutput(
                label="not_implemented_v1",
                reason="breadth_state_used_as_v1_fragility_proxy",
            ),
            transition_risk=transition_risk,
            strategy_response=strategy_response,
        )

    def classify_window(
        self,
        *,
        end_date: date,
        market_data: pd.DataFrame,
        lookback_days: int,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
    ) -> RegimeTimeline:
        end_date = as_date(end_date)
        require_nyse_trading_day(end_date)
        from regime_detection.calendar import nyse_calendar

        if lookback_days <= 0:
            raise ValueError(f"lookback_days must be > 0. Got: {lookback_days}")
        candidate_start = end_date - pd.Timedelta(days=max(30, lookback_days * 3))
        sessions = list(
            nyse_calendar()
            .schedule(start_date=candidate_start, end_date=end_date)
            .index.date
        )
        if len(sessions) < lookback_days:
            raise ValueError(
                "Insufficient NYSE trading-day coverage for requested lookback_days. "
                f"Requested={lookback_days}, available={len(sessions)}, "
                f"candidate_start={candidate_start.date().isoformat()}, end_date={end_date.isoformat()}."
            )
        selected = sessions[-lookback_days:]
        outputs = [
            self.classify(
                as_of_date=day,
                market_data=market_data,
                breadth_data=breadth_data,
                vix_data=vix_data,
                event_calendar=event_calendar,
                config=config,
            )
            for day in selected
        ]
        cfg = config if config is not None else self._config
        return RegimeTimeline(
            engine_version=engine_version(),
            config_version=cfg.config_version,
            market="SPY",
            start_date=selected[0],
            end_date=selected[-1],
            trading_calendar=cfg.trading_calendar,
            outputs=outputs,
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
    has_spy_asof = ((df["symbol"] == "SPY") & (dates == as_of_date)).any()
    if not bool(has_spy_asof):
        raise ValueError(f"market_data must include SPY row for as_of_date={as_of_date.isoformat()}")

    # V1 requires NYSE-session aligned dates only. Reject non-trading-day rows to prevent
    # distortion of trading-day lookbacks.
    # IMPORTANT: do this in one calendar query for the full range; checking day-by-day is too slow.
    uniq_dates = sorted({d for d in dates.dropna().unique()})
    if uniq_dates:
        from regime_detection.calendar import nyse_calendar

        start = min(uniq_dates)
        end = max(uniq_dates)
        sessions = nyse_calendar().schedule(start_date=start, end_date=end).index.date
        session_set = set(sessions)
        bad_dates = [d for d in uniq_dates if d not in session_set]
        if bad_dates:
            raise ValueError(
                "market_data contains non-NYSE session dates (forbidden in V1). "
                f"Examples: {bad_dates[:5]}"
            )


def _spy_ohlcv_frame(df: pd.DataFrame, *, as_of_date: date) -> pd.DataFrame:
    s = df[df["symbol"] == "SPY"].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    s = s.set_index("date")
    return s[["open", "high", "low", "close", "volume"]]


def _symbol_close_series(df: pd.DataFrame, *, symbol: str, as_of_date: date) -> pd.Series:
    s = df[df["symbol"] == symbol].copy()
    if s.empty:
        raise ValueError(f"market_data missing required symbol for V1: {symbol}")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")
    s = s[s["date"].dt.date <= as_of_date]
    out = pd.Series(s["close"].to_numpy(), index=pd.to_datetime(s["date"]))
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
        position_size_multiplier=0.75,
        allow_trend_following=True,
        allow_mean_reversion=True,
        leverage_allowed=False,
        allow_buy_dip=True,
        allow_breakout=True,
        allow_shorts=True,
        require_confirmation_for_new_longs=True,
        require_confirmation_for_shorts=True,
        log_for_review=True,
        reason="unknown_or_unmapped_regime",
        modifiers_applied=[],
    )


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


def _axis_history(
    *,
    as_of_date: date,
    spy_ohlcv: pd.DataFrame,
    rsp_close: pd.Series,
    vix_proxy_close: pd.Series | None,
    event_calendar: pd.DataFrame | None,
    cfg: RegimeConfig,
) -> list[dict[str, str]]:
    index = spy_ohlcv.index
    history_sessions = list(index.date)[-60:]
    td_features = compute_trend_direction_features(spy_ohlcv["close"])
    td_raw = [trend_direction_raw_label_for_day(td_features, day)[0] for day in index]
    td_stable, td_active = apply_trend_direction_hysteresis(
        dates=index,
        raw_labels=td_raw,
        deescalation_days=cfg.hysteresis.trend_direction_deescalation_days,
    )

    tc_features = compute_trend_character_features(
        close=spy_ohlcv["close"],
        high=spy_ohlcv["high"],
        low=spy_ohlcv["low"],
    )
    tc_raw = [trend_character_raw_label_for_day(tc_features, day)[0] for day in index]
    tc_stable, tc_active = apply_asymmetric_hysteresis(
        raw_labels=tc_raw,
        risk_rank=TREND_CHARACTER_RISK_RANK,
        deescalation_days=cfg.hysteresis.trend_character_deescalation_days,
    )

    vs_features = compute_volatility_features(close=spy_ohlcv["close"], vix_proxy_close=vix_proxy_close)
    vs_raw = [volatility_raw_label_for_day(vs_features, day)[0] for day in index]
    vs_stable, vs_active = apply_asymmetric_hysteresis(
        raw_labels=vs_raw,
        risk_rank=VOLATILITY_RISK_RANK,
        deescalation_days=cfg.hysteresis.volatility_deescalation_days,
    )

    bs_features = compute_breadth_features(spy_close=spy_ohlcv["close"], rsp_close=rsp_close.reindex(index))
    bs_raw = [breadth_raw_label_for_day(bs_features, day)[0] for day in index]
    bs_stable, bs_active = apply_asymmetric_hysteresis(
        raw_labels=bs_raw,
        risk_rank=BREADTH_RISK_RANK,
        deescalation_days=cfg.hysteresis.breadth_deescalation_days,
    )

    history: list[dict[str, str]] = []
    event_cache: dict[date, EventCalendarOutput] = {}
    day_to_position = {session_date: idx for idx, session_date in enumerate(index.date)}
    for day in history_sessions:
        pos = day_to_position[day]
        if day not in event_cache:
            event_cache[day] = classify_event_calendar(
                as_of_date=day,
                event_calendar=event_calendar,
                config=cfg,
            )
        ec = event_cache[day]
        history.append(
            {
                "trend_direction_stable": td_stable[pos],
                "trend_direction_active": td_active[pos],
                "trend_character_stable": tc_stable[pos],
                "trend_character_active": tc_active[pos],
                "volatility_state_stable": vs_stable[pos],
                "volatility_state_active": vs_active[pos],
                "breadth_state_stable": bs_stable[pos],
                "breadth_state_active": bs_active[pos],
                "event_calendar_stable": ec.stable_label,
                "event_calendar_active": ec.active_label,
            }
        )
    return history


def _history_metrics(history: list[dict[str, str]]) -> tuple[bool, int | None, list[str]]:
    if not history:
        return False, None, []
    trend_direction_stable_history = [row["trend_direction_stable"] for row in history]
    stable_changed_today = False
    last_switch_days_ago: int | None = None
    stable_keys = [
        "trend_direction_stable",
        "trend_character_stable",
        "volatility_state_stable",
        "breadth_state_stable",
    ]
    for idx in range(1, len(history)):
        changed = any(history[idx][key] != history[idx - 1][key] for key in stable_keys)
        if changed:
            days_ago = len(history) - 1 - idx
            last_switch_days_ago = days_ago
            if idx == len(history) - 1:
                stable_changed_today = True
    return stable_changed_today, last_switch_days_ago, trend_direction_stable_history
