from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.models import DataQuality


def assess_series_input_quality(
    *,
    as_of_date: date,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    raw_label: str,
    unknown_reason: str,
    max_freshness_days: int,
    min_completeness: float,
) -> DataQuality:
    dt = pd.Timestamp(as_of_date)
    windows = [_window_to_asof(series=series, as_of_date=dt, required_trading_days=required_trading_days) for series in required_inputs]
    if any(len(window) < required_trading_days for window in windows):
        return DataQuality(
            status="insufficient_history",
            freshness_days=None,
            completeness=None,
            reason=unknown_reason,
        )

    completeness = min(float(window.notna().mean()) for window in windows)
    freshness_days = max(_freshness_days(window=window, as_of_date=dt) for window in windows)
    insufficient_threshold = max(0.0, min_completeness - 0.20)

    if freshness_days > max_freshness_days:
        return DataQuality(
            status="stale_data",
            freshness_days=freshness_days,
            completeness=completeness,
            reason="stale_data",
        )
    if completeness < insufficient_threshold:
        return DataQuality(
            status="insufficient_data",
            freshness_days=freshness_days,
            completeness=completeness,
            reason="insufficient_data",
        )
    if raw_label == "unknown":
        return DataQuality(
            status="insufficient_history",
            freshness_days=None,
            completeness=None,
            reason=unknown_reason,
        )
    if completeness < min_completeness:
        return DataQuality(
            status="degraded",
            freshness_days=freshness_days,
            completeness=completeness,
            reason="incomplete_data",
        )
    return DataQuality(
        status="ok",
        freshness_days=freshness_days,
        completeness=completeness,
        reason=None,
    )


def quality_forces_unknown(dq: DataQuality) -> bool:
    return dq.status in {"insufficient_data", "insufficient_history", "stale_data"}


def _window_to_asof(*, series: pd.Series, as_of_date: pd.Timestamp, required_trading_days: int) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.loc[:as_of_date].tail(required_trading_days)


def _freshness_days(*, window: pd.Series, as_of_date: pd.Timestamp) -> int:
    non_null = window.dropna()
    if non_null.empty:
        return 10**9
    last_valid = pd.Timestamp(non_null.index[-1])
    return int((as_of_date.normalize() - last_valid.normalize()).days)
