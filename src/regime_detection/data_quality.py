from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.models import DataQuality


# spec §2.8: completeness < 0.70 → label=unknown, reason=insufficient_data.
# The 0.70 floor is a spec constant, not a function of min_completeness.
INSUFFICIENT_COMPLETENESS_FLOOR = 0.70


def assess_series_input_quality(
    *,
    as_of_date: date,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    raw_label: str,
    max_freshness_days: int,
    min_completeness: float,
    skip_raw_label_short_circuit: bool = False,
) -> DataQuality:
    """Assess quality of required input series at ``as_of_date``.

    When ``skip_raw_label_short_circuit=True``, the ``raw_label == "unknown"``
    branch is bypassed — callers who compute the raw label AFTER quality (e.g.
    V2 NetworkFragilitySeriesClassifier) want a pure-quality assessment of the
    inputs themselves; they then re-check with ``quality_forces_unknown`` and
    map back to ``unknown`` if rules don't fire. Default keeps the V1 semantics
    where a ``raw_label=='unknown'`` upstream signal forces an insufficient-
    history status.
    """
    dt = pd.Timestamp(as_of_date)
    dt_normalized = dt.normalize()
    windows = [
        _window_to_asof(series=series, as_of_date=dt, required_trading_days=required_trading_days)
        for series in required_inputs
    ]
    if any(len(window) < required_trading_days for window in windows):
        return DataQuality(
            status="insufficient_history",
            freshness_days=None,
            completeness=None,
            reason="required_feature_is_nan",
        )

    completeness = min(float(window.notna().mean()) for window in windows)
    freshness_days = max(
        _freshness_days(window=window, as_of_date_normalized=dt_normalized) for window in windows
    )

    if freshness_days > max_freshness_days:
        return DataQuality(
            status="stale_data",
            freshness_days=freshness_days,
            completeness=completeness,
            reason="stale_data",
        )
    if completeness < INSUFFICIENT_COMPLETENESS_FLOOR:
        return DataQuality(
            status="insufficient_data",
            freshness_days=freshness_days,
            completeness=completeness,
            reason="insufficient_data",
        )
    if raw_label == "unknown" and not skip_raw_label_short_circuit:
        return DataQuality(
            status="insufficient_history",
            freshness_days=None,
            completeness=None,
            reason="required_feature_is_nan",
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
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing:
        # Hot path: avoid label slicing over the entire prefix on every call.
        # searchsorted + iloc keeps the same trailing required_trading_days
        # semantics while operating on integer bounds only.
        end = idx.searchsorted(as_of_date, side="right")
        start = max(0, end - required_trading_days)
        return series.iloc[start:end]
    # Slow path: legacy callers with non-datetime or unsorted indexes. Behavior
    # is byte-identical to the prior implementation.
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.loc[:as_of_date].tail(required_trading_days)


def _freshness_days(*, window: pd.Series, as_of_date_normalized: pd.Timestamp) -> int:
    last_valid = window.last_valid_index()
    if last_valid is None:
        return 10**9
    return int((as_of_date_normalized - pd.Timestamp(last_valid).normalize()).days)
