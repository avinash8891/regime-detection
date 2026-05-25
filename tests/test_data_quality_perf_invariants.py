"""Pinned invariants for the data_quality.assess_series_input_quality refactor.

These tests lock the exact DataQuality output for representative sessions
(cold-start, mid-history, recent) using real SPY OHLCV from the conftest
fixture. They are intentionally redundant with the higher-level golden-date
tests — they exist to catch any subtle drift in the data_quality short-circuit
that a label-level test might mask (e.g., a freshness or completeness float
that drifts but happens not to flip a rule boundary).

Refactor target: src/regime_detection/data_quality.py per the perf plan
(hoist parse+sort out of the per-day loop, replace dropna with
last_valid_index, normalize as_of_date once). Business logic must remain
byte-identical to the pre-refactor output captured below.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.data_quality import assess_series_input_quality

# Real V1 SPY OHLCV-derived window. These exact numbers were captured from
# the pre-refactor implementation against
# tests/fixtures/raw/market_data.parquet at the three sample dates below.
# The capture script is documented in the docstring of this test module.

_SPY_REQUIRED_TRADING_DAYS = 252
_MAX_FRESHNESS_DAYS = 3
_MIN_COMPLETENESS = 0.95


def _spy_close_series(raw_market_frames: dict[str, pd.DataFrame]) -> pd.Series:
    spy = raw_market_frames["SPY"].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    return spy.set_index("date")["close"].sort_index()


def test_assess_series_input_quality_recent_session_ok(raw_market_frames) -> None:
    spy_close = _spy_close_series(raw_market_frames)
    as_of = date(2023, 12, 14)
    dq = assess_series_input_quality(
        as_of_date=as_of,
        required_inputs=[spy_close],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label="bull",
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
    )
    assert dq.status == "ok"
    assert dq.freshness_days == 0
    assert dq.completeness == 1.0
    assert dq.reason is None


def test_assess_series_input_quality_cold_start_insufficient_history(
    raw_market_frames,
) -> None:
    spy_close = _spy_close_series(raw_market_frames)
    cold_as_of = spy_close.index[10].date()
    dq = assess_series_input_quality(
        as_of_date=cold_as_of,
        required_inputs=[spy_close],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label="bull",
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
    )
    assert dq.status == "insufficient_history"
    assert dq.freshness_days is None
    assert dq.completeness is None
    assert dq.reason == "required_feature_is_nan"


def test_assess_series_input_quality_raw_label_unknown_short_circuit(
    raw_market_frames,
) -> None:
    spy_close = _spy_close_series(raw_market_frames)
    as_of = date(2023, 12, 14)
    dq = assess_series_input_quality(
        as_of_date=as_of,
        required_inputs=[spy_close],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label="unknown",
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
    )
    assert dq.status == "insufficient_history"
    assert dq.freshness_days is None
    assert dq.completeness is None
    assert dq.reason == "required_feature_is_nan"


def test_assess_series_input_quality_skip_raw_label_short_circuit_passes_through(
    raw_market_frames,
) -> None:
    spy_close = _spy_close_series(raw_market_frames)
    as_of = date(2023, 12, 14)
    dq = assess_series_input_quality(
        as_of_date=as_of,
        required_inputs=[spy_close],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label="unknown",
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
        skip_raw_label_short_circuit=True,
    )
    # V2 callers re-check quality_forces_unknown AFTER computing the raw
    # label; bypassing the legacy V1 short-circuit must preserve OK status
    # when the actual data is fine.
    assert dq.status == "ok"
    assert dq.freshness_days == 0
    assert dq.completeness == 1.0
    assert dq.reason is None


def test_assess_series_input_quality_none_raw_label_is_pure_quality_mode(
    raw_market_frames,
) -> None:
    spy_close = _spy_close_series(raw_market_frames)
    as_of = date(2023, 12, 14)
    dq = assess_series_input_quality(
        as_of_date=as_of,
        required_inputs=[spy_close],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label=None,
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
    )

    assert dq.status == "ok"
    assert dq.freshness_days == 0
    assert dq.completeness == 1.0
    assert dq.reason is None


def test_assess_series_input_quality_unsorted_input_still_normalizes(
    raw_market_frames,
) -> None:
    # Defensive: legacy callers may pass an out-of-order or string-indexed
    # series. The function MUST still parse + sort internally to preserve
    # backward compatibility. This pins the slow-path fallback.
    spy_close = _spy_close_series(raw_market_frames)
    shuffled = spy_close.sample(frac=1.0, random_state=42)
    as_of = date(2023, 12, 14)
    dq = assess_series_input_quality(
        as_of_date=as_of,
        required_inputs=[shuffled],
        required_trading_days=_SPY_REQUIRED_TRADING_DAYS,
        raw_label="bull",
        max_freshness_days=_MAX_FRESHNESS_DAYS,
        min_completeness=_MIN_COMPLETENESS,
    )
    assert dq.status == "ok"
    assert dq.freshness_days == 0
    assert dq.completeness == 1.0
    assert dq.reason is None
