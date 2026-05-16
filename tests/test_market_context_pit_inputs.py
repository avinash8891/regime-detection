"""TDD RED tests for V2 Slice 2.8b — MarketContext PIT input scaffolding.

Pins the seam by which slice 2.8c will plumb PIT constituent intervals and
constituent OHLCV through ``build_market_context``. The two new fields are
optional (default ``None``) and must round-trip unchanged — NO reindexing
against the SPY session index (PIT intervals are date-range rows, not a
session-indexed series; reindexing would corrupt them).

Spec refs:
    docs/regime_engine_v2_spec.md §1D PIT breadth seam.
    Implementation Ambiguity Log #54–#59 (PIT breadth resolutions).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.engine import RegimeEngine
from regime_detection.market_context import MarketContext, build_market_context


def _aapl_pit_intervals_df() -> pd.DataFrame:
    """One-row PIT intervals frame matching the writer schema (real ticker)."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": date(1980, 12, 12),
                "end_date": date(2024, 12, 31),
                "source": "fja05680/sp500",
                "source_url": "https://github.com/fja05680/sp500",
                "bias_warning": "survivorship_biased_constituent_universe",
            }
        ]
    )


def _ohlcv_for_ticker(ticker: str) -> pd.DataFrame:
    """Small but real-shaped OHLCV frame (date-indexed) for a single ticker."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
            "adjusted_close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=idx,
    ).rename_axis("date")


def _minimal_required_context_kwargs() -> dict:
    """Build the minimum kwargs needed to construct a MarketContext directly."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    spy_ohlcv = pd.DataFrame(
        {
            "open": [470.0, 471.0, 472.0, 473.0, 474.0],
            "high": [472.0, 473.0, 474.0, 475.0, 476.0],
            "low": [469.0, 470.0, 471.0, 472.0, 473.0],
            "close": [471.0, 472.0, 473.0, 474.0, 475.0],
            "volume": [80_000_000] * 5,
        },
        index=idx,
    )
    rsp_close = pd.Series([160.0, 161.0, 162.0, 163.0, 164.0], index=idx, name="close")
    return {
        "end_date": idx[-1].date(),
        "config": RegimeEngine().config,
        "sessions": tuple(idx.date),
        "spy_ohlcv": spy_ohlcv,
        "rsp_close": rsp_close,
        "vix_proxy_close": None,
    }


# -----------------------------------------------------------------------------
# Direct-construct tests (drive the new optional fields onto MarketContext).
# -----------------------------------------------------------------------------


def test_market_context_accepts_pit_constituent_intervals() -> None:
    intervals = _aapl_pit_intervals_df()
    ctx = MarketContext(
        **_minimal_required_context_kwargs(),
        pit_constituent_intervals=intervals,
    )
    # Round-trips unchanged (same shape, same values, no reindex).
    assert ctx.pit_constituent_intervals is intervals
    assert ctx.pit_constituent_intervals.shape == (1, 6)
    assert ctx.pit_constituent_intervals.loc[0, "ticker"] == "AAPL"
    assert ctx.pit_constituent_intervals.loc[0, "start_date"] == date(1980, 12, 12)


def test_market_context_accepts_constituent_ohlcv() -> None:
    aapl = _ohlcv_for_ticker("AAPL")
    msft = _ohlcv_for_ticker("MSFT")
    ohlcv = {"AAPL": aapl, "MSFT": msft}
    ctx = MarketContext(
        **_minimal_required_context_kwargs(),
        constituent_ohlcv=ohlcv,
    )
    assert ctx.constituent_ohlcv is ohlcv
    assert set(ctx.constituent_ohlcv.keys()) == {"AAPL", "MSFT"}
    # Per-ticker frame round-trips exactly (no reindex onto SPY sessions).
    pd.testing.assert_frame_equal(ctx.constituent_ohlcv["AAPL"], aapl)
    pd.testing.assert_frame_equal(ctx.constituent_ohlcv["MSFT"], msft)


def test_market_context_defaults_pit_inputs_to_none() -> None:
    ctx = MarketContext(**_minimal_required_context_kwargs())
    assert ctx.pit_constituent_intervals is None
    assert ctx.constituent_ohlcv is None


# -----------------------------------------------------------------------------
# build_market_context plumbing tests (mirror conftest fixture pattern).
# -----------------------------------------------------------------------------


def _make_long_ohlcv(symbols, dates) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "symbol": sym,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_build_market_context_plumbs_pit_constituent_intervals(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    intervals = _aapl_pit_intervals_df()

    ctx = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
        pit_constituent_intervals=intervals,
    )

    # Must be plumbed through unchanged — NO reindexing against SPY sessions.
    assert ctx.pit_constituent_intervals is intervals
    pd.testing.assert_frame_equal(ctx.pit_constituent_intervals, intervals)


def test_build_market_context_plumbs_constituent_ohlcv(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    aapl = _ohlcv_for_ticker("AAPL")
    msft = _ohlcv_for_ticker("MSFT")
    ohlcv = {"AAPL": aapl, "MSFT": msft}

    ctx = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
        constituent_ohlcv=ohlcv,
    )

    assert ctx.constituent_ohlcv is not None
    assert set(ctx.constituent_ohlcv.keys()) == {"AAPL", "MSFT"}
    # Equality on per-ticker frames; explicitly NOT reindexed to SPY sessions.
    pd.testing.assert_frame_equal(ctx.constituent_ohlcv["AAPL"], aapl)
    pd.testing.assert_frame_equal(ctx.constituent_ohlcv["MSFT"], msft)


def test_build_market_context_rejects_constituent_ohlcv_missing_required_column(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    bad_aapl = _ohlcv_for_ticker("AAPL").drop(columns=["adjusted_close"])

    with pytest.raises(
        ValueError,
        match=r"constituent_ohlcv frame missing required columns.*ticker='AAPL'.*adjusted_close",
    ):
        build_market_context(
            end_date=as_of,
            market_data=market_df_for_asof(as_of),
            config=RegimeEngine().config,
            constituent_ohlcv={"AAPL": bad_aapl},
        )


def test_build_market_context_rejects_constituent_ohlcv_without_datetime_index(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    bad_msft = _ohlcv_for_ticker("MSFT").reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match=r"constituent_ohlcv frame must use a DatetimeIndex date index.*ticker='MSFT'",
    ):
        build_market_context(
            end_date=as_of,
            market_data=market_df_for_asof(as_of),
            config=RegimeEngine().config,
            constituent_ohlcv={"MSFT": bad_msft},
        )
