from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.engine import RegimeEngine
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    NETWORK_FRAGILITY_UNIVERSE,
    REGIONAL_BANKS_SYMBOL,
    SECTOR_ETFS,
)
from regime_detection.loaders import (
    load_cross_asset_closes,
    load_macro_series,
    load_sector_etf_closes,
)
from regime_detection.market_context import (
    build_market_context,
    slice_context_to_end_date,
    slice_context_to_recent_sessions,
)


# ---------- universe constants -----------------------------------------------


def test_network_fragility_universe_is_22_symbols_per_v2_section_3_1() -> None:
    assert len(NETWORK_FRAGILITY_UNIVERSE) == 22
    assert len(set(NETWORK_FRAGILITY_UNIVERSE)) == 22  # no duplicates
    assert len(SECTOR_ETFS) == 11
    assert len(CROSS_ASSET_SYMBOLS) == 10
    assert REGIONAL_BANKS_SYMBOL == "KRE"
    assert "SPY" not in NETWORK_FRAGILITY_UNIVERSE  # SPY is the index, not in the universe
    assert "RSP" not in NETWORK_FRAGILITY_UNIVERSE  # RSP is V1 breadth proxy


# ---------- load_sector_etf_closes -------------------------------------------


def _make_long_ohlcv(symbols: list[str], dates: list[date]) -> pd.DataFrame:
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


def test_load_sector_etf_closes_from_dataframe_returns_one_series_per_symbol() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    df = _make_long_ohlcv(list(SECTOR_ETFS), dates)

    out = load_sector_etf_closes(df, universe=SECTOR_ETFS)

    assert set(out.keys()) == set(SECTOR_ETFS)
    for sym, series in out.items():
        assert len(series) == 3
        assert series.iloc[0] == 100.5
        assert series.index[0] == pd.Timestamp(date(2024, 1, 2))


def test_load_sector_etf_closes_rejects_missing_universe_symbol() -> None:
    dates = [date(2024, 1, 2)]
    df = _make_long_ohlcv(["XLB", "XLC"], dates)

    with pytest.raises(ValueError, match=r"Source missing required symbols.*XLE"):
        load_sector_etf_closes(df, universe=("XLB", "XLC", "XLE"))


def test_load_sector_etf_closes_without_universe_returns_all_symbols() -> None:
    df = _make_long_ohlcv(["XLB", "XLC"], [date(2024, 1, 2)])

    out = load_sector_etf_closes(df)

    assert set(out.keys()) == {"XLB", "XLC"}


def test_load_sector_etf_closes_raises_when_columns_missing() -> None:
    df = pd.DataFrame({"date": [date(2024, 1, 2)], "symbol": ["XLB"]})  # no `close`

    with pytest.raises(ValueError, match="Source missing required columns"):
        load_sector_etf_closes(df, universe=("XLB",))


# ---------- load_cross_asset_closes ------------------------------------------


def test_load_cross_asset_closes_returns_v2_universe() -> None:
    df = _make_long_ohlcv(list(CROSS_ASSET_SYMBOLS), [date(2024, 1, 2)])

    out = load_cross_asset_closes(df, universe=CROSS_ASSET_SYMBOLS)

    assert set(out.keys()) == set(CROSS_ASSET_SYMBOLS)


# ---------- load_macro_series ------------------------------------------------


def _make_long_macro(series_ids: list[str], dates: list[date]) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "series_id": sid,
                    "value": 4.0 + 0.01 * i,
                    "realtime_start": pd.Timestamp(d),
                    "realtime_end": pd.Timestamp(d),
                }
            )
    return pd.DataFrame(rows)


def test_load_macro_series_returns_v2_fred_ids() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    df = _make_long_macro(["DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"], dates)

    out = load_macro_series(df, series_ids=("DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"))

    assert set(out.keys()) == {"DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"}
    assert len(out["DGS2"]) == 2
    assert out["DGS2"].iloc[0] == 4.0


def test_load_macro_series_rejects_missing_series_id() -> None:
    df = _make_long_macro(["DGS2"], [date(2024, 1, 2)])

    with pytest.raises(ValueError, match=r"Source missing required series_ids.*DGS10"):
        load_macro_series(df, series_ids=("DGS2", "DGS10"))


# ---------- MarketContext propagation ----------------------------------------


def _build_v2_context(market_df_for_asof, as_of: date) -> tuple:
    sector_dates = pd.bdate_range("2022-06-01", end=as_of, freq="C").date.tolist()
    sector_df = _make_long_ohlcv(list(SECTOR_ETFS), sector_dates)
    cross_df = _make_long_ohlcv(list(CROSS_ASSET_SYMBOLS), sector_dates)
    macro_df = _make_long_macro(["DGS2", "DGS10", "DTWEXBGS"], sector_dates)

    sector_closes = load_sector_etf_closes(sector_df, universe=SECTOR_ETFS)
    cross_closes = load_cross_asset_closes(cross_df, universe=CROSS_ASSET_SYMBOLS)
    macros = load_macro_series(macro_df, series_ids=("DGS2", "DGS10", "DTWEXBGS"))

    market_data = market_df_for_asof(as_of)
    config = RegimeEngine().config

    ctx = build_market_context(
        end_date=as_of,
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_closes,
        cross_asset_closes=cross_closes,
        macro_series=macros,
    )
    return ctx, sector_closes, cross_closes, macros


def test_market_context_holds_v2_data_dicts(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    ctx, sector_closes, cross_closes, macros = _build_v2_context(market_df_for_asof, as_of)

    assert ctx.sector_etf_closes is not None
    assert set(ctx.sector_etf_closes.keys()) == set(SECTOR_ETFS)
    assert ctx.cross_asset_closes is not None
    assert set(ctx.cross_asset_closes.keys()) == set(CROSS_ASSET_SYMBOLS)
    assert ctx.macro_series is not None
    assert set(ctx.macro_series.keys()) == {"DGS2", "DGS10", "DTWEXBGS"}


def test_market_context_reindexes_v2_series_to_spy_session_index(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    # Every V2 series must share the SPY OHLCV index after build_market_context.
    spy_index = ctx.spy_ohlcv.index
    for series in (ctx.sector_etf_closes or {}).values():
        assert series.index.equals(spy_index)
    for series in (ctx.cross_asset_closes or {}).values():
        assert series.index.equals(spy_index)
    for series in (ctx.macro_series or {}).values():
        assert series.index.equals(spy_index)


def test_slice_context_to_recent_sessions_propagates_v2_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    sliced = slice_context_to_recent_sessions(context=ctx, required_sessions=30)

    assert sliced.sector_etf_closes is not None
    assert set(sliced.sector_etf_closes.keys()) == set(SECTOR_ETFS)
    spy_index = sliced.spy_ohlcv.index
    for series in sliced.sector_etf_closes.values():
        assert series.index.equals(spy_index)


def test_slice_context_to_end_date_propagates_v2_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    earlier = date(2023, 12, 1)
    sliced = slice_context_to_end_date(context=ctx, end_date=earlier)

    assert sliced.sector_etf_closes is not None
    assert sliced.cross_asset_closes is not None
    assert sliced.macro_series is not None
    spy_index = sliced.spy_ohlcv.index
    for series in sliced.sector_etf_closes.values():
        assert series.index.equals(spy_index)


# ---------- Engine threading -------------------------------------------------


def test_engine_classify_accepts_v2_data_kwargs_without_breaking_v1_output(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    sector_dates = pd.bdate_range("2022-06-01", end=as_of, freq="C").date.tolist()
    sector_df = _make_long_ohlcv(list(SECTOR_ETFS), sector_dates)
    sector_closes = load_sector_etf_closes(sector_df, universe=SECTOR_ETFS)

    # Engine accepts the V2 kwarg.
    out_with_v2 = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        sector_etf_closes=sector_closes,
    )
    out_without_v2 = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
    )

    # V1 wire fields are unchanged whether or not V2 data is passed —
    # the v2 fragility classifier hasn't shipped yet, so network_fragility
    # still emits the v2 "unknown" placeholder regardless.
    assert out_with_v2.network_fragility.active_label == "unknown"
    assert out_without_v2.network_fragility.active_label == "unknown"
    assert out_with_v2.trend_direction.model_dump() == out_without_v2.trend_direction.model_dump()
