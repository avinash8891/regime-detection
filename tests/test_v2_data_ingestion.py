from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.engine import RegimeEngine
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    INDEX_SYMBOL,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)
from regime_detection.loaders import (
    load_aggregate_forward_eps_revision_series,
    load_cpi_nowcast_series,
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
    """v2 spec §3.1 ships 11 sector_etfs + 11 cross_asset_etfs (incl. SPY)
    = 22 assets. SPY is the index; the engine reads SPY's close from
    context.spy_ohlcv rather than from cross_asset_closes."""
    assert len(NETWORK_FRAGILITY_UNIVERSE) == 22
    assert len(set(NETWORK_FRAGILITY_UNIVERSE)) == 22  # no duplicates
    assert len(SECTOR_ETFS) == 11
    assert len(CROSS_ASSET_SYMBOLS) == 10  # 11 cross_asset_etfs minus SPY (the index)
    assert INDEX_SYMBOL == "SPY"
    assert "SPY" in NETWORK_FRAGILITY_UNIVERSE  # v2 §3.1 line 537
    assert "KRE" not in NETWORK_FRAGILITY_UNIVERSE  # KRE is slice 4 credit/funding
    assert "RSP" not in NETWORK_FRAGILITY_UNIVERSE  # RSP is V1 breadth proxy
    # Spec-exact set, per v2 §3.1 lines 524-547.
    assert set(NETWORK_FRAGILITY_UNIVERSE) == {
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "HYG", "LQD", "USO", "UUP",
    }


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


def _make_long_macro(
    series_ids: list[str],
    dates: list[date],
    *,
    logical_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        for i, d in enumerate(dates):
            row = {
                "date": pd.Timestamp(d),
                "series_id": sid,
                "value": 4.0 + 0.01 * i,
                "realtime_start": pd.Timestamp(d),
                "realtime_end": pd.Timestamp(d),
            }
            if logical_names is not None:
                row["logical_name"] = logical_names[sid]
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_macro_series_returns_series_id_when_no_logical_name() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    df = _make_long_macro(["DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"], dates)

    out = load_macro_series(df, series_ids=("DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"))

    # Without logical_name column, series_id is the canonical key — no aliases.
    assert {"DGS2", "DGS10", "DTWEXBGS", "SOFR", "NFCI"}.issubset(out)
    assert "dgs2" not in out
    assert "dgs10" not in out
    assert len(out["DGS2"]) == 2
    assert out["DGS2"].iloc[0] == 4.0


def test_load_macro_series_uses_logical_name_as_sole_key() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    logical_names = {
        "DGS2": "2y_yield",
        "DGS10": "10y_yield",
        "DTWEXBGS": "broad_usd_index",
        "BAMLH0A0HYM2": "hy_oas",
        "BAMLC0A4CBBB": "ig_bbb_oas",
        "CPIAUCSL": "cpi_all_items",
    }
    df = _make_long_macro(
        list(logical_names),
        dates,
        logical_names=logical_names,
    )

    out = load_macro_series(df)

    # logical_name is the sole key — series_id and legacy aliases absent.
    assert {"2y_yield", "10y_yield", "broad_usd_index", "hy_oas", "ig_bbb_oas", "cpi_all_items"}.issubset(out)
    assert "DGS2" not in out
    assert "DGS10" not in out
    assert "dgs2" not in out
    assert "dgs10" not in out
    assert out["hy_oas"].iloc[0] == 4.0
    assert out["2y_yield"].iloc[0] == 4.0


def test_load_macro_series_rejects_missing_series_id() -> None:
    df = _make_long_macro(["DGS2"], [date(2024, 1, 2)])

    with pytest.raises(ValueError, match=r"Source missing required series_ids.*DGS10"):
        load_macro_series(df, series_ids=("DGS2", "DGS10"))


# ---------- load_cpi_nowcast_series (v2 §2B / ADR 0006) ----------------------


def test_load_cpi_nowcast_series_returns_sorted_date_indexed_series() -> None:
    """Wide-form (date, cpi_nowcast) -> a single date-indexed Series, sorted
    ascending, named for the macro_series key the feature store reads."""
    df = pd.DataFrame(
        {
            # deliberately out of order to prove the loader sorts
            "date": [
                pd.Timestamp("2026-03-01"),
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-02-01"),
            ],
            "cpi_nowcast": [0.008441, 0.001350, 0.002486],
        }
    )

    out = load_cpi_nowcast_series(df)

    assert out.name == "cpi_nowcast"
    assert list(out.index) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
    ]
    assert out.loc[pd.Timestamp("2026-03-01")] == 0.008441


def test_load_cpi_nowcast_series_rejects_missing_column() -> None:
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [0.0013]})

    with pytest.raises(
        ValueError, match=r"cpi_nowcast source missing required columns.*cpi_nowcast"
    ):
        load_cpi_nowcast_series(df)


# ---------- load_aggregate_forward_eps_revision_series (v2 §2B / Log #48) -----


def _make_eps_weekly_history(
    observation_dates: list[date], forward_eps: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": observation_dates,
            "observation_label": ["current"] * len(observation_dates),
            "forward_estimate_value": forward_eps,
            "source": ["S&P Global aggregate forward EPS workbook"]
            * len(observation_dates),
        }
    )


def test_load_aggregate_forward_eps_revision_series_hand_computed() -> None:
    """5 accumulated weekly rows -> the 5th carries the hand-computed
    4-week revision (fwd[t] - fwd[t-4]) / fwd[t-4]; rows 0-3 are cold-start
    NaN."""
    history = _make_eps_weekly_history(
        [
            date(2026, 1, 7),
            date(2026, 1, 14),
            date(2026, 1, 21),
            date(2026, 1, 28),
            date(2026, 2, 4),
        ],
        [270.00, 271.00, 272.00, 273.00, 277.40],
    )

    out = load_aggregate_forward_eps_revision_series(history)

    assert out.index[4] == pd.Timestamp("2026-02-04")
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx((277.40 - 270.00) / 270.00)


def test_load_aggregate_forward_eps_revision_series_cold_start_all_nan() -> None:
    """At or below the 4-week lookback the revision series is entirely NaN —
    the §2B earnings labels stay dark (V1 §2.7 cold-start)."""
    history = _make_eps_weekly_history(
        [date(2026, 1, 7), date(2026, 1, 14), date(2026, 1, 21), date(2026, 1, 28)],
        [270.00, 271.00, 272.00, 273.00],
    )

    out = load_aggregate_forward_eps_revision_series(history)

    assert out.isna().all()


def test_load_aggregate_forward_eps_revision_series_rejects_missing_column() -> None:
    df = pd.DataFrame(
        {"observation_date": [date(2026, 1, 7)], "observation_label": ["current"]}
    )

    with pytest.raises(
        ValueError,
        match=r"aggregate forward EPS source missing required columns.*forward_estimate_value",
    ):
        load_aggregate_forward_eps_revision_series(df)


# ---------- MarketContext propagation ----------------------------------------


def _build_v2_context(market_df_for_asof, as_of: date) -> tuple:
    sector_dates = pd.bdate_range("2022-06-01", end=as_of, freq="C").date.tolist()
    sector_df = _make_long_ohlcv(list(SECTOR_ETFS), sector_dates)
    cross_df = _make_long_ohlcv(list(CROSS_ASSET_SYMBOLS), sector_dates)
    macro_df = _make_long_macro(
        ["DGS2", "DGS10", "DTWEXBGS"],
        sector_dates,
        logical_names={"DGS2": "2y_yield", "DGS10": "10y_yield", "DTWEXBGS": "broad_usd_index"},
    )

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
    assert {"2y_yield", "10y_yield", "broad_usd_index"}.issubset(ctx.macro_series)
    assert "DGS2" not in ctx.macro_series
    assert "DGS10" not in ctx.macro_series


def test_market_context_reindexes_v2_series_to_spy_session_index(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    # Session-aligned series (ETFs, cross-asset) share the SPY OHLCV index.
    spy_index = ctx.spy_ohlcv.index
    for series in (ctx.sector_etf_closes or {}).values():
        assert series.index.equals(spy_index)
    for series in (ctx.cross_asset_closes or {}).values():
        assert series.index.equals(spy_index)
    # Macro series retain their original FRED observation index so that
    # pre-SPY history is preserved for z-score normalizer warmup (ADR-0008).
    for series in (ctx.macro_series or {}).values():
        assert len(series) > 0


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


def test_slice_context_to_recent_sessions_preserves_pit_breadth_seams(
    market_df_for_asof,
) -> None:
    """Regression: slice_context_to_recent_sessions used to drop
    pit_constituent_intervals + constituent_ohlcv on the rebuilt
    MarketContext, silently disabling §1D PIT breadth downstream."""
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    pit_intervals = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": date(1980, 12, 12),
                "end_date": None,
                "source": "test",
                "source_url": "test://",
                "bias_warning": "test_only",
            }
        ]
    )
    constituent_ohlcv = {
        "AAPL": pd.DataFrame(
            {
                "open": [195.0],
                "high": [196.0],
                "low": [194.0],
                "close": [195.0],
                "volume": [1_000_000],
                "adjusted_close": [195.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp(as_of)], name="date"),
        ),
    }
    ctx_with_pit = ctx.model_copy(
        update={
            "pit_constituent_intervals": pit_intervals,
            "constituent_ohlcv": constituent_ohlcv,
        }
    )

    sliced = slice_context_to_recent_sessions(
        context=ctx_with_pit, required_sessions=30
    )

    assert sliced.pit_constituent_intervals is not None
    assert sliced.constituent_ohlcv is not None
    assert list(sliced.pit_constituent_intervals["ticker"]) == ["AAPL"]
    assert set(sliced.constituent_ohlcv.keys()) == {"AAPL"}


def test_slice_context_to_end_date_preserves_pit_breadth_seams(
    market_df_for_asof,
) -> None:
    """Regression: slice_context_to_end_date used to drop
    pit_constituent_intervals + constituent_ohlcv on the rebuilt
    MarketContext, silently disabling §1D PIT breadth downstream."""
    as_of = date(2023, 12, 14)
    ctx, _, _, _ = _build_v2_context(market_df_for_asof, as_of)

    pit_intervals = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "start_date": date(1986, 3, 13),
                "end_date": None,
                "source": "test",
                "source_url": "test://",
                "bias_warning": "test_only",
            }
        ]
    )
    constituent_ohlcv = {
        "MSFT": pd.DataFrame(
            {
                "open": [370.0],
                "high": [371.0],
                "low": [369.0],
                "close": [370.0],
                "volume": [1_000_000],
                "adjusted_close": [370.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp(as_of)], name="date"),
        ),
    }
    ctx_with_pit = ctx.model_copy(
        update={
            "pit_constituent_intervals": pit_intervals,
            "constituent_ohlcv": constituent_ohlcv,
        }
    )

    earlier = date(2023, 12, 1)
    sliced = slice_context_to_end_date(context=ctx_with_pit, end_date=earlier)

    assert sliced.pit_constituent_intervals is not None
    assert sliced.constituent_ohlcv is not None
    assert list(sliced.pit_constituent_intervals["ticker"]) == ["MSFT"]
    assert set(sliced.constituent_ohlcv.keys()) == {"MSFT"}


def test_engine_classify_threads_pit_constituent_inputs_into_context(
    market_df_for_asof,
) -> None:
    """Regression: RegimeEngine.classify must accept pit_constituent_intervals
    + constituent_ohlcv kwargs so PIT §1D breadth seams are reachable from
    the public engine entrypoint, not only via direct build_market_context."""
    as_of = date(2023, 12, 14)

    pit_intervals = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": date(1980, 12, 12),
                "end_date": None,
                "source": "test",
                "source_url": "test://",
                "bias_warning": "test_only",
            }
        ]
    )
    constituent_ohlcv = {
        "AAPL": pd.DataFrame(
            {
                "open": [195.0],
                "high": [196.0],
                "low": [194.0],
                "close": [195.0],
                "volume": [1_000_000],
                "adjusted_close": [195.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp(as_of)], name="date"),
        ),
    }

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        pit_constituent_intervals=pit_intervals,
        constituent_ohlcv=constituent_ohlcv,
    )

    # V1 wire fields remain stable when V2 PIT kwargs are passed (the §1D
    # PIT-aware breadth output is opt-in and does not change V1 ETF-proxy
    # breadth_state).
    assert out.breadth_state.active_label is not None


def test_engine_classify_threads_aaii_sentiment_into_context(
    market_df_for_asof,
) -> None:
    """Regression: RegimeEngine.classify must accept aaii_sentiment kwarg so
    the v2 §1A `euphoria` predicate (ADR 0004 / Log #32 closure) is reachable
    from the public engine entrypoint. AAII rows carry publication_date so
    the per-session forward-fill respects V1 §2.2 stateless-replay."""
    as_of = date(2023, 12, 14)

    # Two weekly AAII rows, both with publication_date <= as_of, so the
    # session at as_of inherits the most recent one's bull_bear_spread_8w_ma.
    aaii_sentiment = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2023-12-07"),
                "publication_date": pd.Timestamp("2023-12-07"),
                "bullish": 0.45,
                "neutral": 0.30,
                "bearish": 0.25,
                "bull_bear_spread": 20.0,
                "bull_bear_spread_8w_ma": 18.0,
            },
            {
                "date": pd.Timestamp("2023-12-14"),
                "publication_date": pd.Timestamp("2023-12-14"),
                "bullish": 0.50,
                "neutral": 0.25,
                "bearish": 0.25,
                "bull_bear_spread": 25.0,
                "bull_bear_spread_8w_ma": 22.0,
            },
        ]
    )

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        aaii_sentiment=aaii_sentiment,
    )

    # V1 wire fields remain stable when the new V2 aaii_sentiment kwarg is
    # passed — euphoria firing depends on three additional non-sentiment
    # conjuncts (close > SMA_200, return_126d > 0.20, vol-rising). We
    # assert only that the engine accepts the kwarg end-to-end and emits
    # a valid trend_direction output.
    assert out.trend_direction.active_label is not None


def test_build_market_context_rejects_malformed_date_values(market_df_for_asof) -> None:
    """Regression: _normalize_market_data_for_runtime used pd.to_datetime
    with errors='coerce', silently turning malformed date strings into NaT
    which the validator's dropna() then swept under the rug. Bad-date rows
    must fail loud at the ingestion boundary."""
    as_of = date(2023, 12, 14)
    good_df = market_df_for_asof(as_of)

    # Inject one row with a malformed date string. pd.to_datetime(errors='raise')
    # will reject the whole frame at normalization time.
    bad_row = good_df.iloc[0].copy()
    bad_row["date"] = "not-a-date-at-all"
    corrupted = pd.concat(
        [good_df, pd.DataFrame([bad_row])], ignore_index=True
    )

    with pytest.raises(ValueError, match=r"malformed date"):
        RegimeEngine().classify(
            as_of_date=as_of,
            market_data=corrupted,
        )


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
