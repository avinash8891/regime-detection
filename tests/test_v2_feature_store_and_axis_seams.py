from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.axis_series import (
    NetworkFragilitySeriesClassifier,
    build_axis_series_bundle,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import NetworkFragilityFeatures, build_feature_store
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.loaders import load_sector_etf_closes
from regime_detection.market_context import build_market_context


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


def _build_context_with_sector_data(market_df_for_asof, as_of: date):
    sector_dates = pd.bdate_range("2022-06-01", end=as_of, freq="C").date.tolist()
    sector_df = _make_long_ohlcv(list(SECTOR_ETFS), sector_dates)
    sector_closes = load_sector_etf_closes(sector_df, universe=SECTOR_ETFS)
    return build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
        sector_etf_closes=sector_closes,
    )


# ---------- feature_store seam -----------------------------------------------


def test_feature_store_network_fragility_is_none_without_sector_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )

    store = build_feature_store(context)

    assert store.network_fragility is None


def test_feature_store_populates_network_fragility_stub_with_sector_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = _build_context_with_sector_data(market_df_for_asof, as_of)

    store = build_feature_store(context)

    assert store.network_fragility is not None
    assert isinstance(store.network_fragility, NetworkFragilityFeatures)


# ---------- axis classifier stub --------------------------------------------


def test_network_fragility_classifier_returns_none_without_sector_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    store = build_feature_store(context)

    result = NetworkFragilitySeriesClassifier().build(context, store)

    assert result is None


def test_network_fragility_classifier_returns_per_day_unknowns_with_sector_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = _build_context_with_sector_data(market_df_for_asof, as_of)
    store = build_feature_store(context)

    result = NetworkFragilitySeriesClassifier().build(context, store)

    assert result is not None
    assert set(result.keys()) == set(context.sessions)
    for day, output in result.items():
        assert output.raw_label == "unknown"
        assert output.stable_label == "unknown"
        assert output.active_label == "unknown"
        assert output.evidence == {"reason": "v2_classifier_not_yet_implemented"}
        assert output.data_quality.status == "insufficient_history"
        assert output.data_quality.reason == "required_feature_is_nan"
        assert output.mode == "sector_cross_asset_22"


# ---------- bundle wiring ---------------------------------------------------


def test_axis_bundle_network_fragility_is_none_in_pure_v1_mode(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=store)

    assert bundle.network_fragility is None


def test_axis_bundle_network_fragility_present_with_sector_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    context = _build_context_with_sector_data(market_df_for_asof, as_of)
    store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=store)

    assert bundle.network_fragility is not None
    assert len(bundle.network_fragility) == len(context.sessions)


# ---------- timeline integration --------------------------------------------


def test_timeline_emits_network_fragility_unknown_in_pure_v1_mode(market_df_for_asof) -> None:
    """Regression: without V2 data, network_fragility still emits the v2
    'unknown' placeholder shape locked in Phase C."""
    as_of = date(2023, 12, 14)
    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
    )

    assert out.network_fragility.raw_label == "unknown"
    assert out.network_fragility.stable_label == "unknown"
    assert out.network_fragility.active_label == "unknown"
    assert out.network_fragility.mode == "sector_cross_asset_22"


def test_timeline_pulls_network_fragility_from_axis_bundle_when_sector_data_present(
    market_df_for_asof,
) -> None:
    """When sector data is passed through, timeline.py reads from the
    AxisSeriesBundle entry (slice-1 hand-off seam)."""
    as_of = date(2023, 12, 14)
    sector_dates = pd.bdate_range("2022-06-01", end=as_of, freq="C").date.tolist()
    sector_df = _make_long_ohlcv(list(SECTOR_ETFS), sector_dates)
    sector_closes = load_sector_etf_closes(sector_df, universe=SECTOR_ETFS)

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        sector_etf_closes=sector_closes,
    )

    # Stub classifier still emits unknown, but the data path went through
    # the bundle (verified by the per-day evidence string being identical
    # to NetworkFragilitySeriesClassifier's output).
    assert out.network_fragility.evidence == {"reason": "v2_classifier_not_yet_implemented"}
    assert out.network_fragility.data_quality.status == "insufficient_history"
