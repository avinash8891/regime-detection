from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.axis_series import (
    build_network_fragility_axis_series,
    build_axis_series_bundle,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import NetworkFragilityFeatures, build_feature_store
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)
from regime_detection.market_context import build_market_context


_REAL_V2_AS_OF = date(2026, 5, 13)


def _build_context_with_real_v2_universe(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    as_of: date = _REAL_V2_AS_OF,
):
    missing = sorted(
        set(NETWORK_FRAGILITY_UNIVERSE).difference(v2_close_series_by_symbol)
    )
    if missing:
        raise AssertionError(f"real V2 OHLCV fixture missing symbols: {missing}")

    return build_market_context(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        config=RegimeEngine().config,
        sector_etf_closes={s: v2_close_series_by_symbol[s] for s in SECTOR_ETFS},
        cross_asset_closes={
            s: v2_close_series_by_symbol[s] for s in CROSS_ASSET_SYMBOLS
        },
    )


# ---------- feature_store seam -----------------------------------------------


def test_feature_store_network_fragility_is_none_without_sector_data(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )

    store = build_feature_store(context)

    assert store.network_fragility is None


def test_feature_store_populates_network_fragility_with_real_v2_universe(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
) -> None:
    context = _build_context_with_real_v2_universe(
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
    )

    store = build_feature_store(context)

    assert store.network_fragility is not None
    assert isinstance(store.network_fragility, NetworkFragilityFeatures)
    assert store.network_fragility.largest_eigenvalue_share_percentile_504d.loc[
        pd.Timestamp(_REAL_V2_AS_OF)
    ] == pytest.approx(0.8591269841269841)


# ---------- axis classifier stub --------------------------------------------


def test_network_fragility_classifier_returns_none_without_sector_data(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    store = build_feature_store(context)

    result = build_network_fragility_axis_series(context, store)

    assert result is None


def test_network_fragility_classifier_returns_real_fixture_outputs(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
) -> None:
    """Slice 1.4: with tracked real V2 universe data the classifier emits
    deterministic per-day outputs from the v2 §3.3 label set."""
    context = _build_context_with_real_v2_universe(
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
    )
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )

    result = build_network_fragility_axis_series(context, store)

    assert result is not None
    assert set(result.keys()) == set(context.sessions)
    allowed_labels = {
        "diversified_normal",
        "stock_picker_dispersion",
        "rising_fragility",
        "correlation_concentration",
        "correlation_to_one",
        "systemic_stress",
        "unknown",
    }
    for output in result.values():
        assert output.raw_label in allowed_labels
        assert output.stable_label in allowed_labels
        assert output.active_label in allowed_labels
        assert output.mode == "sector_cross_asset_22"
    assert result[_REAL_V2_AS_OF].active_label == "correlation_concentration"


# ---------- bundle wiring ---------------------------------------------------


def test_axis_bundle_network_fragility_is_none_in_pure_v1_mode(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=store)

    assert bundle.network_fragility is None


def test_axis_bundle_network_fragility_present_with_real_v2_universe(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
) -> None:
    context = _build_context_with_real_v2_universe(
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
    )
    store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=store)

    assert bundle.network_fragility is not None
    assert len(bundle.network_fragility) == len(context.sessions)
    assert (
        bundle.network_fragility[_REAL_V2_AS_OF].active_label
        == "correlation_concentration"
    )


# ---------- timeline integration --------------------------------------------


def test_timeline_emits_network_fragility_unknown_in_pure_v1_mode(
    market_df_for_asof,
) -> None:
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
    real_v2_classify_window_2026_05_13,
) -> None:
    """When sector data is passed through, timeline.py reads from the
    AxisSeriesBundle entry (slice-1 hand-off seam). ``classify(as_of, ...)``
    is equivalent to ``classify_window(end_date=as_of,
    lookback_days=1, ...).outputs[-1]`` per
    ``test_classify_delegates_to_classify_window_with_single_day_lookback``;
    we use the cross-worker cached timeline (see conftest).
    """
    out = real_v2_classify_window_2026_05_13.outputs[-1]
    assert out.as_of_date == _REAL_V2_AS_OF

    assert (
        "v2_classifier_not_yet_implemented"
        not in out.network_fragility.evidence.get("reason", "")
    )
    assert out.network_fragility.mode == "sector_cross_asset_22"
    assert out.network_fragility.active_label == "correlation_concentration"
    assert out.network_fragility.evidence["rule_evidence"][
        "largest_eigenvalue_share_percentile_504d"
    ] == pytest.approx(0.8591269841269841)
