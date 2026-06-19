from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.axis_series import (
    AxisSeriesBundle,
    build_network_fragility_axis_series,
    build_axis_series_bundle,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import (
    FeatureStore,
    NetworkFragilityFeatures,
    build_feature_store,
)
from regime_detection.market_context import MarketContext, build_market_context
from regime_detection.models import NetworkFragilityOutput

_REAL_V2_AS_OF = date(2026, 5, 13)


@dataclass(frozen=True)
class _RealV2NetworkFragilityArtifacts:
    context: MarketContext
    store: FeatureStore
    network_fragility_axis: dict[date, NetworkFragilityOutput] | None
    bundle: AxisSeriesBundle


def _build_context_with_real_v2_universe(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
    as_of: date = _REAL_V2_AS_OF,
):
    market_data = v2_market_df_for_asof(as_of)
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    return build_market_context(
        end_date=as_of,
        market_data=market_data,
        config=kwargs["config"],
        event_calendar=kwargs["event_calendar"],
        sector_etf_closes=kwargs["sector_etf_closes"],
        cross_asset_closes=kwargs["cross_asset_closes"],
        macro_series=kwargs["macro_series"],
        pit_constituent_intervals=kwargs["pit_constituent_intervals"],
        constituent_ohlcv=kwargs["constituent_ohlcv"],
        aaii_sentiment=kwargs["aaii_sentiment"],
        news_sentiment=kwargs["news_sentiment"],
        central_bank_text_releases=kwargs["central_bank_text_releases"],
        cpi_first_release=kwargs["cpi_first_release"],
    )


def _build_real_v2_network_fragility_artifacts(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> _RealV2NetworkFragilityArtifacts:
    context = _build_context_with_real_v2_universe(
        v2_market_df_for_asof,
        synthetic_v2_kwargs_for_market_data,
    )
    store = build_feature_store(context, **context.config.v2_feature_build_configs())
    network_fragility_axis = build_network_fragility_axis_series(context, store)
    bundle = build_axis_series_bundle(context=context, feature_store=store)
    return _RealV2NetworkFragilityArtifacts(
        context=context,
        store=store,
        network_fragility_axis=network_fragility_axis,
        bundle=bundle,
    )


@pytest.fixture(scope="session")
def real_v2_network_fragility_artifacts(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> _RealV2NetworkFragilityArtifacts:
    if worker_id == "master":
        return _build_real_v2_network_fragility_artifacts(
            v2_market_df_for_asof,
            synthetic_v2_kwargs_for_market_data,
        )

    shared_dir = Path(tmp_path_factory.getbasetemp()).parent
    cache_path = shared_dir / "real_v2_network_fragility_artifacts_2026_05_13.pkl"
    lock_path = shared_dir / "real_v2_network_fragility_artifacts_2026_05_13.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        result = _build_real_v2_network_fragility_artifacts(
            v2_market_df_for_asof,
            synthetic_v2_kwargs_for_market_data,
        )
        tmp = cache_path.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(result))
        tmp.replace(cache_path)
        return result
    except FileExistsError:
        pass

    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes())
        time.sleep(0.2)
    raise RuntimeError(
        "real_v2_network_fragility_artifacts build timed out waiting on "
        f"peer worker; cache_path={cache_path}"
    )


# ---------- feature_store seam -----------------------------------------------


def test_feature_store_network_fragility_fails_loudly_without_sector_data(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )

    with pytest.raises(RuntimeError, match="sentiment_score"):
        build_feature_store(context)


def test_feature_store_populates_network_fragility_with_real_v2_universe(
    real_v2_network_fragility_artifacts,
) -> None:
    store = real_v2_network_fragility_artifacts.store

    assert store.network_fragility is not None
    assert store.availability["network_fragility"].available is True
    assert store.availability["network_fragility"].reason == "populated"
    assert isinstance(store.network_fragility, NetworkFragilityFeatures)
    assert store.network_fragility.largest_eigenvalue_share_percentile_504d.loc[
        pd.Timestamp(_REAL_V2_AS_OF)
    ] == pytest.approx(0.97)


def test_feature_store_reports_configured_v2_seam_missing_inputs(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )

    with pytest.raises(RuntimeError, match="sentiment_score"):
        build_feature_store(
            context,
            monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        )


# ---------- axis classifier stub --------------------------------------------


def test_network_fragility_classifier_fails_loudly_without_sector_data(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    with pytest.raises(RuntimeError, match="sentiment_score"):
        build_feature_store(context)


def test_network_fragility_classifier_returns_real_fixture_outputs(
    real_v2_network_fragility_artifacts,
) -> None:
    """Slice 1.4: with tracked real V2 universe data the classifier emits
    deterministic per-day outputs from the v2 §3.3 label set."""
    context = real_v2_network_fragility_artifacts.context
    result = real_v2_network_fragility_artifacts.network_fragility_axis

    assert result is not None
    assert set(result.keys()) == set(context.sessions)
    allowed_labels = {
        "diversified_normal",
        "decorrelated_calm",
        "rotation_watch",
        "stock_picker_dispersion",
        "idiosyncratic_crisis",
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
        assert output.mode == "sector_cross_asset_24"
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
    with pytest.raises(RuntimeError, match="sentiment_score"):
        build_feature_store(context)


def test_axis_bundle_network_fragility_present_with_real_v2_universe(
    real_v2_network_fragility_artifacts,
) -> None:
    context = real_v2_network_fragility_artifacts.context
    bundle = real_v2_network_fragility_artifacts.bundle

    assert bundle.network_fragility is not None
    assert len(bundle.network_fragility) == len(context.sessions)
    assert (
        bundle.network_fragility[_REAL_V2_AS_OF].active_label
        == "correlation_concentration"
    )


# ---------- timeline integration --------------------------------------------


def test_timeline_emits_network_fragility_unknown_in_pure_v1_mode(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    """Default V2 timeline fails loudly when required V2 inputs are absent."""
    as_of = date(2023, 12, 14)
    with pytest.raises(ValueError) as excinfo:
        RegimeEngine().classify(
            as_of_date=as_of,
            market_data=market_df_for_asof(as_of),
            event_calendar=event_calendar_df,
        )
    message = str(excinfo.value)
    assert "ClassifyRequest missing configured V2 inputs" in message
    assert "network_fragility: sector_etf_closes" in message


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
    assert out.network_fragility.mode == "sector_cross_asset_24"
    assert out.network_fragility.active_label == "correlation_concentration"
    assert out.network_fragility.evidence["rule_evidence"][
        "largest_eigenvalue_share_percentile_504d"
    ] == pytest.approx(0.8630952380952381)
