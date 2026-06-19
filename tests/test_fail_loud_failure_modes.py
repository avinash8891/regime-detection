from __future__ import annotations

import json
import logging
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from regime_detection.config import RegimeConfig, load_default_regime_config
from regime_detection.event_calendar import (
    classify_event_calendar,
    compute_event_window_just_passed,
)
from regime_detection.engine import RegimeEngine
from regime_detection.hmm_state import compute_hmm_features
from regime_detection.clustering import compute_clustering_features
from regime_detection.change_point import compute_change_point_features
from regime_detection.central_bank_text import to_daily_score_series
from regime_detection._feature_specs import _build_sentiment_score_series
from regime_detection.feature_store_runtime import FeatureSpec, _run_feature_specs
from regime_detection.loaders import (
    load_central_bank_text_score,
    load_cross_asset_closes,
    load_cpi_vintages_first_release,
    load_macro_series,
    load_sector_etf_closes,
)
from regime_detection.market_context import MarketContext
from regime_detection.timeline import (
    _AlignedV2Evidence,
    _build_cluster_output,
    _build_hmm_output,
)
from regime_detection.transition_risk import (
    TransitionRuleFlags,
    compose_transition_risk_output,
)
from regime_detection.transition_risk_history import _apply_transition_state_debounce
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compute_transition_score,
)


def test_regime_engine_requires_explicit_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REGIME_DETECTION_ALLOW_DEFAULT_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="config_path is required"):
        RegimeEngine()


@pytest.mark.parametrize("section", ["network_fragility", "sentiment_score"])
def test_v2_config_rejects_missing_runtime_sections(section: str) -> None:
    data = load_default_regime_config().model_dump(mode="python")
    data[section] = None

    with pytest.raises(ValidationError, match=section):
        RegimeConfig.model_validate(data)


def test_zero_symbol_or_series_loads_fail_loudly() -> None:
    close_frame = pd.DataFrame(columns=["date", "symbol", "close"])
    macro_frame = pd.DataFrame(columns=["date", "series_id", "value"])

    with pytest.raises(ValueError, match="load_sector_etf_closes returned 0 symbols"):
        load_sector_etf_closes(close_frame)
    with pytest.raises(ValueError, match="load_cross_asset_closes returned 0 symbols"):
        load_cross_asset_closes(close_frame)
    with pytest.raises(ValueError, match="load_macro_series returned 0 series"):
        load_macro_series(macro_frame)


def test_event_calendar_absence_fails_loudly_for_direct_callers() -> None:
    cfg = load_default_regime_config()
    sessions = tuple(pd.bdate_range(start="2024-01-29", end="2024-02-12").date)

    with pytest.raises(ValueError, match="event_calendar is required"):
        classify_event_calendar(
            as_of_date=date(2024, 2, 1),
            event_calendar=None,
            config=cfg,
        )
    with pytest.raises(ValueError, match="event_calendar is required"):
        compute_event_window_just_passed(
            normalized_event_calendar=None,
            sessions=sessions,
            trailing_sessions=3,
        )


def test_regime_engine_logs_rejected_request(
    raw_market_data: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = load_default_regime_config()

    with caplog.at_level(logging.ERROR, logger="regime_detection.engine"):
        with pytest.raises(ValueError, match="event_calendar is required"):
            RegimeEngine().classify(
                as_of_date=date(2024, 2, 1),
                market_data=raw_market_data,
                config=cfg,
            )

    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "classify_request_failed"
    assert payload["end_date"] == "2024-02-01"
    assert payload["error_type"] == "ValueError"
    assert payload["operator_action"] == "fix request inputs or config"


def test_optional_evidence_absence_fails_loudly_for_direct_callers() -> None:
    sessions = pd.DatetimeIndex(pd.bdate_range(start="2024-01-01", periods=5))
    empty_releases = pd.DataFrame(
        columns=[
            "release_date",
            "hawkish_count",
            "dovish_count",
            "total_tokens",
            "net_score",
            "source",
        ]
    )
    empty_cpi = pd.DataFrame(columns=["date", "value", "realtime_start"])

    with pytest.raises(ValueError, match="aaii_sentiment is required"):
        _build_sentiment_score_series(aaii_sentiment=None, session_index=sessions)
    with pytest.raises(ValueError, match="central_bank_text source is required"):
        load_central_bank_text_score()
    with pytest.raises(ValueError, match="central_bank_text releases are required"):
        to_daily_score_series(empty_releases, session_index=sessions)
    with pytest.raises(ValueError, match="cpi_vintages source must not be empty"):
        load_cpi_vintages_first_release(empty_cpi)


def test_model_cold_start_failures_are_loud_for_direct_callers() -> None:
    cfg = load_default_regime_config()
    assert cfg.hmm is not None
    assert cfg.clustering is not None
    assert cfg.change_point is not None
    index = pd.DatetimeIndex(pd.bdate_range(start="2024-01-01", periods=5))
    short = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02], index=index)

    with pytest.raises(RuntimeError, match="HMM insufficient history"):
        compute_hmm_features(
            return_1d=short,
            realized_vol_21d=short,
            drawdown_63d=short,
            volume_zscore_20d=short,
            avg_pairwise_corr_63d=short,
            config=cfg.hmm,
        )
    with pytest.raises(RuntimeError, match="GMM insufficient history"):
        compute_clustering_features(
            return_21d=short,
            return_63d=short,
            realized_vol_21d=short,
            drawdown_63d=short,
            adx_14=short,
            avg_pairwise_corr_63d=short,
            pct_above_50dma=short,
            config=cfg.clustering,
        )
    with pytest.raises(RuntimeError, match="BOCPD insufficient history"):
        compute_change_point_features(realized_vol_21d=short, config=cfg.change_point)


def test_feature_spec_build_returning_none_fails_loudly() -> None:
    class _State:
        value: object | None = None
        context = SimpleNamespace(config=SimpleNamespace(config_version="core3-v2.0.0"))

    spec: FeatureSpec[object | None, _State] = FeatureSpec(
        name="hmm",
        policy="none",
        required_inputs=("x",),
        resolve=lambda _state: {"x": 1},
        build=lambda x: None,
        store=lambda state, value: setattr(state, "value", value),
    )

    with pytest.raises(RuntimeError, match="feature spec 'hmm' returned None"):
        _run_feature_specs((spec,), _State())


def _one_day_context_with_config(config: RegimeConfig) -> MarketContext:
    day = date(2024, 1, 2)
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    return MarketContext(
        end_date=day,
        config=config,
        sessions=(day,),
        spy_ohlcv=pd.DataFrame({"close": [100.0]}, index=idx),
        rsp_close=pd.Series([100.0], index=idx),
        vix_proxy_close=None,
    )


def test_required_hmm_and_cluster_label_maps_fail_loudly() -> None:
    day = date(2024, 1, 2)
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    cfg = load_default_regime_config()
    assert cfg.hmm is not None
    assert cfg.clustering is not None
    context = _one_day_context_with_config(
        cfg.model_copy(
            update={
                "hmm": cfg.hmm.model_copy(update={"state_label_map": None}),
                "clustering": cfg.clustering.model_copy(
                    update={"cluster_label_map": {0: "only_one_cluster"}}
                ),
            }
        )
    )

    hmm_aligned = _AlignedV2Evidence(
        cp_score_aligned=None,
        cp_days_since_aligned=None,
        cp_method=None,
        cluster_id_aligned=None,
        cluster_distance_aligned=None,
        cluster_model_version=None,
        cluster_n_clusters=None,
        hmm_top_state_aligned=pd.Series([1], index=idx),
        hmm_top_state_prob_aligned=pd.Series([0.8], index=idx),
        hmm_top_state_full=pd.Series([1], index=idx),
        hmm_n_states=4,
        hmm_model_version="hmm_4state_v1.0",
    )
    cluster_aligned = _AlignedV2Evidence(
        cp_score_aligned=None,
        cp_days_since_aligned=None,
        cp_method=None,
        cluster_id_aligned=pd.Series([1], index=idx),
        cluster_distance_aligned=pd.Series([0.2], index=idx),
        cluster_model_version="gmm_8cluster_v1.0",
        cluster_n_clusters=8,
        hmm_top_state_aligned=None,
        hmm_top_state_prob_aligned=None,
        hmm_top_state_full=None,
        hmm_n_states=None,
        hmm_model_version=None,
    )

    with pytest.raises(RuntimeError, match="state_label_map_required"):
        _build_hmm_output(
            aligned=hmm_aligned,
            working_context=context,
            selected_day_index=0,
            day=day,
        )
    with pytest.raises(RuntimeError, match="cluster_label_map_invalid"):
        _build_cluster_output(
            aligned=cluster_aligned,
            working_context=context,
            selected_day_index=0,
        )


def test_transition_score_rejects_missing_components() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    components = {key: 0.1 for key in cfg.weights}
    components["credit_stress"] = None

    with pytest.raises(RuntimeError, match="transition score missing components"):
        compute_transition_score(
            components=components,
            weights=cfg.weights,
            minimum_component_weight_coverage=cfg.minimum_component_weight_coverage,
        )


def test_transition_risk_rejects_missing_score_before_state_overrides() -> None:
    with pytest.raises(RuntimeError, match="transition score inputs not ready"):
        compose_transition_risk_output(
            score=ComposedTransitionScore(
                score=None,
                interpretation=None,
                components=None,
            ),
            flags=TransitionRuleFlags(
                crisis=True,
                bear_stress=False,
                fragile_bull=False,
                recovery_attempt=False,
                sideways_stress=False,
                event_transition_watch=False,
                post_switch_cooldown=False,
                insufficient_data=True,
                stable_changed_today=False,
                days_since_axis_switch=None,
                axis_switch_count=0,
                recent_axis_switch_count=0,
            ),
        )


def test_transition_state_debounce_requires_seed_for_restart_safety() -> None:
    day = date(2024, 1, 2)

    with pytest.raises(RuntimeError, match="initial_active_state is required"):
        _apply_transition_state_debounce(
            sessions=[day],
            raw_outputs={
                day: compose_transition_risk_output(
                    score=ComposedTransitionScore(
                        score=0.1,
                        interpretation="stable",
                        components={"trend_break": 0.1},
                    ),
                    flags=TransitionRuleFlags(
                        crisis=False,
                        bear_stress=False,
                        fragile_bull=False,
                        recovery_attempt=False,
                        sideways_stress=False,
                        event_transition_watch=False,
                        post_switch_cooldown=False,
                        insufficient_data=False,
                        stable_changed_today=False,
                        days_since_axis_switch=None,
                        axis_switch_count=0,
                        recent_axis_switch_count=0,
                    ),
                )
            },
            state_confirmation_days={"stable": 1},
            initial_active_state=None,
        )
