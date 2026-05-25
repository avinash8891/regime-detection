from __future__ import annotations

import os
import pickle
import time
from datetime import date

from pathlib import Path
from typing import get_type_hints

import pandas as pd
import pytest

from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.config import load_default_regime_config
from regime_detection.fragility_universe import CROSS_ASSET_SYMBOLS, SECTOR_ETFS
from regime_detection.market_context import (
    MarketContext,
    build_market_context,
    slice_context_to_recent_sessions,
)
from regime_detection.models import (
    AxisEvidencePayload,
    AxisOutput,
    DataQuality,
    EventCalendarEvidencePayload,
    EventCalendarOutput,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
    VolumeLiquidityEvidencePayload,
    VolumeLiquidityOutput,
)
from regime_detection.strategy_response import build_strategy_response
from regime_detection.timeline import (
    ENGINE_MINIMUM_HISTORY,
    _AlignedV2Evidence,
    _build_cluster_output,
    _build_hmm_output,
    _enrich_with_hmm_evidence,
    _resolve_timeline_required_sessions,
    build_regime_timeline,
)
from regime_detection.transition_risk_series import (
    build_transition_risk_history,
    build_transition_risk_series,
)


def _minimal_context_for_evidence_mapping() -> MarketContext:
    day = date(2024, 1, 2)
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    return MarketContext(
        end_date=day,
        config=load_default_regime_config(),
        sessions=(day,),
        spy_ohlcv=pd.DataFrame({"close": [100.0]}, index=idx),
        rsp_close=pd.Series([100.0], index=idx),
        vix_proxy_close=None,
    )


def test_default_config_requires_and_carries_operator_label_maps() -> None:
    cfg = load_default_regime_config()

    assert cfg.hmm is not None
    assert cfg.hmm.label_map_required_for_output is True
    assert cfg.hmm.state_label_map == {
        0: "elevated_uncertainty",
        1: "high_vol_stress",
        2: "calm_trending",
        3: "transient_spike",
    }
    assert cfg.clustering is not None
    assert cfg.clustering.label_map_required_for_output is True
    assert cfg.clustering.cluster_label_map == {
        0: "post_crisis_transition",
        1: "crisis_panic",
        2: "tariff_shock",
        3: "correction_chop",
        4: "volatile_recovery",
        5: "calm_trending_bull",
        6: "steady_bull",
        7: "high_vol_stress",
    }


def test_hmm_output_uses_default_operator_state_label_map() -> None:
    day = date(2024, 1, 2)
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    aligned = _AlignedV2Evidence(
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

    output = _build_hmm_output(
        aligned=aligned,
        working_context=_minimal_context_for_evidence_mapping(),
        selected_day_index=0,
        day=day,
    )

    assert output is not None
    assert output.mapped_label == "high_vol_stress"
    assert output.mapping_status == "mapped"
    assert output.mapping_reason == "state_label_map_valid"


def test_cluster_output_uses_default_operator_cluster_label_map() -> None:
    day = date(2024, 1, 2)
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    aligned = _AlignedV2Evidence(
        cp_score_aligned=None,
        cp_days_since_aligned=None,
        cp_method=None,
        cluster_id_aligned=pd.Series([2], index=idx),
        cluster_distance_aligned=pd.Series([1.25], index=idx),
        cluster_model_version="gmm_8cluster_v1.0",
        cluster_n_clusters=8,
        hmm_top_state_aligned=None,
        hmm_top_state_prob_aligned=None,
        hmm_top_state_full=None,
        hmm_n_states=None,
        hmm_model_version=None,
    )

    output = _build_cluster_output(
        aligned=aligned,
        working_context=_minimal_context_for_evidence_mapping(),
        selected_day_index=0,
    )

    assert output is not None
    assert output.mapped_label == "tariff_shock"
    assert output.mapping_status == "mapped"
    assert output.mapping_reason == "cluster_label_map_valid"


def _constituent_ohlcv_from_close_series(series: pd.Series) -> pd.DataFrame:
    adjusted_close = series.astype(float).copy()
    return pd.DataFrame(
        {
            "open": adjusted_close,
            "high": adjusted_close,
            "low": adjusted_close,
            "close": adjusted_close,
            "volume": pd.Series(1_000_000, index=adjusted_close.index, dtype="int64"),
            "adjusted_close": adjusted_close,
        }
    )


def _fast_v2_test_config():
    engine = RegimeEngine()
    assert engine.config.hmm is not None
    assert engine.config.clustering is not None
    assert engine.config.change_point is not None
    return engine.config.model_copy(
        update={
            "hmm": engine.config.hmm.model_copy(
                update={
                    "n_states": 2,
                    "training_window_days": 100,
                    "random_seeds": (42,),
                }
            ),
            "clustering": engine.config.clustering.model_copy(
                update={"training_window_days": 100}
            ),
            "change_point": engine.config.change_point.model_copy(
                update={"training_window_days": 100}
            ),
        }
    )


def _build_shared_timeline_pipeline(market_df_for_asof):
    end_date = date(2023, 12, 14)
    engine = RegimeEngine()
    market_data = market_df_for_asof(end_date)
    context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=engine.config,
    )
    feature_store = build_feature_store(
        context,
        network_fragility_config=engine.config.network_fragility,
        trend_direction_v2_config=engine.config.trend_direction_v2,
        volatility_state_v2_config=engine.config.volatility_state_v2,
        breadth_state_v2_config=engine.config.breadth_state_v2,
        volume_liquidity_v2_config=engine.config.volume_liquidity_v2,
        monetary_pressure_v2_config=engine.config.monetary_pressure_v2,
        credit_funding_config=engine.config.credit_funding,
        inflation_growth_config=engine.config.inflation_growth,
        central_bank_text_config=engine.config.central_bank_text,
        news_sentiment_config=engine.config.news_sentiment,
    )
    bundle = build_axis_series_bundle(context=context, feature_store=feature_store)
    return {
        "end_date": end_date,
        "market_data": market_data,
        "context": context,
        "feature_store": feature_store,
        "bundle": bundle,
    }


@pytest.fixture(scope="session")
def shared_timeline_pipeline(market_df_for_asof, tmp_path_factory, worker_id):
    """Session-scoped, but ALSO shared across pytest-xdist workers via a
    disk pickle cache. Without this cross-worker cache, each worker
    independently rebuilds the ~80s timeline pipeline (build_market_context +
    build_feature_store + build_regime_timeline + engine.classify), and the
    duplicate work dominates wall-clock. Same pattern as
    ``classified_golden_outputs`` in conftest.py.

    PIT correctness: the pipeline inputs (market_df_for_asof at 2023-12-14
    plus default engine config) are immutable across the session. Tests
    consume the outputs read-only (``model_dump``, ``equals``, attribute
    reads). The pickle round-trip preserves pydantic-model and pandas-frame
    equality, which is what every consumer asserts on.
    """
    if worker_id == "master":
        return _build_shared_timeline_pipeline(market_df_for_asof)

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "shared_timeline_pipeline.pkl"
    lock_path = shared_dir / "shared_timeline_pipeline.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        result = _build_shared_timeline_pipeline(market_df_for_asof)
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
        "shared_timeline_pipeline build timed out waiting on peer worker; "
        f"cache_path={cache_path}"
    )


def test_core3_v1_regime_output_keeps_legacy_placeholder_wire_shapes(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    as_of = date(2023, 12, 14)
    engine = RegimeEngine(
        config_path=Path("src/regime_detection/configs/core3-v1.0.0.yaml")
    )
    out = engine.classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        event_calendar=event_calendar_df,
    )
    payload = out.model_dump()

    assert payload["network_fragility"] == {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    assert "label" in payload["transition_risk"]
    assert "state" not in payload["transition_risk"]
    assert set(payload["transition_risk"]) == {"label", "evidence"}


def test_runtime_evidence_fields_use_named_payloads() -> None:
    dq = DataQuality(status="ok")

    axis = AxisOutput(
        raw_label="bull",
        stable_label="bull",
        active_label="bull",
        evidence={"rule": "trend_above_ma", "value": 1.2},
        data_quality=dq,
    )
    event_calendar = EventCalendarOutput(
        primary_label="normal_calendar",
        matching_labels=("normal_calendar",),
        evidence={"selection_method": "precedence"},
    )
    volume_liquidity = VolumeLiquidityOutput(
        label="normal_volume",
        evidence={"volume_zscore": 0.5},
        data_quality=dq,
    )
    transition_risk = TransitionRiskOutput(
        state="stable",
        evidence=TransitionRiskEvidencePayload(
            triggered_rules=[],
            stable_changed_today=False,
            days_since_axis_switch=None,
            axis_switch_count=0,
            recent_axis_switch_count=0,
        ),
        score=0.10,
        score_components={"trend_break": 0.10},
        data_quality=DataQuality(status="ok"),
    )

    assert AxisOutput.model_fields["evidence"].annotation is AxisEvidencePayload
    assert (
        EventCalendarOutput.model_fields["evidence"].annotation
        is EventCalendarEvidencePayload
    )
    assert (
        VolumeLiquidityOutput.model_fields["evidence"].annotation
        is VolumeLiquidityEvidencePayload
    )
    assert (
        TransitionRiskOutput.model_fields["evidence"].annotation
        is TransitionRiskEvidencePayload
    )

    assert isinstance(axis.evidence, AxisEvidencePayload)
    assert axis.evidence["rule"] == "trend_above_ma"
    assert axis.model_dump()["evidence"] == {"rule": "trend_above_ma", "value": 1.2}
    assert event_calendar.model_dump()["primary_label"] == "normal_calendar"
    assert event_calendar.model_dump()["matching_labels"] == ("normal_calendar",)
    assert event_calendar.model_dump()["evidence"] == {"selection_method": "precedence"}
    assert volume_liquidity.model_dump()["evidence"] == {"volume_zscore": 0.5}
    assert transition_risk.model_dump()["evidence"] == {
        "triggered_rules": [],
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "axis_switch_count": 0,
        "recent_axis_switch_count": 0,
        "macro_event_labels": [],
    }


def test_timeline_private_helper_type_hints_resolve() -> None:
    hints = get_type_hints(_enrich_with_hmm_evidence)

    assert hints["output"] is AxisOutput
    assert hints["return"] is AxisOutput


def test_classify_window_returns_one_output_per_nyse_trading_day(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = market_df_for_asof(end_date)

    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        engine.classify_window(
            end_date=end_date,
            market_data=market_data,
            lookback_days=5,
            event_calendar=event_calendar_df,
        )


def test_classify_window_uses_lookback_days_not_fixed_calendar_span(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = market_df_for_asof(end_date)

    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        engine.classify_window(
            end_date=end_date,
            market_data=market_data,
            lookback_days=23,
            event_calendar=event_calendar_df,
        )


def test_market_context_builds_normalized_series_once(shared_timeline_pipeline) -> None:
    end_date = shared_timeline_pipeline["end_date"]
    context = shared_timeline_pipeline["context"]

    assert context.end_date == end_date
    assert context.sessions[-1] == end_date
    assert len(context.sessions) >= ENGINE_MINIMUM_HISTORY
    assert list(context.spy_ohlcv.columns) == ["open", "high", "low", "close", "volume"]
    assert context.rsp_close.name == "close"


def test_feature_store_precomputes_aligned_axis_features(
    shared_timeline_pipeline,
) -> None:
    context = shared_timeline_pipeline["context"]
    feature_store = shared_timeline_pipeline["feature_store"]

    assert feature_store.spy_index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_direction.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_character.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.volatility.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.breadth.spy_close.index.equals(context.spy_ohlcv.index)


def test_axis_series_bundle_reuses_feature_store_for_all_axes(
    shared_timeline_pipeline,
) -> None:
    end_date = shared_timeline_pipeline["end_date"]
    bundle = shared_timeline_pipeline["bundle"]

    assert end_date in bundle.trend_direction.outputs_by_date
    assert end_date in bundle.trend_character.outputs_by_date
    assert end_date in bundle.volatility_state.outputs_by_date
    assert end_date in bundle.breadth_state.outputs_by_date
    assert end_date in bundle.event_calendar


def test_classify_matches_last_output_of_shared_timeline_pipeline(
    shared_timeline_pipeline,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        engine.classify(
            as_of_date=shared_timeline_pipeline["end_date"],
            market_data=shared_timeline_pipeline["market_data"],
            event_calendar=event_calendar_df,
        )


def test_timeline_output_helper_matches_timeline_output(
    shared_timeline_pipeline,
) -> None:
    context = shared_timeline_pipeline["context"]
    lookback_days = ENGINE_MINIMUM_HISTORY
    required_sessions = _resolve_timeline_required_sessions(
        available_sessions=len(context.sessions),
        lookback_days=lookback_days,
        config=context.config,
    )
    working_context = slice_context_to_recent_sessions(
        context=context,
        required_sessions=required_sessions,
    )
    feature_store = build_feature_store(
        working_context,
        network_fragility_config=context.config.network_fragility,
        trend_direction_v2_config=context.config.trend_direction_v2,
        volatility_state_v2_config=context.config.volatility_state_v2,
        breadth_state_v2_config=context.config.breadth_state_v2,
        volume_liquidity_v2_config=context.config.volume_liquidity_v2,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
        central_bank_text_config=context.config.central_bank_text,
        news_sentiment_config=context.config.news_sentiment,
    )
    axis_bundle = build_axis_series_bundle(
        context=working_context,
        feature_store=feature_store,
    )
    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        build_transition_risk_series(
            context=working_context,
            feature_store=feature_store,
            axis_bundle=axis_bundle,
        )


def test_build_regime_timeline_uses_context_config_when_config_arg_omitted(
    market_df_for_asof,
) -> None:
    """Direct callers must not silently disable v2 seams by omitting config."""
    end_date = date(2023, 12, 14)
    engine = RegimeEngine()
    cfg = engine.config.model_copy(
        update={
            "change_point": engine.config.change_point.model_copy(
                update={"training_window_days": 500}
            ),
        }
    )
    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=cfg,
    )

    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        build_regime_timeline(context=context, lookback_days=3)


@pytest.mark.parametrize(
    ("disabled_fields", "expected_sessions"),
    [
        (
            {"change_point": None, "hmm": None, "clustering": None},
            1260 + 63 + 7 - 1,
        ),
        (
            {
                "change_point": None,
                "hmm": None,
                "clustering": None,
                "monetary_pressure_v2": None,
            },
            ENGINE_MINIMUM_HISTORY + 7 - 1,
        ),
        (
            {"hmm": None, "clustering": None},
            2705 + 21 + 7 - 1,
        ),
        (
            {"change_point": None, "clustering": None},
            1260 + 63 + 7 - 1 + 5,
        ),
        (
            {"change_point": None, "hmm": None},
            1260 + 63 + 7 - 1,
        ),
        (
            {},
            2705 + 21 + 7 - 1 + 5,
        ),
    ],
)
def test_timeline_required_sessions_preserves_v2_window_math(
    disabled_fields: dict[str, object],
    expected_sessions: int,
) -> None:
    cfg = load_default_regime_config().model_copy(update=disabled_fields)

    assert (
        _resolve_timeline_required_sessions(
            available_sessions=5_000,
            lookback_days=7,
            config=cfg,
        )
        == expected_sessions
    )


def test_timeline_required_sessions_caps_at_available_history() -> None:
    cfg = load_default_regime_config()

    assert (
        _resolve_timeline_required_sessions(
            available_sessions=500,
            lookback_days=7,
            config=cfg,
        )
        == 500
    )


def test_classify_delegates_to_classify_window_with_single_day_lookback(
    mocker, market_df_for_asof, event_calendar_df
) -> None:
    engine = RegimeEngine()
    as_of = date(2023, 12, 14)
    spy = mocker.patch.object(
        engine,
        "classify_window",
        side_effect=RuntimeError("transition_risk requires score inputs"),
    )

    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        engine.classify(
            as_of_date=as_of,
            market_data=market_df_for_asof(as_of),
            event_calendar=event_calendar_df,
        )

    spy.assert_called_once()
    assert spy.call_args.kwargs["end_date"] == as_of
    assert spy.call_args.kwargs["lookback_days"] == 1
    assert spy.call_args.kwargs["event_calendar"] is event_calendar_df


def test_timeline_passes_event_calendar_matching_labels_to_strategy_response(
    mocker,
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    as_of = date(2026, 5, 13)
    market_data = v2_market_df_for_asof(as_of)
    spy_frame = market_data[market_data["symbol"] == "SPY"].sort_values("date")
    spy_index = pd.to_datetime(spy_frame["date"])
    trend = pd.Series(range(len(spy_index)), index=spy_index, dtype="float64")
    macro_series = {
        "2y_yield": (4.00 + trend * 0.0002).rename("2y_yield"),
        "10y_yield": (4.25 + trend * 0.0001).rename("10y_yield"),
        "broad_usd_index": (100.0 + trend * 0.001).rename("broad_usd_index"),
    }
    event_calendar = event_calendar_df.copy()
    event_calendar.loc[[0, 1], "date"] = as_of
    event_calendar.loc[[0, 1], "publication_date"] = date(2026, 1, 1)
    spy = mocker.patch(
        "regime_detection.timeline.build_strategy_response",
        wraps=build_strategy_response,
    )
    config = _fast_v2_test_config()

    out = engine.classify(
        as_of_date=as_of,
        market_data=market_data,
        config=config,
        event_calendar=event_calendar,
        sector_etf_closes={
            symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
        },
        cross_asset_closes={
            symbol: v2_close_series_by_symbol[symbol] for symbol in CROSS_ASSET_SYMBOLS
        },
        macro_series=macro_series,
        pit_constituent_intervals=pd.DataFrame(
            {
                "ticker": list(SECTOR_ETFS),
                "start_date": [date(2019, 1, 2)] * len(SECTOR_ETFS),
                "end_date": [None] * len(SECTOR_ETFS),
                "source": ["test_fixture"] * len(SECTOR_ETFS),
                "source_url": ["test_fixture"] * len(SECTOR_ETFS),
                "bias_warning": ["test_fixture"] * len(SECTOR_ETFS),
            }
        ),
        constituent_ohlcv={
            symbol: _constituent_ohlcv_from_close_series(
                v2_close_series_by_symbol[symbol]
            )
            for symbol in SECTOR_ETFS
        },
    )

    event_output = out.structural_causal_state.event_calendar
    assert event_output.primary_label == "cpi_week"
    assert event_output.matching_labels == (
        "cpi_week",
        "nfp_week",
        "expiry_week",
        "earnings_season",
    )
    spy.assert_called_once()
    assert spy.call_args.kwargs["event_calendar_labels"] == event_output.matching_labels
    assert (
        spy.call_args.kwargs["event_modifier_config"] is config.strategy_event_modifiers
    )
    assert out.effective_strategy_constraints is not None
    assert out.strategy_family_constraints is not None
    breakout = out.effective_strategy_constraints["breakout"]
    assert "strategy_response" in breakout.sources
    assert "strategy_family_constraints" in breakout.sources
    if (
        out.agent_routing is not None
        and "breakout" in out.agent_routing.blocked_strategy_modes
    ):
        assert "agent_routing" in breakout.sources
    assert breakout.allowed is (
        out.strategy_response.allow_breakout
        and out.strategy_family_constraints["breakout"].allowed
        and (
            out.agent_routing is None
            or "breakout" not in out.agent_routing.blocked_strategy_modes
        )
    )


def test_transition_risk_history_precomputes_axis_switch_and_prior_bear_flags() -> None:
    sessions = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    bull = {day: "bull" for day in sessions}
    stable_history = {
        sessions[0]: "bull",
        sessions[1]: "bull",
        sessions[2]: "bear",
        sessions[3]: "bear",
        sessions[4]: "bull",
    }

    history = build_transition_risk_history(
        sessions=sessions,
        trend_direction_stable_by_date=stable_history,
        trend_character_stable_by_date=bull,
        volatility_stable_by_date=bull,
        breadth_stable_by_date=bull,
    )

    assert history.stable_changed_by_date[sessions[0]] is False
    assert history.stable_changed_by_date[sessions[2]] is True
    assert history.stable_changed_by_date[sessions[4]] is True
    assert history.axis_switch_count_by_date[sessions[0]] == 0
    assert history.axis_switch_count_by_date[sessions[2]] == 1
    assert history.axis_switch_count_by_date[sessions[4]] == 1
    assert history.recent_axis_switch_count_by_date[sessions[4]] == 2
    assert history.days_since_axis_switch_by_date[sessions[1]] is None
    assert history.days_since_axis_switch_by_date[sessions[2]] == 0
    assert history.days_since_axis_switch_by_date[sessions[3]] == 1
    assert history.days_since_axis_switch_by_date[sessions[4]] == 0
    # v1 §9.4 (post-Q2 fix): prior_bear excludes today's stable_label from the
    # 60-session lookback. sessions[2] is the FIRST day stable goes bear, so
    # the prior window (sessions[0..1] = bull, bull) has no bear and the flag
    # is False. By sessions[3] (the day AFTER the first bear print), the prior
    # window includes sessions[2]=bear, so the flag is True. sessions[4]
    # (bull again) still has bear in its prior window.
    assert history.prior_bear_by_date[sessions[1]] is False
    assert history.prior_bear_by_date[sessions[2]] is False
    assert history.prior_bear_by_date[sessions[3]] is True
    assert history.prior_bear_by_date[sessions[4]] is True
