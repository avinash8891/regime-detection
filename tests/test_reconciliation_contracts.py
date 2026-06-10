from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.trend_character import compute_features


def _vix_data_from_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    vixy = market_data[market_data["symbol"] == "VIXY"].copy()
    return vixy[["date", "close"]]


def _reconciliation_fixture_config():
    from regime_detection.engine import RegimeEngine

    engine = RegimeEngine()
    cfg = engine.config
    assert cfg.hmm is not None
    assert cfg.clustering is not None
    assert cfg.change_point is not None
    assert cfg.network_fragility is not None
    return cfg.model_copy(
        update={
            "network_fragility": cfg.network_fragility.model_copy(
                update={
                    "percentile_lookback_days": 100,
                    "dispersion_percentile_lookback_days": 100,
                }
            ),
            "hmm": cfg.hmm.model_copy(
                update={
                    "n_states": 2,
                    "training_window_days": 100,
                    "random_seeds": (42, 7, 13),
                    "model_version": "hmm_2state_test_v1.0",
                    "state_label_map": {
                        0: "test_low_risk",
                        1: "test_high_risk",
                    },
                }
            ),
            "clustering": cfg.clustering.model_copy(
                update={"training_window_days": 100}
            ),
            "change_point": cfg.change_point.model_copy(
                update={"training_window_days": 100}
            ),
        }
    )


def _synthetic_kwargs_without_config(synthetic_v2_kwargs_for_market_data, market_data):
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    kwargs.pop("config", None)
    return kwargs


def test_classify_uses_vix_data_when_vix_proxy_missing_from_market_data(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)

    market_df = v2_market_df_for_asof(as_of)
    vix_df = _vix_data_from_market_data(market_df)
    market_df = market_df[market_df["symbol"] != "VIXY"].copy()
    market_df = market_df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        vix_data=vix_df,
        config=_reconciliation_fixture_config(),
        **_synthetic_kwargs_without_config(
            synthetic_v2_kwargs_for_market_data, market_df
        ),
    )
    assert (
        out.volatility_state.evidence["rule_evidence"]["vix_percentile_252d"]
        is not None
    )


def test_market_context_requires_true_vix_in_market_data(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    market_df = market_df_for_asof(as_of)
    market_df = market_df[market_df["symbol"] != "VIX"].copy()

    from regime_detection.engine import RegimeEngine
    from regime_detection.market_context import build_market_context

    with pytest.raises(
        ValueError, match="market_data missing required symbol for V1: VIX"
    ):
        build_market_context(
            end_date=as_of,
            market_data=market_df,
            config=RegimeEngine().config,
        )


def test_market_context_uses_true_vix_not_vixy_proxy(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    market_df = market_df_for_asof(as_of)
    vix_mask = market_df["symbol"] == "VIX"
    market_df.loc[vix_mask, "close"] = market_df.loc[vix_mask, "close"] + 1000.0

    from regime_detection.engine import RegimeEngine
    from regime_detection.market_context import build_market_context

    context = build_market_context(
        end_date=as_of,
        market_data=market_df,
        config=RegimeEngine().config,
    )

    expected = market_df[vix_mask].sort_values("date").iloc[-1]["close"]
    assert context.vix_proxy_close is not None
    assert context.vix_proxy_close.iloc[-1] == expected


def test_trend_character_adx_cold_start_stays_nan(raw_market_frames) -> None:
    spy = (
        raw_market_frames["SPY"]
        .copy()
        .assign(date=pd.to_datetime(raw_market_frames["SPY"]["date"]))
    )
    spy = spy.sort_values("date").head(20).set_index("date")

    features = compute_features(
        close=spy["close"],
        high=spy["high"],
        low=spy["low"],
    )

    assert pd.isna(features.adx_14.iloc[13])


def test_breadth_data_quality_does_not_block_pit_breadth_when_rsp_gaps(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = v2_market_df_for_asof(as_of)
    rsp_mask = market_df["symbol"] == "RSP"
    recent_rsp_idx = market_df[rsp_mask].tail(50).index[:7]
    market_df.loc[recent_rsp_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        vix_data=_vix_data_from_market_data(market_df),
        config=_reconciliation_fixture_config(),
        **_synthetic_kwargs_without_config(
            synthetic_v2_kwargs_for_market_data, market_df
        ),
    )

    assert out.breadth_state.active_label == "broadening_breadth"
    assert out.breadth_state.data_quality.status == "degraded"
    assert out.breadth_state.data_quality.reason == "incomplete_data"
    assert out.breadth_state.evidence["rule_evidence"]["v1_raw_label"] == "unknown"
    assert (
        out.breadth_state.evidence["rule_evidence"]["reason"] == "insufficient_history"
    )


def test_trend_direction_data_quality_insufficient_data_fails_loudly(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = v2_market_df_for_asof(as_of)
    spy_mask = market_df["symbol"] == "SPY"
    recent_spy_idx = market_df[spy_mask].tail(200).index[:70]
    market_df.loc[recent_spy_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    with pytest.raises(RuntimeError, match="transition score inputs not ready"):
        RegimeEngine().classify(
            as_of_date=as_of,
            market_data=market_df,
            vix_data=_vix_data_from_market_data(market_df),
            config=_reconciliation_fixture_config(),
            **_synthetic_kwargs_without_config(
                synthetic_v2_kwargs_for_market_data, market_df
            ),
        )


def test_trend_direction_data_quality_stale_data_fails_loudly(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = v2_market_df_for_asof(as_of)
    spy_mask = market_df["symbol"] == "SPY"
    trailing_spy_idx = market_df[spy_mask].tail(4).index
    market_df.loc[trailing_spy_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    with pytest.raises(RuntimeError, match="transition score inputs not ready"):
        RegimeEngine().classify(
            as_of_date=as_of,
            market_data=market_df,
            vix_data=_vix_data_from_market_data(market_df),
            config=_reconciliation_fixture_config(),
            **_synthetic_kwargs_without_config(
                synthetic_v2_kwargs_for_market_data, market_df
            ),
        )
