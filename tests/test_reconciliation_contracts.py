from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.trend_character import compute_features


def test_classify_uses_vix_data_when_vix_proxy_missing_from_market_data(
    raw_market_frames,
    market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    vixy = raw_market_frames["VIXY"]
    as_of = date(2023, 12, 14)

    market_df = market_df_for_asof(as_of)
    market_df = market_df[market_df["symbol"] != "VIXY"].copy()
    market_df = market_df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    vix_df = vixy.copy()
    vix_df = vix_df[vix_df["date"] <= as_of].copy()
    vix_df = vix_df[["date", "close"]]

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        vix_data=vix_df,
        **synthetic_v2_kwargs_for_market_data(market_df),
    )
    assert out.volatility_state.evidence["rule_evidence"]["vix_percentile_252d"] is not None


def test_trend_character_adx_cold_start_stays_nan(raw_market_frames) -> None:
    spy = raw_market_frames["SPY"].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").head(20).set_index("date")

    features = compute_features(
        close=spy["close"],
        high=spy["high"],
        low=spy["low"],
    )

    assert pd.isna(features.adx_14.iloc[13])


def test_breadth_data_quality_does_not_block_pit_breadth_when_rsp_gaps(
    market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = market_df_for_asof(as_of)
    rsp_mask = market_df["symbol"] == "RSP"
    recent_rsp_idx = market_df[rsp_mask].tail(50).index[:7]
    market_df.loc[recent_rsp_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        **synthetic_v2_kwargs_for_market_data(market_df),
    )

    assert out.breadth_state.active_label is not None
    assert out.breadth_state.data_quality.status == "degraded"


def test_trend_direction_data_quality_insufficient_data_can_override_non_unknown_label(
    market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = market_df_for_asof(as_of)
    spy_mask = market_df["symbol"] == "SPY"
    recent_spy_idx = market_df[spy_mask].tail(200).index[:70]
    market_df.loc[recent_spy_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        **synthetic_v2_kwargs_for_market_data(market_df),
    )

    assert out.trend_direction.active_label == "unknown"
    assert out.trend_direction.data_quality.status == "insufficient_data"
    assert out.trend_direction.data_quality.reason == "insufficient_data"


def test_trend_direction_data_quality_stale_data_overrides_insufficient_history(
    market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    as_of = date(2023, 12, 14)
    market_df = market_df_for_asof(as_of)
    spy_mask = market_df["symbol"] == "SPY"
    trailing_spy_idx = market_df[spy_mask].tail(4).index
    market_df.loc[trailing_spy_idx, "close"] = pd.NA

    from regime_detection.engine import RegimeEngine

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_df,
        **synthetic_v2_kwargs_for_market_data(market_df),
    )

    assert out.trend_direction.active_label == "unknown"
    assert out.trend_direction.data_quality.status == "stale_data"
    assert out.trend_direction.data_quality.reason == "stale_data"
