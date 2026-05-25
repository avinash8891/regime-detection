from __future__ import annotations

from dataclasses import fields
from datetime import date

import numpy as np
import pandas as pd

from regime_detection.config import NewsSentimentConfig, load_default_regime_config
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.market_context import build_market_context
from regime_detection.trend_direction import (
    compute_features as compute_trend_direction_features,
)


def _sector_etf_closes(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    closes: dict[str, pd.Series] = {}
    for i, symbol in enumerate(SECTOR_ETFS):
        values = 100.0 * np.exp(np.arange(len(index)) * (0.0003 + i * 0.00002))
        closes[symbol] = pd.Series(values, index=index, name=symbol)
    return closes


def test_build_news_sentiment_score_series_preserves_existing_alignment_and_smoothing() -> (
    None
):
    from regime_detection.feature_store import _build_news_sentiment_score_series

    sessions = pd.bdate_range(start="2024-03-04", end="2024-03-08", freq="B")
    news = pd.Series(
        [0.10, 0.30, -0.20],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-03-04"),
                pd.Timestamp("2024-03-06"),
                pd.Timestamp("2024-03-08"),
            ]
        ),
        name="news_sentiment",
    )

    score = _build_news_sentiment_score_series(
        news_sentiment=news,
        session_index=sessions,
        config=NewsSentimentConfig(smoothing_window_sessions=2),
    )

    assert score is not None
    assert score.name == "news_sentiment_score"
    pd.testing.assert_series_equal(
        score,
        pd.Series(
            [0.10, 0.10, 0.20, 0.30, 0.05],
            index=sessions,
            name="news_sentiment_score",
        ),
    )


def test_feature_store_builder_registry_runs_builders_in_declared_order(
    market_df_for_asof,
) -> None:
    from regime_detection.feature_store import (
        _FeatureStoreBuilder,
        _FeatureStoreBuildState,
        _run_feature_store_builders,
    )

    cfg = load_default_regime_config()
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=cfg,
    )
    spy_close = context.spy_ohlcv["close"]
    assert isinstance(spy_close, pd.Series)
    state = _FeatureStoreBuildState(
        context=context,
        spy_ohlcv=context.spy_ohlcv,
        spy_close=spy_close,
    )
    calls: list[tuple[str, object]] = []

    def moving_average_builder(build_state: _FeatureStoreBuildState) -> None:
        calls.append(("moving_average", build_state.spy_close.name))
        build_state.sma_50 = pd.Series([1.0], name="sma_50")

    def volatility_builder(build_state: _FeatureStoreBuildState) -> None:
        output_name = (
            build_state.sma_50.name if build_state.sma_50 is not None else None
        )
        calls.append(("volatility", output_name))

    _run_feature_store_builders(
        (
            _FeatureStoreBuilder("moving_average", moving_average_builder),
            _FeatureStoreBuilder("volatility", volatility_builder),
        ),
        state,
    )

    assert calls == [("moving_average", "close"), ("volatility", "sma_50")]


def test_feature_store_build_state_uses_typed_intermediate_fields() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState

    state_fields = {field.name for field in fields(_FeatureStoreBuildState)}

    assert "values" not in state_fields
    assert {
        "trend_direction",
        "trend_character",
        "volatility",
        "breadth",
        "sma_50",
        "network_fragility",
        "trend_direction_v2",
        "volatility_state_v2",
        "breadth_state_v2",
        "volume_liquidity_v2",
        "monetary",
        "hmm",
        "clustering",
        "change_point",
        "credit_funding",
        "inflation_growth",
        "sentiment_score",
        "news_sentiment_score",
        "realized_vol_21d",
    }.issubset(state_fields)


def test_default_feature_store_builder_registry_orders_trend_news_before_trend_v2() -> (
    None
):
    from regime_detection.feature_store import _FEATURE_STORE_BUILDERS, FeatureStore

    builder_names = tuple(builder.name for builder in _FEATURE_STORE_BUILDERS)
    feature_fields = tuple(
        name for name in FeatureStore.model_fields if name != "spy_index"
    )

    assert set(feature_fields).issubset(builder_names)
    assert builder_names.index("trend_direction") < builder_names.index(
        "news_sentiment_score"
    )
    assert builder_names.index("news_sentiment_score") < builder_names.index(
        "trend_direction_v2"
    )
    assert builder_names.index("volatility_state_v2") < builder_names.index(
        "breadth_state_v2"
    )
    assert builder_names.index("volatility_state_v2") < builder_names.index(
        "realized_vol_21d"
    )
    assert builder_names.index("breadth_state_v2") < builder_names.index(
        "realized_vol_21d"
    )


def test_feature_store_registry_preserves_trend_and_news_outputs(
    market_df_for_asof,
) -> None:
    from regime_detection.feature_store import _build_news_sentiment_score_series

    cfg = load_default_regime_config()
    as_of = date(2023, 12, 14)
    base_context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=cfg,
    )
    news = pd.Series(
        [0.2, -0.1, 0.4],
        index=pd.DatetimeIndex(
            [
                base_context.spy_ohlcv.index[-5],
                base_context.spy_ohlcv.index[-3],
                base_context.spy_ohlcv.index[-1],
            ]
        ),
        name="news_sentiment",
    )
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=cfg,
        news_sentiment=news,
    )

    store = build_feature_store(
        context,
        trend_direction_v2_config=cfg.trend_direction_v2,
        news_sentiment_config=NewsSentimentConfig(smoothing_window_sessions=2),
    )

    spy_close = context.spy_ohlcv["close"].squeeze()
    assert isinstance(spy_close, pd.Series)
    expected_trend = compute_trend_direction_features(spy_close)
    pd.testing.assert_series_equal(store.trend_direction.close, expected_trend.close)
    pd.testing.assert_series_equal(store.trend_direction.sma_50, expected_trend.sma_50)
    pd.testing.assert_series_equal(
        store.trend_direction.sma_200, expected_trend.sma_200
    )
    pd.testing.assert_series_equal(
        store.trend_direction.return_63d,
        expected_trend.return_63d,
    )

    assert store.trend_direction_v2 is not None
    expected_news = _build_news_sentiment_score_series(
        news_sentiment=context.news_sentiment,
        session_index=pd.DatetimeIndex(context.spy_ohlcv.index),
        config=NewsSentimentConfig(smoothing_window_sessions=2),
    )
    assert expected_news is not None
    pd.testing.assert_series_equal(
        store.trend_direction_v2.news_sentiment_score,
        expected_news,
    )


def test_feature_store_reuses_realized_vol_21d_for_trainable_evidence_layers(
    monkeypatch, market_df_for_asof
) -> None:
    cfg = load_default_regime_config()
    as_of = date(2023, 12, 14)
    bootstrap = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=cfg,
    )
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=cfg,
        sector_etf_closes=_sector_etf_closes(
            pd.DatetimeIndex(bootstrap.spy_ohlcv.index)
        ),
    )

    calls: list[int] = []

    def counting_realized_vol(close: pd.Series, window: int) -> pd.Series:
        if window == 21:
            calls.append(window)
        return pd.Series(0.2, index=close.index, name="realized_vol_21d")

    monkeypatch.setattr(
        "regime_detection.feature_store.realized_vol", counting_realized_vol
    )
    monkeypatch.setattr(
        "regime_detection.feature_store.compute_hmm_features", lambda **_: None
    )
    monkeypatch.setattr(
        "regime_detection.feature_store.compute_clustering_features", lambda **_: None
    )
    monkeypatch.setattr(
        "regime_detection.feature_store.compute_change_point_features", lambda **_: None
    )

    build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
        volatility_state_v2_config=cfg.volatility_state_v2,
        breadth_state_v2_config=cfg.breadth_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
    )

    assert calls == [21]
