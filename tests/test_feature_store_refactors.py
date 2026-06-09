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
    from regime_detection._feature_specs import _build_news_sentiment_score_series

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


def test_feature_specs_registry_preserves_required_ordering_invariants() -> None:
    """After PR 2, all 20 features live in _FEATURE_SPECS. Ordering invariants
    that previously crossed the spec/builder boundary now apply within
    _FEATURE_SPECS only. The orchestrator runs specs in registry order."""
    from regime_detection.feature_store import _FEATURE_SPECS, FeatureStore

    spec_names = tuple(spec.name for spec in _FEATURE_SPECS)
    feature_fields = tuple(
        name
        for name in FeatureStore.model_fields
        if name not in {"spy_index", "availability"}
    )

    # Every user-visible FeatureStore field must be covered by a spec.
    # (Intermediate state specs like sentiment_score are also in _FEATURE_SPECS
    # but are not FeatureStore fields — they use report=False.)
    assert set(feature_fields).issubset(set(spec_names))

    # sentiment_score and news_sentiment_score must run before trend_direction_v2
    # because trend_direction_v2 reads state.sentiment_score and
    # state.news_sentiment_score in its resolve function.
    assert spec_names.index("sentiment_score") < spec_names.index("trend_direction_v2")
    assert spec_names.index("news_sentiment_score") < spec_names.index(
        "trend_direction_v2"
    )

    # volatility_state_v2 must run before breadth_state_v2 and realized_vol_21d
    # to preserve the historical build order (no functional dependency required;
    # this pin prevents accidental reordering).
    assert spec_names.index("volatility_state_v2") < spec_names.index(
        "breadth_state_v2"
    )
    assert spec_names.index("volatility_state_v2") < spec_names.index(
        "realized_vol_21d"
    )
    assert spec_names.index("breadth_state_v2") < spec_names.index("realized_vol_21d")

    # realized_vol_21d and drawdown_63d must run before hmm/clustering/change_point
    # (those features read the intermediate series from state).
    for derived in ("hmm", "clustering", "change_point"):
        assert spec_names.index("realized_vol_21d") < spec_names.index(
            derived
        ), f"realized_vol_21d must precede {derived}"
    for derived in ("hmm", "clustering"):
        assert spec_names.index("drawdown_63d") < spec_names.index(
            derived
        ), f"drawdown_63d must precede {derived}"

    # network_fragility and trend_direction_v2 must run before clustering
    # (clustering reads both in its resolve).
    assert spec_names.index("network_fragility") < spec_names.index("clustering")
    assert spec_names.index("trend_direction_v2") < spec_names.index("clustering")


def test_feature_store_registry_preserves_trend_and_news_outputs(
    market_df_for_asof,
) -> None:
    from regime_detection._feature_specs import _build_news_sentiment_score_series

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
        "regime_detection._feature_specs.realized_vol", counting_realized_vol
    )
    monkeypatch.setattr(
        "regime_detection._feature_specs.compute_hmm_features", lambda **_: None
    )
    monkeypatch.setattr(
        "regime_detection._feature_specs.compute_clustering_features", lambda **_: None
    )
    monkeypatch.setattr(
        "regime_detection._feature_specs.compute_change_point_features", lambda **_: None
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
