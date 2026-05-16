from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from regime_detection.config import NewsSentimentConfig, load_default_regime_config
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.market_context import build_market_context


def _sector_etf_closes(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    closes: dict[str, pd.Series] = {}
    for i, symbol in enumerate(SECTOR_ETFS):
        values = 100.0 * np.exp(np.arange(len(index)) * (0.0003 + i * 0.00002))
        closes[symbol] = pd.Series(values, index=index, name=symbol)
    return closes


def test_build_news_sentiment_score_series_preserves_existing_alignment_and_smoothing() -> None:
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
        sector_etf_closes=_sector_etf_closes(bootstrap.spy_ohlcv.index),
    )

    calls: list[int] = []

    def counting_realized_vol(close: pd.Series, window: int) -> pd.Series:
        if window == 21:
            calls.append(window)
        return pd.Series(0.2, index=close.index, name="realized_vol_21d")

    monkeypatch.setattr("regime_detection.feature_store.realized_vol", counting_realized_vol)
    monkeypatch.setattr("regime_detection.feature_store.compute_hmm_features", lambda **_: None)
    monkeypatch.setattr("regime_detection.feature_store.compute_clustering_features", lambda **_: None)
    monkeypatch.setattr("regime_detection.feature_store.compute_change_point_features", lambda **_: None)

    build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
        volatility_state_v2_config=cfg.volatility_state_v2,
        breadth_state_v2_config=cfg.breadth_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
    )

    assert calls == [21]
