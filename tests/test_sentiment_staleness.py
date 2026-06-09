"""Tests for the AAII and news-sentiment max-staleness guards.

The staleness guard NaN-outs forward-filled sentiment_score values when the
last real AAII reading is older than ``SentimentScoreConfig.max_staleness_sessions``
NYSE sessions. This prevents the euphoria gate from firing on arbitrarily
stale AAII data if the survey stops publishing.

An identical guard exists for SF Fed news sentiment in
``NewsSentimentConfig.max_staleness_sessions`` and
``_build_news_sentiment_score_series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection._config_layer2 import NewsSentimentConfig, SentimentScoreConfig
from regime_detection._feature_specs import (
    _build_news_sentiment_score_series,
    _build_sentiment_score_series,
)


def _make_aaii_frame(
    publication_dates: list[str],
    spreads: list[float],
) -> pd.DataFrame:
    """Build a minimal AAII DataFrame with publication_date and bull_bear_spread_8w_ma."""
    return pd.DataFrame(
        {
            "publication_date": pd.to_datetime(publication_dates),
            "bull_bear_spread_8w_ma": spreads,
        }
    )


def _make_session_index(start: str, periods: int) -> pd.DatetimeIndex:
    """Generate a business-day session index (NYSE proxy)."""
    return pd.bdate_range(start=start, periods=periods, freq="B")


class TestSentimentStalenessGuard:
    """If the last AAII reading is older than max_staleness_sessions,
    sentiment_score should be NaN -- not a stale forward-fill."""

    def test_aaii_sentiment_goes_nan_after_max_staleness(self) -> None:
        """100 NYSE sessions, 5 weekly AAII readings in first 25 sessions,
        then nothing. With max_staleness_sessions=40:
          - Session 30 (few sessions stale) -> valid
          - Session 70 (many sessions stale) -> NaN
          - Session 99 (far past staleness) -> NaN
        """
        sessions = _make_session_index("2024-01-02", periods=100)

        # Place 5 weekly readings in the first ~25 sessions (enough to pass
        # the 4-reading cold-start warmup).
        pub_dates = [str(sessions[i].date()) for i in [0, 5, 10, 15, 20]]
        spreads = [0.10, 0.12, 0.11, 0.13, 0.14]
        aaii = _make_aaii_frame(pub_dates, spreads)

        config = SentimentScoreConfig(max_staleness_sessions=40)

        result = _build_sentiment_score_series(
            aaii_sentiment=aaii,
            session_index=sessions,
            config=config,
        )

        assert result is not None

        # Session 30 is ~10 sessions after last reading at session 20 -> valid
        assert not np.isnan(
            result.iloc[30]
        ), f"Session 30 should be valid (only ~10 sessions stale), got {result.iloc[30]}"

        # Session 70 is ~50 sessions after last reading at session 20 -> stale (> 40)
        assert np.isnan(
            result.iloc[70]
        ), f"Session 70 should be NaN (50 sessions stale > 40 max), got {result.iloc[70]}"

        # Last session is ~79 sessions after last reading -> stale
        assert np.isnan(
            result.iloc[99]
        ), f"Session 99 should be NaN (79 sessions stale > 40 max), got {result.iloc[99]}"

    def test_no_staleness_without_config(self) -> None:
        """Without a SentimentScoreConfig, ffill persists indefinitely
        (backward-compatible default behavior)."""
        sessions = _make_session_index("2024-01-02", periods=100)

        pub_dates = [str(sessions[i].date()) for i in [0, 5, 10, 15, 20]]
        spreads = [0.10, 0.12, 0.11, 0.13, 0.14]
        aaii = _make_aaii_frame(pub_dates, spreads)

        result = _build_sentiment_score_series(
            aaii_sentiment=aaii,
            session_index=sessions,
            config=None,
        )

        assert result is not None

        # Even 79 sessions after last reading, the value should persist
        assert not np.isnan(
            result.iloc[99]
        ), "Without config, ffill should persist indefinitely"

    def test_staleness_boundary_exact(self) -> None:
        """Verify the exact boundary: session at max_staleness_sessions is valid,
        session at max_staleness_sessions + 1 is NaN."""
        sessions = _make_session_index("2024-01-02", periods=80)

        # 5 readings to pass warmup, last real reading at session index 10
        pub_dates = [str(sessions[i].date()) for i in [0, 3, 5, 7, 10]]
        spreads = [0.10, 0.11, 0.12, 0.13, 0.14]
        aaii = _make_aaii_frame(pub_dates, spreads)

        config = SentimentScoreConfig(max_staleness_sessions=20)

        result = _build_sentiment_score_series(
            aaii_sentiment=aaii,
            session_index=sessions,
            config=config,
        )

        assert result is not None

        # Session 30 is exactly 20 sessions after session 10 -> valid (<=20)
        assert not np.isnan(
            result.iloc[30]
        ), f"Session 30 should be valid (exactly at staleness boundary), got {result.iloc[30]}"

        # Session 31 is 21 sessions after session 10 -> stale (>20)
        assert np.isnan(
            result.iloc[31]
        ), f"Session 31 should be NaN (1 session past staleness boundary), got {result.iloc[31]}"

    def test_staleness_resets_on_new_reading(self) -> None:
        """A new AAII reading resets the staleness counter."""
        sessions = _make_session_index("2024-01-02", periods=100)

        # 4 readings for warmup, then a gap, then a new reading at session 60
        pub_dates = [str(sessions[i].date()) for i in [0, 3, 5, 7, 60]]
        spreads = [0.10, 0.11, 0.12, 0.13, 0.20]
        aaii = _make_aaii_frame(pub_dates, spreads)

        config = SentimentScoreConfig(max_staleness_sessions=30)

        result = _build_sentiment_score_series(
            aaii_sentiment=aaii,
            session_index=sessions,
            config=config,
        )

        assert result is not None

        # Session 40 is 33 sessions after last warmup reading at 7 -> stale
        assert np.isnan(
            result.iloc[40]
        ), "Session 40 should be NaN (stale before the new reading at 60)"

        # Session 70 is 10 sessions after the reading at 60 -> valid
        assert not np.isnan(
            result.iloc[70]
        ), f"Session 70 should be valid (10 sessions after new reading), got {result.iloc[70]}"

        # Session 95 is 35 sessions after reading at 60 -> stale again (>30)
        assert np.isnan(
            result.iloc[95]
        ), f"Session 95 should be NaN (35 sessions after last reading > 30 max), got {result.iloc[95]}"

    def test_config_validation_rejects_zero_staleness(self) -> None:
        """max_staleness_sessions must be > 0."""
        with pytest.raises(Exception):
            SentimentScoreConfig(max_staleness_sessions=0)

    def test_config_default_is_40(self) -> None:
        """Default max_staleness_sessions is 40 (approx 8 weeks)."""
        config = SentimentScoreConfig()
        assert config.max_staleness_sessions == 40


class TestNewsSentimentStalenessGuard:
    """SF Fed news-sentiment staleness guard mirrors the AAII guard.

    After ffill, sessions whose last real observation is older than
    ``NewsSentimentConfig.max_staleness_sessions`` NYSE sessions are NaN-ed out.
    """

    def test_news_sentiment_goes_nan_after_max_staleness(self) -> None:
        """100 NYSE sessions, daily readings for first 20 sessions, then nothing.
        With max_staleness_sessions=30:
          - Session 40 (20 sessions stale) -> valid
          - Session 60 (40 sessions stale) -> NaN
        """
        sessions = _make_session_index("2024-01-02", periods=100)

        # Daily readings for first 20 sessions.
        raw_index = sessions[:20]
        raw_values = np.linspace(0.1, 0.3, 20)
        news_sentiment = pd.Series(raw_values, index=raw_index, name="news_sentiment")

        config = NewsSentimentConfig(max_staleness_sessions=30)

        result = _build_news_sentiment_score_series(
            news_sentiment=news_sentiment,
            session_index=sessions,
            config=config,
        )

        assert result is not None

        # Session 40 is 20 sessions after last reading at session 19 -> valid (<= 30)
        assert not np.isnan(
            result.iloc[40]
        ), f"Session 40 should be valid (20 sessions stale <= 30 max), got {result.iloc[40]}"

        # Session 60 is 40 sessions after last reading at session 19 -> stale (> 30)
        assert np.isnan(
            result.iloc[60]
        ), f"Session 60 should be NaN (40 sessions stale > 30 max), got {result.iloc[60]}"

    def test_news_sentiment_config_default_is_63(self) -> None:
        """Default max_staleness_sessions is 63 (approx 3 months)."""
        config = NewsSentimentConfig()
        assert config.max_staleness_sessions == 63

    def test_news_sentiment_config_rejects_zero_staleness(self) -> None:
        """max_staleness_sessions must be > 0."""
        with pytest.raises(Exception):
            NewsSentimentConfig(max_staleness_sessions=0)
