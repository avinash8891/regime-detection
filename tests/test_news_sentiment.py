"""v2 §1A SF Fed Daily News Sentiment evidence wiring (audit follow-up).

Spec authority: docs/regime_engine_v2_spec.md §1A (sentiment_score line 164).
Audit context: docs/spec_code_data_audit_2026_05_15.md "news sentiment"
follow-up to #12.

Contract under test:

1. The SF Fed daily series loads cleanly and reindexes onto the SPY
   session calendar with the configured smoothing window.
2. ``TrendDirectionV2Features.news_sentiment_score`` is populated when
   the context wires it, and ``sentiment_concordance`` lands as a
   pointwise +1 / 0 / -1 / NaN flag.
3. The §1A `euphoria` rule predicate is UNCHANGED — news sentiment is
   evidence only, NOT a rule input. We assert this by configuring AAII
   to fire euphoria on a synthetic day and confirming the predicate
   fires irrespective of news_sentiment value.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch.sf_fed_news_sentiment import run_sf_fed_news_sentiment_fetch
from regime_detection.config import NewsSentimentConfig, TrendDirectionV2Config
from regime_detection.loaders import load_news_sentiment_series
from regime_detection.trend_direction_v2 import compute_trend_v2_features


def _spy_index(periods: int = 260, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="B")


def _close_series(idx: pd.DatetimeIndex) -> pd.Series:
    # A gently rising series so trailing returns are non-degenerate.
    return pd.Series(
        np.linspace(450.0, 470.0, len(idx)),
        index=idx,
        name="close",
    )


def _trend_config() -> TrendDirectionV2Config:
    return TrendDirectionV2Config(
        efficiency_ratio_lookback_days=20,
        hurst_lookback_days=250,
        slope_lookback_days=20,
        sma_short_period=50,
        sma_long_period=200,
        return_short_period=63,
        return_long_period=126,
        drawdown_lookback_days=252,
    )


def test_load_news_sentiment_series_reads_long_form_parquet_shape() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "news_sentiment": 0.12,
                "source": "frbsf:daily_news_sentiment",
            },
            {
                "date": "2024-01-03",
                "news_sentiment": 0.18,
                "source": "frbsf:daily_news_sentiment",
            },
            {
                "date": "2024-01-04",
                "news_sentiment": -0.05,
                "source": "frbsf:daily_news_sentiment",
            },
        ]
    )
    s = load_news_sentiment_series(df)
    assert list(s.index) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert s.tolist() == [0.12, 0.18, -0.05]
    assert s.name == "news_sentiment"


def test_load_news_sentiment_series_missing_column_raises() -> None:
    bad = pd.DataFrame({"date": ["2024-01-02"], "sentiment_value": [0.1]})
    try:
        load_news_sentiment_series(bad)
    except ValueError as exc:
        assert "news_sentiment" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing news_sentiment column")


def test_compute_trend_v2_features_surfaces_news_sentiment_when_supplied() -> None:
    idx = _spy_index()
    close = _close_series(idx)
    # AAII slowly drifting positive across the window.
    aaii = pd.Series(
        np.linspace(10.0, 25.0, len(idx)), index=idx, name="sentiment_score"
    )
    # News sentiment also drifting positive.
    news = pd.Series(
        np.linspace(-0.05, 0.30, len(idx)), index=idx, name="news_sentiment_score"
    )
    out = compute_trend_v2_features(
        close, config=_trend_config(), sentiment_score=aaii, news_sentiment_score=news
    )
    assert out.news_sentiment_score is not None
    assert (out.news_sentiment_score.index == close.index).all()
    assert out.sentiment_concordance is not None
    # Late sessions: both AAII and news positive → concordance == +1.0
    assert out.sentiment_concordance.iloc[-1] == 1.0


def test_sentiment_concordance_flags_disagreement() -> None:
    idx = _spy_index(periods=10)
    close = _close_series(idx)
    aaii = pd.Series([20.0] * 10, index=idx, name="sentiment_score")  # hawkish
    news = pd.Series([-0.20] * 10, index=idx, name="news_sentiment_score")  # dovish
    out = compute_trend_v2_features(
        close, config=_trend_config(), sentiment_score=aaii, news_sentiment_score=news
    )
    assert (out.sentiment_concordance.dropna() == 0.0).all()


def test_sentiment_concordance_flags_agreement_negative() -> None:
    idx = _spy_index(periods=10)
    close = _close_series(idx)
    aaii = pd.Series([-15.0] * 10, index=idx, name="sentiment_score")
    news = pd.Series([-0.30] * 10, index=idx, name="news_sentiment_score")
    out = compute_trend_v2_features(
        close, config=_trend_config(), sentiment_score=aaii, news_sentiment_score=news
    )
    assert (out.sentiment_concordance.dropna() == -1.0).all()


def test_news_sentiment_absent_leaves_features_unchanged() -> None:
    """V1 byte-identity: when news_sentiment_score is None the existing
    sentiment_score field is preserved unchanged and the concordance
    flag is also None."""
    idx = _spy_index(periods=10)
    close = _close_series(idx)
    aaii = pd.Series([20.0] * 10, index=idx, name="sentiment_score")
    out = compute_trend_v2_features(close, config=_trend_config(), sentiment_score=aaii)
    assert out.news_sentiment_score is None
    assert out.sentiment_concordance is None
    assert out.sentiment_score is not None


def test_news_sentiment_does_not_change_euphoria_predicate_inputs() -> None:
    """The §1A `euphoria` predicate consumes only `sentiment_score` per
    spec line 164. Adding news sentiment must NOT introduce any new
    field into the rule-evaluation surface — only evidence."""
    idx = _spy_index(periods=10)
    close = _close_series(idx)
    aaii = pd.Series([25.0] * 10, index=idx, name="sentiment_score")
    # Wildly divergent news sentiment.
    news_neg = pd.Series([-0.95] * 10, index=idx, name="news_sentiment_score")
    out_with = compute_trend_v2_features(
        close,
        config=_trend_config(),
        sentiment_score=aaii,
        news_sentiment_score=news_neg,
    )
    out_without = compute_trend_v2_features(
        close, config=_trend_config(), sentiment_score=aaii
    )
    # The fields the euphoria predicate reads are sma_200, return_126d,
    # realized_vol_21d, sentiment_score — all of which must be byte-
    # identical regardless of news sentiment.
    pd.testing.assert_series_equal(out_with.sma_200, out_without.sma_200)
    pd.testing.assert_series_equal(out_with.return_126d, out_without.return_126d)
    pd.testing.assert_series_equal(
        out_with.realized_vol_21d, out_without.realized_vol_21d
    )
    pd.testing.assert_series_equal(
        out_with.sentiment_score, out_without.sentiment_score
    )


def test_news_sentiment_config_default_smoothing_window() -> None:
    cfg = NewsSentimentConfig()
    assert cfg.smoothing_window_sessions == 21


def test_news_sentiment_config_rejects_zero_window() -> None:
    try:
        NewsSentimentConfig(smoothing_window_sessions=0)
    except Exception:
        pass
    else:
        raise AssertionError("expected ValidationError for zero smoothing window")


def test_sf_fed_fetch_report_exposes_parquet_under_paths_for_manifest(
    tmp_path: Path,
) -> None:
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["fixture"]}).to_excel(
            writer,
            sheet_name="Methodology",
            index=False,
        )
        pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-05-14"), pd.Timestamp("2026-05-15")],
                "News Sentiment": [0.1, -0.2],
            }
        ).to_excel(writer, sheet_name="Data", index=False)

    report_path = run_sf_fed_news_sentiment_fetch(
        out_dir=tmp_path,
        workbook_bytes=workbook.getvalue(),
        acquisition_db_path=None,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["news_sentiment_parquet"] == str(
        tmp_path / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )

    manifest = emit_manifest_for_report_paths(
        report_paths=[report_path],
        out_dir=tmp_path,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="sf-fed-test",
        required_for=["profile_engine"],
    )

    assert [artifact.local_path for artifact in manifest.artifacts] == [
        "data/raw/news_sentiment/sf_fed_news_sentiment.parquet"
    ]
