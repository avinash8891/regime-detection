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
import sqlite3
import urllib.error
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch import sf_fed_news_sentiment
from regime_data_fetch.sf_fed_news_sentiment import (
    SFFedNewsSentimentFetchError,
    fetch_workbook_bytes,
    parse_workbook,
    run_sf_fed_news_sentiment_fetch,
)
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


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_BytesResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _news_sentiment_workbook_bytes(
    *,
    data_sheet_name: str = "Data",
    include_methodology: bool = True,
    rows: list[dict[str, object]] | None = None,
) -> bytes:
    workbook = io.BytesIO()
    if rows is None:
        rows = [
            {"Date": pd.Timestamp("2026-05-14"), "News Sentiment": 0.1},
            {"Date": pd.Timestamp("2026-05-15"), "News Sentiment": -0.2},
        ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        if include_methodology:
            pd.DataFrame({"note": ["fixture"]}).to_excel(
                writer,
                sheet_name="Methodology",
                index=False,
            )
        pd.DataFrame(rows).to_excel(writer, sheet_name=data_sheet_name, index=False)
    return workbook.getvalue()


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
    report_path = run_sf_fed_news_sentiment_fetch(
        out_dir=tmp_path,
        workbook_bytes=_news_sentiment_workbook_bytes(),
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


def test_parse_workbook_falls_back_to_second_sheet_when_data_sheet_is_renamed() -> None:
    frame = parse_workbook(
        _news_sentiment_workbook_bytes(data_sheet_name="Daily News Sentiment")
    )

    assert frame.to_dict(orient="records") == [
        {
            "date": pd.Timestamp("2026-05-14"),
            "news_sentiment": 0.1,
            "source": "frbsf:daily_news_sentiment",
            "source_url": sf_fed_news_sentiment.SF_FED_NEWS_SENTIMENT_URL,
        },
        {
            "date": pd.Timestamp("2026-05-15"),
            "news_sentiment": -0.2,
            "source": "frbsf:daily_news_sentiment",
            "source_url": sf_fed_news_sentiment.SF_FED_NEWS_SENTIMENT_URL,
        },
    ]


def test_parse_workbook_single_sheet_shape_failure_is_loud() -> None:
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"Date": [pd.Timestamp("2026-05-14")]}).to_excel(
            writer,
            sheet_name="OnlySheet",
            index=False,
        )

    with pytest.raises(SFFedNewsSentimentFetchError, match="unexpected shape"):
        parse_workbook(workbook.getvalue())


def test_fetch_workbook_bytes_wraps_urlerror_with_source_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(req, *, timeout: int):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = req.headers["User-agent"]
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(sf_fed_news_sentiment.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(
        SFFedNewsSentimentFetchError, match="failed to download"
    ) as excinfo:
        fetch_workbook_bytes(timeout=19)

    assert captured["url"] == sf_fed_news_sentiment.SF_FED_NEWS_SENTIMENT_URL
    assert captured["timeout"] == 19
    assert "Chrome/126.0.0.0" in captured["user_agent"]
    assert isinstance(excinfo.value.__cause__, urllib.error.URLError)
    assert "network unreachable" in str(excinfo.value)


def test_run_sf_fed_fetch_reads_operator_staged_workbook_and_records_artifact_ledger(
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "manual" / "news_sentiment_data.xlsx"
    workbook_path.parent.mkdir()
    workbook_path.write_bytes(_news_sentiment_workbook_bytes())
    acquisition_db = tmp_path / "acquisition.db"

    report_path = run_sf_fed_news_sentiment_fetch(
        out_dir=tmp_path,
        workbook_path=workbook_path,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["rows"] == 2
    assert report["min_date"] == "2026-05-14"
    assert report["max_date"] == "2026-05-15"
    assert not (tmp_path / "news_sentiment" / "sf_fed_news_sentiment.xlsx").exists()
    with closing(sqlite3.connect(acquisition_db)) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, source_identifier, start_date, end_date FROM artifacts"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind, row_count, min_date, max_date FROM derived_outputs ORDER BY output_id"
        ).fetchall()
        lineage_count = conn.execute(
            "SELECT count(*) FROM artifact_lineage"
        ).fetchone()[0]

    assert fetch_runs == [("sf_fed_news_sentiment", "ok")]
    assert artifacts == [
        (
            "frbsf:daily_news_sentiment",
            "xlsx",
            sf_fed_news_sentiment.SF_FED_NEWS_SENTIMENT_URL,
            "2026-05-14",
            "2026-05-15",
        )
    ]
    assert outputs == [
        ("sf_fed_news_sentiment_parquet", 2, "2026-05-14", "2026-05-15"),
        ("sf_fed_news_sentiment_report", 2, "2026-05-14", "2026-05-15"),
    ]
    assert lineage_count == 1


def test_run_sf_fed_fetch_live_download_writes_raw_workbook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _news_sentiment_workbook_bytes()

    def fake_urlopen(req, *, timeout: int):
        assert req.full_url == sf_fed_news_sentiment.SF_FED_NEWS_SENTIMENT_URL
        assert timeout == 11
        return _BytesResponse(payload)

    monkeypatch.setattr(sf_fed_news_sentiment.urllib.request, "urlopen", fake_urlopen)

    report_path = run_sf_fed_news_sentiment_fetch(out_dir=tmp_path, timeout=11)

    report = json.loads(report_path.read_text())
    raw_path = tmp_path / "news_sentiment" / "sf_fed_news_sentiment.xlsx"
    parquet_path = tmp_path / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    assert raw_path.read_bytes() == payload
    assert report["paths"]["news_sentiment_parquet"] == str(parquet_path)
    assert pd.read_parquet(parquet_path)["news_sentiment"].tolist() == [0.1, -0.2]


def test_run_sf_fed_fetch_marks_acquisition_run_failed_on_bad_workbook(
    tmp_path: Path,
) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    with pytest.raises(SFFedNewsSentimentFetchError, match="unexpected shape"):
        run_sf_fed_news_sentiment_fetch(
            out_dir=tmp_path,
            workbook_bytes=_news_sentiment_workbook_bytes(
                include_methodology=False,
                rows=[{"Date": pd.Timestamp("2026-05-14")}],
            ),
            acquisition_db_path=acquisition_db,
        )

    with closing(sqlite3.connect(acquisition_db)) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status, notes FROM fetch_runs"
        ).fetchall()

    assert len(fetch_runs) == 1
    assert fetch_runs[0][0] == "sf_fed_news_sentiment"
    assert fetch_runs[0][1] == "failed"
    assert "unexpected shape" in fetch_runs[0][2]
