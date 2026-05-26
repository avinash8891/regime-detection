"""SF Fed news sentiment coverage sanity check against the engine window.

Audit follow-up to V2 §1A news-sentiment wiring (see
`docs/regime_engine_v2_spec.md` Ambiguity Log #74 and
`docs/spec_code_data_audit_2026_05_15.md` §4.1).

This file ships TWO bodies of checks:

1. **Unit assertions on the loader semantics** — always run; no live
   parquet required. Verify the contract is right *in principle*.

2. **Live-parquet coverage assertions** — skip cleanly when the real
   `data/raw/news_sentiment/sf_fed_news_sentiment.parquet` is not in
   the checkout. These are the integration-style guards that catch
   real fetch breakage:

   - Every NYSE session in the engine window (`2016-01-04` →
     `max(spy_index)`) must have a non-NaN news_sentiment value after
     forward-fill. The SF Fed publishes truly daily (verified
     empirically: every consecutive-publish gap in the 2016+ window
     is exactly 1 day), so ffill onto NYSE sessions should yield zero
     NaN; if even one session is missing the fetch is broken.

   - The newest news_sentiment row must be no more than 30 calendar
     days behind today. The SF Fed refreshes weekly, so 30 days
     catches a stalled fetch with safety margin.

   - The maximum consecutive-publish gap must be ≤ 7 calendar days
     (matches the SF Fed's weekly publication cadence with
     publication-day-of-week jitter; tighter than the freshness gate
     to surface mid-history holes rather than tail staleness).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_detection.loaders import load_news_sentiment_series
from regime_shared.pandas_compat import cow_safe_assign

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NEWS_PARQUET = (
    _REPO_ROOT / "data" / "raw" / "news_sentiment" / "sf_fed_news_sentiment.parquet"
)
_DAILY_OHLCV_DIR = _REPO_ROOT / "data" / "raw" / "daily_ohlcv"

# Engine OHLCV floor — see `docs/regime_engine_v1_data_requirements.md`
# §1.4 and `market_data_fetch_plan.md` §2.3.
_ENGINE_WINDOW_START = pd.Timestamp("2016-01-04")

# Empirically-grounded thresholds (see `/tmp/inspect_news_gaps.py` 2026-05
# audit: real-data max consecutive-publish gap in the 2016+ window is
# 1 day; SF Fed publishes weekly per their methodology, hence 7d gap +
# 30d freshness with safety margin).
_MAX_GAP_DAYS = 7
_MAX_STALENESS_DAYS = 30


# ---------------------------------------------------------------------------
# Unit assertions (run always)
# ---------------------------------------------------------------------------


def test_load_news_sentiment_series_preserves_full_history_when_supplied() -> None:
    """The loader must not silently truncate to the engine window —
    truncation is the engine's job, not the data layer's."""
    df = pd.DataFrame(
        [
            {"date": "1990-01-02", "news_sentiment": -0.1},
            {"date": "2016-01-04", "news_sentiment": 0.05},
            {"date": "2024-12-31", "news_sentiment": 0.12},
        ]
    )
    s = load_news_sentiment_series(df)
    assert s.index.min() == pd.Timestamp("1990-01-02")
    assert s.index.max() == pd.Timestamp("2024-12-31")
    assert len(s) == 3


def test_reindex_with_ffill_fills_session_calendar_when_data_is_dense() -> None:
    """When the news series is daily (matches the live SF Fed cadence),
    ffill onto a sparser NYSE-session calendar must yield zero NaN."""
    daily = pd.Series(
        np.linspace(-0.1, 0.1, 30),
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
        name="news_sentiment",
    )
    # NYSE sessions are a subset (no weekends) — must always fill.
    nyse_subset = pd.date_range("2024-01-02", periods=20, freq="B")
    aligned = daily.reindex(nyse_subset, method="ffill")
    assert aligned.isna().sum() == 0


def test_reindex_with_ffill_surfaces_gaps_when_data_is_sparse() -> None:
    """If the source has a gap larger than the reindex target's first
    session, ffill produces NaN there. This is the failure mode the
    live-parquet test below catches."""
    sparse = pd.Series(
        [0.0, 0.5],
        index=[pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-20")],
        name="news_sentiment",
    )
    # First target session is BEFORE the first published value.
    target = pd.date_range("2024-01-01", "2024-01-25", freq="B")
    aligned = sparse.reindex(target, method="ffill")
    assert aligned.iloc[0:5].isna().all()  # before first publish
    assert aligned.loc["2024-01-22":].notna().all()  # after second publish


def _load_spy_session_index_from_source(source: Path) -> pd.DatetimeIndex:
    df = pd.read_parquet(source)
    spy = df[df["symbol"] == "SPY"].copy()
    spy = cow_safe_assign(spy, {"date": pd.to_datetime(spy["date"])})
    spy = spy.sort_values("date")
    return pd.DatetimeIndex(spy["date"])


def _write_news_coverage_fixture_parquets(tmp_path: Path) -> tuple[Path, Path]:
    news_path = tmp_path / "sf_fed_news_sentiment.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", "2016-01-15", freq="D"),
            "news_sentiment": np.linspace(-0.2, 0.2, 15),
            "source": "frbsf:daily_news_sentiment",
        }
    ).to_parquet(news_path, index=False)

    daily_path = tmp_path / "daily_ohlcv.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2016-01-04", "2016-01-15", freq="B"),
            "symbol": "SPY",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000,
        }
    ).to_parquet(daily_path, index=False)
    return news_path, daily_path


def test_fixture_news_sentiment_covers_every_engine_session_after_ffill(
    tmp_path: Path,
) -> None:
    news_path, daily_path = _write_news_coverage_fixture_parquets(tmp_path)

    raw = load_news_sentiment_series(news_path)
    spy_idx = _load_spy_session_index_from_source(daily_path)
    engine_sessions = spy_idx[spy_idx >= _ENGINE_WINDOW_START]
    aligned = raw.reindex(engine_sessions, method="ffill")

    assert len(engine_sessions) == 10
    assert aligned.isna().sum() == 0
    assert aligned.index.min() == pd.Timestamp("2016-01-04")
    assert aligned.index.max() == pd.Timestamp("2016-01-15")


def test_fixture_news_sentiment_gap_and_freshness_contracts(tmp_path: Path) -> None:
    news_path, _ = _write_news_coverage_fixture_parquets(tmp_path)

    raw = load_news_sentiment_series(news_path)
    engine_slice = raw[raw.index >= _ENGINE_WINDOW_START]
    gaps = engine_slice.index.to_series().diff().dt.days.dropna()
    newest = raw.index.max()
    today = pd.Timestamp("2016-02-01")

    assert int(gaps.max()) <= _MAX_GAP_DAYS
    assert (today - newest).days <= _MAX_STALENESS_DAYS


# ---------------------------------------------------------------------------
# Live-parquet integration assertions (skip when fetch not materialized)
# ---------------------------------------------------------------------------


def _load_spy_session_index() -> pd.DatetimeIndex:
    return _load_spy_session_index_from_source(_DAILY_OHLCV_DIR)


@pytest.mark.skipif(
    not _NEWS_PARQUET.exists() or not _DAILY_OHLCV_DIR.exists(),
    reason=(
        "Live coverage check requires both "
        "data/raw/news_sentiment/sf_fed_news_sentiment.parquet and "
        "data/raw/daily_ohlcv/ to be materialized. Run "
        "`scripts/fetch_regime_engine_v1_data.py` first."
    ),
)
def test_live_news_sentiment_covers_every_nyse_session_in_engine_window() -> None:
    """The strict coverage guarantee: zero NaN after ffill onto the SPY
    NYSE session calendar in the 2016+ engine window. If this fails the
    SF Fed fetch has a real hole — re-run the fetcher and investigate."""
    raw = load_news_sentiment_series(_NEWS_PARQUET)
    spy_idx = _load_spy_session_index()
    engine_sessions = spy_idx[spy_idx >= _ENGINE_WINDOW_START]
    assert len(engine_sessions) > 0, "SPY OHLCV has no rows ≥ 2016-01-04"
    aligned = raw.reindex(engine_sessions, method="ffill")
    nans = aligned[aligned.isna()]
    assert nans.empty, (
        f"news_sentiment has {len(nans)} NaN sessions after ffill onto "
        f"NYSE calendar; first 5: {list(nans.index[:5])}"
    )


@pytest.mark.skipif(
    not _NEWS_PARQUET.exists(),
    reason="Live news sentiment parquet not in checkout.",
)
def test_live_news_sentiment_max_consecutive_publish_gap_is_small() -> None:
    """The SF Fed publishes truly daily across the 2016+ window
    (empirically verified). Any gap > 7 days within the window means
    publication interruption — surface it loudly."""
    raw = load_news_sentiment_series(_NEWS_PARQUET)
    engine_slice = raw[raw.index >= _ENGINE_WINDOW_START]
    gaps = engine_slice.index.to_series().diff().dt.days.dropna()
    if gaps.empty:
        pytest.skip("News parquet has no 2016+ rows.")
    max_gap = int(gaps.max())
    assert max_gap <= _MAX_GAP_DAYS, (
        f"news_sentiment max consecutive-publish gap is {max_gap} days "
        f"(threshold {_MAX_GAP_DAYS}); largest gap ends at "
        f"{gaps.idxmax().date()}"
    )


@pytest.mark.skipif(
    not _NEWS_PARQUET.exists(),
    reason="Live news sentiment parquet not in checkout.",
)
def test_live_news_sentiment_is_fresh_within_30_days() -> None:
    """Catch stalled-fetch staleness. The SF Fed refreshes weekly per
    their methodology page, so 30 days is a 4x safety margin."""
    raw = load_news_sentiment_series(_NEWS_PARQUET)
    if raw.empty:
        pytest.skip("News parquet is empty.")
    newest = raw.index.max()
    today = pd.Timestamp(dt.date.today())
    staleness = (today - newest).days
    assert staleness <= _MAX_STALENESS_DAYS, (
        f"news_sentiment newest row is {staleness} days behind today "
        f"({newest.date()} vs {today.date()}); threshold "
        f"{_MAX_STALENESS_DAYS}. Re-run "
        f"`run_sf_fed_news_sentiment_fetch` to refresh."
    )
