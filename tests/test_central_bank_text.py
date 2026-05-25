"""Tests for v2 §2A central-bank-text classifier (audit M1).

Spec authority: docs/regime_engine_v2_spec.md §2A lines 2578-2586.
Resolution: docs/spec_code_data_audit_2026_05_15.md §3.1.

Per CLAUDE.md testing rules:
- Use real FOMC / Powell phrasing for body_text fixtures (not toy "test
  text"). The lexicon-scorer's job is to discriminate real central-
  banking prose, so the inputs must look like real central-banking prose.
- Test the integration path (loader → feature_store → MonetaryPressureV2
  Features.central_bank_text_score), not just the unit.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from regime_detection.central_bank_text import (
    DOVISH_TERMS,
    HAWKISH_TERMS,
    combine_release_frames,
    score_release_frame,
    score_text,
    to_daily_score_series,
)
from regime_detection.loaders import load_central_bank_text_score

# Real FOMC-style phrasing. These are intentionally faithful to actual
# FOMC minutes wording so the lexicon test reflects production semantics.
_HAWKISH_FOMC_BODY = (
    "Participants observed that inflationary pressures remained "
    "elevated and that the labor market was tight. Most members "
    "judged that further policy firming would be appropriate to "
    "withdraw accommodation, hike the federal funds rate, and "
    "anchor expectations. Upside risks to inflation persisted."
)

_DOVISH_FOMC_BODY = (
    "Several participants noted disinflation across core categories "
    "and softening in labor market slack. A majority judged that "
    "patience and an accommodative stance remained appropriate; rate "
    "cuts would be considered if downside risks to growth materialize. "
    "The Committee remains supportive of the economy."
)

_NEUTRAL_BODY = (
    "The Committee discussed economic projections and reviewed staff "
    "outlooks. Members exchanged views on developments in domestic "
    "and international markets."
)


def test_score_text_hawkish_body_yields_strongly_positive_net_score() -> None:
    score = score_text(_HAWKISH_FOMC_BODY)
    assert score.hawkish_count > 0
    assert score.dovish_count == 0 or score.dovish_count < score.hawkish_count
    assert score.net_score > 0.5  # strongly hawkish


def test_score_text_dovish_body_yields_strongly_negative_net_score() -> None:
    score = score_text(_DOVISH_FOMC_BODY)
    assert score.dovish_count > 0
    assert score.hawkish_count == 0 or score.hawkish_count < score.dovish_count
    assert score.net_score < -0.5  # strongly dovish


def test_score_text_neutral_body_yields_nan() -> None:
    """A document with no lexicon hits is NaN per the audit M1 contract."""
    score = score_text(_NEUTRAL_BODY)
    assert score.hawkish_count == 0
    assert score.dovish_count == 0
    assert np.isnan(score.net_score)


def test_score_text_empty_string_is_safe() -> None:
    score = score_text("")
    assert score.hawkish_count == 0
    assert score.dovish_count == 0
    assert np.isnan(score.net_score)


def test_score_text_is_deterministic() -> None:
    """V1 §2.2 stateless replay — same input must yield identical output."""
    a = score_text(_HAWKISH_FOMC_BODY)
    b = score_text(_HAWKISH_FOMC_BODY)
    assert a == b


def test_lexicons_have_no_overlapping_terms() -> None:
    """Hawkish and dovish vocabularies must be disjoint to avoid
    self-cancelling on neutral phrasing."""
    overlap = set(HAWKISH_TERMS) & set(DOVISH_TERMS)
    assert overlap == set()


def test_score_release_frame_returns_one_row_per_release() -> None:
    fomc_releases = pd.DataFrame(
        [
            {
                "release_timestamp": "2024-01-31 14:00:00",
                "body_text": _HAWKISH_FOMC_BODY,
            },
            {
                "release_timestamp": "2024-03-20 14:00:00",
                "body_text": _DOVISH_FOMC_BODY,
            },
        ]
    )
    out = score_release_frame(
        fomc_releases, date_column="release_timestamp", source_label="fomc_minutes"
    )
    assert len(out) == 2
    assert set(out.columns) == {
        "release_date",
        "hawkish_count",
        "dovish_count",
        "total_tokens",
        "net_score",
        "source",
    }
    # Sorted by release_date
    assert out.iloc[0]["release_date"] == dt.date(2024, 1, 31)
    assert out.iloc[0]["net_score"] > 0
    assert out.iloc[1]["net_score"] < 0


def test_combine_release_frames_concatenates_and_sorts() -> None:
    fomc = pd.DataFrame(
        [{"release_timestamp": "2024-01-31", "body_text": _HAWKISH_FOMC_BODY}]
    )
    powell = pd.DataFrame(
        [{"publication_timestamp": "2024-02-15", "body_text": _DOVISH_FOMC_BODY}]
    )
    fomc_scored = score_release_frame(
        fomc, date_column="release_timestamp", source_label="fomc_minutes"
    )
    powell_scored = score_release_frame(
        powell, date_column="publication_timestamp", source_label="powell_speech"
    )
    combined = combine_release_frames(fomc_scored, powell_scored)
    assert len(combined) == 2
    assert list(combined["source"]) == ["fomc_minutes", "powell_speech"]


def test_to_daily_score_series_forward_fills_per_v1_replay_rule() -> None:
    """V1 §2.2 — each session reads the latest score with
    release_date <= as_of_date. NEVER consult a future-dated reading."""
    scored = pd.DataFrame(
        [
            {
                "release_date": dt.date(2024, 1, 31),
                "hawkish_count": 5,
                "dovish_count": 0,
                "total_tokens": 100,
                "net_score": 1.0,
                "source": "fomc_minutes",
            },
            {
                "release_date": dt.date(2024, 3, 20),
                "hawkish_count": 0,
                "dovish_count": 5,
                "total_tokens": 100,
                "net_score": -1.0,
                "source": "fomc_minutes",
            },
        ]
    )
    sessions = pd.date_range("2024-01-30", "2024-04-01", freq="B")
    daily = to_daily_score_series(
        scored, session_index=sessions, smoothing_window_sessions=1
    )
    # 2024-01-30 — BEFORE the first release: NaN (no past reading).
    assert np.isnan(daily.loc["2024-01-30"])
    # 2024-02-01 — AFTER the first release: hawkish carries.
    assert daily.loc["2024-02-01"] == 1.0
    # 2024-03-19 — DAY BEFORE the dovish release: still hawkish.
    assert daily.loc["2024-03-19"] == 1.0
    # 2024-03-20 — release date itself: dovish takes over.
    assert daily.loc["2024-03-20"] == -1.0
    # 2024-04-01 — well after dovish release: dovish carries.
    assert daily.loc["2024-04-01"] == -1.0


def test_to_daily_score_series_empty_releases_returns_all_nan() -> None:
    sessions = pd.date_range("2024-01-01", "2024-01-10", freq="B")
    daily = to_daily_score_series(
        pd.DataFrame(
            columns=[
                "release_date",
                "hawkish_count",
                "dovish_count",
                "total_tokens",
                "net_score",
                "source",
            ]
        ),
        session_index=sessions,
    )
    assert daily.isna().all()
    assert daily.name == "central_bank_text_score"


def test_to_daily_score_series_dedupes_same_date_picks_higher_token_row() -> None:
    """When FOMC and Powell collide on the same date, the row with more
    body tokens (likely more material content) wins (default strategy)."""
    scored = pd.DataFrame(
        [
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 1,
                "dovish_count": 0,
                "total_tokens": 50,
                "net_score": 1.0,
                "source": "powell_speech",
            },
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 0,
                "dovish_count": 1,
                "total_tokens": 500,  # FOMC minutes are longer
                "net_score": -1.0,
                "source": "fomc_minutes",
            },
        ]
    )
    sessions = pd.date_range("2024-02-01", "2024-02-05", freq="B")
    daily = to_daily_score_series(
        scored, session_index=sessions, smoothing_window_sessions=1
    )
    assert daily.loc["2024-02-01"] == -1.0  # the longer FOMC row won


def _two_source_collision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 2,
                "dovish_count": 0,
                "total_tokens": 100,
                "net_score": 1.0,
                "source": "powell_speech",
            },
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 0,
                "dovish_count": 3,
                "total_tokens": 300,
                "net_score": -1.0,
                "source": "fomc_minutes",
            },
        ]
    )


def test_same_date_aggregation_token_weighted_average() -> None:
    """token_weighted_average: net_score = (100*1 + 300*(-1)) / 400 = -0.5."""
    sessions = pd.date_range("2024-02-01", "2024-02-05", freq="B")
    daily = to_daily_score_series(
        _two_source_collision_frame(),
        session_index=sessions,
        smoothing_window_sessions=1,
        same_date_aggregation="token_weighted_average",
    )
    assert daily.loc["2024-02-01"] == pytest.approx(-0.5, abs=1e-9)


def test_same_date_aggregation_fomc_priority_picks_fomc() -> None:
    """fomc_priority: FOMC minutes win even if Powell speech is longer."""
    # Inverted token counts: Powell longer but FOMC still wins.
    scored = pd.DataFrame(
        [
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 5,
                "dovish_count": 0,
                "total_tokens": 1000,
                "net_score": 1.0,
                "source": "powell_speech",
            },
            {
                "release_date": dt.date(2024, 2, 1),
                "hawkish_count": 0,
                "dovish_count": 2,
                "total_tokens": 100,
                "net_score": -1.0,
                "source": "fomc_minutes",
            },
        ]
    )
    sessions = pd.date_range("2024-02-01", "2024-02-05", freq="B")
    daily = to_daily_score_series(
        scored,
        session_index=sessions,
        smoothing_window_sessions=1,
        same_date_aggregation="fomc_priority",
    )
    assert daily.loc["2024-02-01"] == -1.0  # FOMC dovish row wins


def test_same_date_aggregation_unknown_strategy_raises() -> None:
    sessions = pd.date_range("2024-02-01", "2024-02-05", freq="B")
    try:
        to_daily_score_series(
            _two_source_collision_frame(),
            session_index=sessions,
            same_date_aggregation="majority_vote",  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "same_date_aggregation" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown strategy")


def test_load_central_bank_text_score_integrates_fomc_and_powell_frames() -> None:
    fomc_df = pd.DataFrame(
        [
            {
                "release_timestamp": "2024-01-31 14:00:00",
                "body_text": _HAWKISH_FOMC_BODY,
            },
        ]
    )
    powell_df = pd.DataFrame(
        [
            {"publication_timestamp": "2024-02-15", "body_text": _DOVISH_FOMC_BODY},
        ]
    )
    out = load_central_bank_text_score(
        fomc_minutes_source=fomc_df, powell_speeches_source=powell_df
    )
    assert len(out) == 2
    assert set(out["source"]) == {"fomc_minutes", "powell_speech"}


def test_load_central_bank_text_score_empty_inputs_returns_empty_frame() -> None:
    out = load_central_bank_text_score()
    assert out.empty


def test_load_central_bank_text_score_respects_max_release_age() -> None:
    fomc_df = pd.DataFrame(
        [
            {
                "release_timestamp": "2020-01-31",
                "body_text": _HAWKISH_FOMC_BODY,
            },  # too old
            {"release_timestamp": "2024-01-31", "body_text": _DOVISH_FOMC_BODY},
        ]
    )
    out = load_central_bank_text_score(
        fomc_minutes_source=fomc_df,
        max_release_age_days=365,
        as_of_date=pd.Timestamp("2024-12-31"),
    )
    assert len(out) == 1
    assert out.iloc[0]["release_date"] == dt.date(2024, 1, 31)
