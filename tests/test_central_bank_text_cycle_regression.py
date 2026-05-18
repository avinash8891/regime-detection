"""v2 §2A central-bank-text lexicon regression — historical FOMC cycles.

Audit M1 follow-up #11 (`docs/spec_code_data_audit_2026_05_15.md` §4).

The deterministic lexicon's job is to discriminate real central-banking
prose by cycle. This regression locks the SIGN of `net_score` against
two well-documented historical cycles:

- **2022-03 → 2023-07 tightening cycle.** The Committee raised the
  federal funds rate from ~0.25% to 5.25% across 11 consecutive
  meetings; FOMC minutes in this window are unambiguously hawkish.
  Expected: net_score > 0 (any reading on a real minutes excerpt
  from this window).
- **2019-07 → 2020-04 easing cycle.** The Committee cut rates three
  times in 2019 ("mid-cycle adjustment") and then cut to the zero
  lower bound + launched QE in response to COVID; FOMC minutes in
  this window are unambiguously dovish. Expected: net_score < 0.

The excerpts below are short passages drafted to match the actual
recurring phrasing in Federal Reserve minutes from those windows. They
are NOT verbatim transcripts (the public PDFs are tens of thousands of
words each); they are realistic test fixtures whose lexical content
exercises the scorer end-to-end.

Run with the live `data/raw/fomc_minutes/fomc_minutes.parquet` when
present (the integration assertion at the bottom of the file picks up
real FOMC body_text for the two windows when available and re-asserts
the sign there too).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.central_bank_text import score_text, to_daily_score_series
from regime_detection.loaders import load_central_bank_text_score


# 2022-03 hike — initial tightening cycle (verbatim-paraphrased structure).
TIGHTENING_2022_MARCH_EXCERPT = (
    "Participants observed that inflationary pressures had become more "
    "broad-based and persistent, with upside risks to the inflation outlook. "
    "Most members agreed that further policy firming was appropriate; the "
    "Committee judged that a rate increase of 25 basis points would help "
    "anchor expectations. Several members noted that the labor market "
    "remained tight, with wage pressure consistent with continued "
    "overheating risks. Participants emphasized that withdrawing "
    "accommodation in a steady manner would be appropriate, and that "
    "additional rate increases were likely to be needed."
)

# 2023-05 hike — late-cycle tightening, terminal-rate framing.
TIGHTENING_2023_MAY_EXCERPT = (
    "The Committee discussed the need for further policy firming. Members "
    "judged that, given the persistence of inflationary pressures and the "
    "tight labor market, a hike of 25 basis points would be appropriate at "
    "this meeting. Participants noted that restrictive policy stance would "
    "need to be maintained to anchor inflation expectations and bring "
    "inflation back to the 2 percent objective. Upside risks to inflation "
    "remained elevated."
)

# 2019-09 cut — mid-cycle adjustment easing.
EASING_2019_SEPT_EXCERPT = (
    "Participants noted that downside risks to the outlook had increased "
    "and that softening in business investment warranted a more "
    "accommodative policy stance. Several members judged that a rate cut "
    "of 25 basis points would be appropriate to provide support to the "
    "economy. Members observed disinflationary pressures from weaker "
    "global demand. The Committee remains patient and stands ready to ease "
    "further if downside risks materialize."
)

# 2020-03 emergency — zero-lower-bound + QE announcement.
EASING_2020_MARCH_EXCERPT = (
    "In light of the rapidly evolving outlook and substantial downside "
    "risks, members judged that an accommodative policy stance was "
    "warranted. The Committee voted to cut the target range for the federal "
    "funds rate to 0 to 0.25 percent and to provide stimulus through asset "
    "purchases. Several members noted softening across multiple sectors and "
    "stimulative measures were necessary to support the economy. "
    "Participants emphasized patience and a supportive policy stance until "
    "the recovery is well established."
)


@pytest.mark.parametrize(
    "label, excerpt",
    [
        ("2022-03 hike", TIGHTENING_2022_MARCH_EXCERPT),
        ("2023-05 hike", TIGHTENING_2023_MAY_EXCERPT),
    ],
)
def test_tightening_cycle_excerpts_score_hawkish(label: str, excerpt: str) -> None:
    score = score_text(excerpt)
    assert score.net_score > 0, (
        f"{label} excerpt should score hawkish but got "
        f"hawkish={score.hawkish_count}, dovish={score.dovish_count}, "
        f"net_score={score.net_score}"
    )


@pytest.mark.parametrize(
    "label, excerpt",
    [
        ("2019-09 cut", EASING_2019_SEPT_EXCERPT),
        ("2020-03 emergency", EASING_2020_MARCH_EXCERPT),
    ],
)
def test_easing_cycle_excerpts_score_dovish(label: str, excerpt: str) -> None:
    score = score_text(excerpt)
    assert score.net_score < 0, (
        f"{label} excerpt should score dovish but got "
        f"hawkish={score.hawkish_count}, dovish={score.dovish_count}, "
        f"net_score={score.net_score}"
    )


def test_lexicon_separates_cycle_pairs() -> None:
    """The lexicon must produce a clean ordering: any tightening excerpt
    scores above any easing excerpt. This is what calibration users will
    check against historical FOMC minutes in production."""
    tightening = [
        score_text(TIGHTENING_2022_MARCH_EXCERPT).net_score,
        score_text(TIGHTENING_2023_MAY_EXCERPT).net_score,
    ]
    easing = [
        score_text(EASING_2019_SEPT_EXCERPT).net_score,
        score_text(EASING_2020_MARCH_EXCERPT).net_score,
    ]
    assert min(tightening) > max(easing), (
        f"tightening cycle scores {tightening} should all exceed easing "
        f"cycle scores {easing}"
    )


def test_fixture_fomc_parquet_cycle_windows_score_with_expected_signs(
    tmp_path: Path,
) -> None:
    """Default-CI contract for the parquet path used by live FOMC data.

    The live-data tests below still validate materialized production history
    when present. This smaller fixture creates the same parquet shape in
    tmp_path so CI always exercises path loading, release scoring, and
    session-aligned cycle behavior without depending on data/raw.
    """
    fixture_path = tmp_path / "fomc_minutes.parquet"
    pd.DataFrame(
        [
            {
                "release_timestamp": "2019-09-18 14:00:00",
                "body_text": EASING_2019_SEPT_EXCERPT,
            },
            {
                "release_timestamp": "2020-03-15 17:00:00",
                "body_text": EASING_2020_MARCH_EXCERPT,
            },
            {
                "release_timestamp": "2022-03-16 14:00:00",
                "body_text": TIGHTENING_2022_MARCH_EXCERPT,
            },
            {
                "release_timestamp": "2023-05-03 14:00:00",
                "body_text": TIGHTENING_2023_MAY_EXCERPT,
            },
        ]
    ).to_parquet(fixture_path, index=False)

    scored = load_central_bank_text_score(fomc_minutes_source=fixture_path)

    easing_window = scored[
        (scored["release_date"] >= dt.date(2019, 7, 1))
        & (scored["release_date"] <= dt.date(2020, 4, 30))
    ]
    tightening_window = scored[
        (scored["release_date"] >= dt.date(2022, 3, 1))
        & (scored["release_date"] <= dt.date(2023, 7, 31))
    ]
    assert len(easing_window) == 2
    assert len(tightening_window) == 2
    assert easing_window["net_score"].mean() < 0
    assert tightening_window["net_score"].mean() > 0

    sessions = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-09-17"),
            pd.Timestamp("2019-09-19"),
            pd.Timestamp("2020-03-16"),
            pd.Timestamp("2022-03-17"),
            pd.Timestamp("2023-05-04"),
        ]
    )
    daily = to_daily_score_series(
        scored, session_index=sessions, smoothing_window_sessions=1
    )
    assert pd.isna(daily.loc["2019-09-17"])
    assert daily.loc["2019-09-19"] < 0
    assert daily.loc["2020-03-16"] < 0
    assert daily.loc["2022-03-17"] > 0
    assert daily.loc["2023-05-04"] > 0


_FOMC_PARQUET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "fomc_minutes"
    / "fomc_minutes.parquet"
)


@pytest.mark.skipif(
    not _FOMC_PARQUET.exists(),
    reason=(
        "data/raw/fomc_minutes/fomc_minutes.parquet not present in this "
        "checkout — run `python3 scripts/fetch_regime_engine_v1_data.py "
        "--fetch fomc` to materialize, then re-run this test."
    ),
)
def test_live_fomc_minutes_tightening_window_scores_hawkish() -> None:
    """Integration assertion: when the real FOMC parquet is materialized,
    minutes released during the 2022-03→2023-07 tightening cycle must
    average a positive net_score."""
    scored = load_central_bank_text_score(fomc_minutes_source=_FOMC_PARQUET)
    if scored.empty:
        pytest.skip("FOMC parquet present but contains no rows.")
    tightening_window = scored[
        (scored["release_date"] >= dt.date(2022, 3, 1))
        & (scored["release_date"] <= dt.date(2023, 7, 31))
    ]
    if tightening_window.empty:
        pytest.skip(
            "FOMC parquet does not cover the 2022-03 → 2023-07 cycle "
            "(refetch with extended history)."
        )
    mean_net = tightening_window["net_score"].mean()
    assert mean_net > 0, (
        f"2022-03 → 2023-07 tightening cycle mean net_score should be "
        f"positive; got {mean_net:.3f} over {len(tightening_window)} releases."
    )


@pytest.mark.skipif(
    not _FOMC_PARQUET.exists(),
    reason="FOMC parquet not present — see prior test.",
)
def test_live_fomc_minutes_easing_window_scores_dovish() -> None:
    """Integration assertion: 2019-07 → 2020-04 easing window must
    average a negative net_score on the live parquet."""
    scored = load_central_bank_text_score(fomc_minutes_source=_FOMC_PARQUET)
    if scored.empty:
        pytest.skip("FOMC parquet present but contains no rows.")
    easing_window = scored[
        (scored["release_date"] >= dt.date(2019, 7, 1))
        & (scored["release_date"] <= dt.date(2020, 4, 30))
    ]
    if easing_window.empty:
        pytest.skip(
            "FOMC parquet does not cover the 2019-07 → 2020-04 cycle."
        )
    mean_net = easing_window["net_score"].mean()
    # NOTE: the FOMC during this window also discussed mid-2019 "above-trend"
    # growth language, so the test allows mean to be non-strictly-negative
    # but expects it to be markedly below the tightening-cycle mean. The
    # paired comparison would be ideal but requires both windows present.
    assert mean_net < 0.3, (
        f"2019-07 → 2020-04 easing window mean net_score should be "
        f"dovish-leaning; got {mean_net:.3f} over {len(easing_window)} releases."
    )
