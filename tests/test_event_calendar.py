from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from regime_detection.config import load_default_regime_config, load_regime_config
from regime_detection.axis_series import build_event_calendar_series
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.market_context import build_market_context, slice_context_to_recent_sessions
from regime_detection.loaders import load_event_calendar
from regime_data_fetch.event_calendar import (
    EventCalendarFetchError,
    ScheduledEvent,
    _build_url_text_fetcher,
    validate_fomc_listing_integrity,
)
from regime_data_fetch.event_calendar_reporting import (
    build_candidate_artifact_records,
    build_group_a_report,
    build_group_b_report,
)
from regime_data_fetch.event_sources.models import EventCandidate, PromotionDecision, ValidationResult
from regime_data_fetch.event_sources.deterministic_budget import DeterministicBudgetAdapter


FOMC_FIXTURES = Path("tests/fixtures/raw/fomc")


def test_url_text_fetcher_raises_and_logs_url_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise URLError("fixture timeout")

    monkeypatch.setattr("regime_data_fetch.event_calendar_group_a.urlopen", failing_urlopen)

    with caplog.at_level("ERROR", logger="regime_data_fetch.event_calendar"):
        with pytest.raises(EventCalendarFetchError, match="fixture timeout"):
            _build_url_text_fetcher("https://example.invalid/calendar")()

    assert "event calendar source fetch failed" in caplog.text
    assert "https://example.invalid/calendar" in caplog.text


def _empty_hf_central_bank_parquet_bytes(tmp_path: Path) -> bytes:
    parquet_path = tmp_path / "empty_hf_central_bank.parquet"
    pd.DataFrame(
        columns=["central_bank", "doc_type", "title", "url", "meeting_date"]
    ).to_parquet(parquet_path, index=False)
    return parquet_path.read_bytes()


def test_event_calendar_reporting_builds_candidate_records_and_group_reports(tmp_path: Path) -> None:
    ecb_candidate = EventCandidate(
        date=dt.date(2026, 4, 30),
        event_type="ECB_decision",
        market="GLOBAL",
        importance="medium",
        source_id="ecb.europa.eu:monetary-policy-decisions",
        source_url="https://www.ecb.europa.eu/example",
        raw_title="Monetary policy decisions",
        raw_snippet="ECB decision snippet",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=True,
        release_timestamp_et=dt.datetime(2026, 4, 30, 8, 0, tzinfo=dt.timezone.utc),
        window_days=(0, 1),
        candidate_id="ecb-20260430",
    )
    budget_candidate = EventCandidate(
        date=dt.date(2026, 9, 30),
        event_type="budget",
        market="US",
        importance="medium",
        source_id="usa.gov:federal-budget-process",
        source_url="https://www.usa.gov/federal-budget-process",
        raw_title="Federal budget deadline",
        raw_snippet="Budget process snippet",
        is_future_scheduled=True,
        confidence="high",
        requires_manual_review=False,
        event_subtype="fy_deadline",
        candidate_id="budget-20260930",
    )
    validations = [
        ValidationResult(
            candidate_key=("ECB_decision", dt.date(2026, 4, 30)),
            validator_id="hf_central_bank",
            verdict="confirm",
            evidence_url="https://evidence.example/ecb",
            evidence_snippet="confirmed",
        )
    ]
    decisions = [
        PromotionDecision(
            candidate_key=("ECB_decision", dt.date(2026, 4, 30)),
            outcome="quarantine",
            final_confidence="low",
            source_count=2,
            requires_manual_review=True,
            reason="contradictory evidence",
        ),
        PromotionDecision(
            candidate_key=("budget", dt.date(2026, 9, 30)),
            outcome="promote",
            final_confidence="high",
            source_count=1,
            requires_manual_review=False,
            reason="deterministic schedule",
        ),
    ]

    records = build_candidate_artifact_records(
        candidates=[ecb_candidate, budget_candidate],
        validations=validations,
        decisions=decisions,
    )

    assert records.candidates[0]["promotion_outcome"] == "quarantine"
    assert records.candidates[0]["confidence"] == "low"
    assert records.candidates[0]["release_timestamp_et"] == "2026-04-30T08:00:00+00:00"
    assert records.candidates[0]["window_days"] == [0, 1]
    assert records.validations == [
        {
            "event_type": "ECB_decision",
            "date": "2026-04-30",
            "validator_id": "hf_central_bank",
            "verdict": "confirm",
            "evidence_url": "https://evidence.example/ecb",
            "evidence_snippet": "confirmed",
        }
    ]
    assert records.quarantine == [records.candidates[0]]

    output_paths = {
        "candidates": tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "event_candidates.parquet",
        "validations": tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "event_validations.parquet",
        "quarantine": tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "quarantine.parquet",
    }
    group_a_report = build_group_a_report(
        candidates=[ecb_candidate, budget_candidate],
        decisions=decisions,
        output_paths=output_paths,
        repo_root=tmp_path,
    )
    group_b_report = build_group_b_report(
        candidates=[ecb_candidate, budget_candidate],
        decisions=decisions,
        approval_overlay=[],
    )

    assert group_a_report["candidates"] == {"ECB_decision": 1}
    assert group_a_report["quarantined"] == {"ECB_decision": 1}
    assert group_a_report["paths"]["candidates"] == "data/raw/event_calendar/candidates/event_candidates.parquet"
    assert group_b_report["candidates"] == {"budget": 1}
    assert group_b_report["promoted"] == {"budget": 1}


def test_load_event_calendar_yaml_defaults_publication_date() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.yaml"
    df = load_event_calendar(path)

    assert set(df["type"]) == {"FOMC", "CPI", "NFP", "ad_hoc"}
    fomc_row = df[df["type"] == "FOMC"].iloc[0]
    assert fomc_row["publication_date"] == date(2023, 10, 21)


def test_load_event_calendar_yaml_accepts_v2_manual_types_and_window_days(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-11-03"',
                '    market: "US"',
                '    type: "election"',
                '    importance: "high"',
                "    window_days: [-5, +10]",
                '  - date: "2026-12-10"',
                '    market: "GLOBAL"',
                '    type: "global_rate_decision"',
                '    importance: "medium"',
                '  - date: "2026-06-15"',
                '    market: "US"',
                '    type: "geopolitical_event"',
                '    importance: "high"',
            ]
        )
        + "\n"
    )

    us = load_event_calendar(path, market="US")
    assert set(us["type"]) == {
        "election",
        "geopolitical_event",
        "global_rate_decision",
    }
    election = us[us["type"] == "election"].iloc[0]
    assert election["window_days"] == [-5, 10]
    assert election["publication_date"] == date(2026, 8, 5)

    global_events = load_event_calendar(path, market="GLOBAL")
    assert list(global_events["type"]) == ["global_rate_decision"]


def test_load_event_calendar_csv_defaults_publication_date() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.csv"
    df = load_event_calendar(path)

    nfp_row = df[df["type"] == "NFP"].iloc[0]
    assert nfp_row["publication_date"] == date(2023, 10, 22)


def test_load_event_calendar_rejects_malformed_publication_date() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-19",
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": "not-a-date",
            }
        ]
    )

    with pytest.raises(ValueError):
        load_event_calendar(df)


def test_event_calendar_uses_publication_date_and_precedence() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 19),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
            {
                "date": date(2024, 1, 18),
                "market": "US",
                "type": "CPI",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2024, 1, 17),
        event_calendar=events,
        config=cfg,
    )

    assert out.primary_label == "fed_week"
    assert set(out.matching_labels) >= {"fed_week", "cpi_week", "expiry_week", "earnings_season"}
    assert not hasattr(out, "raw_label")
    assert not hasattr(out, "stable_label")
    assert not hasattr(out, "active_label")


def test_event_calendar_blocks_unpublished_future_scheduled_event() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 31),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2024, 1, 25),
            }
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2024, 1, 17),
        event_calendar=events,
        config=cfg,
    )

    assert out.primary_label == "expiry_week"
    assert out.matching_labels == ("expiry_week", "earnings_season")
    assert "fed_week" not in out.matching_labels


def test_event_calendar_v2_election_uses_default_trading_day_window() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
            }
        ]
    )

    start_out = classify_event_calendar(
        as_of_date=date(2026, 10, 27),
        event_calendar=events,
        config=cfg,
    )
    end_out = classify_event_calendar(
        as_of_date=date(2026, 11, 17),
        event_calendar=events,
        config=cfg,
    )
    before_out = classify_event_calendar(
        as_of_date=date(2026, 10, 26),
        event_calendar=events,
        config=cfg,
    )

    assert start_out.primary_label == "election_window"
    assert end_out.primary_label == "election_window"
    assert before_out.primary_label != "election_window"


def test_event_calendar_v2_precedence_and_labels() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
                "publication_date": date(2026, 8, 1),
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "geopolitical_event",
                "importance": "high",
                "publication_date": date(2026, 11, 3),
                "approved_label": "geopolitical_event",
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "budget",
                "importance": "medium",
                "publication_date": date(2026, 8, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 11, 3),
        event_calendar=events,
        config=cfg,
    )

    assert out.primary_label == "geopolitical_event"
    assert set(out.matching_labels) >= {
        "geopolitical_event",
        "election_window",
        "budget_week",
    }


def test_event_calendar_outputs_primary_and_matching_labels_for_overlaps() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 6, 17),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2026, 1, 1),
            },
            {
                "date": date(2026, 6, 17),
                "market": "GLOBAL",
                "type": "ECB_decision",
                "importance": "high",
                "publication_date": date(2026, 1, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 6, 17),
        event_calendar=events,
        config=cfg,
    )

    assert out.primary_label == "fed_week"
    assert out.matching_labels == ("fed_week", "global_rate_decision", "expiry_week")
    assert not hasattr(out, "raw_label")
    assert not hasattr(out, "stable_label")
    assert not hasattr(out, "active_label")


def test_event_calendar_v1_config_ignores_v2_event_labels() -> None:
    cfg = load_regime_config(Path("src/regime_detection/configs/core3-v1.0.0.yaml"))
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 6, 15),
                "market": "US",
                "type": "election",
                "importance": "high",
                "publication_date": date(2026, 6, 1),
            },
            {
                "date": date(2026, 6, 15),
                "market": "US",
                "type": "budget",
                "importance": "medium",
                "publication_date": date(2026, 6, 1),
            },
            {
                "date": date(2026, 6, 15),
                "market": "GLOBAL",
                "type": "global_rate_decision",
                "importance": "medium",
                "publication_date": date(2026, 6, 1),
            },
            {
                "date": date(2026, 6, 15),
                "market": "US",
                "type": "geopolitical_event",
                "importance": "high",
                "publication_date": date(2026, 6, 15),
                "approved_label": "geopolitical_event",
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 6, 15),
        event_calendar=events,
        config=cfg,
    )

    assert out.primary_label == "normal_calendar"
    assert out.matching_labels == ("normal_calendar",)


def test_budget_week_fires_when_fiscal_year_deadline_falls_on_weekend() -> None:
    cfg = load_default_regime_config()
    candidate = DeterministicBudgetAdapter(as_of_date=dt.date(2017, 1, 1)).fetch(
        start_year=2017,
        end_year=2017,
        store=None,
        run_id=None,
    )[0]
    events = pd.DataFrame(
        [
            {
                "date": candidate.date,
                "market": candidate.market,
                "type": candidate.event_type,
                "importance": candidate.importance,
            }
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2017, 9, 29),
        event_calendar=events,
        config=cfg,
    )

    assert candidate.date == date(2017, 9, 29)
    assert out.primary_label == "budget_week"
    assert out.matching_labels == ("budget_week",)


def test_build_event_calendar_series_matches_point_classifier(market_df_for_asof) -> None:
    cfg = load_default_regime_config()
    end_date = date(2024, 1, 17)
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 19),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
            {
                "date": date(2024, 1, 18),
                "market": "US",
                "type": "CPI",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
        ]
    )

    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=cfg,
        event_calendar=events,
    )
    context = slice_context_to_recent_sessions(context=context, required_sessions=10)
    outputs = build_event_calendar_series(context)

    for day in context.sessions:
        expected = classify_event_calendar(
            as_of_date=day,
            event_calendar=events,
            config=cfg,
        )
        assert outputs[day].model_dump() == expected.model_dump()


def test_build_event_calendar_series_matches_point_classifier_for_holiday_shifted_expiry(
    market_df_for_asof,
) -> None:
    cfg = load_default_regime_config()
    end_date = date(2022, 4, 14)
    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=cfg,
    )
    context = slice_context_to_recent_sessions(context=context, required_sessions=10)
    outputs = build_event_calendar_series(context)

    for day in context.sessions:
        expected = classify_event_calendar(
            as_of_date=day,
            event_calendar=None,
            config=cfg,
        )
        assert outputs[day].model_dump() == expected.model_dump()


def test_validate_fomc_listing_integrity_detects_missing_structured_dates() -> None:
    html = """
    <a href="/monetarypolicy/fomcminutes20230201.htm">HTML</a>
    <a href="/monetarypolicy/fomcminutes20230322.htm">HTML</a>
    """
    parsed_entries = [
        ScheduledEvent(
            date=dt.date(2023, 3, 22),
            release_timestamp_et=dt.datetime(2023, 4, 12, 14, 0, tzinfo=dt.timezone.utc),
            market="US",
            type="FOMC",
            importance="high",
            source="federalreserve.gov:fomccalendars",
        )
    ]

    try:
        validate_fomc_listing_integrity(html=html, parsed_entries=parsed_entries, min_year=2023)
    except EventCalendarFetchError as exc:
        assert "mismatch" in str(exc).lower()
        assert "2023-02-01" in str(exc)
    else:
        raise AssertionError("Expected EventCalendarFetchError for missing FOMC dates")
