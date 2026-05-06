from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from regime_data_fetch.pmi import (
    PMIFetchError,
    PMIObservation,
    choose_latest_available,
    parse_dbnomics_html,
    parse_tradingeconomics_html,
    release_timestamp_for_period,
    run_pmi_fetch,
)


FIXTURES = Path("tests/fixtures/raw/pmi")


def test_parse_dbnomics_html_extracts_observations() -> None:
    html = (FIXTURES / "dbnomics_manufacturing.html").read_text()
    observations = parse_dbnomics_html(
        html,
        series_name="manufacturing",
        source_url="https://db.nomics.world/ISM/pmi/pm?tab=table",
    )

    assert len(observations) == 4
    assert observations[0].period == "2024-01"
    assert observations[0].value == 49.1
    assert observations[-1].period == "2024-04"
    assert observations[-1].source == "dbnomics"


def test_parse_tradingeconomics_html_extracts_latest_observation() -> None:
    html = (FIXTURES / "tradingeconomics_manufacturing.html").read_text()
    obs = parse_tradingeconomics_html(
        html,
        series_name="manufacturing",
        source_url="https://tradingeconomics.com/united-states/business-confidence",
    )

    assert obs.period == "2026-04"
    assert obs.value == 52.7
    assert obs.source == "tradingeconomics"


def test_parse_tradingeconomics_html_extracts_services_observation() -> None:
    html = (FIXTURES / "tradingeconomics_services.html").read_text()
    obs = parse_tradingeconomics_html(
        html,
        series_name="services",
        source_url="https://tradingeconomics.com/united-states/non-manufacturing-pmi",
    )

    assert obs.period == "2026-04"
    assert obs.value == 53.6
    assert obs.source == "tradingeconomics"


def test_release_timestamp_for_period_uses_next_month_business_day() -> None:
    manufacturing_ts = release_timestamp_for_period(series_name="manufacturing", period="2026-03")
    services_ts = release_timestamp_for_period(series_name="services", period="2026-03")

    assert manufacturing_ts.isoformat() == "2026-04-01T10:00:00-04:00"
    assert services_ts.isoformat() == "2026-04-03T10:00:00-04:00"


def test_choose_latest_available_respects_release_timestamp() -> None:
    observations = [
        PMIObservation(
            series_name="manufacturing",
            period="2026-03",
            value=52.7,
            release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2026-03"),
            source="dbnomics",
            source_url="https://db.nomics.world/ISM/pmi/pm?tab=table",
        ),
        PMIObservation(
            series_name="manufacturing",
            period="2026-04",
            value=53.1,
            release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2026-04"),
            source="dbnomics",
            source_url="https://db.nomics.world/ISM/pmi/pm?tab=table",
        ),
    ]

    chosen = choose_latest_available(
        observations=observations,
        as_of_timestamp=dt.datetime(2026, 4, 30, 16, 0, tzinfo=observations[0].release_timestamp.tzinfo),
    )
    assert chosen.period == "2026-03"


def test_run_pmi_fetch_falls_back_to_backup(monkeypatch, tmp_path: Path) -> None:
    def failing_primary(*, as_of_date: dt.date) -> list[PMIObservation]:
        raise PMIFetchError("primary down")

    def backup_fetcher(*, as_of_date: dt.date) -> list[PMIObservation]:
        return [
            PMIObservation(
                series_name="manufacturing",
                period="2026-04",
                value=52.7,
                release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2026-04"),
                source="tradingeconomics",
                source_url="https://tradingeconomics.com/united-states/business-confidence",
            ),
            PMIObservation(
                series_name="services",
                period="2026-04",
                value=53.6,
                release_timestamp=release_timestamp_for_period(series_name="services", period="2026-04"),
                source="tradingeconomics",
                source_url="https://tradingeconomics.com/united-states/non-manufacturing-pmi",
            ),
        ]

    report_path = run_pmi_fetch(
        out_dir=tmp_path,
        as_of_date=dt.date(2026, 5, 15),
        primary_fetcher=failing_primary,
        backup_fetcher=backup_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["selected_source"] == "tradingeconomics"
    assert report["counts"]["rows"] == 2
    assert (tmp_path / "pmi" / "us_ism_pmi.parquet").exists()


def test_run_pmi_fetch_falls_back_when_primary_data_is_stale(tmp_path: Path) -> None:
    def stale_primary(*, as_of_date: dt.date) -> list[PMIObservation]:
        del as_of_date
        return [
            PMIObservation(
                series_name="manufacturing",
                period="2025-12",
                value=10.3,
                release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2025-12"),
                source="dbnomics",
                source_url="https://db.nomics.world/ISM/pmi/pm?tab=table",
            ),
            PMIObservation(
                series_name="services",
                period="2025-08",
                value=52.0,
                release_timestamp=release_timestamp_for_period(series_name="services", period="2025-08"),
                source="dbnomics",
                source_url="https://db.nomics.world/ISM/nm-pmi/pm?tab=table",
            ),
        ]

    def backup_fetcher(*, as_of_date: dt.date) -> list[PMIObservation]:
        del as_of_date
        return [
            PMIObservation(
                series_name="manufacturing",
                period="2026-04",
                value=52.7,
                release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2026-04"),
                source="tradingeconomics",
                source_url="https://tradingeconomics.com/united-states/business-confidence",
            ),
            PMIObservation(
                series_name="services",
                period="2026-04",
                value=53.6,
                release_timestamp=release_timestamp_for_period(series_name="services", period="2026-04"),
                source="tradingeconomics",
                source_url="https://tradingeconomics.com/united-states/non-manufacturing-pmi",
            ),
        ]

    report_path = run_pmi_fetch(
        out_dir=tmp_path,
        as_of_date=dt.date(2026, 5, 15),
        primary_fetcher=stale_primary,
        backup_fetcher=backup_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["selected_source"] == "tradingeconomics"
    assert report["attempts"][0]["source"] == "dbnomics"
    assert report["attempts"][0]["status"] == "failure"
    assert "stale" in report["attempts"][0]["error"].lower()


def test_run_pmi_fetch_raises_when_all_sources_fail(tmp_path: Path) -> None:
    def failing(*, as_of_date: dt.date) -> list[PMIObservation]:
        raise PMIFetchError("down")

    try:
        run_pmi_fetch(
            out_dir=tmp_path,
            as_of_date=dt.date(2026, 5, 15),
            primary_fetcher=failing,
            backup_fetcher=failing,
        )
    except PMIFetchError as exc:
        assert "All PMI sources failed" in str(exc)
    else:
        raise AssertionError("Expected PMIFetchError")
