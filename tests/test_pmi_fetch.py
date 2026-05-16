from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd

from regime_data_fetch.pmi import (
    DEFAULT_MANUAL_PMI_HISTORY_DIR,
    PMIFetchError,
    PMIFetchBundle,
    PMIObservation,
    choose_latest_available,
    load_manual_pmi_history,
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
    assert report["history_source"] == "tradingeconomics"
    assert report["counts"]["rows"] == 2
    assert report["counts"]["history_rows"] == 2
    assert (tmp_path / "pmi" / "us_ism_pmi.parquet").exists()
    assert (tmp_path / "pmi" / "us_ism_pmi_history.parquet").exists()


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
    assert report["history_source"] == "dbnomics"
    assert report["attempts"][0]["source"] == "dbnomics"
    assert report["attempts"][0]["status"] == "failure"
    assert "stale" in report["attempts"][0]["error"].lower()
    assert report["counts"]["history_rows"] == 2

    history_df = pd.read_parquet(tmp_path / "pmi" / "us_ism_pmi_history.parquet")
    assert history_df["period"].tolist() == ["2025-08", "2025-12"]
    assert history_df["source"].tolist() == ["dbnomics", "dbnomics"]


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


def test_run_pmi_fetch_records_raw_pages_and_outputs_in_sqlite(tmp_path: Path) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    def primary_fetcher(*, as_of_date: dt.date) -> PMIFetchBundle:
        del as_of_date
        return PMIFetchBundle(
            source_name="dbnomics",
            raw_pages={
                "manufacturing": "<html>mfg</html>",
                "services": "<html>svc</html>",
            },
            observations=[
                PMIObservation(
                    series_name="manufacturing",
                    period="2026-04",
                    value=52.7,
                    release_timestamp=release_timestamp_for_period(series_name="manufacturing", period="2026-04"),
                    source="dbnomics",
                    source_url="https://db.nomics.world/ISM/pmi/pm?tab=table",
                ),
                PMIObservation(
                    series_name="services",
                    period="2026-04",
                    value=53.6,
                    release_timestamp=release_timestamp_for_period(series_name="services", period="2026-04"),
                    source="dbnomics",
                    source_url="https://db.nomics.world/ISM/nm-pmi/pm?tab=table",
                ),
            ],
        )

    report_path = run_pmi_fetch(
        out_dir=tmp_path,
        as_of_date=dt.date(2026, 5, 15),
        primary_fetcher=primary_fetcher,
        backup_fetcher=primary_fetcher,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind ORDER BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("pmi", "ok")]
    assert artifacts == [("dbnomics:pmi", "html", 2)]
    assert outputs == [
        ("pmi_parquet",),
        ("pmi_history_parquet",),
        ("pmi_report",),
    ]


def test_load_manual_pmi_history_covers_backtest_aligned_window() -> None:
    rows = load_manual_pmi_history(history_dir=DEFAULT_MANUAL_PMI_HISTORY_DIR)

    assert rows[0].period == "2016-01"
    assert rows[-1].period == "2026-04"
    assert len(rows) == 248
    assert {row.series_name for row in rows} == {"manufacturing", "services"}


def test_run_pmi_fetch_uses_manual_history_dir_and_records_sqlite(tmp_path: Path) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    report_path = run_pmi_fetch(
        out_dir=tmp_path,
        as_of_date=dt.date(2026, 5, 7),
        acquisition_db_path=acquisition_db,
        manual_history_dir=DEFAULT_MANUAL_PMI_HISTORY_DIR,
    )

    report = json.loads(report_path.read_text())
    assert report["selected_source"] == "manual_investing_history"
    assert report["history_source"] == "manual_investing_history"
    assert report["counts"]["rows"] == 2
    assert report["counts"]["history_rows"] == 248
    assert report["paths"]["manual_pmi_manufacturing_tsv"] == {
        "path": str(DEFAULT_MANUAL_PMI_HISTORY_DIR / "ism_manufacturing_pmi.tsv"),
        "local_path": "data/manual_inputs/pmi/ism_manufacturing_pmi.tsv",
    }

    latest_df = pd.read_parquet(tmp_path / "pmi" / "us_ism_pmi.parquet")
    assert latest_df["period"].tolist() == ["2026-04", "2026-04"]
    assert latest_df["value"].tolist() == [52.7, 53.6]
    assert latest_df["source"].tolist() == ["investing_manual", "investing_manual"]

    history_df = pd.read_parquet(tmp_path / "pmi" / "us_ism_pmi_history.parquet")
    assert history_df["period"].min() == "2016-01"
    assert history_df["period"].max() == "2026-04"
    assert len(history_df) == 248

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind ORDER BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("pmi", "ok")]
    assert artifacts == [("investing:pmi", "tsv", 2)]
    assert outputs == [
        ("pmi_parquet",),
        ("pmi_history_parquet",),
        ("pmi_report",),
    ]
