from __future__ import annotations

import json
from pathlib import Path

from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch.artifact_manifest import load_manifest
from regime_data_fetch.manifest_inputs import resolve_runner_input_paths
from regime_data_fetch.sf_fed_news_sentiment import SF_FED_NEWS_SENTIMENT_PARQUET


def _store_uri(root: Path, key: str) -> str:
    return (root.resolve() / key).as_uri()


def test_emit_manifest_for_report_paths_uploads_existing_report_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    macro = out_dir / "macro" / "fred_macro_series.parquet"
    report = out_dir / "macro_fetch_report.json"
    macro.parent.mkdir(parents=True)
    macro.write_bytes(b"macro")
    report.write_text(
        json.dumps(
            {
                "paths": {
                    "macro_parquet": str(macro),
                    "missing_optional": str(out_dir / "missing.parquet"),
                }
            }
        )
    )
    store_root = tmp_path / "store"
    manifest_path = tmp_path / "manifest.yaml"

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(store_root),
        manifest_path=manifest_path,
        artifact_set="regime_engine_test",
        required_for=["v2_calibration"],
    )

    loaded = load_manifest(manifest_path)
    assert loaded == manifest
    assert [artifact.name for artifact in loaded.artifacts] == ["fred_macro_series"]
    assert loaded.artifacts[0].uri == _store_uri(
        store_root, "canonical/macro/fred_macro_series.parquet"
    )
    assert loaded.artifacts[0].local_path == "data/raw/macro/fred_macro_series.parquet"
    assert loaded.artifacts[0].required_for == ("v2_calibration",)
    assert (store_root / "canonical" / "macro" / "fred_macro_series.parquet").read_bytes() == b"macro"


def test_emit_manifest_for_report_paths_expands_partitioned_parquet_directories(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    part = out_dir / "daily_ohlcv" / "symbol=SPY" / "part.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"spy")
    report = out_dir / "fetch_report.json"
    report.write_text(json.dumps({"paths": {"daily_ohlcv_parquet": str(out_dir / "daily_ohlcv")}}))

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="market",
        required_for=["profile_engine_30d"],
    )

    assert [artifact.local_path for artifact in manifest.artifacts] == [
        "data/raw/daily_ohlcv/symbol=SPY/part.parquet"
    ]
    assert (tmp_path / "store" / "canonical" / "daily_ohlcv" / "symbol=SPY" / "part.parquet").read_bytes() == b"spy"


def test_emit_manifest_for_report_paths_allows_multi_file_symbol_partitions(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    symbol_dir = out_dir / "daily_ohlcv_762" / "symbol=SPY"
    part_0 = symbol_dir / "part-0.parquet"
    part_1 = symbol_dir / "part-1.parquet"
    symbol_dir.mkdir(parents=True)
    part_0.write_bytes(b"spy-0")
    part_1.write_bytes(b"spy-1")
    macro = out_dir / "macro" / "fred_macro_series.parquet"
    pit = out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
    events = out_dir / "event_calendar" / "us_events.yaml"
    for path, payload in [
        (macro, b"macro"),
        (pit, b"pit"),
        (events, b"events: []\n"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    report = out_dir / "combined_report.json"
    report.write_text(
        json.dumps(
            {
                "paths": {
                    "profile_constituent_tree": str(out_dir / "daily_ohlcv_762"),
                    "macro_parquet": str(macro),
                    "pit_constituents_parquet": str(pit),
                    "event_calendar_yaml": str(events),
                }
            }
        )
    )
    manifest_path = tmp_path / "manifest.yaml"

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=manifest_path,
        artifact_set="profile",
        required_for=["profile_engine_30d"],
    )
    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=tmp_path / "materialized" / "data" / "raw",
        runner_name="profile_engine_30d",
        cli_values={},
        cli_overrides=set(),
    )

    daily_names = [
        artifact.name
        for artifact in manifest.artifacts
        if artifact.local_path.startswith("data/raw/daily_ohlcv_762/")
    ]
    assert daily_names == [
        "constituent_ohlcv_SPY_part_0_parquet",
        "constituent_ohlcv_SPY_part_1_parquet",
    ]
    assert resolved.daily_dir == tmp_path / "materialized" / "data" / "raw" / "daily_ohlcv_762"


def test_emit_manifest_for_report_paths_fails_when_no_files_found(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)
    report = out_dir / "fetch_report.json"
    report.write_text(json.dumps({"paths": {"not_a_path": "s3://bucket/object"}}))

    import pytest

    with pytest.raises(ValueError, match="no exportable artifact files in report"):
        emit_manifest_for_report_paths(
            report_paths=[report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="empty_not_ok",
            required_for=[],
        )


def test_emit_manifest_for_report_paths_fails_for_missing_report_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)

    import pytest

    with pytest.raises(FileNotFoundError, match="manifest report path does not exist"):
        emit_manifest_for_report_paths(
            report_paths=[out_dir / "missing_report.json"],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="missing-report",
            required_for=[],
        )


def test_emit_manifest_for_report_paths_fails_for_non_json_report_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)
    report = out_dir / "report.txt"
    report.write_text("not json")

    import pytest

    with pytest.raises(ValueError, match="manifest report path must be JSON"):
        emit_manifest_for_report_paths(
            report_paths=[report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="non-json-report",
            required_for=[],
        )


def test_emit_manifest_for_report_paths_fails_when_one_report_has_no_exportable_artifacts(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    macro = out_dir / "macro" / "fred_macro_series.parquet"
    macro.parent.mkdir(parents=True)
    macro.write_bytes(b"macro")
    good_report = out_dir / "macro_fetch_report.json"
    good_report.write_text(json.dumps({"paths": {"macro_parquet": str(macro)}}))
    bad_report = out_dir / "sf_fed_news_sentiment_fetch_report.json"
    bad_report.write_text(
        json.dumps(
            {
                "source": "frbsf:daily_news_sentiment",
                "parquet": str(
                    out_dir / "news_sentiment" / SF_FED_NEWS_SENTIMENT_PARQUET
                ),
            }
        )
    )

    import pytest

    with pytest.raises(ValueError, match="no exportable artifact files in report"):
        emit_manifest_for_report_paths(
            report_paths=[good_report, bad_report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="mixed",
            required_for=["profile_engine_30d"],
        )


def test_emit_manifest_for_report_paths_allows_explicit_non_materializable_report(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    macro = out_dir / "macro" / "fred_macro_series.parquet"
    macro.parent.mkdir(parents=True)
    macro.write_bytes(b"macro")
    good_report = out_dir / "macro_fetch_report.json"
    good_report.write_text(json.dumps({"paths": {"macro_parquet": str(macro)}}))
    ledger_report = out_dir / "ledger_report.json"
    ledger_report.write_text(
        json.dumps({"materializable": False, "paths": {"acquisition_db": "ignored"}})
    )

    manifest = emit_manifest_for_report_paths(
        report_paths=[good_report, ledger_report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="mixed",
        required_for=["profile_engine_30d"],
    )

    assert [artifact.name for artifact in manifest.artifacts] == ["fred_macro_series"]


def test_emit_manifest_for_report_paths_skips_acquisition_db_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)
    acquisition_db = out_dir / "acquisition.db"
    acquisition_db.write_bytes(b"sqlite")
    report = out_dir / "fetch_report.json"
    report.write_text(json.dumps({"paths": {"acquisition_db": str(acquisition_db)}}))

    import pytest

    with pytest.raises(ValueError, match="no exportable artifact files in report"):
        emit_manifest_for_report_paths(
            report_paths=[report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="skip-db",
            required_for=[],
        )


def test_emit_manifest_for_report_paths_skips_files_outside_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)
    outside = tmp_path / "elsewhere" / "acquisition.db"
    outside.parent.mkdir()
    outside.write_bytes(b"sqlite")
    report = out_dir / "fetch_report.json"
    report.write_text(json.dumps({"paths": {"acquisition_db": str(outside)}}))

    import pytest

    with pytest.raises(ValueError, match="no exportable artifact files in report"):
        emit_manifest_for_report_paths(
            report_paths=[report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="skip-outside",
            required_for=[],
        )


def test_emit_manifest_for_report_paths_exports_repo_relative_event_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    out_dir = repo_root / "data" / "raw"
    event_config = repo_root / "configs" / "events" / "us_events.yaml"
    report = out_dir / "events" / "us_event_calendar_report.json"
    event_config.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    event_config.write_text("events: []\n")
    report.write_text(json.dumps({"paths": {"us_events_yaml": str(event_config)}}))

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        repo_root=repo_root,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="events",
        required_for=["profile_engine_30d"],
    )

    assert manifest.artifacts[0].local_path == "data/raw/event_calendar/us_events.yaml"
    assert manifest.artifacts[0].uri == _store_uri(
        tmp_path / "store", "canonical/event_calendar/us_events.yaml"
    )
    assert (
        tmp_path / "store" / "canonical" / "event_calendar" / "us_events.yaml"
    ).read_text() == "events: []\n"


def test_emit_manifest_for_report_paths_honors_explicit_materialized_local_path(tmp_path: Path) -> None:
    source_root = tmp_path / "archive" / "daily_ohlcv_762"
    source_file = source_root / "symbol=SPY" / "ohlcv.parquet"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"spy-762")
    out_dir = tmp_path / "repo" / "data" / "raw"
    report = out_dir / "daily_ohlcv_local_sqlite_import_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "paths": {
                    "profile_constituent_tree": {
                        "path": str(source_root),
                        "local_path": "data/raw/daily_ohlcv_762",
                    }
                }
            }
        )
    )

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="profile",
        required_for=["profile_engine_30d"],
    )

    assert [artifact.local_path for artifact in manifest.artifacts] == [
        "data/raw/daily_ohlcv_762/symbol=SPY/ohlcv.parquet"
    ]
    assert [artifact.name for artifact in manifest.artifacts] == ["constituent_ohlcv_SPY"]
    assert (tmp_path / "store" / "canonical" / "daily_ohlcv_762" / "symbol=SPY" / "ohlcv.parquet").read_bytes() == b"spy-762"


def test_emit_manifest_for_report_paths_exports_sf_fed_news_sentiment_report(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    parquet = out_dir / "news_sentiment" / SF_FED_NEWS_SENTIMENT_PARQUET
    report = out_dir / "sf_fed_news_sentiment_fetch_report.json"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"sf-fed-news")
    report.write_text(
        json.dumps(
            {
                "source": "frbsf:daily_news_sentiment",
                "paths": {"news_sentiment_parquet": str(parquet)},
            }
        )
    )

    manifest = emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=tmp_path / "manifest.yaml",
        artifact_set="sf-fed-news",
        required_for=["profile_engine_30d"],
    )

    assert [artifact.name for artifact in manifest.artifacts] == [
        "sf_fed_news_sentiment"
    ]
    assert manifest.artifacts[0].local_path == (
        f"data/raw/news_sentiment/{SF_FED_NEWS_SENTIMENT_PARQUET}"
    )
    assert (
        tmp_path / "store" / "canonical" / "news_sentiment" / SF_FED_NEWS_SENTIMENT_PARQUET
    ).read_bytes() == b"sf-fed-news"


def test_emitted_manifest_resolves_profile_runner_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    daily = out_dir / "daily_ohlcv_762" / "symbol=SPY" / "ohlcv.parquet"
    macro = out_dir / "macro" / "fred_macro_series.parquet"
    pit = out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
    events = out_dir / "event_calendar" / "us_events.yaml"
    for path, payload in [
        (daily, b"spy"),
        (macro, b"macro"),
        (pit, b"pit"),
        (events, b"events: []\n"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    report = out_dir / "combined_report.json"
    report.write_text(
        json.dumps(
            {
                "paths": {
                    "profile_constituent_tree": str(out_dir / "daily_ohlcv_762"),
                    "macro_parquet": str(macro),
                    "pit_constituents_parquet": str(pit),
                    "event_calendar_yaml": str(events),
                }
            }
        )
    )
    manifest_path = tmp_path / "manifest.yaml"

    emit_manifest_for_report_paths(
        report_paths=[report],
        out_dir=out_dir,
        artifact_store_root=str(tmp_path / "store"),
        manifest_path=manifest_path,
        artifact_set="profile",
        required_for=["profile_engine_30d"],
    )
    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=tmp_path / "materialized" / "data" / "raw",
        runner_name="profile_engine_30d",
        cli_values={},
        cli_overrides=set(),
    )

    assert resolved.daily_dir == tmp_path / "materialized" / "data" / "raw" / "daily_ohlcv_762"
    assert resolved.constituent_tree == resolved.daily_dir
    assert resolved.macro_parquet == tmp_path / "materialized" / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    assert resolved.pit_parquet == tmp_path / "materialized" / "data" / "raw" / "pit_constituents" / "sp500_ticker_intervals.parquet"
    assert resolved.event_calendar == tmp_path / "materialized" / "data" / "raw" / "event_calendar" / "us_events.yaml"
