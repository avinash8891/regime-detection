from __future__ import annotations

import json
from pathlib import Path

from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch.artifact_manifest import load_manifest


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
    assert [artifact.name for artifact in loaded.artifacts] == ["macro_parquet"]
    assert loaded.artifacts[0].uri == "canonical/macro/fred_macro_series.parquet"
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


def test_emit_manifest_for_report_paths_fails_when_no_files_found(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    out_dir.mkdir(parents=True)
    report = out_dir / "fetch_report.json"
    report.write_text(json.dumps({"paths": {"not_a_path": "s3://bucket/object"}}))

    import pytest

    with pytest.raises(ValueError, match="no existing artifact files"):
        emit_manifest_for_report_paths(
            report_paths=[report],
            out_dir=out_dir,
            artifact_store_root=str(tmp_path / "store"),
            manifest_path=tmp_path / "manifest.yaml",
            artifact_set="empty_not_ok",
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

    with pytest.raises(ValueError, match="no existing artifact files"):
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

    assert manifest.artifacts[0].local_path == "configs/events/us_events.yaml"
    assert manifest.artifacts[0].uri == "canonical/configs/events/us_events.yaml"
    assert (tmp_path / "store" / "canonical" / "configs" / "events" / "us_events.yaml").read_text() == "events: []\n"
