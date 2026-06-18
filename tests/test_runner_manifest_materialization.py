from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    write_manifest,
)
from regime_data_fetch.artifact_store import sha256_file


def _store_uri(root: Path, key: str) -> str:
    return (root.resolve() / key).as_uri()


def _load_script_module(name: str, script_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, repo_root / script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_walkforward_gate_subprocess_materializes_manifest_defaults(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    daily_source = store_root / "canonical" / "daily_ohlcv_762" / "part.parquet"
    daily_source.parent.mkdir(parents=True)
    pd.DataFrame([{"date": pd.Timestamp("2026-05-15"), "symbol": "SPY"}]).to_parquet(
        daily_source,
        index=False,
    )
    macro_source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    macro_source.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-05-15"), "series_id": "DGS10", "value": 4.0}]
    ).to_parquet(
        macro_source,
        index=False,
    )
    pmi_source = store_root / "canonical" / "pmi" / "us_ism_pmi_history.parquet"
    pmi_source.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-04",
                "value": 52.7,
                "release_timestamp": "2026-05-01T14:00:00Z",
            }
        ]
    ).to_parquet(
        pmi_source,
        index=False,
    )
    manifest = ArtifactManifest(
        artifact_set="runner-test",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "daily_ohlcv_part",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/daily_ohlcv_762/part.parquet"
                    ),
                    "local_path": "data/raw/daily_ohlcv_762/part.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["v2_calibration"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "fred_macro_series",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(macro_source),
                    "required_for": ["v2_calibration"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "ism_pmi_history",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/pmi/us_ism_pmi_history.parquet"
                    ),
                    "local_path": "data/raw/pmi/us_ism_pmi_history.parquet",
                    "sha256": sha256_file(pmi_source),
                    "required_for": ["v2_calibration"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data" / "raw"
    write_manifest(manifest, manifest_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_v2_walkforward_gate.py",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / "out.md"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (data_root / "daily_ohlcv_762" / "part.parquet").exists()
    assert (data_root / "macro" / "fred_macro_series.parquet").exists()
    assert (data_root / "pmi" / "us_ism_pmi_history.parquet").exists()


@pytest.mark.parametrize(
    ("script_path", "required_for", "extra_args"),
    [
        (
            "scripts/run_v2_walkforward_gate.py",
            "v2_calibration",
            [
                "--output",
                "out.md",
                "--start-date",
                "2026-05-15",
                "--end-date",
                "2026-05-15",
            ],
        ),
        (
            "scripts/run_v2_shadow_ab_gate.py",
            "v2_calibration",
            ["--output", "out.md", "--n-sessions", "1"],
        ),
        ("scripts/run_v2_calibration.py", "v2_calibration", []),
        (
            "scripts/run_historical_walkforward.py",
            "historical_walkforward",
            [
                "--market-data",
                "missing-market.parquet",
                "--output-root",
                "walkforward-out",
                "--start-date",
                "2026-05-15",
                "--end-date",
                "2026-05-15",
            ],
        ),
        (
            "scripts/profile_engine.py",
            "profile_engine",
            ["--config-path", "missing-config.yaml"],
        ),
        (
            "scripts/audit_layer2_30d.py",
            "audit_layer2_30d",
            ["--config-path", "missing-config.yaml"],
        ),
    ],
)
def test_runner_entrypoints_materialize_manifest_before_loading_inputs(
    tmp_path: Path,
    script_path: str,
    required_for: str,
    extra_args: list[str],
) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / required_for / "marker.txt"
    source.parent.mkdir(parents=True)
    source.write_text(f"{required_for}\n")
    manifest = ArtifactManifest(
        artifact_set=f"{required_for}-runner-test",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": f"{required_for}_marker",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, f"canonical/{required_for}/marker.txt"
                    ),
                    "local_path": f"data/raw/{required_for}/marker.txt",
                    "sha256": sha256_file(source),
                    "required_for": [required_for],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data" / "raw"
    write_manifest(manifest, manifest_path)

    def tmp_arg(arg: str) -> str:
        if arg.startswith("--"):
            return arg
        if arg.endswith((".md", ".parquet", ".yaml")) or arg.endswith("-out"):
            return str(tmp_path / arg)
        return arg

    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            *[tmp_arg(arg) for arg in extra_args],
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (data_root / required_for / "marker.txt").read_text() == f"{required_for}\n"


def test_historical_walkforward_binds_manifest_resolved_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / "store"

    daily_source = store_root / "canonical" / "daily" / "SPY.parquet"
    macro_source = store_root / "canonical" / "macro-alt" / "macro.parquet"
    pit_source = store_root / "canonical" / "pit-alt" / "pit.parquet"
    event_source = store_root / "canonical" / "events-alt" / "events.yaml"
    for source in (daily_source, macro_source, pit_source, event_source):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source.name)

    manifest = ArtifactManifest(
        artifact_set="historical-walkforward-path-binding",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "daily_ohlcv_parquet_SPY",
                    "stage": "canonical",
                    "uri": _store_uri(store_root, "canonical/daily/SPY.parquet"),
                    "local_path": "data/raw/custom_daily/symbol=SPY/ohlcv.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["historical_walkforward"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "fred_macro_series",
                    "stage": "canonical",
                    "uri": _store_uri(store_root, "canonical/macro-alt/macro.parquet"),
                    "local_path": "data/raw/manifest_macro/custom_macro.parquet",
                    "sha256": sha256_file(macro_source),
                    "required_for": ["historical_walkforward"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "sp500_pit_constituents",
                    "stage": "canonical",
                    "uri": _store_uri(store_root, "canonical/pit-alt/pit.parquet"),
                    "local_path": "data/raw/manifest_pit/custom_pit.parquet",
                    "sha256": sha256_file(pit_source),
                    "required_for": ["historical_walkforward"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "event_calendar_us",
                    "stage": "canonical",
                    "uri": _store_uri(store_root, "canonical/events-alt/events.yaml"),
                    "local_path": "data/raw/manifest_events/custom_events.yaml",
                    "sha256": sha256_file(event_source),
                    "required_for": ["historical_walkforward"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data" / "raw"
    write_manifest(manifest, manifest_path)

    runner = _load_script_module(
        "run_historical_walkforward_manifest_test",
        "scripts/run_historical_walkforward.py",
    )
    captured: dict[str, Path | None] = {}

    def fake_run_walkforward(**kwargs):
        captured["event_calendar_path"] = kwargs["event_calendar_path"]
        captured["macro_parquet_path"] = kwargs["macro_parquet_path"]
        return {"success_count": 0}

    monkeypatch.setattr(runner, "run_walkforward", fake_run_walkforward)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_historical_walkforward.py",
            "--market-data",
            str(tmp_path / "market.parquet"),
            "--output-root",
            str(tmp_path / "out"),
            "--start-date",
            "2026-05-15",
            "--end-date",
            "2026-05-15",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
        ],
    )

    assert runner.main() == 0
    assert (
        captured["event_calendar_path"]
        == data_root / "manifest_events" / "custom_events.yaml"
    )
    assert (
        captured["macro_parquet_path"]
        == data_root / "manifest_macro" / "custom_macro.parquet"
    )
