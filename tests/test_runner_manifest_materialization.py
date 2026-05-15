from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from regime_data_fetch.artifact_manifest import ArtifactManifest, ManifestArtifact, write_manifest
from regime_data_fetch.artifact_store import sha256_file
from scripts import run_v2_walkforward_gate


def test_walkforward_gate_parse_args_materializes_manifest_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "store"
    daily_source = store_root / "canonical" / "daily_ohlcv" / "part.parquet"
    daily_source.parent.mkdir(parents=True)
    pd.DataFrame([{"date": pd.Timestamp("2026-05-15"), "symbol": "SPY"}]).to_parquet(
        daily_source,
        index=False,
    )
    macro_source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    macro_source.parent.mkdir(parents=True)
    pd.DataFrame([{"date": pd.Timestamp("2026-05-15"), "series_id": "DGS10", "value": 4.0}]).to_parquet(
        macro_source,
        index=False,
    )
    pmi_source = store_root / "canonical" / "manual_inputs" / "pmi" / "ism_manufacturing_pmi.tsv"
    pmi_source.parent.mkdir(parents=True)
    pmi_source.write_text("period\trelease_date_local\ttime_local\tactual\n2026-04\t01-05-2026\t10:00\t52.7\n")
    manifest = ArtifactManifest(
        artifact_set="runner-test",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "daily_ohlcv_part",
                    "stage": "canonical",
                    "uri": "canonical/daily_ohlcv/part.parquet",
                    "local_path": "data/raw/daily_ohlcv/part.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["v2_calibration"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": "canonical/macro/fred_macro_series.parquet",
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(macro_source),
                    "required_for": ["v2_calibration"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "manual_pmi",
                    "stage": "canonical",
                    "uri": "canonical/manual_inputs/pmi/ism_manufacturing_pmi.tsv",
                    "local_path": "data/manual_inputs/pmi/ism_manufacturing_pmi.tsv",
                    "sha256": sha256_file(pmi_source),
                    "required_for": ["v2_calibration"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data" / "raw"
    repo_root = tmp_path / "repo"
    write_manifest(manifest, manifest_path)
    monkeypatch.setattr(run_v2_walkforward_gate, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_walkforward_gate.py",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / "out.md"),
        ],
    )

    args = run_v2_walkforward_gate._parse_args()

    assert args.daily_dir == data_root / "daily_ohlcv"
    assert args.macro_parquet == data_root / "macro" / "fred_macro_series.parquet"
    assert args.end_date.isoformat() == "2026-05-15"
    assert (data_root / "daily_ohlcv" / "part.parquet").exists()
    assert args.macro_parquet.exists()
    assert (repo_root / "data" / "manual_inputs" / "pmi" / "ism_manufacturing_pmi.tsv").exists()
