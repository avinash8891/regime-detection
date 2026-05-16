from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    write_manifest,
)
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
                    "name": "pmi_history",
                    "stage": "canonical",
                    "uri": "canonical/pmi/us_ism_pmi_history.parquet",
                    "local_path": "data/raw/pmi/us_ism_pmi_history.parquet",
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
    assert args.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"
    assert args.pmi_path.exists()


def test_manifest_materialization_is_wired_for_all_runner_entrypoints() -> None:
    expected = {
        "scripts/run_v2_walkforward_gate.py": 'required_for="v2_calibration"',
        "scripts/run_v2_shadow_ab_gate.py": 'required_for="v2_calibration"',
        "scripts/run_v2_calibration.py": 'required_for="v2_calibration"',
        "scripts/run_historical_walkforward.py": 'required_for="historical_walkforward"',
        "scripts/profile_engine_30d.py": 'required_for="profile_engine_30d"',
        "scripts/audit_layer2_30d.py": 'required_for="audit_layer2_30d"',
    }
    for path, required_for in expected.items():
        text = Path(path).read_text()
        assert "materialize_if_requested" in text, path
        assert "--manifest" in text, path
        assert "--artifact-store" in text, path
        assert "--data-root" in text, path
        assert required_for in text, path


def test_emitted_manifest_tags_all_runner_use_cases_by_default() -> None:
    text = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert (
        'default="profile_engine_30d,v2_calibration,historical_walkforward,audit_layer2_30d"'
        in text
    )
