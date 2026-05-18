"""Tests for ``scripts/publish_canonical_snapshot.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "publish_canonical_snapshot.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "publish_canonical_snapshot", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_canonical_snapshot"] = module
    spec.loader.exec_module(module)
    return module


pcs = _load_script_module()


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="snappy",
        coerce_timestamps="us",
        use_deprecated_int96_timestamps=False,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def manifest_setup(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    """Build a 2-artifact manifest backed by canonical parquet bytes on disk."""
    data_root = tmp_path / "data" / "raw"
    a_rel = Path("macro") / "fred_macro_series.parquet"
    b_rel = Path("pmi") / "us_ism_pmi_history.parquet"
    a_path = data_root / a_rel
    b_path = data_root / b_rel

    df_a = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "series_id": ["UNRATE", "UNRATE", "UNRATE"],
            "value": [3.7, 3.8, 3.9],
        }
    )
    df_b = pd.DataFrame(
        {
            "release_date": ["2024-02-01", "2024-03-01"],
            "headline_pmi": [49.1, 50.3],
        }
    )

    # Use the script's canonicalization to compute the on-disk + sha values.
    _write_parquet(a_path, df_a)
    _write_parquet(b_path, df_b)
    canonical_a = pcs._canonicalize_parquet_bytes(a_path)
    canonical_b = pcs._canonicalize_parquet_bytes(b_path)
    a_path.write_bytes(canonical_a)
    b_path.write_bytes(canonical_b)

    manifest = {
        "artifact_set": "test_publish",
        "created_at_utc": "2026-05-17T00:00:00Z",
        "storage_root": str(tmp_path / "store"),
        "artifacts": [
            {
                "name": "fred_macro_series",
                "stage": "canonical",
                "uri": f"canonical/{a_rel.as_posix()}",
                "local_path": f"data/raw/{a_rel.as_posix()}",
                "sha256": hashlib.sha256(canonical_a).hexdigest(),
                "schema_version": None,
                "rows": pq.ParquetFile(a_path).metadata.num_rows,
                "min_date": "2024-01-01",
                "max_date": "2024-01-03",
                "required_for": ["profile_engine"],
            },
            {
                "name": "ism_pmi_history",
                "stage": "canonical",
                "uri": f"canonical/{b_rel.as_posix()}",
                "local_path": f"data/raw/{b_rel.as_posix()}",
                "sha256": hashlib.sha256(canonical_b).hexdigest(),
                "schema_version": None,
                "rows": pq.ParquetFile(b_path).metadata.num_rows,
                "min_date": "2024-02-01",
                "max_date": "2024-03-01",
                "required_for": ["profile_engine"],
            },
        ],
    }

    manifest_path = tmp_path / "manifest.yaml"
    import yaml

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path, data_root, manifest, {"a_path": a_path, "b_path": b_path}


def _run_main(args: list[str]) -> int:
    return pcs.main(args)


def test_publish_is_idempotent_when_data_unchanged(manifest_setup):
    manifest_path, data_root, _, _ = manifest_setup
    before = manifest_path.read_text()

    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--skip-upload",
        ]
    )
    assert rc == 0
    after_first = manifest_path.read_text()
    assert after_first == before  # no changes -> no rewrite

    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--skip-upload",
        ]
    )
    assert rc == 0
    assert manifest_path.read_text() == before


def test_publish_updates_manifest_on_sha_change(manifest_setup):
    manifest_path, data_root, original_manifest, paths = manifest_setup
    old_sha = original_manifest["artifacts"][0]["sha256"]

    # Mutate the fred parquet by adding a row, written through canonical form
    # so the new sha is reproducible.
    new_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "series_id": ["UNRATE", "UNRATE", "UNRATE", "UNRATE"],
            "value": [3.7, 3.8, 3.9, 4.0],
        }
    )
    _write_parquet(paths["a_path"], new_df)

    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--skip-upload",
        ]
    )
    assert rc == 0

    import yaml

    updated = yaml.safe_load(manifest_path.read_text())
    a = next(x for x in updated["artifacts"] if x["name"] == "fred_macro_series")
    assert a["sha256"] != old_sha
    assert a["sha256"] == _sha256_file(paths["a_path"])
    assert a["rows"] == 4
    assert a["max_date"] == "2024-01-04"

    # The other artifact must not be touched.
    b = next(x for x in updated["artifacts"] if x["name"] == "ism_pmi_history")
    assert b["sha256"] == original_manifest["artifacts"][1]["sha256"]


def test_dry_run_does_not_mutate_anything(manifest_setup):
    manifest_path, data_root, _, paths = manifest_setup
    manifest_before = manifest_path.read_text()
    a_bytes_before = paths["a_path"].read_bytes()

    # Mutate disk so dry-run has something to report.
    new_df = pd.DataFrame(
        {"date": ["2024-01-01"], "series_id": ["X"], "value": [1.0]}
    )
    _write_parquet(paths["a_path"], new_df)
    a_bytes_mutated = paths["a_path"].read_bytes()

    rc = _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert manifest_path.read_text() == manifest_before
    assert paths["a_path"].read_bytes() == a_bytes_mutated  # unchanged by dry-run
    assert a_bytes_mutated != a_bytes_before  # sanity: we did mutate disk


def test_check_mode_returns_nonzero_on_drift(manifest_setup, caplog):
    import logging

    manifest_path, data_root, _, paths = manifest_setup

    new_df = pd.DataFrame(
        {"date": ["2099-01-01"], "series_id": ["DRIFT"], "value": [42.0]}
    )
    _write_parquet(paths["a_path"], new_df)

    with caplog.at_level(logging.INFO, logger="publish_canonical_snapshot"):
        rc = _run_main(
            [
                "--manifest",
                str(manifest_path),
                "--data-root",
                str(data_root),
                "--check",
            ]
        )
    assert rc == 1
    combined = "\n".join(rec.message for rec in caplog.records)
    assert "fred_macro_series" in combined
    assert "DRIFT" in combined


def test_canonicalize_is_a_fixed_point_for_datetime_float_parquet(tmp_path: Path):
    """Regression: pandas <-> pyarrow round-trip embedded metadata that made
    canon(canon(x)) != canon(x) for parquets with datetime + float columns.
    Real Cleveland Fed nowcast file triggered this. Canon must be idempotent.
    """
    import datetime as dt

    path = tmp_path / "sample.parquet"
    df = pd.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "value": [0.123456789, 0.987654321],
        }
    )
    _write_parquet(path, df)

    pass1 = pcs._canonicalize_parquet_bytes(path)
    path.write_bytes(pass1)
    pass2 = pcs._canonicalize_parquet_bytes(path)
    path.write_bytes(pass2)
    pass3 = pcs._canonicalize_parquet_bytes(path)

    assert pass1 == pass2 == pass3, (
        f"canonicalize is not a fixed point: "
        f"sha1={hashlib.sha256(pass1).hexdigest()[:12]} "
        f"sha2={hashlib.sha256(pass2).hexdigest()[:12]} "
        f"sha3={hashlib.sha256(pass3).hexdigest()[:12]}"
    )
