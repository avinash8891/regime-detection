from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from regime_data_fetch.artifact_store import (
    ArtifactHashMismatchError,
    ArtifactOverwriteError,
    LocalArtifactStore,
    S3ArtifactStore,
    sha256_bytes,
    sha256_file,
)


def test_local_artifact_store_put_get_and_verify_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"regime-data")
    store = LocalArtifactStore(tmp_path / "store")

    stored = store.put_file(source, "canonical/macro/fred_macro_series.parquet")
    destination = tmp_path / "materialized" / "fred_macro_series.parquet"

    copied = store.get_file(stored.uri, destination, expected_sha256=stored.sha256)

    assert stored.uri == "canonical/macro/fred_macro_series.parquet"
    assert stored.size_bytes == len(b"regime-data")
    assert stored.sha256 == sha256_file(source)
    assert copied == destination
    assert destination.read_bytes() == b"regime-data"


def test_local_artifact_store_rejects_hash_mismatch_on_get(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}')
    store = LocalArtifactStore(tmp_path / "store")
    stored = store.put_file(source, "raw_capture/fred/response.json")

    with pytest.raises(ArtifactHashMismatchError, match="sha256 mismatch"):
        store.get_file(stored.uri, tmp_path / "out.json", expected_sha256="0" * 64)


def test_local_artifact_store_rejects_different_bytes_for_existing_key(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("date,value\n2026-05-15,1\n")
    second.write_text("date,value\n2026-05-15,2\n")
    store = LocalArtifactStore(tmp_path / "store")

    original = store.put_file(first, "normalized/aaii/aaii_sentiment.parquet")

    with pytest.raises(ArtifactOverwriteError, match="already exists"):
        store.put_file(second, original.uri)


def test_local_artifact_store_allows_idempotent_same_bytes_for_existing_key(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    store = LocalArtifactStore(tmp_path / "store")

    original = store.put_file(first, "normalized/aaii/aaii_sentiment.parquet")
    duplicate = store.put_file(second, original.uri)

    assert duplicate == original
    assert duplicate.sha256 == sha256_bytes(b"same")


def test_s3_artifact_store_removes_temp_file_when_download_hash_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3Client:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            del bucket, key
            Path(filename).write_bytes(b"corrupt")

    fake_boto3 = types.SimpleNamespace(client=lambda service: FakeS3Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    store = S3ArtifactStore("s3://regime-data/artifacts")
    destination = tmp_path / "data" / "raw" / "macro.parquet"

    with pytest.raises(ArtifactHashMismatchError, match="sha256 mismatch"):
        store.get_file("canonical/macro.parquet", destination, expected_sha256=sha256_bytes(b"good"))

    assert not destination.exists()
    assert list(destination.parent.glob(".macro.parquet.*.tmp")) == []
