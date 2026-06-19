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
    build_artifact_store,
    content_addressed_key,
    sha256_bytes,
    sha256_file,
    strip_content_address,
)


def _store_uri(root: Path, key: str) -> str:
    return (root.resolve() / key).as_uri()


def test_content_addressed_key_embeds_sha_before_extension() -> None:
    sha = sha256_bytes(b"sp500-constituents")
    key = content_addressed_key(
        "canonical/pit_constituents/sp500_ticker_intervals.parquet", sha
    )
    assert key == f"canonical/pit_constituents/sp500_ticker_intervals.{sha}.parquet"


def test_content_addressed_key_is_deterministic_and_content_separating() -> None:
    base = "canonical/pit_constituents/sp500_ticker_intervals.parquet"
    sha_a = sha256_bytes(b"constituents-rev-A")
    sha_b = sha256_bytes(b"constituents-rev-B")
    assert content_addressed_key(base, sha_a) == content_addressed_key(base, sha_a)
    assert content_addressed_key(base, sha_a) != content_addressed_key(base, sha_b)


def test_content_addressed_key_handles_extensionless_name() -> None:
    sha = sha256_bytes(b"events")
    assert (
        content_addressed_key("canonical/event_calendar/us_events", sha)
        == f"canonical/event_calendar/us_events.{sha}"
    )


def test_content_addressed_key_rejects_non_hex_sha() -> None:
    with pytest.raises(ValueError, match="sha256"):
        content_addressed_key(
            "canonical/macro/fred_macro_series.parquet", "not-a-real-sha"
        )


def test_content_addressed_key_rejects_path_escape() -> None:
    sha = sha256_bytes(b"x")
    with pytest.raises(ValueError, match="relative"):
        content_addressed_key("../escape.parquet", sha)


def test_strip_content_address_round_trips_and_is_idempotent() -> None:
    base = "canonical/pit_constituents/sp500_ticker_intervals.parquet"
    sha_a = sha256_bytes(b"constituents-rev-A")
    sha_b = sha256_bytes(b"constituents-rev-B")

    addressed = content_addressed_key(base, sha_a)
    assert strip_content_address(addressed) == base
    # Idempotent on a legacy/logical key (no embedded sha).
    assert strip_content_address(base) == base
    # Re-addressing a previously-addressed key uses the base, never doubling shas.
    assert content_addressed_key(strip_content_address(addressed), sha_b) == (
        content_addressed_key(base, sha_b)
    )


def test_republish_changed_content_does_not_clobber_pinned_artifact(
    tmp_path: Path,
) -> None:
    """Regression (June-18 clobber): a re-publish of changed content must not
    overwrite the object an older lockfile pins. Content-addressed keys keep both
    revisions resolvable, with NO overwrite flag."""
    store = LocalArtifactStore(tmp_path / "store")
    logical = "canonical/pit_constituents/sp500_ticker_intervals.parquet"

    rev_a = tmp_path / "rev_a.parquet"
    rev_a.write_bytes(b"sp500-constituents-1246-rows")
    rev_b = tmp_path / "rev_b.parquet"
    rev_b.write_bytes(b"sp500-constituents-1243-rows")
    sha_a = sha256_file(rev_a)
    sha_b = sha256_file(rev_b)

    stored_a = store.put_file(rev_a, content_addressed_key(logical, sha_a))
    # Later run re-fetches changed content and re-publishes — no overwrite=True.
    stored_b = store.put_file(rev_b, content_addressed_key(logical, sha_b))

    assert stored_a.uri != stored_b.uri
    # The older pinned artifact is byte-unchanged and still resolves.
    out_a = store.get_file(
        stored_a.uri, tmp_path / "out_a.parquet", expected_sha256=sha_a
    )
    assert out_a.read_bytes() == b"sp500-constituents-1246-rows"
    # The newer artifact resolves independently.
    out_b = store.get_file(
        stored_b.uri, tmp_path / "out_b.parquet", expected_sha256=sha_b
    )
    assert out_b.read_bytes() == b"sp500-constituents-1243-rows"


def test_local_artifact_store_put_get_and_verify_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"regime-data")
    store_root = tmp_path / "store"
    store = LocalArtifactStore(store_root)

    stored = store.put_file(source, "canonical/macro/fred_macro_series.parquet")
    destination = tmp_path / "materialized" / "fred_macro_series.parquet"

    copied = store.get_file(stored.uri, destination, expected_sha256=stored.sha256)

    assert stored.uri == _store_uri(
        store_root, "canonical/macro/fred_macro_series.parquet"
    )
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


def test_local_artifact_store_rejects_different_bytes_for_existing_key(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("date,value\n2026-05-15,1\n")
    second.write_text("date,value\n2026-05-15,2\n")
    store = LocalArtifactStore(tmp_path / "store")

    original = store.put_file(first, "normalized/aaii/aaii_sentiment.parquet")

    with pytest.raises(ArtifactOverwriteError, match="already exists"):
        store.put_file(second, original.uri)


def test_local_artifact_store_allows_idempotent_same_bytes_for_existing_key(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    store = LocalArtifactStore(tmp_path / "store")

    original = store.put_file(first, "normalized/aaii/aaii_sentiment.parquet")
    duplicate = store.put_file(second, original.uri)

    assert duplicate == original
    assert duplicate.sha256 == sha256_bytes(b"same")


def test_build_artifact_store_accepts_windows_drive_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    store = build_artifact_store(r"C:\regime-data")

    assert isinstance(store, LocalArtifactStore)
    assert store.root == Path(r"C:\regime-data")


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
        store.get_file(
            "canonical/macro.parquet",
            destination,
            expected_sha256=sha256_bytes(b"good"),
        )

    assert not destination.exists()
    assert list(destination.parent.glob(".macro.parquet.*.tmp")) == []


def test_s3_artifact_store_put_file_and_bytes_share_idempotent_overwrite_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClientError(Exception):
        def __init__(self, response: dict[str, object], operation_name: str) -> None:
            super().__init__(operation_name)
            self.response = response

    class FakeS3Client:
        def __init__(self) -> None:
            self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            del Bucket
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            payload, metadata = self.objects[Key]
            return {"Metadata": metadata, "ContentLength": len(payload)}

        def upload_file(
            self, filename: str, bucket: str, key: str, ExtraArgs: dict[str, object]
        ) -> None:
            del bucket
            self.objects[key] = (Path(filename).read_bytes(), ExtraArgs["Metadata"])

        def put_object(
            self, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
        ) -> None:
            del Bucket
            self.objects[Key] = (Body, Metadata)

    client = FakeS3Client()
    fake_boto3 = types.SimpleNamespace(client=lambda service: client)
    fake_botocore = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(ClientError=ClientError)
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore.exceptions)
    store = S3ArtifactStore("s3://regime-data/artifacts")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same")
    second.write_bytes(b"different")

    stored = store.put_file(first, "canonical/a.csv")
    duplicate = store.put_bytes(b"same", stored.uri)

    assert duplicate == stored
    assert stored.uri == "s3://regime-data/artifacts/canonical/a.csv"
    with pytest.raises(ArtifactOverwriteError, match="different bytes"):
        store.put_file(second, "canonical/a.csv")
