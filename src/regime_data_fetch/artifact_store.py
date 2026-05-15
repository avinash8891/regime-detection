from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class ArtifactStoreError(RuntimeError):
    pass


class ArtifactHashMismatchError(ArtifactStoreError):
    pass


class ArtifactOverwriteError(ArtifactStoreError):
    pass


@dataclass(frozen=True)
class StoredArtifact:
    uri: str
    sha256: str
    size_bytes: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArtifactStore:
    def put_file(self, source_path: Path, key: str) -> StoredArtifact:
        raise NotImplementedError

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        raise NotImplementedError

    def get_file(self, uri: str, destination_path: Path, *, expected_sha256: str) -> Path:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source_path: Path, key: str) -> StoredArtifact:
        source_path = source_path.resolve()
        relative_key = _normalize_key(key)
        destination = self.root / relative_key
        source_sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size

        if destination.exists():
            existing_sha = sha256_file(destination)
            if existing_sha != source_sha:
                raise ArtifactOverwriteError(
                    f"artifact key already exists with different bytes: {relative_key}"
                )
            return StoredArtifact(uri=relative_key, sha256=existing_sha, size_bytes=destination.stat().st_size)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return StoredArtifact(uri=relative_key, sha256=source_sha, size_bytes=size_bytes)

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        relative_key = _normalize_key(key)
        destination = self.root / relative_key
        payload_sha = sha256_bytes(payload)
        size_bytes = len(payload)

        if destination.exists():
            existing_sha = sha256_file(destination)
            if existing_sha != payload_sha:
                raise ArtifactOverwriteError(
                    f"artifact key already exists with different bytes: {relative_key}"
                )
            return StoredArtifact(uri=relative_key, sha256=existing_sha, size_bytes=destination.stat().st_size)

        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(destination, payload)
        return StoredArtifact(uri=relative_key, sha256=payload_sha, size_bytes=size_bytes)

    def get_file(self, uri: str, destination_path: Path, *, expected_sha256: str) -> Path:
        relative_key = _normalize_key(uri)
        source = self.root / relative_key
        if not source.exists():
            raise FileNotFoundError(source)

        actual_sha = sha256_file(source)
        if actual_sha != expected_sha256:
            raise ArtifactHashMismatchError(
                f"sha256 mismatch for {relative_key}: expected {expected_sha256}, got {actual_sha}"
            )

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_path(destination_path)
        try:
            shutil.copy2(source, tmp_path)
            copied_sha = sha256_file(tmp_path)
            if copied_sha != expected_sha256:
                raise ArtifactHashMismatchError(
                    f"sha256 mismatch after materializing {relative_key}: "
                    f"expected {expected_sha256}, got {copied_sha}"
                )
            tmp_path.replace(destination_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return destination_path


def build_artifact_store(root_uri: str | Path) -> ArtifactStore:
    root_text = str(root_uri)
    parsed = urlparse(root_text)
    if parsed.scheme in {"", "file"}:
        root = Path(parsed.path if parsed.scheme == "file" else root_text)
        return LocalArtifactStore(root)
    if parsed.scheme == "s3":
        return S3ArtifactStore(root_text)
    raise ValueError(f"unsupported artifact store URI scheme: {parsed.scheme}")


class S3ArtifactStore(ArtifactStore):
    def __init__(self, root_uri: str) -> None:
        self.root_uri = root_uri.rstrip("/")
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "S3 artifact storage requires optional dependency boto3. "
                "Install boto3 or use a local/file artifact store for tests."
            ) from exc
        parsed = urlparse(self.root_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"invalid s3 artifact root: {root_uri}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client("s3")

    def put_file(self, source_path: Path, key: str) -> StoredArtifact:
        relative_key = _normalize_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size
        try:
            existing = self.client.head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:  # botocore is optional; avoid importing its exception type.
            response = getattr(exc, "response", {})
            error_code = str(response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            existing_sha = existing.get("Metadata", {}).get("sha256")
            existing_size = int(existing.get("ContentLength", -1))
            if existing_sha == sha and existing_size == size_bytes:
                return StoredArtifact(uri=relative_key, sha256=sha, size_bytes=size_bytes)
            raise ArtifactOverwriteError(
                f"s3 artifact key already exists with different bytes: s3://{self.bucket}/{object_key}"
            )
        self.client.upload_file(
            str(source_path),
            self.bucket,
            object_key,
            ExtraArgs={"Metadata": {"sha256": sha}},
        )
        return StoredArtifact(uri=relative_key, sha256=sha, size_bytes=size_bytes)

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        relative_key = _normalize_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_bytes(payload)
        size_bytes = len(payload)
        try:
            existing = self.client.head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:  # botocore is optional; avoid importing its exception type.
            response = getattr(exc, "response", {})
            error_code = str(response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            existing_sha = existing.get("Metadata", {}).get("sha256")
            existing_size = int(existing.get("ContentLength", -1))
            if existing_sha == sha and existing_size == size_bytes:
                return StoredArtifact(uri=relative_key, sha256=sha, size_bytes=size_bytes)
            raise ArtifactOverwriteError(
                f"s3 artifact key already exists with different bytes: s3://{self.bucket}/{object_key}"
            )
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=payload,
            Metadata={"sha256": sha},
        )
        return StoredArtifact(uri=relative_key, sha256=sha, size_bytes=size_bytes)

    def get_file(self, uri: str, destination_path: Path, *, expected_sha256: str) -> Path:
        relative_key = _normalize_key(uri)
        object_key = _join_s3_key(self.prefix, relative_key)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_path(destination_path)
        try:
            self.client.download_file(self.bucket, object_key, str(tmp_path))
            actual_sha = sha256_file(tmp_path)
            if actual_sha != expected_sha256:
                raise ArtifactHashMismatchError(
                    f"sha256 mismatch for s3://{self.bucket}/{object_key}: "
                    f"expected {expected_sha256}, got {actual_sha}"
                )
            tmp_path.replace(destination_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return destination_path


def _normalize_key(key: str) -> str:
    parsed = urlparse(key)
    if parsed.scheme == "file":
        key = parsed.path
    elif parsed.scheme == "s3":
        key = parsed.path
    normalized = str(Path(key))
    if normalized.startswith("../") or normalized == ".." or Path(normalized).is_absolute():
        raise ValueError(f"artifact key must be relative within the store: {key}")
    return normalized


def _join_s3_key(prefix: str, key: str) -> str:
    return "/".join(part.strip("/") for part in (prefix, key) if part.strip("/"))


def _temporary_path(destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    handle.close()
    return Path(handle.name)


def _write_bytes_atomically(destination_path: Path, payload: bytes) -> None:
    tmp_path = _temporary_path(destination_path)
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(destination_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
