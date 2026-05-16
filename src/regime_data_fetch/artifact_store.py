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
    """Persisted artifact metadata.

    ``uri`` is a fully qualified backend URI: local stores emit absolute
    ``file://`` URIs and S3 stores emit ``s3://`` URIs. Store methods still
    accept relative keys for callers that address artifacts within a configured
    store root.
    """

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

    def get_file(
        self, uri: str, destination_path: Path, *, expected_sha256: str
    ) -> Path:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._root_resolved = self.root.resolve()

    def put_file(self, source_path: Path, key: str) -> StoredArtifact:
        source_path = source_path.resolve()
        relative_key = self._relative_key(key)
        destination = self.root / relative_key
        source_sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size

        if destination.exists():
            existing_sha = sha256_file(destination)
            if existing_sha != source_sha:
                raise ArtifactOverwriteError(
                    f"artifact key already exists with different bytes: {relative_key}"
                )
            return StoredArtifact(
                uri=self._uri_for_key(relative_key),
                sha256=existing_sha,
                size_bytes=destination.stat().st_size,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return StoredArtifact(
            uri=self._uri_for_key(relative_key),
            sha256=source_sha,
            size_bytes=size_bytes,
        )

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        relative_key = self._relative_key(key)
        destination = self.root / relative_key
        payload_sha = sha256_bytes(payload)
        size_bytes = len(payload)

        if destination.exists():
            existing_sha = sha256_file(destination)
            if existing_sha != payload_sha:
                raise ArtifactOverwriteError(
                    f"artifact key already exists with different bytes: {relative_key}"
                )
            return StoredArtifact(
                uri=self._uri_for_key(relative_key),
                sha256=existing_sha,
                size_bytes=destination.stat().st_size,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(destination, payload)
        return StoredArtifact(
            uri=self._uri_for_key(relative_key),
            sha256=payload_sha,
            size_bytes=size_bytes,
        )

    def get_file(
        self, uri: str, destination_path: Path, *, expected_sha256: str
    ) -> Path:
        relative_key = self._relative_key(uri)
        source = self.root / relative_key
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

    def _relative_key(self, key_or_uri: str) -> str:
        return _normalize_local_key(key_or_uri, self._root_resolved)

    def _uri_for_key(self, relative_key: str) -> str:
        return (self._root_resolved / relative_key).as_uri()


def build_artifact_store(root_uri: str | Path) -> ArtifactStore:
    root_text = str(root_uri)
    if _is_windows_absolute_path(root_text):
        return LocalArtifactStore(Path(root_text))
    parsed = urlparse(root_text)
    if parsed.scheme in {"", "file"}:
        root = Path(parsed.path if parsed.scheme == "file" else root_text)
        return LocalArtifactStore(root)
    if parsed.scheme == "s3":
        return S3ArtifactStore(root_text)
    raise ValueError(f"unsupported artifact store URI scheme: {parsed.scheme}")


def _is_windows_absolute_path(value: str) -> bool:
    return (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in {"\\", "/"}
    )


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
        relative_key = self._relative_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size
        existing = _s3_existing_artifact(
            self.client, bucket=self.bucket, object_key=object_key
        )
        if existing is not None:
            existing_sha, existing_size = existing
            if existing_sha == sha and existing_size == size_bytes:
                return StoredArtifact(
                    uri=self._uri_for_key(relative_key),
                    sha256=sha,
                    size_bytes=size_bytes,
                )
            raise ArtifactOverwriteError(
                f"s3 artifact key already exists with different bytes: s3://{self.bucket}/{object_key}"
            )
        self.client.upload_file(
            str(source_path),
            self.bucket,
            object_key,
            ExtraArgs={"Metadata": {"sha256": sha}},
        )
        return StoredArtifact(
            uri=self._uri_for_key(relative_key), sha256=sha, size_bytes=size_bytes
        )

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        relative_key = self._relative_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_bytes(payload)
        size_bytes = len(payload)
        existing = _s3_existing_artifact(
            self.client, bucket=self.bucket, object_key=object_key
        )
        if existing is not None:
            existing_sha, existing_size = existing
            if existing_sha == sha and existing_size == size_bytes:
                return StoredArtifact(
                    uri=self._uri_for_key(relative_key),
                    sha256=sha,
                    size_bytes=size_bytes,
                )
            raise ArtifactOverwriteError(
                f"s3 artifact key already exists with different bytes: s3://{self.bucket}/{object_key}"
            )
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=payload,
            Metadata={"sha256": sha},
        )
        return StoredArtifact(
            uri=self._uri_for_key(relative_key), sha256=sha, size_bytes=size_bytes
        )

    def get_file(
        self, uri: str, destination_path: Path, *, expected_sha256: str
    ) -> Path:
        relative_key = self._relative_key(uri)
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

    def _relative_key(self, key_or_uri: str) -> str:
        return _normalize_s3_key(
            key_or_uri, bucket=self.bucket, prefix=self.prefix
        )

    def _uri_for_key(self, relative_key: str) -> str:
        object_key = _join_s3_key(self.prefix, relative_key)
        return f"s3://{self.bucket}/{object_key}"


def _normalize_local_key(key: str, root: Path) -> str:
    parsed = urlparse(key)
    if parsed.scheme == "file":
        path = Path(parsed.path).resolve()
        try:
            return str(path.relative_to(root))
        except ValueError as exc:
            raise ValueError(
                f"file artifact URI must stay within the store root: {key}"
            ) from exc
    if parsed.scheme:
        raise ValueError(f"unsupported local artifact URI scheme: {parsed.scheme}")
    return _normalize_relative_key(key)


def _normalize_s3_key(key: str, *, bucket: str, prefix: str) -> str:
    parsed = urlparse(key)
    if parsed.scheme == "s3":
        if parsed.netloc != bucket:
            raise ValueError(
                f"s3 artifact URI bucket {parsed.netloc} does not match store bucket {bucket}"
            )
        object_key = parsed.path.strip("/")
        prefix = prefix.strip("/")
        if prefix:
            if object_key == prefix:
                raise ValueError(f"s3 artifact URI must include a key: {key}")
            if not object_key.startswith(f"{prefix}/"):
                raise ValueError(
                    f"s3 artifact URI must stay within the store prefix: {key}"
                )
            object_key = object_key[len(prefix) + 1 :]
        return _normalize_relative_key(object_key)
    if parsed.scheme:
        raise ValueError(f"unsupported s3 artifact URI scheme: {parsed.scheme}")
    return _normalize_relative_key(key)


def _normalize_relative_key(key: str) -> str:
    normalized = str(Path(key))
    if (
        normalized.startswith("../")
        or normalized == ".."
        or Path(normalized).is_absolute()
    ):
        raise ValueError(f"artifact key must be relative within the store: {key}")
    return normalized


def _join_s3_key(prefix: str, key: str) -> str:
    return "/".join(part.strip("/") for part in (prefix, key) if part.strip("/"))


def _s3_existing_artifact(
    client: object, *, bucket: str, object_key: str
) -> tuple[str | None, int] | None:
    try:
        existing = client.head_object(Bucket=bucket, Key=object_key)
    except Exception as exc:
        # boto3 is optional here, so do not import botocore just to name its
        # ClientError. Tests and local file-store users should not need it.
        response = getattr(exc, "response", {})
        error_code = str(response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return existing.get("Metadata", {}).get("sha256"), int(
        existing.get("ContentLength", -1)
    )


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
