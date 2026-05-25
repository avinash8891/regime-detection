from __future__ import annotations

import hashlib
import importlib
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
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
    """Persisted artifact metadata returned by ``put_*`` operations.

    ``uri`` is a fully qualified backend URI: local stores emit absolute
    ``file://`` URIs and S3 stores emit ``s3://`` URIs. Store methods still
    accept relative keys for callers that address artifacts within a configured
    store root.

    ``sha256`` is mandatory here: every put_* path computes the digest as it
    writes, so callers can always trust the value. The "unverifiable" case
    (object exists in the backend but its digest is unrecoverable) belongs to
    :class:`StoreCheckResult`, not this type.
    """

    uri: str
    sha256: str
    size_bytes: int


class StoreCheckStatus(str, Enum):
    """Result classes for :meth:`ArtifactStore.check_file`.

    ``MATCH``        store has the bytes and they hash to ``expected_sha256``.
    ``DRIFT``        store has bytes but the digest differs.
    ``MISSING``      backend has no object at the URI.
    ``UNVERIFIABLE`` object exists but the backend cannot vouch for the digest
                     (e.g. an S3 object uploaded without the ``sha256``
                     user-metadata header — its ETag would not be a usable md5).
                     The local check should be the source of truth in this case;
                     ``UNVERIFIABLE`` is a warning, not a failure.
    """

    MATCH = "match"
    DRIFT = "drift"
    MISSING = "missing"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class StoreCheckResult:
    """Outcome of comparing a manifest sha against what the store holds.

    ``status`` is the discriminator. ``observed_sha`` is populated for
    ``MATCH`` and ``DRIFT`` and is ``None`` otherwise — the type doesn't
    encode that tighter at the class level, but the four-way enum + a single
    optional field is much easier to read at call sites than a bare
    ``(status_str, sha_or_none)`` tuple.
    """

    status: StoreCheckStatus
    observed_sha: str | None = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArtifactStore:
    def put_file(
        self, source_path: Path, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        raise NotImplementedError

    def put_bytes(
        self, payload: bytes, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        raise NotImplementedError

    def get_file(
        self, uri: str, destination_path: Path, *, expected_sha256: str
    ) -> Path:
        raise NotImplementedError

    def check_file(
        self,
        uri: str,
        *,
        expected_sha256: str,
        known_path: Path | None = None,
        known_sha: str | None = None,
    ) -> StoreCheckResult:
        """Compare ``expected_sha256`` to the digest of whatever the backend
        currently holds at ``uri`` and return a typed verdict.

        Backends must hash using the same scheme they use for ``put_*``
        (raw bytes for non-parquet, canonicalized bytes for parquet artifacts
        stored via ``publish_canonical_snapshot``). When the backend has the
        object but cannot recover a verifiable digest, return
        ``StoreCheckResult(status=UNVERIFIABLE)`` — *do not* return ``DRIFT``,
        which would imply a verified mismatch.

        ``known_path`` / ``known_sha`` are an optional caller-supplied hint:
        when the implementation can prove the backing object is the same file
        the caller has already hashed (e.g. a local-store root resolves to
        the same on-disk path), it may reuse ``known_sha`` instead of
        re-hashing. Implementations that cannot prove equality (e.g. S3) must
        ignore the hint.
        """
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._root_resolved = self.root.resolve()

    def put_file(
        self, source_path: Path, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        source_path = source_path.resolve()
        relative_key = self._relative_key(key)
        destination = self.root / relative_key
        source_sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size

        existing = self._existing_or_raise(
            destination, relative_key, source_sha, overwrite=overwrite
        )
        if existing is not None:
            return existing

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return StoredArtifact(
            uri=self._uri_for_key(relative_key),
            sha256=source_sha,
            size_bytes=size_bytes,
        )

    def put_bytes(
        self, payload: bytes, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        relative_key = self._relative_key(key)
        destination = self.root / relative_key
        payload_sha = sha256_bytes(payload)
        size_bytes = len(payload)

        existing = self._existing_or_raise(
            destination, relative_key, payload_sha, overwrite=overwrite
        )
        if existing is not None:
            return existing

        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(destination, payload)
        return StoredArtifact(
            uri=self._uri_for_key(relative_key),
            sha256=payload_sha,
            size_bytes=size_bytes,
        )

    def _existing_or_raise(
        self,
        destination: Path,
        relative_key: str,
        new_sha: str,
        *,
        overwrite: bool,
    ) -> StoredArtifact | None:
        if not destination.exists():
            return None
        existing_sha = sha256_file(destination)
        if existing_sha == new_sha:
            return StoredArtifact(
                uri=self._uri_for_key(relative_key),
                sha256=existing_sha,
                size_bytes=destination.stat().st_size,
            )
        if not overwrite:
            raise ArtifactOverwriteError(
                f"artifact key already exists with different bytes: {relative_key}"
            )
        return None

    def get_file(
        self, uri: str, destination_path: Path, *, expected_sha256: str
    ) -> Path:
        relative_key = self._relative_key(uri)
        source = self.root / relative_key
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

    def check_file(
        self,
        uri: str,
        *,
        expected_sha256: str,
        known_path: Path | None = None,
        known_sha: str | None = None,
    ) -> StoreCheckResult:
        relative_key = self._relative_key(uri)
        source = self.root / relative_key
        if not source.exists():
            return StoreCheckResult(status=StoreCheckStatus.MISSING)
        if (
            known_path is not None
            and known_sha is not None
            and known_path.exists()
            and known_path.resolve() == source.resolve()
        ):
            digest = known_sha
        else:
            digest = sha256_file(source)
        if digest == expected_sha256:
            return StoreCheckResult(status=StoreCheckStatus.MATCH, observed_sha=digest)
        return StoreCheckResult(status=StoreCheckStatus.DRIFT, observed_sha=digest)

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

    def put_file(
        self, source_path: Path, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        relative_key = self._relative_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_file(source_path)
        size_bytes = source_path.stat().st_size
        existing = self._existing_or_raise(
            relative_key, object_key, sha, size_bytes, overwrite=overwrite
        )
        if existing is not None:
            return existing
        self.client.upload_file(
            str(source_path),
            self.bucket,
            object_key,
            ExtraArgs={"Metadata": {"sha256": sha}},
        )
        return StoredArtifact(
            uri=self._uri_for_key(relative_key), sha256=sha, size_bytes=size_bytes
        )

    def put_bytes(
        self, payload: bytes, key: str, *, overwrite: bool = False
    ) -> StoredArtifact:
        relative_key = self._relative_key(key)
        object_key = _join_s3_key(self.prefix, relative_key)
        sha = sha256_bytes(payload)
        size_bytes = len(payload)
        existing = self._existing_or_raise(
            relative_key, object_key, sha, size_bytes, overwrite=overwrite
        )
        if existing is not None:
            return existing
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=payload,
            Metadata={"sha256": sha},
        )
        return StoredArtifact(
            uri=self._uri_for_key(relative_key), sha256=sha, size_bytes=size_bytes
        )

    def _existing_or_raise(
        self,
        relative_key: str,
        object_key: str,
        new_sha: str,
        size_bytes: int,
        *,
        overwrite: bool,
    ) -> StoredArtifact | None:
        existing = _s3_existing_artifact(
            self.client, bucket=self.bucket, object_key=object_key
        )
        if existing is None:
            return None
        existing_sha, existing_size = existing
        if existing_sha == new_sha and existing_size == size_bytes:
            return StoredArtifact(
                uri=self._uri_for_key(relative_key),
                sha256=new_sha,
                size_bytes=size_bytes,
            )
        if not overwrite:
            raise ArtifactOverwriteError(
                f"s3 artifact key already exists with different bytes: s3://{self.bucket}/{object_key}"
            )
        return None

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

    def check_file(
        self,
        uri: str,
        *,
        expected_sha256: str,
        known_path: Path | None = None,
        known_sha: str | None = None,
    ) -> StoreCheckResult:
        # S3 cannot prove a local file is the same bytes as the remote
        # object, so the known_path/known_sha hints are intentionally ignored.
        del known_path, known_sha
        relative_key = self._relative_key(uri)
        object_key = _join_s3_key(self.prefix, relative_key)
        existing = _s3_existing_artifact(
            self.client, bucket=self.bucket, object_key=object_key
        )
        if existing is None:
            return StoreCheckResult(status=StoreCheckStatus.MISSING)
        existing_sha, _existing_size = existing
        if existing_sha is None:
            return StoreCheckResult(status=StoreCheckStatus.UNVERIFIABLE)
        if existing_sha == expected_sha256:
            return StoreCheckResult(
                status=StoreCheckStatus.MATCH, observed_sha=existing_sha
            )
        return StoreCheckResult(
            status=StoreCheckStatus.DRIFT, observed_sha=existing_sha
        )

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
        except ValueError:
            pass
        # Workspace-portable fallback: a manifest authored on one machine may
        # encode an absolute ``file://`` URI whose prefix points at a different
        # checkout (e.g. ``/Users/alice/.../<store-name>/canonical/foo.parquet``)
        # than the local store root (``/Users/bob/.../<store-name>``). When the
        # store-root's basename appears as a directory component in the URI's
        # path, treat everything after that component as the relative key.
        # This keeps the store-relative contract (no escapes) while letting
        # manifests survive being moved between workspaces.
        anchor = root.name
        if anchor:
            parts = path.parts
            try:
                last_idx = len(parts) - 1 - parts[::-1].index(anchor)
                tail = parts[last_idx + 1 :]
                if tail:
                    return _normalize_relative_key(str(Path(*tail)))
            except ValueError:
                pass
        raise ValueError(
            f"file artifact URI must stay within the store root: {key}"
        )
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
        if _is_s3_not_found_error(exc):
            return None
        raise
    return existing.get("Metadata", {}).get("sha256"), int(
        existing.get("ContentLength", -1)
    )


def _is_s3_not_found_error(exc: Exception) -> bool:
    try:
        botocore_exceptions = importlib.import_module("botocore.exceptions")
    except ImportError:
        return False
    client_error_type = getattr(botocore_exceptions, "ClientError", None)
    if client_error_type is None or not isinstance(exc, client_error_type):
        return False
    response = getattr(exc, "response", {})
    error_code = str(response.get("Error", {}).get("Code", ""))
    return error_code in {"404", "NoSuchKey", "NotFound"}


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
