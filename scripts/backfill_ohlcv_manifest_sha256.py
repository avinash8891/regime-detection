#!/usr/bin/env python3
"""Backfill sha256 digests for sentinel-hash entries in an OHLCV manifest.

The OHLCV-only lockfile at
``manifests/runs/profile_ready_daily_ohlcv_762_2016_20260515.yaml`` was
regenerated from S3 metadata when the original 15,367-line lockfile was
removed from git, but only 5 of 1086 entries were given true sha256 digests
at regeneration time. The remaining 1081 entries carry the empty-string
sentinel ``e3b0c44298...b7852b855``. ``materialize_manifest`` skips sentinel
entries at fetch time, so a fresh-workspace materialize promotes only 5 of
1086 symbol parquets to ``data/raw/daily_ohlcv_762/`` (see audit P1-3 and
``src/regime_data_fetch/materialization.py:50``).

This script HEADs each sentinel object and reads sha256 from S3 object
metadata (which the canonical upload path populates via
``S3ArtifactStore.put_file``). It does not download the parquet — HEAD is
cheap and the metadata is the source of truth for the canonical store.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_ohlcv_sha256")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "manifests"
        / "runs"
        / "profile_ready_daily_ohlcv_762_2016_20260515.yaml",
        help="Path to the OHLCV manifest to rewrite in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe and report counts without writing the manifest back.",
    )
    return parser.parse_args()


def _split_storage_root(storage_root: str) -> tuple[str, str]:
    parsed = urlparse(storage_root)
    if parsed.scheme != "s3":
        raise SystemExit(
            f"storage_root must be an s3:// URI; got {storage_root!r}"
        )
    return parsed.netloc, parsed.path.strip("/")


def main() -> int:
    args = _parse_args()
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    yaml_loader.width = 4096

    with args.manifest.open("r") as handle:
        manifest = yaml_loader.load(handle)

    bucket, prefix = _split_storage_root(manifest["storage_root"])
    s3 = boto3.client("s3")

    artifacts = manifest.get("artifacts") or []
    sentinel_indices = [
        idx
        for idx, art in enumerate(artifacts)
        if str(art.get("sha256", "")) == EMPTY_SHA256
    ]
    logger.info(
        "scanning %d artifacts; %d carry the empty-sha sentinel and need a digest",
        len(artifacts),
        len(sentinel_indices),
    )

    updated = 0
    missing_metadata: list[str] = []
    for position, idx in enumerate(sentinel_indices):
        artifact = artifacts[idx]
        uri = artifact.get("uri")
        if not uri:
            raise SystemExit(f"artifact {artifact.get('name')!r} has empty uri")
        # uri is a relative key inside storage_root
        object_key = "/".join(part.strip("/") for part in (prefix, uri) if part.strip("/"))
        response = s3.head_object(Bucket=bucket, Key=object_key)
        meta_sha = (response.get("Metadata") or {}).get("sha256")
        if not meta_sha:
            missing_metadata.append(artifact.get("name", object_key))
            continue
        if not isinstance(meta_sha, str) or len(meta_sha) != 64:
            raise SystemExit(
                f"unexpected sha256 metadata shape for {object_key!r}: {meta_sha!r}"
            )
        artifact["sha256"] = meta_sha
        updated += 1
        if (position + 1) % 100 == 0:
            logger.info("backfilled %d/%d", position + 1, len(sentinel_indices))

    logger.info(
        "backfill complete: updated=%d, missing_metadata=%d, untouched=%d",
        updated,
        len(missing_metadata),
        len(artifacts) - updated - len(missing_metadata),
    )
    if missing_metadata:
        logger.warning(
            "%d S3 objects had no sha256 metadata; left as sentinels: %s",
            len(missing_metadata),
            ", ".join(missing_metadata[:10])
            + ("..." if len(missing_metadata) > 10 else ""),
        )

    if args.dry_run:
        logger.info("--dry-run: not writing manifest")
        return 0

    with args.manifest.open("w") as handle:
        yaml_loader.dump(manifest, handle)
    logger.info("wrote %s", args.manifest)
    return 0 if not missing_metadata else 1


if __name__ == "__main__":
    raise SystemExit(main())
