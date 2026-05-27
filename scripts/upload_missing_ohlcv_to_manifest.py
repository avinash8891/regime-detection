"""Upload missing PIT-constituent OHLCV parquets to S3 and add manifest entries.

Reads each symbol parquet from the repaired local tree, canonicalizes it,
uploads to S3 under the merged runtime manifest's storage_root, then appends
entries to the merged manifest.

Run once; idempotent on re-run (skips symbols already in the manifest).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from regime_data_fetch.artifact_store import (
    build_artifact_store,
    sha256_bytes,
)  # noqa: E402
from regime_data_fetch.canonical_parquet import (  # noqa: E402
    canonicalize_parquet_bytes,
)
from regime_data_fetch.cli_common import load_operator_env_files  # noqa: E402

LOGGER = logging.getLogger("upload_missing_ohlcv")

_DEFAULT_SOURCE_TREE = _REPO_ROOT / ".context" / "daily_ohlcv_2016_20260515_repaired"
_MERGED_MANIFEST = _REPO_ROOT / "manifests" / "runs" / "regime_engine_2026-05-17.yaml"

def _make_yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def _load_manifest(path: Path, yaml):
    with path.open("r") as fh:
        return yaml.load(fh)


def _dump_manifest(payload, yaml, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        yaml.dump(payload, fh)
    os.replace(tmp, path)


def _manifest_symbol_set(payload) -> set[str]:
    return {
        a["name"].replace("daily_ohlcv_762_", "")
        for a in payload["artifacts"]
        if a["name"].startswith("daily_ohlcv_762_")
    }


def _date_range(canon: bytes) -> tuple[str | None, str | None]:
    import pandas as pd

    table = pq.read_table(io.BytesIO(canon))
    if "date" not in table.column_names:
        return None, None
    series = table.column("date").to_pandas()
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return None, None
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def _build_artifact_entry(symbol: str, canon: bytes, sha: str) -> dict:
    row_count = pq.ParquetFile(io.BytesIO(canon)).metadata.num_rows
    min_d, max_d = _date_range(canon)
    uri = f"canonical/daily_ohlcv_762/symbol={symbol}/ohlcv.parquet"
    local_path = f"data/raw/daily_ohlcv_762/symbol={symbol}/ohlcv.parquet"
    entry: dict = {
        "name": f"daily_ohlcv_762_{symbol}",
        "stage": "canonical",
        "uri": uri,
        "local_path": local_path,
        "sha256": sha,
        "schema_version": None,
        "rows": row_count,
    }
    if min_d:
        entry["min_date"] = min_d
    if max_d:
        entry["max_date"] = max_d
    entry["required_for"] = [
        "profile_engine",
        "v2_calibration",
        "historical_walkforward",
        "audit_layer2_30d",
    ]
    return entry


def _insertion_index(artifacts: list, symbol: str) -> int:
    """Return index just after the last daily_ohlcv_762 entry that sorts before symbol."""
    last = -1
    for i, a in enumerate(artifacts):
        name = a.get("name", "")
        if not name.startswith("daily_ohlcv_762_"):
            continue
        existing_sym = name[len("daily_ohlcv_762_") :]
        if existing_sym <= symbol:
            last = i
    return last + 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, default=_DEFAULT_SOURCE_TREE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    load_operator_env_files(repo_root=_REPO_ROOT)

    yaml = _make_yaml()
    merged = _load_manifest(_MERGED_MANIFEST, yaml)

    storage_root: str = merged["storage_root"]
    store = None if args.dry_run else build_artifact_store(storage_root)

    merged_symbols = _manifest_symbol_set(merged)

    source_tree = args.source_tree
    if not source_tree.exists():
        LOGGER.error("source tree not found: %s", source_tree)
        return 1

    candidate_dirs = sorted(
        d for d in source_tree.iterdir() if d.is_dir() and d.name.startswith("symbol=")
    )

    uploaded = 0
    skipped = 0
    errors: list[str] = []

    for sym_dir in candidate_dirs:
        symbol = sym_dir.name.split("=", 1)[1]
        parquet_path = sym_dir / "ohlcv.parquet"
        if not parquet_path.exists():
            continue

        in_merged = symbol in merged_symbols
        if in_merged:
            skipped += 1
            continue

        LOGGER.info("processing %s", symbol)
        try:
            canon = canonicalize_parquet_bytes(parquet_path)
        except Exception as exc:
            LOGGER.error("  canonicalize failed: %s", exc)
            errors.append(symbol)
            continue

        sha = sha256_bytes(canon)
        entry = _build_artifact_entry(symbol, canon, sha)
        uri = entry["uri"]

        if not args.dry_run:
            assert store is not None
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp.write(canon)
                tmp_path = Path(tmp.name)
            try:
                store.put_file(tmp_path, uri, overwrite=True)
            finally:
                tmp_path.unlink(missing_ok=True)

        if not in_merged:
            idx = _insertion_index(merged["artifacts"], symbol)
            merged["artifacts"].insert(idx, entry)

        LOGGER.info(
            "  sha=%s rows=%s min=%s max=%s %s",
            sha[:12],
            entry["rows"],
            entry.get("min_date"),
            entry.get("max_date"),
            "DRY-RUN" if args.dry_run else "uploaded",
        )
        uploaded += 1

    LOGGER.info(
        "done: uploaded=%d skipped=%d errors=%d", uploaded, skipped, len(errors)
    )
    if errors:
        LOGGER.error("symbols with errors: %s", errors)
        return 1

    if not args.dry_run and uploaded > 0:
        _dump_manifest(merged, yaml, _MERGED_MANIFEST)
        LOGGER.info("manifest rewritten (%d new entries)", uploaded)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
