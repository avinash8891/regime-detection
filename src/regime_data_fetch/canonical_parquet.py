from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.types as pat

from regime_data_fetch.artifact_store import sha256_bytes, sha256_file

_PARQUET_COMPRESSION = "snappy"
_PARQUET_COERCE_TIMESTAMPS = "us"


def canonical_artifact_digest(path: Path) -> tuple[str, bytes | None]:
    """Digest an artifact for content-addressed storage using ONE sha scheme.

    Parquet files are canonicalized: the returned sha is over the canonical bytes,
    and those bytes are returned so the caller stores them verbatim. Non-parquet
    files are hashed raw (sha over the file as-is) and ``None`` bytes signal "store
    the file unchanged". This keeps every artifact-creation path (emit and
    ``publish_canonical_snapshot``) on the same scheme, so a parquet artifact gets
    the same content address regardless of which path produced it.
    """
    if path.suffix.lower() == ".parquet":
        canonical = canonicalize_parquet_bytes(path)
        return sha256_bytes(canonical), canonical
    return sha256_file(path), None


def canonicalize_parquet_bytes(source: Path | BinaryIO) -> bytes:
    """Return deterministic parquet bytes for a file-like source."""
    payload = _canonicalize_parquet_source(source)
    for _ in range(3):
        next_payload = _canonicalize_parquet_source(io.BytesIO(payload))
        if next_payload == payload:
            return payload
        payload = next_payload
    raise RuntimeError(f"parquet canonicalization did not converge: {source}")


def _canonicalize_parquet_source(source: Path | BinaryIO) -> bytes:
    table = pq.ParquetFile(source).read()
    table = table.replace_schema_metadata(None)
    if any(pat.is_dictionary(field.type) for field in table.schema):
        table = table.from_arrays(
            [
                (
                    column.combine_chunks().dictionary_decode()
                    if pat.is_dictionary(field.type)
                    else column
                )
                for field, column in zip(table.schema, table.itercolumns(), strict=True)
            ],
            names=table.column_names,
        )
    if table.num_rows > 0:
        sort_keys = [
            (field.name, "ascending")
            for field in table.schema
            if _is_sortable_arrow_type(field.type)
        ]
        if not sort_keys:
            indices = _full_row_sort_indices(table)
        elif len(sort_keys) == len(table.schema):
            indices = pc.sort_indices(table, sort_keys=sort_keys)
        else:
            indices = _full_row_sort_indices(table)
        table = table.take(indices)
    return _write_canonical_parquet_bytes(table)


def _full_row_sort_indices(table: Any) -> list[int]:
    rows = table.to_pylist()
    return sorted(
        range(len(rows)),
        key=lambda idx: json.dumps(rows[idx], sort_keys=True, default=str),
    )


def _write_canonical_parquet_bytes(table: Any) -> bytes:
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        compression=_PARQUET_COMPRESSION,
        coerce_timestamps=_PARQUET_COERCE_TIMESTAMPS,
        use_deprecated_int96_timestamps=False,
    )
    return buf.getvalue()


def _is_sortable_arrow_type(arrow_type: Any) -> bool:
    return not (
        pat.is_list(arrow_type)
        or pat.is_large_list(arrow_type)
        or pat.is_fixed_size_list(arrow_type)
        or pat.is_struct(arrow_type)
        or pat.is_map(arrow_type)
        or pat.is_union(arrow_type)
    )
