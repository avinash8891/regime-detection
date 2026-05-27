from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.types as pat

_PARQUET_COMPRESSION = "snappy"
_PARQUET_COERCE_TIMESTAMPS = "us"


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
            return _write_canonical_parquet_bytes(table)
        indices = pc.sort_indices(table, sort_keys=sort_keys)
        table = table.take(indices)
    return _write_canonical_parquet_bytes(table)


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
