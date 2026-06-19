from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from regime_data_fetch.artifact_store import sha256_file
from regime_data_fetch.canonical_parquet import (
    canonical_artifact_digest,
    canonicalize_parquet_bytes,
)


def _write_ohlcv_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2026-05-13", "2026-05-14", "2026-05-15"],
            "open": [101.0, 102.5, 103.1],
            "high": [104.0, 103.9, 105.2],
            "low": [100.5, 101.8, 102.0],
            "close": [103.2, 103.0, 104.8],
            "volume": [1_200_000, 980_000, 1_410_000],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def test_canonical_artifact_digest_uses_canonical_sha_for_parquet(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "daily_ohlcv_762" / "symbol=SPY" / "ohlcv.parquet"
    _write_ohlcv_parquet(parquet)

    sha, canon = canonical_artifact_digest(parquet)

    expected_canon = canonicalize_parquet_bytes(parquet)
    assert canon == expected_canon
    assert sha == hashlib.sha256(expected_canon).hexdigest()
    # The whole point: canonical sha differs from the raw file sha, and is the one
    # publish_canonical_snapshot also pins — so the two creation paths agree.
    assert sha != sha256_file(parquet)


def test_canonical_artifact_digest_is_raw_for_non_parquet(tmp_path: Path) -> None:
    yaml_path = tmp_path / "event_calendar" / "us_events.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("events:\n  - date: '2026-05-15'\n    label: CPI\n")

    sha, canon = canonical_artifact_digest(yaml_path)

    assert canon is None  # signal: store the file as-is
    assert sha == sha256_file(yaml_path)


def test_canonical_artifact_digest_stable_across_row_order(tmp_path: Path) -> None:
    # Same logical OHLCV data written in different row orders must digest identically
    # (this is what raw hashing got wrong and canonicalization fixes).
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    rows = [
        {"date": "2026-05-13", "close": 103.2},
        {"date": "2026-05-14", "close": 103.0},
    ]
    pq.write_table(pa.Table.from_pylist(rows), a)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), b)

    assert canonical_artifact_digest(a)[0] == canonical_artifact_digest(b)[0]
