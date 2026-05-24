from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.constituent_ohlcv_tree import materialize_constituent_ohlcv_tree


def _write_ohlcv(path: Path, close: float = 100.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000,
                "adjusted_close": close,
            }
        ]
    ).to_parquet(path, index=False)


def test_materialize_constituent_ohlcv_tree_writes_canonical_tree_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_ohlcv(source / "symbol=AAPL" / "part.parquet", 100.0)
    _write_ohlcv(source / "symbol=MSFT" / "ohlcv.parquet", 200.0)
    pit = tmp_path / "pit.parquet"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "start_date": "2026-01-01", "end_date": None},
            {"ticker": "MSFT", "start_date": "2020-01-01", "end_date": "2026-05-31"},
            {"ticker": "OLD", "start_date": "2010-01-01", "end_date": "2020-01-01"},
        ]
    ).to_parquet(pit, index=False)

    result = materialize_constituent_ohlcv_tree(
        source_tree=source,
        output_tree=tmp_path / "daily_ohlcv_762",
        pit_parquet_path=pit,
        start_date=dt.date(2026, 5, 1),
        end_date=dt.date(2026, 5, 2),
    )

    assert result.requested_symbols == 2
    assert result.written_symbols == 2
    assert result.missing_symbols == ()
    assert (tmp_path / "daily_ohlcv_762" / "symbol=AAPL" / "ohlcv.parquet").exists()
    assert (tmp_path / "daily_ohlcv_762" / "symbol=MSFT" / "ohlcv.parquet").exists()
    aapl = pd.read_parquet(
        tmp_path / "daily_ohlcv_762" / "symbol=AAPL" / "ohlcv.parquet"
    )
    assert aapl["symbol"].to_list() == ["AAPL"]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["requested_symbols"] == 2
    assert manifest["written_symbols"] == 2
    assert [item["symbol"] for item in manifest["files"]] == ["AAPL", "MSFT"]
    report = json.loads(result.report_path.read_text())
    assert report["counts"] == {
        "requested_symbols": 2,
        "written_symbols": 2,
        "missing_symbols": 0,
    }


def test_materialize_constituent_ohlcv_tree_fails_on_missing_symbol(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_ohlcv(source / "symbol=AAPL" / "part.parquet")
    pit = tmp_path / "pit.parquet"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "start_date": "2026-01-01", "end_date": None},
            {"ticker": "MSFT", "start_date": "2026-01-01", "end_date": None},
        ]
    ).to_parquet(pit, index=False)

    with pytest.raises(FileNotFoundError, match="MSFT"):
        materialize_constituent_ohlcv_tree(
            source_tree=source,
            output_tree=tmp_path / "daily_ohlcv_762",
            pit_parquet_path=pit,
            start_date=dt.date(2026, 5, 1),
            end_date=dt.date(2026, 5, 2),
        )


def test_materialize_constituent_ohlcv_tree_can_record_missing_symbols(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_ohlcv(source / "symbol=AAPL" / "part.parquet")
    pit = tmp_path / "pit.parquet"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "start_date": "2026-01-01", "end_date": None},
            {"ticker": "MSFT", "start_date": "2026-01-01", "end_date": None},
        ]
    ).to_parquet(pit, index=False)

    result = materialize_constituent_ohlcv_tree(
        source_tree=source,
        output_tree=tmp_path / "daily_ohlcv_762",
        pit_parquet_path=pit,
        start_date=dt.date(2026, 5, 1),
        end_date=dt.date(2026, 5, 2),
        allow_missing_symbols=True,
    )

    assert result.written_symbols == 1
    assert result.missing_symbols == ("MSFT",)
    report = json.loads(result.report_path.read_text())
    assert report["missing_symbols"] == ["MSFT"]
