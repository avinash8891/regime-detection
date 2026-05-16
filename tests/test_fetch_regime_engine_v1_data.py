from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import fetch_regime_engine_v1_data as fetch_script


def test_resolve_stock_universe_defaults_pit_parquet_to_out_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_dir = tmp_path / "portable" / "data" / "raw"
    pit_path = out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
    pit_path.parent.mkdir(parents=True)
    pd.DataFrame({"ticker": ["ZZZ", "AAA", "ZZZ"]}).to_parquet(pit_path, index=False)
    monkeypatch.setattr(fetch_script, "REPO_ROOT", tmp_path / "repo")
    args = argparse.Namespace(
        universe_json=None,
        constituent_universe_dir=None,
        pit_parquet=None,
    )

    assert fetch_script._resolve_stock_universe(args, out_dir=out_dir) == ["AAA", "ZZZ"]
