"""Cleveland Fed inflation-nowcast fetcher tests (ADR 0006 / Log #48).

The Cleveland Fed CSV schema cannot be verified without web access, so the
fetcher parameterises the column mapping (date_column / value_column /
value_scale). These tests exercise that parameterisation against a
synthetic CSV fixture shaped like the expected export — a date column plus
several inflation-measure columns.

Per ~/.claude testing rules: real series key (`cpi_nowcast`), real module
constants, no mocks. Network is never touched — the manual-drop path is
the one under test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.cleveland_fed_nowcast import (
    CPI_NOWCAST_PARQUET,
    DEFAULT_VALUE_SCALE,
    MANUAL_REL_PATH,
    ClevelandFedNowcastError,
    parse_cleveland_fed_nowcast_csv,
    run_cleveland_fed_nowcast_fetch,
    update_cpi_nowcast_parquet,
)

# Synthetic CSV shaped like the Cleveland Fed export: a `date` column plus
# month-over-month nowcasts for several inflation measures, published in
# percent. The fetcher's default mapping reads the `CPI` column.
_FIXTURE_CSV = (
    "date,CPI,Core CPI,PCE,Core PCE\n"
    "2026-01-15,0.28,0.31,0.22,0.25\n"
    "2026-02-15,0.34,0.30,0.27,0.24\n"
    "2026-03-15,0.41,0.35,0.33,0.29\n"
)


def _write_manual_csv(out_dir: Path, csv_text: str = _FIXTURE_CSV) -> Path:
    csv_path = out_dir / MANUAL_REL_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text)
    return csv_path


# --- parse ------------------------------------------------------------------


def test_parse_applies_default_percent_to_fraction_scale() -> None:
    df = parse_cleveland_fed_nowcast_csv(_FIXTURE_CSV)
    assert list(df.columns) == ["date", "cpi_nowcast"]
    assert len(df) == 3
    # 0.28 percent m/m -> 0.0028 fractional under DEFAULT_VALUE_SCALE.
    assert df.loc[0, "date"] == pd.Timestamp("2026-01-15")
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.28 * DEFAULT_VALUE_SCALE)
    assert df.loc[2, "cpi_nowcast"] == pytest.approx(0.41 * DEFAULT_VALUE_SCALE)


def test_parse_honours_custom_column_and_scale() -> None:
    """The operator pins the verified schema via the parameters — here a
    different measure column and an already-fractional export."""
    df = parse_cleveland_fed_nowcast_csv(
        _FIXTURE_CSV, value_column="Core PCE", value_scale=1.0
    )
    assert df.loc[1, "cpi_nowcast"] == pytest.approx(0.24)


def test_parse_keeps_last_on_duplicate_date() -> None:
    csv_text = (
        "date,CPI\n"
        "2026-01-15,0.28\n"
        "2026-01-15,0.35\n"  # same date, revised estimate
    )
    df = parse_cleveland_fed_nowcast_csv(csv_text)
    assert len(df) == 1
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.35 * DEFAULT_VALUE_SCALE)


def test_parse_skips_blank_rows() -> None:
    csv_text = "date,CPI\n2026-01-15,0.28\n,\n2026-02-15,0.34\n"
    df = parse_cleveland_fed_nowcast_csv(csv_text)
    assert len(df) == 2


def test_parse_raises_on_missing_value_column() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="missing column 'CPI'"):
        parse_cleveland_fed_nowcast_csv("date,Headline\n2026-01-15,0.28\n")


def test_parse_raises_on_missing_date_column() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="missing column 'date'"):
        parse_cleveland_fed_nowcast_csv("observation,CPI\n2026-01-15,0.28\n")


def test_parse_raises_on_unparseable_value() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="unparseable value"):
        parse_cleveland_fed_nowcast_csv("date,CPI\n2026-01-15,n/a\n")


def test_parse_raises_when_no_usable_rows() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="no usable rows"):
        parse_cleveland_fed_nowcast_csv("date,CPI\n,\n,\n")


# --- update_cpi_nowcast_parquet ---------------------------------------------


def test_update_creates_parquet_when_absent(tmp_path: Path) -> None:
    csv_path = _write_manual_csv(tmp_path)
    out_path = tmp_path / "cleveland_fed_nowcast" / CPI_NOWCAST_PARQUET
    df = update_cpi_nowcast_parquet(csv_path=csv_path, out_path=out_path)
    assert out_path.exists()
    assert len(df) == 3
    reloaded = pd.read_parquet(out_path)
    assert list(reloaded.columns) == ["date", "cpi_nowcast"]


def test_update_merges_and_supersedes_existing(tmp_path: Path) -> None:
    """A re-fetch with a later export supersedes same-date rows and appends
    new ones — the nowcast is revised intra-month as daily data lands."""
    out_path = tmp_path / "cleveland_fed_nowcast" / CPI_NOWCAST_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Seed an existing parquet: Jan + Feb, with a stale Feb value.
    seed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-15"), pd.Timestamp("2026-02-15")],
            "cpi_nowcast": [0.0028, 0.0099],
        }
    )
    seed.to_parquet(out_path, index=False)

    csv_path = _write_manual_csv(tmp_path)  # Jan/Feb/Mar fixture
    df = update_cpi_nowcast_parquet(csv_path=csv_path, out_path=out_path)

    assert len(df) == 3  # Jan superseded, Feb superseded, Mar appended
    feb = df.loc[df["date"] == pd.Timestamp("2026-02-15"), "cpi_nowcast"].iloc[0]
    # Stale 0.0099 replaced by the fresh parse (0.34 percent -> fraction).
    assert feb == pytest.approx(0.34 * DEFAULT_VALUE_SCALE)


def test_update_raises_when_csv_absent(tmp_path: Path) -> None:
    with pytest.raises(ClevelandFedNowcastError, match="No Cleveland Fed nowcast CSV"):
        update_cpi_nowcast_parquet(
            csv_path=tmp_path / "missing.csv",
            out_path=tmp_path / "out.parquet",
        )


# --- run_cleveland_fed_nowcast_fetch ----------------------------------------


def test_run_fetch_produces_parquet_and_report(tmp_path: Path) -> None:
    _write_manual_csv(tmp_path)
    report_path = run_cleveland_fed_nowcast_fetch(out_dir=tmp_path)

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["rows"] == 3
    assert report["min_date"] == "2026-01-15"
    assert report["max_date"] == "2026-03-15"
    assert report["column_mapping"]["value_scale"] == DEFAULT_VALUE_SCALE
    assert "verification_needed" in report

    parquet_path = Path(report["paths"]["cpi_nowcast_parquet"])
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert len(df) == 3


def test_run_fetch_threads_custom_column_mapping(tmp_path: Path) -> None:
    _write_manual_csv(tmp_path)
    report_path = run_cleveland_fed_nowcast_fetch(
        out_dir=tmp_path, value_column="Core CPI", value_scale=1.0
    )
    report = json.loads(report_path.read_text())
    assert report["column_mapping"]["value_column"] == "Core CPI"
    df = pd.read_parquet(report["paths"]["cpi_nowcast_parquet"])
    # Core CPI Jan = 0.31, value_scale=1.0 -> unscaled.
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.31)


def test_run_fetch_raises_when_no_manual_csv(tmp_path: Path) -> None:
    with pytest.raises(ClevelandFedNowcastError, match="No Cleveland Fed nowcast CSV"):
        run_cleveland_fed_nowcast_fetch(out_dir=tmp_path)
