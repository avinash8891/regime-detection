"""V2PRE-010 / V2 §1D line 326-327 / §10 — survivorship-bias ingestion gate.

A point-in-time constituent universe includes removed/delisted members (closed
membership intervals). read_pit_intervals rejects a current-only universe (every
interval open) at load time unless biased research mode is approved. The V1
ETF-proxy breadth fallback (no PIT universe loaded at all) is a separate,
unbiased path and is not affected by this gate.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.pit_constituents import (
    BIAS_WARNING,
    SOURCE_NAME,
    SOURCE_URL,
    is_survivorship_biased_universe,
    read_pit_intervals,
)


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    pit_dir = tmp_path / "pit_constituents"
    pit_dir.mkdir(exist_ok=True)
    path = pit_dir / "sp500_ticker_intervals.parquet"
    pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "source": SOURCE_NAME,
                "source_url": SOURCE_URL,
                "bias_warning": BIAS_WARNING,
            }
            for r in rows
        ]
    ).to_parquet(path, index=False)
    return path


# Tickers deliberately NOT in SOURCE_END_DATE_CORRECTIONS so the all-open
# universe stays all-open through the patch-on-read step.
_CURRENT_ONLY_ROWS = [
    {"ticker": "AAPL", "start_date": "1980-12-12", "end_date": None},
    {"ticker": "MSFT", "start_date": "1986-03-13", "end_date": None},
]
_POINT_IN_TIME_ROWS = [
    {"ticker": "AAPL", "start_date": "1980-12-12", "end_date": None},
    {"ticker": "IBM", "start_date": "1957-03-04", "end_date": "2008-12-31"},
]


def test_is_survivorship_biased_universe_detection() -> None:
    current_only = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "start_date": ["1980-12-12", "1986-03-13"],
            "end_date": [None, None],
        }
    )
    assert is_survivorship_biased_universe(current_only) is True

    point_in_time = pd.DataFrame(
        {
            "ticker": ["AAPL", "IBM"],
            "start_date": ["1980-12-12", "1957-03-04"],
            "end_date": [None, dt.date(2008, 12, 31)],
        }
    )
    assert is_survivorship_biased_universe(point_in_time) is False

    assert is_survivorship_biased_universe(pd.DataFrame({"ticker": ["AAPL"]})) is True


def test_read_pit_intervals_rejects_current_only_universe(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, _CURRENT_ONLY_ROWS)
    with pytest.raises(ValueError, match="survivorship-biased"):
        read_pit_intervals(path)


def test_read_pit_intervals_rejects_current_only_before_source_corrections(
    tmp_path: Path,
) -> None:
    path = _write_parquet(
        tmp_path,
        [
            {"ticker": "DAY", "start_date": "2024-01-01", "end_date": None},
            {"ticker": "AAPL", "start_date": "1980-12-12", "end_date": None},
        ],
    )

    with pytest.raises(ValueError, match="survivorship-biased"):
        read_pit_intervals(path)


def test_read_pit_intervals_allows_current_only_in_research_mode(
    tmp_path: Path,
) -> None:
    path = _write_parquet(tmp_path, _CURRENT_ONLY_ROWS)
    df = read_pit_intervals(path, allow_survivorship_biased_breadth=True)
    assert set(df["ticker"]) == {"AAPL", "MSFT"}


def test_read_pit_intervals_accepts_point_in_time_universe(tmp_path: Path) -> None:
    # Includes a removed member (IBM, closed 2008) → genuine PIT → no opt-in.
    path = _write_parquet(tmp_path, _POINT_IN_TIME_ROWS)
    df = read_pit_intervals(path)
    assert df.loc[df["ticker"] == "IBM", "end_date"].item() == dt.date(2008, 12, 31)
