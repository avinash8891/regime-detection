from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from regime_data_fetch.pit_constituents import (
    BIAS_WARNING,
    SOURCE_NAME,
    SOURCE_URL,
    run_pit_constituents_fetch,
)
from regime_data_fetch.pit_constituents import (  # noqa: E402  (under-test imports)
    members_on,
    read_pit_intervals,
)


# Fixed PIT CSV: AAPL (closed), MSFT (open), IBM (closed).
# Real S&P 500 historical tickers + plausible (hand-picked) interval bounds.
_PIT_CSV = "\n".join(
    [
        "ticker,start_date,end_date",
        "AAPL,1980-12-12,2024-12-31",
        "MSFT,1986-03-13,",
        "IBM,1957-03-04,2008-12-31",
    ]
)


def _build_intervals_parquet(tmp_path: Path) -> Path:
    run_pit_constituents_fetch(
        out_dir=tmp_path,
        csv_fetcher=lambda: _PIT_CSV,
    )
    return tmp_path / "pit_constituents" / "sp500_ticker_intervals.parquet"


def test_read_pit_intervals_round_trips_writer_schema(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)

    df = read_pit_intervals(parquet_path)

    assert list(df.columns) == [
        "ticker",
        "start_date",
        "end_date",
        "source",
        "source_url",
        "bias_warning",
    ]
    # All six columns are object dtype after ISO string -> date conversion.
    assert df["ticker"].dtype == object
    assert df["start_date"].dtype == object
    assert df["end_date"].dtype == object
    assert df["source"].dtype == object
    assert df["source_url"].dtype == object
    assert df["bias_warning"].dtype == object

    by_ticker = {row["ticker"]: row for _, row in df.iterrows()}

    aapl = by_ticker["AAPL"]
    assert aapl["start_date"] == dt.date(1980, 12, 12)
    assert aapl["end_date"] == dt.date(2024, 12, 31)
    assert aapl["source"] == SOURCE_NAME
    assert aapl["source_url"] == SOURCE_URL
    assert aapl["bias_warning"] == BIAS_WARNING

    msft = by_ticker["MSFT"]
    assert msft["start_date"] == dt.date(1986, 3, 13)
    assert msft["end_date"] is None

    ibm = by_ticker["IBM"]
    assert ibm["start_date"] == dt.date(1957, 3, 4)
    assert ibm["end_date"] == dt.date(2008, 12, 31)


def test_members_on_inclusive_lower_bound(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(1980, 12, 12))

    assert "AAPL" in members


def test_members_on_inclusive_upper_bound(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(2008, 12, 31))

    assert "IBM" in members


def test_members_on_excludes_before_start(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(1980, 12, 11))

    assert "AAPL" not in members


def test_members_on_excludes_after_end(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(2009, 1, 1))

    assert "IBM" not in members
    assert "AAPL" in members
    assert "MSFT" in members


def test_members_on_null_end_date_treated_as_open_interval(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(2099, 1, 1))

    assert "MSFT" in members


def test_members_on_returns_frozenset(tmp_path: Path) -> None:
    parquet_path = _build_intervals_parquet(tmp_path)
    df = read_pit_intervals(parquet_path)

    members = members_on(df, dt.date(2000, 1, 1))

    assert isinstance(members, frozenset)


def test_members_on_empty_df_returns_empty_frozenset() -> None:
    empty_df = pd.DataFrame(
        columns=[
            "ticker",
            "start_date",
            "end_date",
            "source",
            "source_url",
            "bias_warning",
        ]
    )

    members = members_on(empty_df, dt.date(2020, 1, 1))

    assert members == frozenset()
