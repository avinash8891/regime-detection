from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd

from regime_shared.pandas_compat import cow_safe_assign, optional_date


def test_cow_safe_assign_replaces_columns_without_copy_on_write_warning() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-05-15"],
            "symbol": ["SPY"],
            "close": [100.0],
        }
    )

    with pd.option_context("mode.copy_on_write", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = cow_safe_assign(frame, {"date": pd.to_datetime(frame["date"])})

    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["symbol"].to_list() == ["SPY"]
    assert out["close"].to_list() == [100.0]


def test_cow_safe_assign_can_select_output_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-05-15"],
            "symbol": ["SPY"],
            "unused": [1],
        }
    )

    out = cow_safe_assign(
        frame,
        {"date": pd.to_datetime(frame["date"])},
        columns=["date", "symbol"],
    )

    assert out.columns.tolist() == ["date", "symbol"]


def test_cow_safe_assign_can_append_replacement_columns() -> None:
    frame = pd.DataFrame({"date": ["2026-05-15"], "symbol": ["SPY"]})

    out = cow_safe_assign(
        frame,
        {
            "date": pd.to_datetime(frame["date"]),
            "release_date_local": pd.DatetimeIndex(["2026-05-15"]),
        },
    )

    assert out.columns.tolist() == ["date", "symbol", "release_date_local"]
    assert out.loc[0, "release_date_local"] == pd.Timestamp("2026-05-15")


def test_optional_date_normalizes_nulls_dates_datetimes_and_iso_strings() -> None:
    assert optional_date(None) is None
    assert optional_date(pd.NA) is None
    assert optional_date(dt.date(2026, 5, 15)) == dt.date(2026, 5, 15)
    assert optional_date(dt.datetime(2026, 5, 15, 12, 0)) == dt.date(2026, 5, 15)
    assert optional_date("2026-05-15") == dt.date(2026, 5, 15)
