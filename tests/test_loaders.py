from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.loaders import (
    load_macro_series,
    load_sector_etf_closes,
)


def test_load_sector_etf_closes_rejects_malformed_dates() -> None:
    source = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "symbol": ["XLB"],
            "close": [100.0],
        }
    )

    with pytest.raises(ValueError, match="malformed date"):
        load_sector_etf_closes(source)


def test_load_sector_etf_closes_rejects_non_numeric_close_values() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "symbol": ["XLB"],
            "close": ["bad-close"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric close"):
        load_sector_etf_closes(source)


def test_load_macro_series_rejects_malformed_dates() -> None:
    source = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "series_id": ["DGS10"],
            "value": [4.25],
        }
    )

    with pytest.raises(ValueError, match="malformed date"):
        load_macro_series(source)


def test_load_macro_series_rejects_non_numeric_values() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "series_id": ["DGS10"],
            "value": ["bad-value"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric value"):
        load_macro_series(source)
