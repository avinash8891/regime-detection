from __future__ import annotations

import pytest

from regime_data_fetch.sqlite_identifiers import quote_sqlite_identifier


def test_quote_sqlite_identifier_accepts_allowlisted_table_name() -> None:
    assert (
        quote_sqlite_identifier(
            "daily_ohlcv_rows", allowed_identifiers={"daily_ohlcv_rows"}
        )
        == '"daily_ohlcv_rows"'
    )


def test_quote_sqlite_identifier_rejects_unknown_table_name() -> None:
    with pytest.raises(ValueError, match="Unexpected SQLite identifier"):
        quote_sqlite_identifier(
            "daily_ohlcv_rows; DROP TABLE fetch_runs",
            allowed_identifiers={"daily_ohlcv_rows"},
        )
