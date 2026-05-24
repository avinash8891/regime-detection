from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd
import pytest

from regime_data_fetch.acquisition_consolidation_normalized import (
    AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE,
    AGGREGATE_EPS_WAYBACK_ROWS_TABLE,
    ALPACA_MARKET_ROWS_TABLE,
    EVENT_CALENDAR_ROWS_TABLE,
    FOMC_MINUTES_ROWS_TABLE,
    MACRO_ROWS_TABLE,
    PIT_CONSTITUENT_ROWS_TABLE,
    PMI_ROWS_TABLE,
    POWELL_SPEECHES_ROWS_TABLE,
    USD_INDEX_ROWS_TABLE,
    _ensure_normalized_tables,
    _import_normalized_output,
)

_OPEN_CONNS: list[sqlite3.Connection] = []


def _normalized_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE fetch_runs (run_id INTEGER PRIMARY KEY, fetch_type TEXT)"
    )
    conn.execute("INSERT INTO fetch_runs (run_id, fetch_type) VALUES (7, 'test')")
    _ensure_normalized_tables(conn)
    _OPEN_CONNS.append(conn)
    return conn


@pytest.fixture(autouse=True)
def _close_normalized_conns():
    yield
    while _OPEN_CONNS:
        _OPEN_CONNS.pop().close()


def test_import_normalized_output_loads_event_calendar_yaml(tmp_path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-01-28"',
                '    release_timestamp_et: "2026-01-28T14:00:00-05:00"',
                '    market: "US"',
                '    type: "FOMC"',
                '    importance: "high"',
                '    source: "federalreserve.gov:fomccalendars"',
            ]
        )
        + "\n"
    )
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn,
        run_id=7,
        output_kind="event_calendar_yaml",
        path=path,
    )

    assert table == EVENT_CALENDAR_ROWS_TABLE
    assert conn.execute(
        "SELECT run_id, event_date, event_type, source FROM event_calendar_rows"
    ).fetchall() == [(7, "2026-01-28", "FOMC", "federalreserve.gov:fomccalendars")]


def test_import_normalized_output_returns_none_for_unknown_kind(tmp_path) -> None:
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn,
        run_id=1,
        output_kind="unsupported_output",
        path=tmp_path / "ignored.parquet",
    )

    assert table is None


def test_import_normalized_output_loads_fred_macro_parquet(tmp_path) -> None:
    path = tmp_path / "fred_macro_series.parquet"
    pd.DataFrame(
        [
            {
                "date": dt.date(2026, 5, 18),
                "series_id": "DGS10",
                "value": 4.45,
                "realtime_start": "2026-05-20",
                "realtime_end": "2026-05-20",
                "logical_name": "ten_year_treasury_yield",
            },
            {
                "date": pd.Timestamp("2026-05-19"),
                "series_id": "DGS10",
                "value": None,
                "realtime_start": "2026-05-20",
                "realtime_end": "2026-05-20",
                "logical_name": "ten_year_treasury_yield",
            },
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="fred_macro_parquet", path=path
    )

    assert table == MACRO_ROWS_TABLE
    assert conn.execute(
        """
        SELECT run_id, dataset_kind, date, series_id, value, realtime_start, realtime_end, logical_name
        FROM macro_rows ORDER BY date
        """
    ).fetchall() == [
        (
            7,
            "series",
            "2026-05-18",
            "DGS10",
            4.45,
            "2026-05-20",
            "2026-05-20",
            "ten_year_treasury_yield",
        ),
        (
            7,
            "series",
            "2026-05-19",
            "DGS10",
            None,
            "2026-05-20",
            "2026-05-20",
            "ten_year_treasury_yield",
        ),
    ]


def test_import_normalized_output_loads_cpi_vintage_macro_parquet(tmp_path) -> None:
    path = tmp_path / "fred_cpi_vintages.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-04-01",
                "series_id": "CPIAUCSL",
                "value": 321.123,
                "realtime_start": "2026-05-13",
                "realtime_end": "2026-05-13",
                "logical_name": "cpi_all_items",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="fred_cpi_vintages_parquet", path=path
    )

    assert table == MACRO_ROWS_TABLE
    assert conn.execute(
        "SELECT dataset_kind, date, series_id, value FROM macro_rows"
    ).fetchall() == [("cpi_vintages", "2026-04-01", "CPIAUCSL", 321.123)]


@pytest.mark.parametrize(
    ("output_kind", "expected_kind"),
    [
        ("pmi_parquet", "latest"),
        ("pmi_history_parquet", "history"),
    ],
)
def test_import_normalized_output_loads_pmi_parquet(
    tmp_path, output_kind: str, expected_kind: str
) -> None:
    path = tmp_path / f"{output_kind}.parquet"
    pd.DataFrame(
        [
            {
                "series_name": "ISM Manufacturing PMI",
                "period": "2026-04",
                "value": 49.2,
                "release_timestamp": "2026-05-01T10:00:00-04:00",
                "source": "ismworld.org",
                "source_url": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind=output_kind, path=path
    )

    assert table == PMI_ROWS_TABLE
    assert conn.execute(
        "SELECT dataset_kind, series_name, period, value, source FROM pmi_rows"
    ).fetchall() == [
        (expected_kind, "ISM Manufacturing PMI", "2026-04", 49.2, "ismworld.org")
    ]


def test_import_normalized_output_loads_pit_constituents_parquet(tmp_path) -> None:
    path = tmp_path / "sp500_ticker_intervals.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start_date": dt.date(1982, 11, 30),
                "end_date": pd.NaT,
                "source": "wikipedia:sp500_changes",
                "source_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "bias_warning": "point_in_time_interval",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="pit_constituents_parquet", path=path
    )

    assert table == PIT_CONSTITUENT_ROWS_TABLE
    assert conn.execute(
        "SELECT ticker, start_date, end_date, source, bias_warning FROM pit_constituent_rows"
    ).fetchall() == [
        (
            "AAPL",
            "1982-11-30",
            None,
            "wikipedia:sp500_changes",
            "point_in_time_interval",
        )
    ]


def test_import_normalized_output_loads_fomc_minutes_parquet(tmp_path) -> None:
    path = tmp_path / "fomc_minutes.parquet"
    pd.DataFrame(
        [
            {
                "meeting_end_date": dt.date(2026, 1, 28),
                "release_timestamp": pd.Timestamp("2026-02-18T19:00:00Z"),
                "title": "Minutes of the Federal Open Market Committee",
                "meeting_date_text": "January 27-28, 2026",
                "body_text": "Participants discussed financial conditions.",
                "source": "federalreserve.gov:fomcminutes",
                "source_url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
                "pdf_url": None,
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="fomc_minutes_parquet", path=path
    )

    assert table == FOMC_MINUTES_ROWS_TABLE
    assert conn.execute(
        "SELECT meeting_end_date, release_timestamp, title, pdf_url FROM fomc_minutes_rows"
    ).fetchall() == [
        (
            "2026-01-28",
            "2026-02-18T19:00:00+00:00",
            "Minutes of the Federal Open Market Committee",
            None,
        )
    ]


def test_import_normalized_output_loads_powell_speeches_parquet(tmp_path) -> None:
    path = tmp_path / "powell_speeches.parquet"
    pd.DataFrame(
        [
            {
                "speech_date": dt.date(2026, 5, 1),
                "publication_timestamp": "2026-05-01T09:00:00-04:00",
                "publication_timestamp_precision": "minute",
                "title": "Economic Outlook",
                "speaker": "Jerome H. Powell",
                "location": "Washington, D.C.",
                "body_text": "Inflation remains a key focus.",
                "source": "federalreserve.gov:powell_speeches",
                "source_url": "https://www.federalreserve.gov/newsevents/speech/powell20260501a.htm",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="powell_speeches_parquet", path=path
    )

    assert table == POWELL_SPEECHES_ROWS_TABLE
    assert conn.execute(
        "SELECT speech_date, title, speaker, source FROM powell_speeches_rows"
    ).fetchall() == [
        (
            "2026-05-01",
            "Economic Outlook",
            "Jerome H. Powell",
            "federalreserve.gov:powell_speeches",
        )
    ]


def test_import_normalized_output_loads_usd_index_parquet(tmp_path) -> None:
    path = tmp_path / "usd_index.parquet"
    pd.DataFrame(
        [
            {
                "date": dt.date(2026, 5, 20),
                "symbol": "DX-Y.NYB",
                "open": 101.0,
                "high": 101.7,
                "low": 100.8,
                "close": 101.3,
                "adjusted_close": 101.3,
                "volume": None,
                "source": "yahoo_chart",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="usd_index_parquet", path=path
    )

    assert table == USD_INDEX_ROWS_TABLE
    assert conn.execute(
        "SELECT date, symbol, open, high, low, close, adjusted_close, volume, source FROM usd_index_rows"
    ).fetchall() == [
        ("2026-05-20", "DX-Y.NYB", 101.0, 101.7, 100.8, 101.3, 101.3, 0, "yahoo_chart")
    ]


def test_import_normalized_output_loads_aggregate_eps_snapshot_parquet(
    tmp_path,
) -> None:
    path = tmp_path / "sp500_eps_snapshots.parquet"
    pd.DataFrame(
        [
            {
                "workbook_as_of_date": dt.date(2026, 5, 16),
                "observation_date": dt.date(2026, 5, 16),
                "observation_label": "current",
                "forward_estimate_label": "2026E",
                "forward_estimate_value": 275.5,
                "estimate_2025e": 250.0,
                "estimate_q4_2025e": 62.0,
                "estimate_2026e": 275.5,
                "price": 5600.0,
                "pe_2025e": 22.4,
                "pe_2026e": 20.3,
                "change_vs_prior_observation_2025e": 0.4,
                "change_vs_prior_observation_q4_2025e": 0.1,
                "change_vs_prior_observation_2026e": 0.5,
                "change_vs_prior_observation_price": 12.0,
                "change_vs_prior_observation_pe_2025e": 0.02,
                "change_vs_prior_observation_pe_2026e": 0.01,
                "source": "spglobal",
                "source_path": "data/raw/spglobal_eps/sp-500-eps-est.xlsx",
                "public_files_discontinued": True,
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="aggregate_eps_parquet", path=path
    )

    assert table == AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE
    assert conn.execute(
        """
        SELECT workbook_as_of_date, observation_date, forward_estimate_label,
               forward_estimate_value, public_files_discontinued
        FROM aggregate_eps_snapshot_rows
        """
    ).fetchall() == [("2026-05-16", "2026-05-16", "2026E", 275.5, 1)]


def test_import_normalized_output_loads_aggregate_eps_wayback_parquet(tmp_path) -> None:
    path = tmp_path / "sp500_eps_wayback_timeline.parquet"
    pd.DataFrame(
        [
            {
                "snapshot_date": dt.date(2026, 5, 17),
                "timestamp": "20260517010203",
                "archive_url": "https://web.archive.org/web/20260517010203/https://www.spglobal.com/spdji/",
                "workbook_as_of_date": dt.date(2026, 5, 16),
                "forward_estimate_label": "2026E",
                "forward_estimate_value": 275.5,
                "estimate_2025e": 250.0,
                "estimate_q4_2025e": 62.0,
                "estimate_2026e": 275.5,
                "price": 5600.0,
                "pe_2025e": 22.4,
                "pe_2026e": 20.3,
                "change_vs_prior_observation_2025e": 0.4,
                "change_vs_prior_observation_q4_2025e": 0.1,
                "change_vs_prior_observation_2026e": 0.5,
                "change_vs_prior_observation_price": 12.0,
                "change_vs_prior_observation_pe_2025e": 0.02,
                "change_vs_prior_observation_pe_2026e": 0.01,
                "public_files_discontinued": False,
                "source": "wayback_machine",
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn,
        run_id=7,
        output_kind="aggregate_eps_wayback_timeline",
        path=path,
    )

    assert table == AGGREGATE_EPS_WAYBACK_ROWS_TABLE
    assert conn.execute(
        """
        SELECT snapshot_date, timestamp, workbook_as_of_date,
               forward_estimate_value, public_files_discontinued, source
        FROM aggregate_eps_wayback_rows
        """
    ).fetchall() == [
        ("2026-05-17", "20260517010203", "2026-05-16", 275.5, 0, "wayback_machine")
    ]


def test_import_normalized_output_loads_alpaca_partitioned_symbol_parquet(
    tmp_path,
) -> None:
    path = tmp_path / "symbol=SPY" / "ohlcv.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "date": dt.date(2026, 5, 20),
                "open": 590.0,
                "high": 592.5,
                "low": 589.0,
                "close": 591.2,
                "volume": 73_000_000,
                "adjusted_close": 591.2,
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    table = _import_normalized_output(
        dst_conn=conn, run_id=7, output_kind="alpaca_daily_ohlcv_parquet", path=path
    )

    assert table == ALPACA_MARKET_ROWS_TABLE
    assert conn.execute(
        "SELECT symbol, date, open, high, low, close, volume, adjusted_close, source_file FROM alpaca_market_rows"
    ).fetchall() == [
        ("SPY", "2026-05-20", 590.0, 592.5, 589.0, 591.2, 73_000_000, 591.2, str(path))
    ]


def test_import_normalized_output_raises_for_missing_parquet(tmp_path) -> None:
    conn = _normalized_conn()
    missing = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError, match="Missing derived output parquet during consolidation"
    ):
        _import_normalized_output(
            dst_conn=conn, run_id=7, output_kind="fred_macro_parquet", path=missing
        )


def test_import_normalized_output_raises_for_malformed_event_yaml(tmp_path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text("events: not-a-list\n")
    conn = _normalized_conn()

    with pytest.raises(RuntimeError, match="Unexpected event calendar YAML shape"):
        _import_normalized_output(
            dst_conn=conn, run_id=7, output_kind="event_calendar_yaml", path=path
        )


def test_import_normalized_output_raises_for_missing_event_yaml(tmp_path) -> None:
    conn = _normalized_conn()
    missing = tmp_path / "events.yaml"

    with pytest.raises(
        FileNotFoundError, match="Missing derived output YAML during consolidation"
    ):
        _import_normalized_output(
            dst_conn=conn, run_id=7, output_kind="event_calendar_yaml", path=missing
        )


def test_import_normalized_output_raises_when_alpaca_path_has_no_symbol_partition(
    tmp_path,
) -> None:
    path = tmp_path / "SPY.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-05-20",
                "open": 590.0,
                "high": 592.5,
                "low": 589.0,
                "close": 591.2,
                "volume": 73_000_000,
                "adjusted_close": 591.2,
            }
        ]
    ).to_parquet(path, index=False)
    conn = _normalized_conn()

    with pytest.raises(RuntimeError, match="Could not infer symbol"):
        _import_normalized_output(
            dst_conn=conn,
            run_id=7,
            output_kind="alpaca_daily_ohlcv_parquet",
            path=path,
        )
