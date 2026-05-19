from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd

from regime_data_fetch.alpaca_daily import DailyBarsFetchResult
from regime_data_fetch.fetch_workflow import (
    V2_CROSS_ASSET_SYMBOLS,
    V2_FRED_SERIES,
    V2_SECTOR_SYMBOLS,
    V2_V1_SHARED_ANCHORS,
    build_market_symbols,
    run_market_fetch,
    run_macro_fetch,
    write_event_calendar_template,
)
from regime_data_fetch.ism import extract_ism_pmi_value, release_timestamp_for


def test_build_market_symbols_supports_v1_v2_and_all() -> None:
    v1 = build_market_symbols(
        scope="v1", stock_symbols=["AAPL", "MSFT"], vix_symbol="VIX"
    )
    assert v1[:4] == ["AAPL", "MSFT", "SPY", "RSP"]
    assert v1[-1] == "VIX"

    v2 = build_market_symbols(scope="v2", stock_symbols=["AAPL"], vix_symbol="VIXY")
    assert v2[:3] == ["SPY", "RSP", "KRE"]
    assert "VIXY" in v2
    assert set(V2_SECTOR_SYMBOLS).issubset(v2)
    assert set(V2_CROSS_ASSET_SYMBOLS).issubset(v2)

    combined = build_market_symbols(
        scope="all", stock_symbols=["AAPL", "AAPL"], vix_symbol="VIX"
    )
    assert combined.count("AAPL") == 1
    assert combined.count("SPY") == 1
    assert combined.count("VIX") == 1
    assert set(V2_V1_SHARED_ANCHORS).issubset(combined)


def test_write_event_calendar_template_includes_v1_and_v2_examples(
    tmp_path: Path,
) -> None:
    template_path = write_event_calendar_template(tmp_path)
    contents = template_path.read_text()

    assert "FOMC" in contents
    assert 'type: "election"' in contents
    assert 'type: "geopolitical_event"' in contents


def test_run_market_fetch_writes_unified_report(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_daily_bars_alpaca(
        *,
        symbols: list[str],
        start_date: dt.date,
        end_date: dt.date,
        adjustment: str = "raw",
        feed: str | None = None,
        batch_size: int = 100,
        verbose: bool = False,
    ) -> DailyBarsFetchResult:
        captured["symbols"] = symbols
        rows = [
            {
                "date": dt.date(2015, 1, 2),
                "symbol": symbol,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "adjusted_close": 10.5,
            }
            for symbol in symbols
        ]
        return DailyBarsFetchResult(df=pd.DataFrame(rows), missing_symbols=[])

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_daily_bars_alpaca",
        fake_fetch_daily_bars_alpaca,
    )
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    report_path = run_market_fetch(
        out_dir=tmp_path,
        scope="all",
        stock_symbols=["AAPL"],
        start=dt.date(2015, 1, 1),
        end=dt.date(2015, 1, 5),
        adjustment="raw",
        alpaca_feed="iex",
        vix_symbol="VIXY",
        allow_vix_proxy=False,
        verbose=False,
    )

    report = json.loads(report_path.read_text())
    assert report["scope"] == "all"
    assert report["counts"]["symbols_requested_for_alpaca"] == len(captured["symbols"])
    assert report["vix"]["source"] == "alpaca"
    assert report["vix"]["symbol"] == "VIXY"
    assert report["paths"]["event_calendar_template"] == str(
        tmp_path / "event_calendar" / "events.template.yaml"
    )


def test_run_market_fetch_records_alpaca_payload_in_sqlite(
    monkeypatch, tmp_path: Path
) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    def fake_fetch_daily_bars_alpaca(
        *,
        symbols: list[str],
        start_date: dt.date,
        end_date: dt.date,
        adjustment: str = "raw",
        feed: str | None = None,
        batch_size: int = 100,
        verbose: bool = False,
    ) -> DailyBarsFetchResult:
        rows = [
            {
                "date": dt.date(2015, 1, 2),
                "symbol": symbol,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "adjusted_close": 10.5,
            }
            for symbol in symbols
        ]
        return DailyBarsFetchResult(df=pd.DataFrame(rows), missing_symbols=["ZZZZ"])

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_daily_bars_alpaca",
        fake_fetch_daily_bars_alpaca,
    )
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    report_path = run_market_fetch(
        out_dir=tmp_path,
        scope="v2",
        stock_symbols=[],
        start=dt.date(2015, 1, 1),
        end=dt.date(2015, 1, 5),
        adjustment="raw",
        alpaca_feed="iex",
        vix_symbol="VIXY",
        allow_vix_proxy=True,
        verbose=False,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifact_count = conn.execute("SELECT count(*) FROM artifacts").fetchone()[0]
        artifact_source = conn.execute(
            "SELECT source_name, artifact_kind FROM artifacts"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

    assert fetch_runs == [("market", "ok")]
    assert artifact_count == 1
    assert artifact_source == [("alpaca:daily_bars", "json")]
    assert outputs == [
        ("alpaca_daily_ohlcv_parquet",),
        ("alpaca_market_fetch_report",),
    ]


def test_run_market_fetch_merges_incremental_daily_ohlcv(
    monkeypatch, tmp_path: Path
) -> None:
    existing_dir = tmp_path / "daily_ohlcv"
    existing_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": dt.date(2026, 5, 14),
                "symbol": "TLT",
                "open": 90.0,
                "high": 91.0,
                "low": 89.0,
                "close": 90.5,
                "volume": 1000,
                "adjusted_close": 90.5,
            },
            {
                "date": dt.date(2026, 5, 15),
                "symbol": "TLT",
                "open": 91.0,
                "high": 92.0,
                "low": 90.0,
                "close": 91.5,
                "volume": 1000,
                "adjusted_close": 91.5,
            },
        ]
    ).to_parquet(existing_dir, index=False, partition_cols=["symbol"])

    def fake_fetch_daily_bars_alpaca(**kwargs) -> DailyBarsFetchResult:
        return DailyBarsFetchResult(
            df=pd.DataFrame(
                [
                    {
                        "date": dt.date(2026, 5, 15),
                        "symbol": "TLT",
                        "open": 91.1,
                        "high": 92.1,
                        "low": 90.1,
                        "close": 91.7,
                        "volume": 1100,
                        "adjusted_close": 91.7,
                    },
                    {
                        "date": dt.date(2026, 5, 16),
                        "symbol": "TLT",
                        "open": 92.0,
                        "high": 93.0,
                        "low": 91.0,
                        "close": 92.5,
                        "volume": 1200,
                        "adjusted_close": 92.5,
                    },
                ]
            ),
            missing_symbols=[],
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_daily_bars_alpaca",
        fake_fetch_daily_bars_alpaca,
    )
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    run_market_fetch(
        out_dir=tmp_path,
        scope="v2",
        stock_symbols=[],
        start=dt.date(2026, 5, 15),
        end=dt.date(2026, 5, 16),
        adjustment="raw",
        alpaca_feed="iex",
        vix_symbol="VIXY",
        allow_vix_proxy=True,
        verbose=False,
    )

    merged = pd.read_parquet(existing_dir).sort_values(["symbol", "date"])
    assert merged[merged["symbol"] == "TLT"][["date", "close"]].to_dict(
        orient="records"
    ) == [
        {"date": dt.date(2026, 5, 14), "close": 90.5},
        {"date": dt.date(2026, 5, 15), "close": 91.7},
        {"date": dt.date(2026, 5, 16), "close": 92.5},
    ]


def test_extract_ism_pmi_value_and_release_timestamp() -> None:
    html = """
    <html><body>
    <h1>April 2026 ISM Manufacturing PMI Report</h1>
    <p>Manufacturing PMI at 52.7%</p>
    </body></html>
    """
    value = extract_ism_pmi_value(html, label="Manufacturing PMI")
    ts = release_timestamp_for(year=2026, month=4, business_day_index=1)

    assert value == 52.7
    assert ts.isoformat() == "2026-04-01T10:00:00-04:00"


def test_run_macro_fetch_writes_macro_and_vintage_reports(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2015-01-01",
                        "value": "1.23",
                        "realtime_start": realtime_start or "2015-01-01",
                        "realtime_end": realtime_end or "2015-01-01",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    report_path = run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2015, 1, 1),
        end=dt.date(2015, 1, 31),
        fred_api_key=None,
        include_cpi_vintages=True,
    )

    report = json.loads(report_path.read_text())
    assert set(V2_FRED_SERIES).issubset(report["series"])
    assert report["series"]["broad_usd_index"]["series_id"] == "DTWEXBGS"
    assert report["series"]["iorb"]["series_id"] == "IORB"
    assert report["series"]["fedfunds"]["series_id"] == "DFF"
    assert report["series"]["ioer_legacy"]["series_id"] == "IOER"
    assert (tmp_path / "macro" / "fred_macro_series.parquet").exists()
    assert (tmp_path / "macro_vintages" / "cpi_all_items_vintages.parquet").exists()


def test_run_macro_fetch_merges_incremental_fred_rows(
    monkeypatch, tmp_path: Path
) -> None:
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-14"),
                "value": 4.10,
                "series_id": "BAMLH0A0HYM2",
                "realtime_start": "2026-05-14",
                "realtime_end": "2026-05-14",
                "logical_name": "hy_oas",
            }
        ]
    ).to_parquet(macro_dir / "fred_macro_series.parquet", index=False)

    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        value = "4.25" if series_id == "BAMLH0A0HYM2" else "1.00"
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2026-05-15",
                        "value": value,
                        "realtime_start": realtime_start or "2026-05-15",
                        "realtime_end": realtime_end or "2026-05-15",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2026, 5, 15),
        end=dt.date(2026, 5, 15),
        fred_api_key=None,
        include_cpi_vintages=False,
    )

    merged = pd.read_parquet(macro_dir / "fred_macro_series.parquet")
    hy = merged[merged["logical_name"] == "hy_oas"].sort_values("date")
    assert hy[["date", "value"]].to_dict(orient="records") == [
        {"date": dt.date(2026, 5, 14), "value": 4.10},
        {"date": dt.date(2026, 5, 15), "value": 4.25},
    ]


def test_run_macro_fetch_drops_fred_null_observations_at_canonical_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2026-05-14",
                        "value": ".",
                        "realtime_start": "2026-05-14",
                        "realtime_end": "2026-05-14",
                    },
                    {
                        "date": "2026-05-15",
                        "value": "4.25",
                        "realtime_start": "2026-05-15",
                        "realtime_end": "2026-05-15",
                    },
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    report_path = run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2026, 5, 14),
        end=dt.date(2026, 5, 15),
        fred_api_key=None,
        include_cpi_vintages=False,
    )

    macro = pd.read_parquet(tmp_path / "macro" / "fred_macro_series.parquet")
    report = json.loads(report_path.read_text())
    assert not macro["value"].isna().any()
    assert report["quality"]["dropped_null_observations"] == len(V2_FRED_SERIES)


def test_run_macro_fetch_preserves_existing_vintages_when_incremental_window_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    vintages_dir = tmp_path / "macro_vintages"
    vintages_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-04-01"),
                "value": 300.0,
                "series_id": "CPIAUCSL",
                "realtime_start": "2026-04-10",
                "realtime_end": "2026-04-10",
                "logical_name": "cpi_all_items_vintages",
            }
        ]
    ).to_parquet(vintages_dir / "cpi_all_items_vintages.parquet", index=False)

    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        if realtime_start is not None:
            return json.dumps({"observations": []})
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2026-05-15",
                        "value": "1.00",
                        "realtime_start": "2026-05-15",
                        "realtime_end": "2026-05-15",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2026, 5, 15),
        end=dt.date(2026, 5, 15),
        fred_api_key=None,
        include_cpi_vintages=True,
    )

    preserved = pd.read_parquet(vintages_dir / "cpi_all_items_vintages.parquet")
    assert preserved[["date", "value"]].to_dict(orient="records") == [
        {"date": dt.date(2026, 4, 1), "value": 300.0}
    ]


def test_run_macro_fetch_cleans_nan_vintages_from_existing_parquet(
    monkeypatch, tmp_path: Path
) -> None:
    """NaN rows left in existing vintages parquet by a prior fetch (e.g. BLS
    government shutdown gaps) must be dropped after merge so the vintages
    parquet never persists NaN values across incremental fetches."""
    vintages_dir = tmp_path / "macro_vintages"
    vintages_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-09-01"),
                "value": 324.245,
                "series_id": "CPIAUCSL",
                "realtime_start": "2025-10-24",
                "realtime_end": "9999-12-31",
                "logical_name": "cpi_all_items_vintages",
            },
            {
                "date": pd.Timestamp("2025-10-01"),
                "value": float("nan"),
                "series_id": "CPIAUCSL",
                "realtime_start": "2025-12-18",
                "realtime_end": "9999-12-31",
                "logical_name": "cpi_all_items_vintages",
            },
            {
                "date": pd.Timestamp("2025-11-01"),
                "value": 325.063,
                "series_id": "CPIAUCSL",
                "realtime_start": "2025-12-18",
                "realtime_end": "9999-12-31",
                "logical_name": "cpi_all_items_vintages",
            },
        ]
    ).to_parquet(vintages_dir / "cpi_all_items_vintages.parquet", index=False)

    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        if realtime_start is not None:
            return json.dumps({"observations": []})
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2026-05-15",
                        "value": "1.00",
                        "realtime_start": "2026-05-15",
                        "realtime_end": "2026-05-15",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2026, 5, 15),
        end=dt.date(2026, 5, 15),
        fred_api_key=None,
        include_cpi_vintages=True,
    )

    result = pd.read_parquet(vintages_dir / "cpi_all_items_vintages.parquet")
    assert result["value"].isna().sum() == 0, (
        f"Expected 0 NaN in vintages after merge, got {result['value'].isna().sum()}"
    )
    assert len(result) == 2
    assert set(result["date"].astype(str)) == {"2025-09-01", "2025-11-01"}


def test_run_macro_fetch_records_raw_fred_json_in_sqlite(
    monkeypatch, tmp_path: Path
) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2015-01-01",
                        "value": "1.23",
                        "realtime_start": realtime_start or "2015-01-01",
                        "realtime_end": realtime_end or "2015-01-01",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    report_path = run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2015, 1, 1),
        end=dt.date(2015, 1, 31),
        fred_api_key=None,
        include_cpi_vintages=True,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifact_count = conn.execute("SELECT count(*) FROM artifacts").fetchone()[0]
        derived_outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

    assert fetch_runs == [("macro", "ok")]
    assert artifact_count == len(V2_FRED_SERIES) + 1
    assert derived_outputs == [
        ("fred_macro_parquet",),
        ("fred_cpi_vintages_parquet",),
        ("fred_macro_report",),
    ]


def test_run_macro_fetch_requires_fred_api_key(tmp_path: Path) -> None:
    try:
        run_macro_fetch(
            out_dir=tmp_path,
            start=dt.date(2015, 1, 1),
            end=dt.date(2015, 1, 31),
            fred_api_key=None,
            include_cpi_vintages=False,
        )
    except SystemExit as exc:
        assert (
            str(exc)
            == "Missing required FRED API key: pass --fred-api-key or set FRED_API_KEY"
        )
    else:
        raise AssertionError("Expected SystemExit when FRED API key is missing")


def test_run_macro_fetch_uses_env_fred_api_key(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_fred_series_json(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        max_retries: int = 4,
        base_sleep_sec: float = 2.0,
    ) -> str:
        captured.setdefault("api_keys", []).append(api_key)
        return json.dumps(
            {
                "observations": [
                    {
                        "date": "2015-01-01",
                        "value": "1.23",
                        "realtime_start": realtime_start or "2015-01-01",
                        "realtime_end": realtime_end or "2015-01-01",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "regime_data_fetch.fetch_workflow.fetch_fred_series_json",
        fake_fetch_fred_series_json,
    )
    monkeypatch.setenv("FRED_API_KEY", "env-key")

    run_macro_fetch(
        out_dir=tmp_path,
        start=dt.date(2015, 1, 1),
        end=dt.date(2015, 1, 31),
        fred_api_key=None,
        include_cpi_vintages=False,
    )

    assert captured["api_keys"]
    assert set(captured["api_keys"]) == {"env-key"}
