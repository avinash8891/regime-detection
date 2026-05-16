from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd

from regime_data_fetch.alpaca_daily import DailyBarsFetchResult
from regime_data_fetch.bls_schedule import build_bls_local_archive_page_fetcher
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
from regime_data_fetch.universe import FIXED_UNIVERSE_SYMBOL_COUNT
from scripts import fetch_regime_engine_v1_data as fetch_script
from scripts.fetch_regime_engine_v1_data import (
    FETCH_MODES,
    OPERATOR_ASSISTED_FETCH_MODES,
    UNATTENDED_FETCH_MODES,
    _should_fetch,
)


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
    assert "monthly_options_expiry" in contents
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


def test_fetch_help_surface_mentions_pmi_and_pit() -> None:
    help_text = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    for mode in FETCH_MODES:
        assert mode in help_text
    assert "--eps-workbook" in help_text
    assert "--eps-wayback-max-snapshots" in help_text
    assert "--eps-wayback-from" in help_text
    assert "--eps-wayback-to" in help_text
    assert "--eps-wayback-stop-after-first-success" in help_text
    assert "--eps-browser-user-data-dir" in help_text
    assert "--eps-browser-executable" in help_text
    assert "--eps-browser-headless" in help_text
    assert "--eps-browser-timeout-ms" in help_text
    assert "--usd-index-csv" in help_text
    assert "--daily-ohlcv-dir" in help_text
    assert "--pit-parquet" in help_text
    assert "--allow-missing-constituent-symbols" in help_text
    assert "--pmi-history-dir" in help_text
    assert "--investing-archive-root" in help_text
    assert "--investing-earnings-loaded-page" in help_text
    assert "--investing-earnings-browser-capture" in help_text
    assert "--investing-browser-user-data-dir" in help_text
    assert "--investing-browser-executable" in help_text
    assert "--investing-browser-headless" in help_text
    assert "--investing-browser-timeout-ms" in help_text
    assert "investing-live" in help_text
    assert "cleveland-fed-nowcast" in help_text
    assert "sf-fed-news-sentiment" in help_text
    assert "--fetch all is reserved for unattended autonomous refreshes" in help_text
    assert "operator-assisted" in help_text.lower()


def test_unattended_usd_ingestion_uses_fred_macro_not_local_csv() -> None:
    runner_text = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    workflow_text = Path("src/regime_data_fetch/fetch_workflow.py").read_text()

    assert '"broad_usd_index": "DTWEXBGS"' in workflow_text
    assert (
        "Routine future USD ingestion uses FRED DTWEXBGS through --fetch macro."
        in runner_text
    )
    assert 'if args.fetch == "usd-index-local":' in runner_text
    assert not _should_fetch("all", "usd-index-local")


def test_fetch_all_excludes_manual_eps_and_wayback_backfill() -> None:
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert 'if args.fetch == "eps":' in script
    assert 'if args.fetch == "eps-wayback":' in script
    assert 'if args.fetch == "eps-spglobal-auto":' in script
    assert not _should_fetch("all", "eps")
    assert not _should_fetch("all", "eps-wayback")
    assert not _should_fetch("all", "eps-spglobal-auto")


def test_fetch_all_excludes_operator_assisted_browser_and_archive_paths() -> None:
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    for fetch_name in [
        "investing-live",
        "investing-archive-local",
        "daily-ohlcv-local-sqlite",
        "usd-index-local",
    ]:
        assert f'if args.fetch == "{fetch_name}":' in script
        assert not _should_fetch("all", fetch_name)


def test_fetch_all_uses_live_constituent_ohlcv_not_local_sqlite_import() -> None:
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert "daily-ohlcv-constituents-alpaca" in script
    assert _should_fetch("all", "daily-ohlcv-constituents-alpaca")
    assert 'if args.fetch == "daily-ohlcv-local-sqlite":' in script
    assert not _should_fetch("all", "daily-ohlcv-local-sqlite")


def test_fetch_all_uses_live_pmi_by_default_not_manual_history() -> None:
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert "--pmi-history-dir" in script
    assert (
        "manual_history_dir=Path(args.pmi_history_dir) if args.pmi_history_dir else None"
        in script
    )
    assert "manual_history_dir=DEFAULT_MANUAL_PMI_HISTORY_DIR" not in script


def test_constituent_ohlcv_requires_fixed_universe_unless_pit_bootstrap_is_explicit() -> (
    None
):
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert "--constituent-universe-dir" in script
    assert "--allow-pit-constituent-universe" in script
    assert "--constituent-universe-expected-count" in script
    assert "load_symbols_from_pit_constituents_parquet" in script
    assert "FIXED_UNIVERSE_SYMBOL_COUNT" in script
    assert FIXED_UNIVERSE_SYMBOL_COUNT == 762
    assert (
        "fixed_universe_symbols=_load_json_symbol_list(Path(args.universe_json)) if args.universe_json else None"
        in script
    )
    assert "allow_pit_universe=args.allow_pit_constituent_universe" in script
    assert "fetch_alpaca_active_stock_symbols" not in script


def test_fetch_mode_sets_make_operator_assisted_boundary_explicit() -> None:
    assert UNATTENDED_FETCH_MODES.isdisjoint(OPERATOR_ASSISTED_FETCH_MODES)
    for mode in OPERATOR_ASSISTED_FETCH_MODES:
        assert not _should_fetch("all", mode), mode
    for mode in UNATTENDED_FETCH_MODES:
        assert _should_fetch("all", mode), mode


def test_fetch_all_dispatches_only_unattended_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called: list[str] = []

    def report_for(name: str):
        def _fake(**kwargs):
            del kwargs
            called.append(name)
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps({"paths": {}}))
            return path

        return _fake

    unattended_callables = {
        "market": "run_market_fetch",
        "macro": "run_macro_fetch",
        "sentiment": "run_sentiment_fetch",
        "events": "run_us_event_calendar_fetch",
        "pmi": "run_pmi_fetch",
        "pit": "run_pit_constituents_fetch",
        "fomc": "run_fomc_minutes_fetch",
        "powell": "run_powell_speeches_fetch",
        "cleveland-fed-nowcast": "run_cleveland_fed_nowcast_fetch",
        "sf-fed-news-sentiment": "run_sf_fed_news_sentiment_fetch",
        "daily-ohlcv-constituents-alpaca": "run_alpaca_constituent_daily_ohlcv_fetch",
    }
    for mode, attr in unattended_callables.items():
        monkeypatch.setattr(fetch_script, attr, report_for(mode))

    def operator_called(name: str):
        def _fake(**kwargs):
            del kwargs
            raise AssertionError(f"operator-assisted fetch was called by --fetch all: {name}")

        return _fake

    operator_callables = {
        "eps": "run_aggregate_eps_fetch",
        "eps-spglobal-auto": "run_aggregate_eps_auto_fetch",
        "eps-wayback": "run_wayback_aggregate_eps_fetch",
        "usd-index-local": "run_local_usd_index_import",
        "daily-ohlcv-local-sqlite": "run_local_daily_ohlcv_sqlite_import",
        "investing-archive-local": "run_local_investing_archive_import",
        "investing-live": "run_investing_live_fetch",
    }
    for mode, attr in operator_callables.items():
        monkeypatch.setattr(fetch_script, attr, operator_called(mode))

    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_regime_engine_v1_data.py",
            "--fetch",
            "all",
            "--scope",
            "v2",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--acquisition-db",
            str(tmp_path / "data" / "raw" / "acquisition.db"),
        ],
    )

    assert fetch_script.main() == 0
    assert set(called) == set(UNATTENDED_FETCH_MODES)
    assert set(called).isdisjoint(OPERATOR_ASSISTED_FETCH_MODES)


def test_emit_manifest_uses_all_runner_use_cases_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_market(**kwargs):
        del kwargs
        report = tmp_path / "market_report.json"
        report.write_text(json.dumps({"paths": {}}))
        return report

    class FakeManifest:
        artifacts = [object(), object()]

    def fake_emit_manifest_for_report_paths(**kwargs):
        captured.update(kwargs)
        return FakeManifest()

    monkeypatch.setattr(fetch_script, "run_market_fetch", fake_market)
    monkeypatch.setattr(
        fetch_script,
        "emit_manifest_for_report_paths",
        fake_emit_manifest_for_report_paths,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_regime_engine_v1_data.py",
            "--fetch",
            "market",
            "--scope",
            "v2",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--emit-manifest",
            str(tmp_path / "manifest.yaml"),
            "--artifact-store",
            str(tmp_path / "store"),
        ],
    )

    assert fetch_script.main() == 0
    assert captured["required_for"] == [
        "profile_engine_30d",
        "v2_calibration",
        "historical_walkforward",
        "audit_layer2_30d",
    ]


def test_event_calendar_fetch_symbol_is_wired() -> None:
    script = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert "run_us_event_calendar_fetch" in script
    assert _should_fetch("all", "events")


def test_build_bls_local_archive_page_fetcher_prefers_local_file(
    tmp_path: Path,
) -> None:
    schedule_dir = tmp_path / "bls"
    schedule_dir.mkdir()
    local_file = schedule_dir / "bls_schedule_2024.html"
    local_file.write_text("Consumer Price Index for March 2024")

    calls: list[str] = []

    def fake_fallback(url: str) -> str:
        calls.append(url)
        return "fallback"

    fetcher = build_bls_local_archive_page_fetcher(
        schedule_dir=schedule_dir,
        fallback_page_fetcher=fake_fallback,
    )

    html = fetcher("https://www.bls.gov/schedule/2024/")

    assert html == "Consumer Price Index for March 2024"
    assert calls == []


def test_fetch_help_surface_mentions_acquisition_db_and_bls_schedule_dir() -> None:
    help_text = Path("scripts/fetch_regime_engine_v1_data.py").read_text()
    assert "--acquisition-db" in help_text
    assert "--bls-schedule-dir" in help_text
    assert "--bls-start-year" in help_text
    assert "--bls-end-year" in help_text
