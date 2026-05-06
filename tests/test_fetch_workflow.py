from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

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
    v1 = build_market_symbols(scope="v1", stock_symbols=["AAPL", "MSFT"], vix_symbol="VIX")
    assert v1[:4] == ["AAPL", "MSFT", "SPY", "RSP"]
    assert v1[-1] == "VIX"

    v2 = build_market_symbols(scope="v2", stock_symbols=["AAPL"], vix_symbol="VIXY")
    assert v2[:3] == ["SPY", "RSP", "KRE"]
    assert "VIXY" in v2
    assert set(V2_SECTOR_SYMBOLS).issubset(v2)
    assert set(V2_CROSS_ASSET_SYMBOLS).issubset(v2)

    combined = build_market_symbols(scope="all", stock_symbols=["AAPL", "AAPL"], vix_symbol="VIX")
    assert combined.count("AAPL") == 1
    assert combined.count("SPY") == 1
    assert combined.count("VIX") == 1
    assert set(V2_V1_SHARED_ANCHORS).issubset(combined)


def test_write_event_calendar_template_includes_v1_and_v2_examples(tmp_path: Path) -> None:
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

    monkeypatch.setattr("regime_data_fetch.fetch_workflow.fetch_daily_bars_alpaca", fake_fetch_daily_bars_alpaca)
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
    assert report["paths"]["event_calendar_template"] == str(tmp_path / "event_calendar" / "events.template.yaml")


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


def test_run_macro_fetch_writes_macro_and_vintage_reports(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch_fred_series(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": dt.date(2015, 1, 1),
                    "series_id": series_id,
                    "value": 1.23,
                    "realtime_start": realtime_start,
                    "realtime_end": realtime_end,
                }
            ]
        )

    monkeypatch.setattr("regime_data_fetch.fetch_workflow.fetch_fred_series", fake_fetch_fred_series)
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
        assert str(exc) == "Missing required FRED API key: pass --fred-api-key or set FRED_API_KEY"
    else:
        raise AssertionError("Expected SystemExit when FRED API key is missing")


def test_run_macro_fetch_uses_env_fred_api_key(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_fred_series(
        *,
        series_id: str,
        start_date: dt.date,
        end_date: dt.date,
        api_key: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        captured.setdefault("api_keys", []).append(api_key)
        return pd.DataFrame(
            [
                {
                    "date": dt.date(2015, 1, 1),
                    "series_id": series_id,
                    "value": 1.23,
                    "realtime_start": realtime_start,
                    "realtime_end": realtime_end,
                }
            ]
        )

    monkeypatch.setattr("regime_data_fetch.fetch_workflow.fetch_fred_series", fake_fetch_fred_series)
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
