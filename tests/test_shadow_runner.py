from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from contextlib import closing

import pandas as pd
import pytest
import yaml

from regime_detection.fragility_universe import SECTOR_ETFS


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_shadow_regime.py"
    spec = importlib.util.spec_from_file_location("run_shadow_regime", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _write_sector_pit_intervals(path: Path) -> Path:
    rows = [
        {"ticker": symbol, "start_date": "2019-01-02", "end_date": ""}
        for symbol in SECTOR_ETFS
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_shadow_test_config(tmp_path: Path) -> Path:
    """Copy core3-v2-fast.yaml into tmp_path, stripped of credit_funding and
    inflation_growth.

    The shadow runner has no parameter for loading macro_series or full
    cross_asset_closes from disk; the canonical fixture (`v2_daily_ohlcv.csv`)
    is OHLCV-only. When credit_funding and inflation_growth are configured,
    the ClassifyRequest validator (engine.py:259) refuses to run without
    macro keys (sofr/iorb/nfci/cpi_all_items/pmi_manufacturing) and extra
    cross_asset closes (KRE/XLY/XLI/XLP/XLU) that the runner can't supply.
    Stripping these two axes keeps the shadow-runner tests scoped to the
    artifact/ledger/duplicate-rejection behavior they actually exercise."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    config = yaml.safe_load(src.read_text())
    config.pop("credit_funding", None)
    config.pop("inflation_growth", None)
    dst = tmp_path / "shadow_test_config.yaml"
    dst.write_text(yaml.safe_dump(config))
    return dst


def _write_v2_macro_parquet(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(
        repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    )
    dates = pd.to_datetime(sorted(source["date"].unique()))
    trend = pd.Series(range(len(dates)), index=dates, dtype="float64")
    logical_series = {
        "2y_yield": 4.00 + trend * 0.0002,
        "10y_yield": 4.25 + trend * 0.0001,
        "broad_usd_index": 100.0 + trend * 0.01,
        "sofr": 5.25 + trend * 0.0001,
        "iorb": 5.40 + trend * 0.0001,
        "nfci": -0.35 + trend * 0.00001,
        "hy_oas": 3.8 + trend * 0.0001,
        "ig_bbb_oas": 1.6 + trend * 0.0001,
        "cpi_all_items": 300.0 + trend * 0.01,
        "pmi_manufacturing": 50.0 + trend * 0.0001,
    }
    rows = [
        {
            "date": observed_date,
            "series_id": logical_name.upper(),
            "logical_name": logical_name,
            "value": value,
        }
        for logical_name, series in logical_series.items()
        for observed_date, value in series.items()
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_shadow_runner_writes_expected_artifacts_and_ledger(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    market_data_path = v2_daily_path
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = _write_shadow_test_config(tmp_path)
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    mod = _load_runner_module()
    result = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
    )

    assert result["status"] == "success"
    assert result["as_of_date"] == "2026-05-13"

    db_path = out_root / "regime_shadow.db"
    assert db_path.exists()
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, output_path, input_archive_path FROM runs"
        ).fetchall()
        replay_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('replay_checks', 'incidents') ORDER BY name"
        ).fetchall()

    assert rows == [
        (
            "2026-05-13",
            "success",
            str(out_root / "outputs" / "2026-05-13.json"),
            str(out_root / "input_archives" / "2026-05-13"),
        )
    ]
    assert replay_tables == [("incidents",), ("replay_checks",)]

    output_path = out_root / "outputs" / "2026-05-13.json"
    payload = json.loads(output_path.read_text())
    assert payload["as_of_date"] == "2026-05-13"
    assert payload["engine_version"].startswith("regime-engine-v")
    from regime_detection.config import load_default_regime_config

    assert payload["config_version"] == load_default_regime_config().config_version

    archive_market = out_root / "input_archives" / "2026-05-13" / "market_data.parquet"
    archived_df = pd.read_parquet(archive_market)
    archived_df = archived_df.assign(date=pd.to_datetime(archived_df["date"]).dt.date)
    assert archived_df["date"].max() == date(2026, 5, 13)

    checksums_path = out_root / "input_archives" / "2026-05-13" / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    assert "market_data.parquet" in checksums
    assert "events.yaml" in checksums


def test_shadow_runner_supplies_macro_series_for_configured_v2_axes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    macro_path = _write_v2_macro_parquet(tmp_path / "fred_macro_series.parquet")
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    mod = _load_runner_module()
    result = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=v2_daily_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
        macro_parquet_path=macro_path,
    )

    assert result["status"] == "success"
    payload = json.loads((out_root / "outputs" / "2026-05-13.json").read_text())
    assert payload["credit_funding_state"] is not None
    assert payload["inflation_growth_state"] is not None


def test_shadow_runner_v1_only_run_does_not_load_v2_macro_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = (
        repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    )
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = (
        repo_root / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
    )

    mod = _load_runner_module()

    def _fail_if_called(**_kwargs):
        raise AssertionError("V1-only shadow run must not load V2 macro artifacts")

    monkeypatch.setattr(mod, "_load_v2_macro_series", _fail_if_called)

    result = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=tmp_path / "shadow_run",
        macro_parquet_path=tmp_path / "missing_v2_macro.parquet",
    )

    assert result["status"] == "success"


def test_shadow_runner_archives_inputs_and_inserts_in_progress_before_classify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    market_data_path = v2_daily_path
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = _write_shadow_test_config(tmp_path)
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    mod = _load_runner_module()
    real_classify = mod.RegimeEngine.classify

    def _checked_classify(self, *, as_of_date, market_data, event_calendar, **kwargs):
        archive_dir = out_root / "input_archives" / "2026-05-13"
        assert (archive_dir / "market_data.parquet").exists()
        assert (archive_dir / "events.yaml").exists()
        assert (archive_dir / "checksums.json").exists()
        with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
            row = conn.execute(
                "SELECT status, input_archive_path FROM runs WHERE as_of_date = ?",
                ("2026-05-13",),
            ).fetchone()
        assert row == ("in_progress", str(archive_dir))
        return real_classify(
            self,
            as_of_date=as_of_date,
            market_data=market_data,
            event_calendar=event_calendar,
            **kwargs,
        )

    monkeypatch.setattr(mod.RegimeEngine, "classify", _checked_classify)

    result = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
    )

    assert result["status"] == "success"


def test_shadow_runner_records_failures_without_silent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    market_data_path = v2_daily_path
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    mod = _load_runner_module()

    def _boom(self, **kwargs):
        raise RuntimeError("forced classify failure")

    monkeypatch.setattr(mod.RegimeEngine, "classify", _boom)

    result = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
    )

    assert result["status"] == "failure"
    assert "forced classify failure" in result["failure_reason"]

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, failure_reason FROM runs"
        ).fetchall()

    assert rows == [("2026-05-13", "failure", "forced classify failure")]
    assert (out_root / "input_archives" / "2026-05-13" / "market_data.parquet").exists()
    assert not (out_root / "outputs" / "2026-05-13.json").exists()


def test_shadow_runner_resets_qualification_on_mid_window_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F-018: two runs share the coarse config_version Literal but the config CONTENT
    # changes between them. The runner must record each config's content hash and,
    # when it differs from the window's prior hash, insert a breaking incident that
    # resets the qualification window. The incident is recorded before classify, so we
    # stub classify out to keep the test fast and content-change-focused.
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    config_a = _write_shadow_test_config(tmp_path)
    # Same config_version, different bytes: appended comment parses identically.
    config_b = tmp_path / "shadow_test_config_changed.yaml"
    config_b.write_text(config_a.read_text() + "\n# F-018 mid-window config change\n")

    mod = _load_runner_module()
    monkeypatch.setattr(
        mod.RegimeEngine,
        "classify",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("stubbed classify")),
    )

    common = dict(
        market_data_path=v2_daily_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
    )
    mod.run_shadow(as_of_date=date(2026, 5, 12), config_path=config_a, **common)
    mod.run_shadow(as_of_date=date(2026, 5, 13), config_path=config_b, **common)

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        run_hashes = conn.execute(
            "SELECT as_of_date, config_sha256 FROM runs ORDER BY as_of_date"
        ).fetchall()
        incidents = conn.execute(
            "SELECT incident_date, description, breaks_qualification FROM incidents"
        ).fetchall()

    hash_a = run_hashes[0][1]
    hash_b = run_hashes[1][1]
    assert hash_a is not None and hash_b is not None
    assert hash_a != hash_b  # content change produced distinct hashes
    assert len(incidents) == 1
    assert incidents[0][0] == "2026-05-13"
    assert "Config content changed mid-window" in incidents[0][1]
    assert incidents[0][2] == 1  # breaks_qualification


def test_shadow_runner_rejects_duplicate_versioned_run_and_non_trading_day(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    market_data_path = v2_daily_path
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = _write_shadow_test_config(tmp_path)
    out_root = tmp_path / "shadow_run"
    pit_path = _write_sector_pit_intervals(tmp_path / "pit.csv")

    mod = _load_runner_module()
    first = mod.run_shadow(
        as_of_date=date(2026, 5, 13),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        output_root=out_root,
        v2_daily_ohlcv_path=v2_daily_path,
        pit_constituent_intervals_path=pit_path,
    )
    assert first["status"] == "success"

    with pytest.raises(sqlite3.IntegrityError):
        mod.run_shadow(
            as_of_date=date(2026, 5, 13),
            market_data_path=market_data_path,
            event_calendar_path=event_calendar_path,
            config_path=config_path,
            output_root=out_root,
            v2_daily_ohlcv_path=v2_daily_path,
            pit_constituent_intervals_path=pit_path,
        )

    with pytest.raises(ValueError, match="NYSE trading day"):
        mod.run_shadow(
            as_of_date=date(2023, 12, 16),
            market_data_path=market_data_path,
            event_calendar_path=event_calendar_path,
            output_root=tmp_path / "weekend_shadow",
        )


def test_shadow_runner_requires_event_calendar_unless_cli_resolves_manifest_default(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    out_root = tmp_path / "shadow_run"
    mod = _load_runner_module()

    with pytest.raises(ValueError, match="event_calendar_path is required"):
        mod.run_shadow(
            as_of_date=date(2023, 12, 14),
            market_data_path=market_data_path,
            output_root=out_root,
        )
    assert not out_root.exists()


def test_shadow_runner_cli_defaults_event_calendar_to_manifest_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_runner_module()
    data_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shadow_regime.py",
            "--as-of-date",
            "2023-12-14",
            "--market-data",
            str(tmp_path / "market.parquet"),
            "--output-root",
            str(tmp_path / "out"),
            "--data-root",
            str(data_root),
        ],
    )

    args = mod._parse_args()

    assert args.event_calendar == data_root / "event_calendar" / "us_events.yaml"
