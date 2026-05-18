from __future__ import annotations

import datetime as dt
import argparse
import json
from pathlib import Path

import pytest

from regime_data_fetch.bls_schedule import build_bls_local_archive_page_fetcher
from regime_data_fetch.fetch_workflow import (
    V2_FRED_SERIES,
)
from regime_data_fetch.universe import FIXED_UNIVERSE_SYMBOL_COUNT
from scripts import fetch_regime_engine_v1_data as fetch_script
from scripts.fetch_regime_engine_v1_data import (
    FETCH_MODE_REGISTRY,
    OPERATOR_ASSISTED_FETCH_MODES,
    UNATTENDED_FETCH_MODES,
    FetchModeSpec,
    _invoke_unattended_fetch_mode,
    _plan_fetch_mode_execution,
    _should_fetch,
)


def test_fetch_help_surface_mentions_pmi_and_pit(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["fetch_regime_engine_v1_data.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        fetch_script.main()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
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
    assert (
        "--fetch all is reserved for unattended autonomous refreshes" in normalized_help
    )
    assert "operator-assisted" in help_text.lower()


def test_unattended_usd_ingestion_uses_fred_macro_not_local_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_macro(**kwargs):
        captured.update(kwargs)
        report = tmp_path / "macro.json"
        report.write_text(json.dumps({"paths": {}}))
        return report

    monkeypatch.setattr(fetch_script, "run_macro_fetch", fake_macro)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_regime_engine_v1_data.py",
            "--fetch",
            "macro",
            "--scope",
            "v2",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--start",
            "2026-05-01",
            "--end",
            "2026-05-02",
        ],
    )

    assert fetch_script.main() == 0
    assert V2_FRED_SERIES["broad_usd_index"] == "DTWEXBGS"
    assert captured["start"] == dt.date(2026, 5, 1)
    assert captured["end"] == dt.date(2026, 5, 2)
    assert not _should_fetch("all", "usd-index-local")


def test_fetch_all_excludes_manual_eps_and_wayback_backfill() -> None:
    assert not _should_fetch("all", "eps")
    assert not _should_fetch("all", "eps-wayback")
    assert not _should_fetch("all", "eps-spglobal-auto")


def test_fetch_all_excludes_operator_assisted_browser_and_archive_paths() -> None:
    for fetch_name in [
        "investing-live",
        "investing-archive-local",
        "daily-ohlcv-local-sqlite",
        "usd-index-local",
    ]:
        assert not _should_fetch("all", fetch_name)


def test_fetch_all_uses_live_constituent_ohlcv_not_local_sqlite_import() -> None:
    assert _should_fetch("all", "daily-ohlcv-constituents-alpaca")
    assert not _should_fetch("all", "daily-ohlcv-local-sqlite")


def test_fetch_all_uses_live_pmi_by_default_not_manual_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_pmi(**kwargs):
        captured.update(kwargs)
        report = tmp_path / "pmi.json"
        report.write_text(json.dumps({"paths": {}}))
        return report

    monkeypatch.setattr(fetch_script, "run_pmi_fetch", fake_pmi)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_regime_engine_v1_data.py",
            "--fetch",
            "pmi",
            "--scope",
            "v2",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--end",
            "2026-05-02",
        ],
    )

    assert fetch_script.main() == 0
    assert captured["as_of_date"] == dt.date(2026, 5, 2)
    assert captured["manual_history_dir"] is None


def test_constituent_ohlcv_requires_fixed_universe_unless_pit_bootstrap_is_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_alpaca_ohlcv(**kwargs):
        captured.update(kwargs)
        report = tmp_path / "daily_ohlcv.json"
        report.write_text(json.dumps({"paths": {}}))
        return report

    monkeypatch.setattr(
        fetch_script,
        "run_alpaca_constituent_daily_ohlcv_fetch",
        fake_alpaca_ohlcv,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_regime_engine_v1_data.py",
            "--fetch",
            "daily-ohlcv-constituents-alpaca",
            "--scope",
            "v2",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--acquisition-db",
            str(tmp_path / "data" / "raw" / "acquisition.db"),
            "--start",
            "2026-05-01",
            "--end",
            "2026-05-02",
        ],
    )

    assert fetch_script.main() == 0
    assert FIXED_UNIVERSE_SYMBOL_COUNT == 762
    assert captured["fixed_universe_symbols"] is None
    assert captured["fixed_universe_dir"] is None
    assert captured["allow_pit_universe"] is False
    assert captured["expected_universe_count"] == FIXED_UNIVERSE_SYMBOL_COUNT


def test_fetch_mode_sets_make_operator_assisted_boundary_explicit() -> None:
    assert UNATTENDED_FETCH_MODES.isdisjoint(OPERATOR_ASSISTED_FETCH_MODES)
    assert (
        set(FETCH_MODE_REGISTRY)
        == UNATTENDED_FETCH_MODES | OPERATOR_ASSISTED_FETCH_MODES
    )
    assert all(spec.name == name for name, spec in FETCH_MODE_REGISTRY.items())
    assert {
        spec.name
        for spec in FETCH_MODE_REGISTRY.values()
        if spec.category == "unattended"
    } == UNATTENDED_FETCH_MODES
    assert {
        spec.name
        for spec in FETCH_MODE_REGISTRY.values()
        if spec.category == "operator-assisted"
    } == OPERATOR_ASSISTED_FETCH_MODES
    for mode in OPERATOR_ASSISTED_FETCH_MODES:
        assert not _should_fetch("all", mode), mode
    for mode in UNATTENDED_FETCH_MODES:
        assert _should_fetch("all", mode), mode


def test_fetch_all_execution_plan_is_serial_by_default() -> None:
    plan = _plan_fetch_mode_execution("all", conservative_concurrency=False)

    registry_unattended_order = [
        name
        for name, spec in FETCH_MODE_REGISTRY.items()
        if spec.category == "unattended"
    ]
    assert [group.modes for group in plan] == [
        (mode,) for mode in registry_unattended_order
    ]
    assert not any(group.concurrent for group in plan)


def test_fetch_all_conservative_concurrency_batches_only_safe_unattended_modes() -> (
    None
):
    plan = _plan_fetch_mode_execution("all", conservative_concurrency=True)

    assert plan[0].modes == ("market",)
    assert plan[0].concurrent is False
    assert plan[-1].modes == ("daily-ohlcv-constituents-alpaca",)
    assert plan[-1].concurrent is False
    concurrent_groups = [group for group in plan if group.concurrent]
    assert len(concurrent_groups) == 1
    assert set(concurrent_groups[0].modes).issubset(UNATTENDED_FETCH_MODES)
    assert set(concurrent_groups[0].modes).isdisjoint(OPERATOR_ASSISTED_FETCH_MODES)
    assert "market" not in concurrent_groups[0].modes
    assert "daily-ohlcv-constituents-alpaca" not in concurrent_groups[0].modes


def test_unattended_fetch_dispatch_uses_registry_invoker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called: dict[str, object] = {}

    def fake_invoke(context: fetch_script.FetchModeInvocation) -> Path:
        called["mode"] = "test-registry-mode"
        called["out_dir"] = context.out_dir
        path = tmp_path / "registry-mode-report.json"
        path.write_text(json.dumps({"paths": {}}))
        return path

    monkeypatch.setitem(
        fetch_script.FETCH_MODE_REGISTRY,
        "test-registry-mode",
        FetchModeSpec("test-registry-mode", "unattended", invoke=fake_invoke),
    )

    assert not hasattr(fetch_script, "_run_unattended_fetch_mode")

    report_path = _invoke_unattended_fetch_mode(
        "test-registry-mode",
        args=argparse.Namespace(fetch="test-registry-mode"),
        out_dir=tmp_path,
        start=dt.date(2026, 5, 1),
        end=dt.date(2026, 5, 2),
        acquisition_db_path=None,
        acquisition_artifact_store_root=None,
    )

    assert report_path == tmp_path / "registry-mode-report.json"
    assert called == {"mode": "test-registry-mode", "out_dir": tmp_path}


def test_fetch_all_dispatches_only_unattended_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called: list[str] = []
    kwargs_by_mode: dict[str, dict[str, object]] = {}

    def report_for(name: str):
        def _fake(**kwargs):
            called.append(name)
            kwargs_by_mode[name] = kwargs
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
            raise AssertionError(
                f"operator-assisted fetch was called by --fetch all: {name}"
            )

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
    daily_kwargs = kwargs_by_mode["daily-ohlcv-constituents-alpaca"]
    assert daily_kwargs["fixed_universe_symbols"] is None
    assert daily_kwargs["fixed_universe_dir"] is None
    assert daily_kwargs["allow_pit_universe"] is False
    assert daily_kwargs["expected_universe_count"] == FIXED_UNIVERSE_SYMBOL_COUNT
    assert kwargs_by_mode["events"]["include_v2_curated_candidates"] is True


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


def test_emit_manifest_without_path_uses_tracked_immutable_run_manifest(
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
        artifacts = [object()]

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
            "--end",
            "2026-05-17",
            "--out-dir",
            str(tmp_path / "data" / "raw"),
            "--emit-manifest",
            "--artifact-store",
            str(tmp_path / "store"),
        ],
    )

    assert fetch_script.main() == 0
    assert captured["manifest_path"] == (
        fetch_script.REPO_ROOT / "manifests" / "runs" / "regime_engine_2026-05-17.yaml"
    )


def test_emit_manifest_rejects_ignored_data_manifest_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            "data/manifests/regime_engine_latest.yaml",
            "--artifact-store",
            str(tmp_path / "store"),
        ],
    )

    with pytest.raises(
        SystemExit, match="manifest lockfiles must be written outside ignored data/"
    ):
        fetch_script.main()


def test_tracked_manifest_rejects_context_artifact_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            "--artifact-store",
            ".context/regime-artifact-store",
        ],
    )

    with pytest.raises(
        SystemExit, match="tracked manifests require durable artifact storage"
    ):
        fetch_script.main()


def test_event_calendar_fetch_symbol_is_wired() -> None:
    assert "events" in UNATTENDED_FETCH_MODES
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


def test_fetch_help_surface_mentions_acquisition_db_and_bls_schedule_dir(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.argv", ["fetch_regime_engine_v1_data.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        fetch_script.main()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--acquisition-db" in help_text
    assert "--bls-schedule-dir" in help_text
    assert "--bls-start-year" in help_text
    assert "--bls-end-year" in help_text
    assert "manifests/runs" in help_text
