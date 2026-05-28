from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from regime_detection import comparison
from regime_detection.models import AxisOutput, DataQuality
from scripts import _v2_calibration_helpers
from scripts import run_v2_shadow_ab_gate, run_v2_walkforward_gate

pytestmark = [pytest.mark.slow, pytest.mark.v2_gate]


def test_gate_reporting_label_uses_granular_status() -> None:
    output = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert run_v2_walkforward_gate._reporting_label(output) == "no_rule_fired"
    assert run_v2_shadow_ab_gate._reporting_label(output) == "no_rule_fired"


def test_gate_scripts_use_comparison_reporting_label_source() -> None:
    assert (
        _v2_calibration_helpers.axis_reporting_label is comparison.axis_reporting_label
    )
    assert run_v2_walkforward_gate._reporting_label is comparison.axis_reporting_label
    assert run_v2_shadow_ab_gate._reporting_label is comparison.axis_reporting_label


def test_walkforward_gate_markdown_uses_comparison_gate_metric_names() -> None:
    markdown = run_v2_walkforward_gate._build_markdown(
        start_date=pd.Timestamp("2026-05-12").date(),
        end_date=pd.Timestamp("2026-05-12").date(),
        sessions=[pd.Timestamp("2026-05-12").date()],
        v1_metrics=run_v2_walkforward_gate._session_metrics_empty(),
        v2_metrics=run_v2_walkforward_gate._session_metrics_empty(),
        v2_axes=run_v2_walkforward_gate._axis_activation_empty(),
        v1_errors=0,
        v2_errors=0,
        engine_version="regime-engine-v-test",
    )

    for metric_name in comparison.V2_GATE_METRIC_NAMES:
        assert f"- {metric_name}" in markdown


def test_walkforward_gate_default_window_handles_partitioned_symbol_schemas(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily"
    spy_dir = daily_dir / "symbol=SPY"
    rsp_dir = daily_dir / "symbol=RSP"
    spy_dir.mkdir(parents=True)
    rsp_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-11", "2026-05-12"]),
            "symbol": pd.Series(["SPY", "SPY"], dtype="string"),
            "close": [100.0, 101.0],
        }
    ).to_parquet(spy_dir / "ohlcv.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-11", "2026-05-13"]),
            "symbol": pd.Series(["RSP", "RSP"], dtype="category"),
            "close": [99.0, 102.0],
        }
    ).to_parquet(rsp_dir / "ohlcv.parquet", index=False)

    start_date, end_date = run_v2_walkforward_gate._resolve_default_window(daily_dir)

    assert end_date == pd.Timestamp("2026-05-13").date()
    assert start_date == pd.Timestamp("2025-02-12").date()


@pytest.mark.parametrize(
    ("v1_errors", "v2_errors", "allow_session_errors", "expected"),
    [
        (0, 0, False, 0),
        (1, 0, False, 1),
        (0, 1, False, 1),
        (1, 1, True, 0),
    ],
)
def test_gate_scripts_fail_on_session_errors_unless_explicitly_allowed(
    v1_errors: int,
    v2_errors: int,
    allow_session_errors: bool,
    expected: int,
) -> None:
    assert (
        run_v2_walkforward_gate._session_error_exit_code(
            v1_errors=v1_errors,
            v2_errors=v2_errors,
            allow_session_errors=allow_session_errors,
        )
        == expected
    )
    assert (
        run_v2_shadow_ab_gate._session_error_exit_code(
            v1_errors=v1_errors,
            v2_errors=v2_errors,
            allow_session_errors=allow_session_errors,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("module", "argv_base"),
    [
        (
            run_v2_walkforward_gate,
            [
                "run_v2_walkforward_gate.py",
                "--daily-dir",
                "unused_daily.parquet",
                "--macro-parquet",
                "unused_macro.parquet",
                "--start-date",
                "2026-05-12",
                "--end-date",
                "2026-05-12",
            ],
        ),
        (
            run_v2_shadow_ab_gate,
            [
                "run_v2_shadow_ab_gate.py",
                "--daily-dir",
                "unused_daily.parquet",
                "--macro-parquet",
                "unused_macro.parquet",
            ],
        ),
    ],
)
def test_gate_scripts_parse_allow_session_errors_flag(
    module,
    argv_base: list[str],
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", argv_base)
    assert module._parse_args().allow_session_errors is False

    monkeypatch.setattr(sys, "argv", [*argv_base, "--allow-session-errors"])
    assert module._parse_args().allow_session_errors is True


def test_shadow_ab_classify_per_session_continues_after_runtime_error() -> None:
    sessions = [pd.Timestamp("2026-05-12").date(), pd.Timestamp("2026-05-13").date()]
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-12", "2026-05-13"]),
            "close": [100.0, 101.0],
        }
    )

    class Output:
        trend_direction = AxisOutput(
            raw_label="up",
            stable_label="up",
            active_label="up",
            evidence={},
            data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
        )
        trend_character = trend_direction
        volatility_state = trend_direction
        breadth_state = trend_direction
        transition_risk = type(
            "TransitionRisk",
            (),
            {
                "state": "stable",
                "score": None,
                "score_components": None,
                "primary_drivers": [],
                "triggered_rules": [],
                "data_quality": DataQuality(
                    status="ok", freshness_days=0, completeness=1.0
                ),
                "evidence": {},
            },
        )()
        agent_routing = None
        change_point = None
        credit_funding_state = None
        credit_funding_effective_state = None
        inflation_growth_state = None
        cluster = None
        monetary_pressure_state = None
        volume_liquidity_state = None
        network_fragility = None

    class Engine:
        def classify(self, **kwargs):
            if kwargs["as_of_date"] == sessions[0]:
                raise RuntimeError("insufficient window")
            return Output()

    v1_records, v2_records, errors = run_v2_shadow_ab_gate._classify_per_session(
        engine=Engine(),
        sessions=sessions,
        market_data=market_data,
        event_calendar=pd.DataFrame(),
        v2_kwargs=None,
        mode_label="test",
    )

    assert errors == 1
    assert list(v1_records) == [sessions[1]]
    assert list(v2_records) == [sessions[1]]
    assert v2_records[sessions[1]]["transition_risk_primary_drivers"] == []
    assert v2_records[sessions[1]]["transition_risk_triggered_rules"] == []


def test_walkforward_gate_tallies_rich_transition_risk_fields() -> None:
    output = type(
        "Output",
        (),
        {
            "transition_risk": type(
                "TransitionRisk",
                (),
                {
                    "state": "fragile_bull",
                    "score_components": {"model_instability": 0.25},
                    "triggered_rules": ["fragile_bull", "state_confirmation_pending"],
                },
            )(),
            "agent_routing": None,
            "change_point": None,
            "credit_funding_state": None,
            "credit_funding_effective_state": None,
            "inflation_growth_state": None,
            "cluster": None,
        },
    )()
    metrics = run_v2_walkforward_gate._session_metrics_empty()

    run_v2_walkforward_gate._tally_output(metrics, output)

    assert metrics["fragile_bull_fired"] == 1
    assert metrics["score_components_dict"] == 1
    assert metrics["model_instability_evidence_on_score"] == 1
    assert metrics["state_confirmation_pending"] == 1


def test_shadow_ab_classify_per_session_propagates_programmer_errors() -> None:
    sessions = [pd.Timestamp("2026-05-12").date()]
    market_data = pd.DataFrame(
        {"date": pd.to_datetime(["2026-05-12"]), "close": [100.0]}
    )

    class Engine:
        def classify(self, **_kwargs):
            raise TypeError("bad call shape")

    with pytest.raises(TypeError, match="bad call shape"):
        run_v2_shadow_ab_gate._classify_per_session(
            engine=Engine(),
            sessions=sessions,
            market_data=market_data,
            event_calendar=pd.DataFrame(),
            v2_kwargs=None,
            mode_label="test",
        )


def _write_v2_gate_parquets(tmp_path: Path) -> tuple[Path, Path]:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "raw" / "v2"
    daily = pd.read_csv(fixture_root / "daily_ohlcv.csv")
    if "VIX" not in set(daily["symbol"]) and "VIXY" in set(daily["symbol"]):
        vix = daily[daily["symbol"] == "VIXY"].copy()
        vix["symbol"] = "VIX"
        daily = pd.concat([daily, vix], ignore_index=True)
    daily = daily.assign(date=pd.to_datetime(daily["date"]))
    daily_path = tmp_path / "daily_ohlcv.parquet"
    daily.to_parquet(daily_path, index=False)

    macro = pd.read_csv(fixture_root / "fred_macro_series.csv")
    macro = macro.assign(date=pd.to_datetime(macro["date"]))
    macro = macro.dropna(subset=["value"]).reset_index(drop=True)
    dates = pd.to_datetime(sorted(daily["date"].unique()))
    trend = pd.Series(range(len(dates)), index=dates, dtype="float64")
    synthetic_macro = {
        "2y_yield": 4.00 + trend * 0.0002,
        "10y_yield": 4.25 + trend * 0.0001,
        "cpi_all_items": 300.0 + trend * 0.01,
        "pmi_manufacturing": 50.0 + trend * 0.0001,
    }
    macro = pd.concat(
        [
            macro,
            pd.DataFrame(
                [
                    {
                        "date": observed_date,
                        "series_id": logical_name.upper(),
                        "logical_name": logical_name,
                        "value": value,
                    }
                    for logical_name, series in synthetic_macro.items()
                    for observed_date, value in series.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    macro_path = tmp_path / "fred_macro_series.parquet"
    macro.to_parquet(macro_path, index=False)
    return daily_path, macro_path


def test_walkforward_gate_main_runs_against_committed_v2_fixtures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daily_path, macro_path = _write_v2_gate_parquets(tmp_path)
    output_path = tmp_path / "walkforward_gate.md"
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_walkforward_gate.py",
            "--daily-dir",
            str(daily_path),
            "--macro-parquet",
            str(macro_path),
            "--event-calendar",
            str(
                Path(__file__).resolve().parent
                / "fixtures"
                / "events"
                / "us_events.yaml"
            ),
            "--config-path",
            str(config_path),
            "--start-date",
            "2026-05-12",
            "--end-date",
            "2026-05-12",
            "--output",
            str(output_path),
            "--allow-session-errors",
        ],
    )

    assert run_v2_walkforward_gate.main() == 0

    markdown = output_path.read_text()
    assert "- Window: 2026-05-12" in markdown
    assert "| sessions classified | 1 | 1 | 0 |" in markdown
    assert "| sessions with credit_funding_state | 0 | 1 | 1 |" in markdown
    assert "| sessions with credit_funding_effective_state | 0 | 1 | 1 |" in markdown


def test_shadow_ab_gate_main_runs_against_committed_v2_fixtures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daily_path, macro_path = _write_v2_gate_parquets(tmp_path)
    output_path = tmp_path / "shadow_ab_gate.md"
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_shadow_ab_gate.py",
            "--daily-dir",
            str(daily_path),
            "--macro-parquet",
            str(macro_path),
            "--event-calendar",
            str(
                Path(__file__).resolve().parent
                / "fixtures"
                / "events"
                / "us_events.yaml"
            ),
            "--config-path",
            str(config_path),
            "--n-sessions",
            "1",
            "--output",
            str(output_path),
            "--allow-session-errors",
        ],
    )

    assert run_v2_shadow_ab_gate.main() == 0

    markdown = output_path.read_text()
    assert "- Window: 2026-05-13" in markdown
    assert "| trend_direction | 0 |" in markdown
    assert "| transition_risk_state | 0 |" in markdown
    assert "- v1-mode errors (sessions): 1" in markdown
    assert "- v2-mode errors (sessions): 0" in markdown
    assert "| transition_risk_primary_drivers | 0 |" in markdown
    assert "| transition_risk_triggered_rules | 0 |" in markdown
    assert "| transition_risk_data_quality | 0 |" in markdown
    assert "| credit_funding_state | 0 |" in markdown
    assert "| credit_funding_effective_state | 0 |" in markdown
    assert "| network_fragility | 0 |" in markdown
