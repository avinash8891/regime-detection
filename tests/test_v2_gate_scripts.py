from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.models import AxisOutput, DataQuality
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
    monkeypatch.setattr(module, "materialize_if_requested", lambda **_: None)

    monkeypatch.setattr(sys, "argv", argv_base)
    assert module._parse_args().allow_session_errors is False

    monkeypatch.setattr(sys, "argv", [*argv_base, "--allow-session-errors"])
    assert module._parse_args().allow_session_errors is True


def _write_v2_gate_parquets(tmp_path: Path) -> tuple[Path, Path]:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "raw" / "v2"
    daily = pd.read_csv(fixture_root / "daily_ohlcv.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily_path = tmp_path / "daily_ohlcv.parquet"
    daily.to_parquet(daily_path, index=False)

    macro = pd.read_csv(fixture_root / "fred_macro_series.csv")
    macro["date"] = pd.to_datetime(macro["date"])
    macro_path = tmp_path / "fred_macro_series.parquet"
    macro.to_parquet(macro_path, index=False)
    return daily_path, macro_path


def test_walkforward_gate_main_runs_against_committed_v2_fixtures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daily_path, macro_path = _write_v2_gate_parquets(tmp_path)
    output_path = tmp_path / "walkforward_gate.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_walkforward_gate.py",
            "--daily-dir",
            str(daily_path),
            "--macro-parquet",
            str(macro_path),
            "--start-date",
            "2026-05-12",
            "--end-date",
            "2026-05-12",
            "--output",
            str(output_path),
        ],
    )

    assert run_v2_walkforward_gate.main() == 0

    markdown = output_path.read_text()
    assert "- Window: 2026-05-12" in markdown
    assert "| sessions classified | 1 | 1 | 0 |" in markdown
    assert "| sessions with credit_funding_state | 0 | 1 | 1 |" in markdown
    assert "| sessions with credit_funding_effective_state | 0 | 1 | 1 |" in markdown
    assert "| credit_funding (classified) | 1 | 100.0% |" in markdown


def test_shadow_ab_gate_main_runs_against_committed_v2_fixtures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daily_path, macro_path = _write_v2_gate_parquets(tmp_path)
    output_path = tmp_path / "shadow_ab_gate.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_shadow_ab_gate.py",
            "--daily-dir",
            str(daily_path),
            "--macro-parquet",
            str(macro_path),
            "--n-sessions",
            "1",
            "--output",
            str(output_path),
        ],
    )

    assert run_v2_shadow_ab_gate.main() == 0

    markdown = output_path.read_text()
    assert "- Window: 2026-05-13" in markdown
    assert "| trend_direction | 0 |" in markdown
    assert "| transition_risk_label | 0 |" in markdown
    assert "| credit_funding_state | 1 |" in markdown
    assert "| credit_funding_effective_state | 1 |" in markdown
    assert "| network_fragility | 1 |" in markdown
