from __future__ import annotations

import json
from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.shadow_storage import open_shadow_db
from regime_detection.shadow_strategy_metrics import (
    ShadowStrategyMetrics,
    compute_shadow_strategy_metrics,
)
from scripts.build_shadow_metrics_report import build_shadow_metrics_report

# A six-session ledger straddling the COVID-crash window (2020-02-24..2020-03-23):
# two calm sessions, three crisis sessions (engine raises crisis_vol/crisis on the
# 02-26 selloff), then a recovery session. Declining SPY into the crash, bounce after.
_LEDGER = [
    ("2020-02-24", 100.0, "low_vol", "stable"),
    ("2020-02-25", 98.0, "low_vol", "stable"),
    ("2020-02-26", 96.0, "crisis_vol", "crisis"),
    ("2020-02-27", 90.0, "crisis_vol", "crisis"),
    ("2020-02-28", 85.0, "crisis_vol", "crisis"),
    ("2020-03-02", 88.0, "low_vol", "stable"),
]


def _build_ledger(out_root: Path) -> None:
    layout = out_root
    (layout / "outputs").mkdir(parents=True)
    last_archive = layout / "input_archives" / _LEDGER[-1][0]
    last_archive.mkdir(parents=True)

    with open_shadow_db(layout / "regime_shadow.db") as conn:
        for as_of, _close, _vol, _trans in _LEDGER:
            conn.execute(
                """
                INSERT INTO runs (
                    run_timestamp, as_of_date, engine_version, config_version,
                    status, input_archive_path, output_path, output_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{as_of}T12:00:00+00:00",
                    as_of,
                    "regime-engine-vtest",
                    "core3-v2.0.0",
                    "success",
                    str(layout / "input_archives" / as_of),
                    str(layout / "outputs" / f"{as_of}.json"),
                    None,
                ),
            )
        conn.commit()

    for as_of, _close, vol, trans in _LEDGER:
        (layout / "outputs" / f"{as_of}.json").write_text(
            json.dumps(
                {
                    "as_of_date": as_of,
                    "volatility_state": {"active_label": vol},
                    "transition_risk": {"state": trans},
                }
            ),
            encoding="utf-8",
        )

    market = pd.DataFrame(
        [
            {"date": date.fromisoformat(as_of), "symbol": "SPY", "close": close}
            for as_of, close, _vol, _trans in _LEDGER
        ]
    )
    market.to_parquet(last_archive / "market_data.parquet", index=False)


def test_shadow_metrics_reports_all_six_section_10_metrics(tmp_path: Path) -> None:
    out_root = tmp_path / "shadow_run"
    _build_ledger(out_root)

    metrics = compute_shadow_strategy_metrics(out_root)

    # All six §10 metric keys present (plus the no-regime baseline).
    field_names = {f.name for f in fields(ShadowStrategyMetrics)}
    assert {
        "strategy_return",
        "max_drawdown",
        "sharpe",
        "false_switch_rate",
        "average_detection_lag",
        "wrong_environment_trades_avoided",
    } <= field_names

    # Defensive overlay went flat on the 02-26 crisis (offset 2 in the covid window).
    assert metrics.average_detection_lag == 2.0
    assert metrics.covered_crash_windows == 1
    # Two flat sessions (02-27, 02-28) had negative SPY returns ⇒ losses dodged.
    assert metrics.wrong_environment_trades_avoided == 2
    # One switch (invested→flat) reverts to invested within 3 sessions; one does not.
    assert metrics.false_switch_rate == 0.5
    # Avoiding the deep crash beats the always-invested baseline on both return and
    # drawdown (less negative).
    assert metrics.strategy_return > metrics.baseline_return
    assert metrics.max_drawdown > metrics.baseline_max_drawdown


def test_shadow_metrics_are_reproducible_on_rerun(tmp_path: Path) -> None:
    out_root = tmp_path / "shadow_run"
    _build_ledger(out_root)

    first = compute_shadow_strategy_metrics(out_root)
    second = compute_shadow_strategy_metrics(out_root)

    assert first == second  # pure function of the ledger — no clock, no randomness


def test_build_shadow_metrics_report_writes_deterministic_json(tmp_path: Path) -> None:
    out_root = tmp_path / "shadow_run"
    _build_ledger(out_root)

    report_path = build_shadow_metrics_report(out_root)
    first_text = report_path.read_text(encoding="utf-8")
    second_text = build_shadow_metrics_report(out_root).read_text(encoding="utf-8")

    assert report_path == out_root / "reports" / "shadow_strategy_metrics.json"
    assert first_text == second_text
    payload = json.loads(first_text)
    assert payload["wrong_environment_trades_avoided"] == 2
    assert payload["average_detection_lag"] == 2.0


def test_shadow_metrics_requires_at_least_two_sessions(tmp_path: Path) -> None:
    out_root = tmp_path / "shadow_run"
    (out_root / "outputs").mkdir(parents=True)
    with open_shadow_db(out_root / "regime_shadow.db") as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_timestamp, as_of_date, engine_version, config_version,
                status, input_archive_path
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2020-02-24T12:00:00+00:00",
                "2020-02-24",
                "regime-engine-vtest",
                "core3-v2.0.0",
                "success",
                str(out_root / "input_archives" / "2020-02-24"),
            ),
        )
        conn.commit()

    try:
        compute_shadow_strategy_metrics(out_root)
    except ValueError as exc:
        assert "require >= 2 successful sessions" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a single-session ledger")
