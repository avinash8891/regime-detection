from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest
import yaml

from regime_detection.models import TransitionRiskState


def test_conftest_market_data_requires_real_combined_market_parquet(
    monkeypatch, tmp_path: Path
) -> None:
    import conftest as project_conftest

    for symbol in ("SPY", "RSP", "VIXY"):
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ]
        ).to_csv(tmp_path / f"{symbol}.csv", index=False)

    project_conftest._load_market_data.cache_clear()
    monkeypatch.setattr(project_conftest, "_RAW_DIR", tmp_path)
    monkeypatch.setattr(
        project_conftest, "_MARKET_PARQUET_PATH", tmp_path / "missing.parquet"
    )

    with pytest.raises(RuntimeError, match="market_data.parquet"):
        project_conftest._load_market_data()

    project_conftest._load_market_data.cache_clear()


def test_conftest_v2_kwargs_reject_full_history_window_before_real_v2_rows() -> None:
    import conftest as project_conftest

    project_conftest._load_market_data.cache_clear()
    market_data = project_conftest._load_market_data()
    event_calendar = pd.DataFrame()
    build_kwargs = project_conftest.synthetic_v2_kwargs_for_market_data.__wrapped__(
        event_calendar
    )

    with pytest.raises(RuntimeError, match="window start=2016-01-04"):
        build_kwargs(market_data[market_data["date"] <= date(2023, 12, 14)])


def test_conftest_v2_kwargs_use_real_v2_fixture_rows_when_window_is_covered() -> None:
    import conftest as project_conftest

    event_calendar = pd.DataFrame()
    build_kwargs = project_conftest.synthetic_v2_kwargs_for_market_data.__wrapped__(
        event_calendar
    )
    v2_daily = project_conftest._load_v2_daily_ohlcv()
    market_data = (
        v2_daily[
            (v2_daily["date"] <= date(2023, 12, 14))
            & (v2_daily["symbol"].isin({"SPY", "RSP", "VIX", "VIXY"}))
        ]
        .copy()
        .reset_index(drop=True)
    )
    kwargs = build_kwargs(market_data)

    qqq_rows = v2_daily[
        (v2_daily["symbol"] == "QQQ") & (v2_daily["date"] <= date(2023, 12, 14))
    ].sort_values("date")
    expected_qqq = qqq_rows.set_index(pd.to_datetime(qqq_rows["date"]))["close"].astype(
        float
    )
    pd.testing.assert_series_equal(
        kwargs["cross_asset_closes"]["QQQ"],
        expected_qqq.rename("QQQ"),
        check_names=True,
    )


def test_fixture_verification_legacy_path_fails_loudly_without_v2_transition_inputs() -> (
    None
):
    """
    Hard gate for Slice 2:
    - golden_dates.yaml is hand-labeled (never engine-generated)
    - legacy raw CSV fixtures do not carry required V2 transition-score inputs
    - the report generator must fail loudly instead of silently fabricating a
      transition_risk fallback
    """
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"

    committed_derived = yaml.safe_load(derived_path.read_text())
    assert committed_derived.get("provenance") == "hand_labeled", (
        "golden_dates.yaml must carry provenance: hand_labeled — "
        "expected values are independently derived, not from engine output"
    )

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    generate_report = getattr(mod, "generate_report")

    with pytest.raises(ValueError) as excinfo:
        generate_report(
            generated_at_utc="2026-05-19T00:00:00+00:00",
            generated_by_commit="test_determinism",
        )
    message = str(excinfo.value)
    assert "ClassifyRequest missing configured V2 inputs" in message


def test_fixture_verification_report_includes_rich_transition_evidence(
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    session = date(2026, 5, 15)
    axis = SimpleNamespace(active_label="bull", evidence={"rule": "sample"})
    transition = SimpleNamespace(
        state="watch",
        evidence={
            "triggered_rules": ["post_switch_cooldown"],
            "axis_switch_count": 1,
            "recent_axis_switch_count": 2,
        },
        score=0.42,
        score_components={"trend_break": 0.30, "macro_event": 1.0},
        primary_drivers=["macro_event"],
        triggered_rules=["post_switch_cooldown"],
        data_quality={"status": "ok"},
    )
    output = SimpleNamespace(
        trend_direction=axis,
        trend_character=SimpleNamespace(active_label="trending", evidence={}),
        volatility_state=SimpleNamespace(active_label="normal_vol", evidence={}),
        breadth_state=SimpleNamespace(active_label="healthy_breadth", evidence={}),
        transition_risk=transition,
    )

    monkeypatch.setattr(
        mod,
        "INTENTS",
        [
            {
                "intent_id": "transition_rich_evidence",
                "intent_date": session.isoformat(),
                "intent": {"transition_risk": "watch"},
                "search_window_trading_days": 0,
                "notes": "synthetic v2 transition evidence",
            }
        ],
    )
    monkeypatch.setattr(
        mod,
        "_load_hand_labeled_expectations",
        lambda: {"transition_rich_evidence": {"transition_risk": "watch"}},
    )
    monkeypatch.setattr(
        mod,
        "_load_market_data",
        lambda: pd.DataFrame({"date": [pd.Timestamp(session)]}),
    )
    monkeypatch.setattr(
        mod, "_classify_all_intents", lambda _market_data: {session: output}
    )
    monkeypatch.setattr(mod, "_sha256_file", lambda _path: "sha256")

    report = mod.generate_report(
        generated_at_utc="2026-05-23T00:00:00+00:00",
        generated_by_commit="test_transition_rich_evidence",
    )

    transition_evidence = report["rows"][0]["predicate_evaluations"]["transition_risk"]
    assert transition_evidence == {
        "evidence": {
            "triggered_rules": ["post_switch_cooldown"],
            "axis_switch_count": 1,
            "recent_axis_switch_count": 2,
        },
        "score": 0.42,
        "score_components": {"trend_break": 0.3, "macro_event": 1.0},
        "primary_drivers": ["macro_event"],
        "triggered_rules": ["post_switch_cooldown"],
        "data_quality": {"status": "ok"},
    }


def test_fixture_verification_requires_combined_market_parquet_for_vix(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    for symbol in ("SPY", "RSP", "VIXY"):
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ]
        ).to_csv(tmp_path / f"{symbol}.csv", index=False)

    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="market_data.parquet"):
        mod._load_market_data()


def test_fixture_transition_risk_expectations_use_current_state_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"
    doc = yaml.safe_load(derived_path.read_text())
    valid_states = set(get_args(TransitionRiskState))

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    labels: list[str] = []
    labels.extend(
        row["expected"]["transition_risk"]
        for row in doc.get("rows", [])
        if "transition_risk" in row.get("expected", {})
    )
    labels.extend(
        item["intent"]["transition_risk"]
        for item in mod.INTENTS
        if "transition_risk" in item.get("intent", {})
    )

    assert labels
    assert sorted(set(labels) - valid_states) == []
