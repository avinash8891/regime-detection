from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest
import yaml

from regime_detection.models import TransitionRiskState


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

    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        generate_report(
            generated_at_utc="2026-05-19T00:00:00+00:00",
            generated_by_commit="test_determinism",
        )


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
