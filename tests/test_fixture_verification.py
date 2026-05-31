from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest
import yaml

from regime_detection.models import TransitionRiskState

_GOLDEN_EXPECTED_KEYS = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state_raw",
    "breadth_state_active",
    "transition_risk",
)

_V2_SPEC_GOLDEN_DATES = {
    "2010-05-06",
    "2011-08-08",
    "2015-08-24",
    "2018-10-10",
    "2020-08-15",
    "2021-01-27",
    "2022-09-26",
    "2023-03-13",
    "2024-08-05",
}

_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES = {
    "2010-05-06": "V2 daily OHLCV fixture must include real VIX rows",
    "2011-08-08": "V2 daily OHLCV fixture must include real VIX rows",
    "2015-08-24": "V2 daily OHLCV fixture must include real VIX rows",
    "2018-10-10": "V2 daily OHLCV fixture must include real VIX rows",
    "2020-08-15": "as_of_date must be an NYSE trading day",
}


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


def test_conftest_v2_kwargs_reject_asof_before_real_v2_rows() -> None:
    import conftest as project_conftest

    project_conftest._load_market_data.cache_clear()
    market_data = project_conftest._load_market_data()
    event_calendar = pd.DataFrame()
    build_kwargs = project_conftest.synthetic_v2_kwargs_for_market_data.__wrapped__(
        event_calendar
    )

    with pytest.raises(RuntimeError, match="as_of=2018-12-31"):
        build_kwargs(market_data[market_data["date"] <= date(2018, 12, 31)])


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


def test_classified_golden_outputs_cover_every_row_without_silent_skips(
    golden_rows: list[dict[str, object]],
    classified_golden_outputs: dict[date, object],
) -> None:
    expected_dates = {date.fromisoformat(str(row["as_of_date"])) for row in golden_rows}

    assert len(expected_dates) == 10
    assert set(classified_golden_outputs) == expected_dates


def test_golden_dates_match_live_labels_without_data_quality_bypass(
    golden_rows: list[dict[str, object]],
    classified_golden_outputs: dict[date, object],
) -> None:
    for row in golden_rows:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        expected = row["expected"]
        output = classified_golden_outputs[as_of]
        actual = {
            "trend_direction": output.trend_direction.active_label,
            "trend_character": output.trend_character.active_label,
            "volatility_state": output.volatility_state.active_label,
            "breadth_state_raw": output.breadth_state.raw_label,
            "breadth_state_active": output.breadth_state.active_label,
            "transition_risk": output.transition_risk.state,
        }

        assert set(_GOLDEN_EXPECTED_KEYS).issubset(expected), as_of
        assert actual == {key: expected[key] for key in _GOLDEN_EXPECTED_KEYS}, as_of


def test_v2_section_9_4_golden_dates_are_registered() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture = yaml.safe_load(
        (
            repo_root / "tests" / "fixtures" / "derived" / "golden_dates_v2.yaml"
        ).read_text()
    )

    assert fixture["source_spec"] == "docs/regime_engine_v2_spec.md#9.4"
    assert {row["as_of_date"] for row in fixture["rows"]} == _V2_SPEC_GOLDEN_DATES
    for row in fixture["rows"]:
        assert row["intent_id"]
        assert row["expected_v2_fields"]


def test_v2_golden_dates_classify_expected_fields(
    v2_classify_kwargs_for_asof,
) -> None:
    from regime_detection.engine import RegimeEngine

    repo_root = Path(__file__).resolve().parents[1]
    fixture = yaml.safe_load(
        (
            repo_root / "tests" / "fixtures" / "derived" / "golden_dates_v2.yaml"
        ).read_text()
    )
    engine = RegimeEngine()
    missing: list[str] = []
    unsupported: dict[str, str] = {}
    classified_dates: set[str] = set()

    for row in fixture["rows"]:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        try:
            output = engine.classify(
                as_of_date=as_of,
                **v2_classify_kwargs_for_asof(as_of),
            )
        except (RuntimeError, ValueError) as exc:
            unsupported[str(as_of)] = str(exc)
            continue

        classified_dates.add(str(as_of))
        dumped = output.model_dump(mode="json", exclude_none=True)

        for field_name, expected in row["expected_v2_fields"].items():
            if field_name == "sequence":
                continue
            if field_name == "transition_evidence":
                transition = dumped.get("transition_risk", {})
                evidence = transition.get("evidence", {})
                drivers = transition.get("primary_drivers", [])
                triggered = transition.get("triggered_rules", [])
                if not evidence and not drivers and not triggered:
                    missing.append(f"{as_of}:{field_name}:missing_output")
                continue
            if field_name == "transition_risk_minimum":
                state = dumped.get("transition_risk", {}).get("state")
                if state is None:
                    missing.append(f"{as_of}:{field_name}:missing_state")
                continue
            if field_name == "credit_funding":
                field_name = "credit_funding_effective_state"
            if dumped.get(field_name) in (None, {}, []):
                missing.append(f"{as_of}:{field_name}:missing_output")

    assert unsupported.keys() == _V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES.keys()
    for as_of, expected_fragment in _V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES.items():
        assert expected_fragment in unsupported[as_of]
    assert classified_dates == _V2_SPEC_GOLDEN_DATES - set(unsupported)
    assert missing == []


def test_golden_date_replacement_set_has_documented_justification() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    justification = (
        repo_root
        / "docs"
        / "verification"
        / "golden_dates_replacement_justification.md"
    ).read_text()

    assert "2020-04-10" in justification
    assert "Good Friday" in justification
    assert "no silent pre-2019 or data-quality skips" in justification


def test_classification_labels_are_independent_of_extra_history_length(
    market_df_for_asof,
    event_calendar_df: pd.DataFrame,
) -> None:
    from regime_detection.config import load_regime_config
    from regime_detection.engine import RegimeEngine

    repo_root = Path(__file__).resolve().parents[1]
    as_of = date(2023, 12, 5)
    market_data = market_df_for_asof(as_of)
    spy_sessions = (
        market_data.loc[market_data["symbol"] == "SPY", "date"]
        .drop_duplicates()
        .sort_values()
    )
    shorter_start = spy_sessions.iloc[-700]
    shorter_market_data = (
        market_data[market_data["date"] >= shorter_start].copy().reset_index(drop=True)
    )
    config = load_regime_config(
        repo_root / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
    )
    engine = RegimeEngine()

    full = engine.classify(
        as_of_date=as_of,
        market_data=market_data,
        config=config,
        event_calendar=event_calendar_df,
    )
    shorter = engine.classify(
        as_of_date=as_of,
        market_data=shorter_market_data,
        config=config,
        event_calendar=event_calendar_df,
    )

    assert {
        "trend_direction": shorter.trend_direction.active_label,
        "trend_character": shorter.trend_character.active_label,
        "volatility_state": shorter.volatility_state.active_label,
        "breadth_state_raw": shorter.breadth_state.raw_label,
        "breadth_state_active": shorter.breadth_state.active_label,
    } == {
        "trend_direction": full.trend_direction.active_label,
        "trend_character": full.trend_character.active_label,
        "volatility_state": full.volatility_state.active_label,
        "breadth_state_raw": full.breadth_state.raw_label,
        "breadth_state_active": full.breadth_state.active_label,
    }


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
