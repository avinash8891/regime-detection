from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.engine import RegimeEngine
from regime_detection.strategy_response import StrategyInputs, build_strategy_response


def _empty_event_calendar() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "market", "type", "importance"])


def _market_df_for_asof(as_of: date) -> pd.DataFrame:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "raw"
    spy = pd.read_csv(fixtures / "SPY.csv")
    rsp = pd.read_csv(fixtures / "RSP.csv")
    vixy = pd.read_csv(fixtures / "VIXY.csv")
    df = pd.concat([spy, rsp, vixy], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= as_of].copy()
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return df[keep]


def test_strategy_response_regresses_on_golden_fixtures() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )

    engine = RegimeEngine()
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        df = _market_df_for_asof(as_of)
        out = engine.classify(as_of_date=as_of, market_data=df, event_calendar=_empty_event_calendar())

        tr = out.transition_risk.label
        sr = out.strategy_response

        if tr == "crisis_override":
            assert sr.position_size_multiplier == 0.25
            assert sr.leverage_allowed is False
            assert sr.hard_max_loss_required is True
            assert sr.block_weak_signals is True
            assert "crisis" in sr.modifiers_applied
        elif tr == "bear_stress_warning":
            assert sr.position_size_multiplier == 0.5
            assert sr.leverage_allowed is False
            assert sr.require_confirmation_for_shorts is True
            assert "bear_stress" in sr.modifiers_applied
        elif tr == "recovery_attempt":
            assert sr.position_size_multiplier == 0.5
            assert sr.require_breadth_confirmation is True
            assert "recovery_attempt" in sr.modifiers_applied
        else:
            # stable / post_switch_cooldown should not trigger the high-priority warnings.
            assert "crisis" not in sr.modifiers_applied
            assert "bear_stress" not in sr.modifiers_applied


def test_strategy_response_bull_fragile_modifier() -> None:
    out = build_strategy_response(
        inp=StrategyInputs(
            trend_direction_active="bull",
            trend_character_active="transition",
            volatility_active="normal_vol",
            breadth_active="divergent_fragile",
            transition_risk_label="bull_fragile_warning",
        )
    )
    assert out.position_size_multiplier == 0.5
    assert out.allow_leverage_expansion is False
    assert out.require_confirmation_for_new_longs is True
    assert "bull_fragile" in out.modifiers_applied


def test_strategy_response_sideways_chop_modifier() -> None:
    out = build_strategy_response(
        inp=StrategyInputs(
            trend_direction_active="bull",
            trend_character_active="chop",
            volatility_active="normal_vol",
            breadth_active="neutral_breadth",
            transition_risk_label="stable",
        )
    )
    assert out.allow_trend_following is False
    assert out.take_profit_faster is True
    assert "sideways_chop" in out.modifiers_applied


def test_strategy_response_unknown_fallback() -> None:
    out = build_strategy_response(
        inp=StrategyInputs(
            trend_direction_active="unknown",
            trend_character_active="unknown",
            volatility_active="unknown",
            breadth_active="unknown",
            transition_risk_label="unknown",
        )
    )
    assert out.leverage_allowed is False
    assert out.reason == "unknown_or_unmapped_regime"
    assert out.log_for_review is True

