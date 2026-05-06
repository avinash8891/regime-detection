from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from regime_detection.engine import RegimeEngine


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


def test_transition_risk_matches_real_data_cases() -> None:
    cases = {
        date(2018, 2, 9): "crisis_override",
        date(2018, 12, 20): "bear_stress_warning",
        date(2019, 9, 11): "post_switch_cooldown",
        date(2020, 4, 29): "recovery_attempt",
        date(2021, 11, 12): "stable",
    }

    engine = RegimeEngine()
    for as_of, expected in cases.items():
        out = engine.classify(as_of_date=as_of, market_data=_market_df_for_asof(as_of))
        assert out.transition_risk.label == expected


def test_strategy_response_matches_crisis_fixture() -> None:
    as_of = date(2018, 2, 9)
    out = RegimeEngine().classify(as_of_date=as_of, market_data=_market_df_for_asof(as_of))

    assert out.transition_risk.label == "crisis_override"
    assert out.strategy_response.position_size_multiplier == 0.25
    assert out.strategy_response.leverage_allowed is False
    assert out.strategy_response.allow_buy_dip is False
    assert out.strategy_response.hard_max_loss_required is True
    assert out.strategy_response.modifiers_applied == ["crisis"]


def test_strategy_response_matches_recovery_attempt_fixture() -> None:
    as_of = date(2020, 4, 29)
    out = RegimeEngine().classify(as_of_date=as_of, market_data=_market_df_for_asof(as_of))

    assert out.transition_risk.label == "recovery_attempt"
    assert out.strategy_response.position_size_multiplier == 0.5
    assert out.strategy_response.require_breadth_confirmation is True
    assert out.strategy_response.allow_leverage_expansion is False
    assert out.strategy_response.modifiers_applied == ["recovery_attempt"]


def test_strategy_response_matches_bull_healthy_low_vol_fixture() -> None:
    as_of = date(2023, 12, 14)
    out = RegimeEngine().classify(as_of_date=as_of, market_data=_market_df_for_asof(as_of))

    assert out.transition_risk.label == "stable"
    assert out.strategy_response.position_size_multiplier == 1.0
    assert out.strategy_response.allow_leverage_expansion is True
    assert out.strategy_response.modifiers_applied == ["bull_healthy_low_vol"]
