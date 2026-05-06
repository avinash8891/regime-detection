from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.calendar import nyse_calendar
from regime_detection.engine import RegimeEngine


def _market_df_until(as_of: date) -> pd.DataFrame:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "raw"
    spy = pd.read_csv(fixtures / "SPY.csv")
    rsp = pd.read_csv(fixtures / "RSP.csv")
    vixy = pd.read_csv(fixtures / "VIXY.csv")
    df = pd.concat([spy, rsp, vixy], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= as_of].copy()
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def test_regime_output_contains_v1_placeholders_and_omits_none_fields() -> None:
    as_of = date(2023, 12, 14)
    out = RegimeEngine().classify(as_of_date=as_of, market_data=_market_df_until(as_of))
    dumped = out.model_dump()

    assert dumped["structural_causal_state"]["monetary_pressure"] == {
        "label": "unknown",
        "reason": "not_implemented_v1",
    }
    assert dumped["network_fragility"] == {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    assert "hard_max_loss_required" not in dumped["strategy_response"]


def test_classify_window_returns_one_output_per_nyse_trading_day() -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = _market_df_until(end_date)

    timeline = engine.classify_window(
        end_date=end_date,
        market_data=market_data,
        lookback_days=5,
    )

    expected_days = nyse_calendar().schedule(
        start_date=date(2023, 12, 8),
        end_date=end_date,
    ).index.date
    assert [row.as_of_date for row in timeline.outputs] == list(expected_days)
    assert timeline.start_date == expected_days[0]
    assert timeline.end_date == end_date


def test_classify_window_uses_lookback_days_not_fixed_calendar_span() -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = _market_df_until(end_date)

    timeline = engine.classify_window(
        end_date=end_date,
        market_data=market_data,
        lookback_days=30,
    )

    assert len(timeline.outputs) == 30
