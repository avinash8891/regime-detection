from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.engine import RegimeEngine


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


def test_volatility_state_matches_pinned_fixtures() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )

    engine = RegimeEngine()
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        df = _market_df_for_asof(as_of)
        out = engine.classify(as_of_date=as_of, market_data=df, event_calendar=_empty_event_calendar())
        assert out.volatility_state.active_label == row["expected"]["volatility_state"]
