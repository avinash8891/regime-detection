from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

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


def test_trend_direction_matches_pinned_fixtures() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )

    engine = RegimeEngine()

    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        df = _market_df_for_asof(as_of)
        out = engine.classify(as_of_date=as_of, market_data=df)
        assert out.trend_direction.active_label == row["expected"]["trend_direction"]

