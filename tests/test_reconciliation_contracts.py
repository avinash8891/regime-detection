from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.engine import RegimeEngine
from regime_detection.trend_character import compute_features


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "raw"
    spy = pd.read_csv(fixtures / "SPY.csv")
    rsp = pd.read_csv(fixtures / "RSP.csv")
    vixy = pd.read_csv(fixtures / "VIXY.csv")
    return spy, rsp, vixy


def test_classify_uses_vix_data_when_vix_proxy_missing_from_market_data() -> None:
    spy, rsp, vixy = _load_raw()
    as_of = date(2023, 12, 14)

    market_df = pd.concat([spy, rsp], ignore_index=True)
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
    market_df = market_df[market_df["date"] <= as_of].copy()
    market_df = market_df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    vix_df = vixy.copy()
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
    vix_df = vix_df[vix_df["date"] <= as_of].copy()
    vix_df = vix_df[["date", "close"]]

    out = RegimeEngine().classify(as_of_date=as_of, market_data=market_df, vix_data=vix_df)
    assert out.volatility_state.evidence["rule_evidence"]["vix_percentile_present"] is True


def test_trend_character_adx_cold_start_stays_nan() -> None:
    spy, _, _ = _load_raw()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").head(20).set_index("date")

    features = compute_features(
        close=spy["close"],
        high=spy["high"],
        low=spy["low"],
    )

    assert pd.isna(features.adx_14.iloc[13])
