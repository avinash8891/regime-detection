from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from regime_detection.config import load_regime_config
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context

_V1_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "regime_detection"
    / "configs"
    / "core3-v1.0.0.yaml"
)


@pytest.fixture(scope="module")
def v1_minimal_availability(market_df_for_asof) -> dict[str, dict[str, object]]:
    """Capture availability dict from a minimal V1 MarketContext.

    Same fixture pattern as tests/test_feature_store_specs.py — real fixture
    data (SPY/RSP/VIX) for a stable historical date. Snapshot pins both the
    keys present (coverage) and their per-feature reason/required_inputs/policy
    (semantics)."""
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=load_regime_config(_V1_CONFIG_PATH),
    )
    store = build_feature_store(context)
    return {name: avail.model_dump() for name, avail in store.availability.items()}


def test_availability_dict_pure_v1_context_snapshot(
    v1_minimal_availability: dict[str, dict[str, object]],
) -> None:
    expected: dict[str, dict[str, object]] = {
        "trend_direction": {
            "feature": "trend_direction",
            "available": True,
            "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        "trend_character": {
            "feature": "trend_character",
            "available": True,
            "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close", "spy_ohlcv.high", "spy_ohlcv.low"),
            "missing_inputs": (),
        },
        "volatility": {
            "feature": "volatility",
            "available": True,
            "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        "breadth": {
            "feature": "breadth",
            "available": True,
            "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close", "rsp_close"),
            "missing_inputs": (),
        },
        "sma_50": {
            "feature": "sma_50",
            "available": True,
            "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        "network_fragility": {
            "feature": "network_fragility",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("sector_etf_closes",),
            "missing_inputs": ("sector_etf_closes",),
        },
        "trend_direction_v2": {
            "feature": "trend_direction_v2",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("trend_direction_v2_config", "spy_ohlcv.close"),
            "missing_inputs": ("trend_direction_v2_config",),
        },
        "volatility_state_v2": {
            "feature": "volatility_state_v2",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("volatility_state_v2_config", "spy_ohlcv.ohlc"),
            "missing_inputs": ("volatility_state_v2_config",),
        },
        "breadth_state_v2": {
            "feature": "breadth_state_v2",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("breadth_state_v2_config", "sector_etf_closes"),
            "missing_inputs": ("breadth_state_v2_config", "sector_etf_closes"),
        },
        "volume_liquidity_v2": {
            "feature": "volume_liquidity_v2",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("volume_liquidity_v2_config", "spy_ohlcv.volume"),
            "missing_inputs": ("volume_liquidity_v2_config",),
        },
        "monetary": {
            "feature": "monetary",
            "available": False,
            "policy": "none",
            "reason": "not_configured",
            "required_inputs": (
                "macro_series",
                "2y_yield",
                "10y_yield",
                "broad_usd_index",
            ),
            "missing_inputs": (),
        },
        "hmm": {
            "feature": "hmm",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": (
                "hmm_config",
                "volume_liquidity_v2",
                "network_fragility",
            ),
            "missing_inputs": (
                "hmm_config",
                "volume_liquidity_v2",
                "network_fragility",
            ),
        },
        "clustering": {
            "feature": "clustering",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": (
                "clustering_config",
                "breadth_state_v2.pct_above_50dma",
                "network_fragility",
                "trend_direction_v2",
            ),
            "missing_inputs": (
                "clustering_config",
                "breadth_state_v2.pct_above_50dma",
                "network_fragility",
                "trend_direction_v2",
            ),
        },
        "change_point": {
            "feature": "change_point",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": ("change_point_config", "realized_vol_21d"),
            "missing_inputs": ("change_point_config",),
        },
        "credit_funding": {
            "feature": "credit_funding",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": (
                "credit_funding_config",
                "cross_asset_closes",
                "macro_series",
            ),
            "missing_inputs": ("credit_funding_config",),
        },
        "inflation_growth": {
            "feature": "inflation_growth",
            "available": False,
            "policy": "none",
            "reason": "missing_required_inputs",
            "required_inputs": (
                "inflation_growth_config",
                "cross_asset_closes",
                "macro_series",
            ),
            "missing_inputs": ("inflation_growth_config",),
        },
    }

    assert v1_minimal_availability == expected
