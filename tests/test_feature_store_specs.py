from __future__ import annotations

from datetime import date

import pytest

from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import _FEATURE_SPECS, _FeatureStoreBuildState
from regime_detection.feature_store_runtime import FeatureSpec
from regime_detection.market_context import build_market_context


def _spec_by_name(name: str) -> FeatureSpec:
    matches = [s for s in _FEATURE_SPECS if s.name == name]
    if not matches:
        raise AssertionError(f"no spec named {name!r} in _FEATURE_SPECS")
    if len(matches) > 1:
        raise AssertionError(f"duplicate specs named {name!r}: {matches}")
    return matches[0]


@pytest.fixture(scope="module")
def v1_minimal_state(market_df_for_asof) -> _FeatureStoreBuildState:
    """Real MarketContext for a stable historical date, wrapped in a build state.

    Uses the same `market_df_for_asof` conftest fixture as
    `tests/test_v2_feature_store_and_axis_seams.py`. No mocks, no synthetic data —
    real SPY/RSP/VIX frames so resolution tests exercise the same code paths as
    end-to-end builds.
    """
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    return _FeatureStoreBuildState(
        context=context,
        spy_ohlcv=context.spy_ohlcv,
        spy_close=context.spy_ohlcv["close"],
    )


def test_trend_direction_resolve_returns_spy_close_kwargs(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    spec = _spec_by_name("trend_direction")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, dict), (
        f"trend_direction.resolve returned {type(resolved).__name__}, expected dict"
    )
    assert set(resolved.keys()) == {"spy_close"}
    assert resolved["spy_close"] is v1_minimal_state.spy_close
