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

    assert isinstance(
        resolved, dict
    ), f"trend_direction.resolve returned {type(resolved).__name__}, expected dict"
    assert set(resolved.keys()) == {"spy_close"}
    assert resolved["spy_close"] is v1_minimal_state.spy_close


def test_trend_character_resolve_returns_ohlcv_kwargs_v1_path(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    spec = _spec_by_name("trend_character")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"close", "high", "low", "volume", "tc_v2_config"}
    assert resolved["close"] is v1_minimal_state.spy_close
    # tc_v2_config is passed through from state.context.config — may be None or a
    # TrendCharacterV2Config depending on which RegimeConfig is in scope. Assert
    # identity rather than None so the test stays correct regardless of config defaults.
    assert (
        resolved["tc_v2_config"] is v1_minimal_state.context.config.trend_character_v2
    )


def test_volatility_resolve_returns_close_and_vix_proxy(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    spec = _spec_by_name("volatility")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"close", "vix_proxy_close"}
    assert resolved["close"] is v1_minimal_state.spy_close
    assert resolved["vix_proxy_close"] is v1_minimal_state.context.vix_proxy_close


def test_breadth_resolve_returns_spy_close_and_aligned_rsp(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    spec = _spec_by_name("breadth")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"spy_close", "rsp_close"}
    assert resolved["spy_close"] is v1_minimal_state.spy_close
    # rsp_close must be reindexed onto spy_ohlcv.index — matches legacy
    # _build_breadth_feature behavior.
    assert list(resolved["rsp_close"].index) == list(v1_minimal_state.spy_ohlcv.index)


def test_sma_50_resolve_returns_spy_close(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    spec = _spec_by_name("sma_50")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"spy_close"}
    assert resolved["spy_close"] is v1_minimal_state.spy_close


def test_sentiment_score_resolve_missing_aaii_returns_unavailable(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    """Pure V1 context has no aaii_sentiment — spec.resolve must report
    aaii_sentiment as the missing input."""
    from regime_detection.feature_store_runtime import _Unavailable

    spec = _spec_by_name("sentiment_score")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, _Unavailable)
    assert "aaii_sentiment" in resolved.missing_inputs


def test_sentiment_score_spec_is_internal_report_false() -> None:
    """sentiment_score is intermediate state — must not emit availability."""
    spec = _spec_by_name("sentiment_score")
    assert spec.report is False


def test_news_sentiment_score_resolve_missing_config_returns_unavailable(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    """Pure V1 context has no news_sentiment_config — spec.resolve must
    report news_sentiment_config as missing."""
    from regime_detection.feature_store_runtime import _Unavailable

    spec = _spec_by_name("news_sentiment_score")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, _Unavailable)
    assert "news_sentiment_config" in resolved.missing_inputs


def test_news_sentiment_score_spec_is_internal_report_false() -> None:
    spec = _spec_by_name("news_sentiment_score")
    assert spec.report is False


def test_trend_direction_v2_resolve_missing_config_returns_unavailable(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    """V1 context has no trend_direction_v2_config — resolve must report
    trend_direction_v2_config as missing, matching legacy report."""
    from regime_detection.feature_store_runtime import _Unavailable

    spec = _spec_by_name("trend_direction_v2")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, _Unavailable)
    assert resolved.missing_inputs == ("trend_direction_v2_config",)


def test_trend_direction_v2_spec_is_user_visible_report_true() -> None:
    spec = _spec_by_name("trend_direction_v2")
    assert spec.report is True
    assert spec.required_inputs == ("trend_direction_v2_config", "spy_ohlcv.close")


def test_network_fragility_resolve_missing_sector_closes_returns_unavailable(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    from regime_detection.feature_store_runtime import _Unavailable

    spec = _spec_by_name("network_fragility")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, _Unavailable)
    assert resolved.missing_inputs == ("sector_etf_closes",)


def test_network_fragility_spec_required_inputs_matches_legacy() -> None:
    spec = _spec_by_name("network_fragility")
    assert spec.required_inputs == ("sector_etf_closes",)
    assert spec.policy == "none"
    assert spec.report is True


def test_volatility_state_v2_resolve_missing_config_returns_unavailable(
    v1_minimal_state: _FeatureStoreBuildState,
) -> None:
    from regime_detection.feature_store_runtime import _Unavailable

    spec = _spec_by_name("volatility_state_v2")
    resolved = spec.resolve(v1_minimal_state)

    assert isinstance(resolved, _Unavailable)
    assert resolved.missing_inputs == ("volatility_state_v2_config",)


def test_volatility_state_v2_spec_required_inputs_matches_legacy() -> None:
    spec = _spec_by_name("volatility_state_v2")
    assert spec.required_inputs == ("volatility_state_v2_config", "spy_ohlcv.ohlc")
    assert spec.policy == "none"
    assert spec.report is True
