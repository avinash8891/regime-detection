from __future__ import annotations

import importlib.util
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.axis_builders.breadth import (
    _derive_breadth_active_label_source,
    build_breadth_axis_series,
)
from regime_detection.axis_builders.credit_funding import (
    build_credit_funding_axis_series,
)
from regime_detection.axis_builders.inflation_growth import (
    build_inflation_growth_axis_series,
)
from regime_detection.axis_builders.monetary_pressure import (
    build_monetary_pressure_axis_series,
)
from regime_detection.axis_builders.network_fragility import (
    build_network_fragility_axis_series,
)
from regime_detection.axis_builders.trend_character import (
    build_trend_character_axis_series,
)
from regime_detection.axis_builders.trend_direction import (
    build_trend_direction_axis_series,
)
from regime_detection.axis_builders.volatility import build_volatility_axis_series
from regime_detection.axis_builders.volume_liquidity import (
    build_volume_liquidity_axis_series,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import CROSS_ASSET_SYMBOLS, SECTOR_ETFS
from regime_detection.market_context import build_market_context

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_V2_AS_OF = date(2026, 5, 13)


@pytest.mark.parametrize(
    ("raw", "stable", "active", "expected"),
    [
        ("weak_breadth", "weak_breadth", "weak_breadth", "etf_proxy"),
        (
            "narrowing_breadth",
            "narrowing_breadth",
            "narrowing_breadth",
            "pit_constituent",
        ),
        (
            "narrowing_breadth",
            "weak_breadth",
            "weak_breadth",
            "hysteresis_from_prior_state",
        ),
    ],
)
def test_breadth_active_label_source_separates_pit_from_hysteresis(
    raw: str,
    stable: str,
    active: str,
    expected: str,
) -> None:
    assert (
        _derive_breadth_active_label_source(
            raw=raw,
            stable=stable,
            active=active,
        )
        == expected
    )


def test_axis_builders_do_not_use_empty_string_raw_label_quality_sentinel() -> None:
    for path in (_REPO_ROOT / "src" / "regime_detection" / "axis_builders").glob(
        "*.py"
    ):
        source = path.read_text()
        assert (
            'raw_label=""' not in source
        ), f"{path.name} uses empty-string raw_label sentinel"


def test_breadth_builder_uses_shared_per_label_hysteresis_helper() -> None:
    source = (
        _REPO_ROOT / "src" / "regime_detection" / "axis_builders" / "breadth.py"
    ).read_text()

    assert "from regime_detection.axis_builders.per_label import" in source
    assert "build_per_label_axis_outputs(" in source
    assert "apply_data_quality_aware_hysteresis" not in source


def _load_test_helper_module(name: str, filename: str):
    path = _REPO_ROOT / "tests" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _build_context(as_of: date, market_df_for_asof):
    return build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )


def _with_tail_nan(context, *, column: str | None, series_name: str | None, count: int):
    spy_ohlcv = context.spy_ohlcv.copy()
    rsp_close = context.rsp_close.copy()
    if column is not None:
        spy_ohlcv.loc[spy_ohlcv.index[-count:], column] = float("nan")
    if series_name == "rsp_close":
        rsp_close.loc[rsp_close.index[-count:]] = float("nan")
    return context.model_copy(update={"spy_ohlcv": spy_ohlcv, "rsp_close": rsp_close})


@pytest.mark.parametrize(
    ("builder", "break_kwargs", "required_days"),
    [
        (build_trend_direction_axis_series, {"column": "close"}, 200),
        (build_trend_character_axis_series, {"column": "high"}, 63),
        (build_volatility_axis_series, {"column": "close"}, 252),
        (build_breadth_axis_series, {"series_name": "rsp_close"}, 50),
    ],
)
def test_core_axis_builders_force_unknown_when_required_input_window_is_missing(
    market_df_for_asof,
    builder,
    break_kwargs,
    required_days,
) -> None:
    context = _build_context(date(2023, 12, 14), market_df_for_asof)
    store = build_feature_store(context)
    broken_context = _with_tail_nan(
        context,
        column=break_kwargs.get("column"),
        series_name=break_kwargs.get("series_name"),
        count=required_days,
    )

    result = builder(broken_context, store)

    output = result.outputs_by_date[context.end_date]
    if builder is build_breadth_axis_series:
        assert output.raw_label != "unknown"
        assert output.data_quality.status == "stale_data"
        return
    assert output.raw_label == "unknown"
    assert output.stable_label == "unknown"
    assert output.active_label == "unknown"
    assert output.data_quality.status in {"insufficient_data", "stale_data"}


def test_breadth_builder_keeps_etf_proxy_when_pit_inputs_are_missing(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
) -> None:
    context = build_market_context(
        end_date=_REAL_V2_AS_OF,
        market_data=v2_market_df_for_asof(_REAL_V2_AS_OF),
        config=RegimeEngine().config,
        sector_etf_closes={
            symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
        },
    )
    store = build_feature_store(
        context,
        breadth_state_v2_config=context.config.breadth_state_v2,
    )
    assert store.breadth_state_v2 is not None
    assert store.breadth_state_v2.pct_above_50dma is None

    result = build_breadth_axis_series(context, store)

    output = result.outputs_by_date[_REAL_V2_AS_OF]
    assert output.mode == "etf_proxy"
    assert output.evidence["proxy"] == "RSP/SPY"
    assert output.evidence["row_provenance_mode"] == "etf_proxy"
    assert output.evidence["active_label_source"] == "etf_proxy"


def test_volume_liquidity_builder_returns_none_when_feature_seam_is_missing(
    market_df_for_asof,
) -> None:
    context = _build_context(date(2023, 12, 14), market_df_for_asof)
    store = build_feature_store(context)
    assert store.volume_liquidity_v2 is None

    assert build_volume_liquidity_axis_series(context, store) is None


def test_monetary_pressure_builder_forces_unknown_when_yield_input_is_missing() -> None:
    helpers = _load_test_helper_module(
        "axis_builder_monetary_helpers", "test_monetary_pressure_classifier.py"
    )
    context = helpers._build_context_with_macro()
    store = build_feature_store(
        context,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
    )
    assert store.monetary is not None
    nan_series = pd.Series(
        float("nan"), index=store.monetary.yield_change_zscore_2y_63d.index
    )
    broken_store = store.model_copy(
        update={
            "monetary": replace(
                store.monetary,
                yield_change_zscore_2y_63d=nan_series,
            )
        }
    )

    outputs = build_monetary_pressure_axis_series(context, broken_store)

    assert outputs is not None
    output = outputs[context.end_date]
    assert output.raw_label == "unknown"
    assert output.data_quality.status in {
        "insufficient_data",
        "insufficient_history",
        "stale_data",
    }


def test_credit_funding_builder_marks_stale_etf_source_unknown() -> None:
    helpers = _load_test_helper_module(
        "axis_builder_credit_helpers", "test_credit_funding_axis_engine.py"
    )
    context = helpers._build_full_synthetic_context(hyg_truncate_sessions=10)
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        credit_funding_config=context.config.credit_funding,
    )

    outputs = build_credit_funding_axis_series(context, store)

    assert outputs is not None
    output = outputs[context.end_date]
    assert output.raw_label == "unknown"
    assert output.data_quality.status == "stale_data"
    assert "etf_stale:HYG" in (output.data_quality.reason or "")


def test_network_fragility_builder_raises_when_supplied_axis_labels_miss_session(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
) -> None:
    context = build_market_context(
        end_date=_REAL_V2_AS_OF,
        market_data=v2_market_df_for_asof(_REAL_V2_AS_OF),
        config=RegimeEngine().config,
        sector_etf_closes={
            symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
        },
        cross_asset_closes={
            symbol: v2_close_series_by_symbol[symbol] for symbol in CROSS_ASSET_SYMBOLS
        },
    )
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
    )
    assert store.network_fragility is not None
    breadth_labels = {day: "broadening" for day in context.sessions}
    breadth_labels.pop(context.end_date)
    volatility_labels = {day: "normal_vol" for day in context.sessions}

    with pytest.raises(KeyError, match="breadth_active_labels_by_date missing session"):
        build_network_fragility_axis_series(
            context,
            store,
            breadth_active_labels_by_date=breadth_labels,
            volatility_active_labels_by_date=volatility_labels,
        )


def test_inflation_growth_builder_marks_stale_cpi_source_unknown() -> None:
    helpers = _load_test_helper_module(
        "axis_builder_inflation_helpers", "test_inflation_growth_axis_engine.py"
    )
    context = helpers._build_synthetic_context(cpi_truncate_calendar_days=90)
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )

    outputs = build_inflation_growth_axis_series(context, store)

    assert outputs is not None
    output = outputs[context.end_date]
    assert output.raw_label == "unknown"
    assert output.data_quality.status == "stale_data"
    assert "cpi_stale" in (output.data_quality.reason or "")


def test_inflation_growth_builder_raises_when_credit_labels_miss_session() -> None:
    helpers = _load_test_helper_module(
        "axis_builder_inflation_missing_label_helpers",
        "test_inflation_growth_axis_engine.py",
    )
    context = helpers._build_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )
    credit_labels = {day: "credit_calm" for day in context.sessions}
    credit_labels.pop(context.end_date)

    with pytest.raises(
        KeyError, match="credit_funding_active_labels_by_date missing session"
    ):
        build_inflation_growth_axis_series(
            context,
            store,
            credit_funding_active_labels_by_date=credit_labels,
        )
