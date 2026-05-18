from __future__ import annotations

import contextlib
from typing import Any, Callable


def _timed_wrapper(
    timer: Any,
    stage_name: str,
    func: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with timer.measure(stage_name):
            return func(*args, **kwargs)

    return wrapped


def _timed_method_wrapper(
    timer: Any,
    stage_name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with timer.measure(stage_name):
            return method(*args, **kwargs)

    return wrapped


def _timed_inflation_growth_builder(
    timer: Any,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(
        context: Any,
        feature_store: Any,
        credit_funding_active_labels_by_date: Any = None,
    ) -> Any:
        import regime_detection.axis_builders.inflation_growth as inflation_growth_builder

        original_assess = inflation_growth_builder.assess_series_input_quality
        original_build_inputs = (
            inflation_growth_builder.build_inflation_growth_rule_inputs_by_date
        )
        original_eval = inflation_growth_builder.evaluate_inflation_growth_rules

        def timed_assess(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(
                "axis_series.inflation_growth.assess_series_input_quality"
            ):
                return original_assess(*args, **kwargs)

        def timed_build_inputs(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(
                "axis_series.inflation_growth.build_rule_inputs_by_date"
            ):
                return original_build_inputs(*args, **kwargs)

        def timed_eval(*args: Any, **kwargs: Any) -> Any:
            with timer.measure("axis_series.inflation_growth.evaluate_rules"):
                return original_eval(*args, **kwargs)

        with timer.measure("axis_series.inflation_growth"):
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _patched_attr(
                        inflation_growth_builder,
                        "assess_series_input_quality",
                        timed_assess,
                    )
                )
                stack.enter_context(
                    _patched_attr(
                        inflation_growth_builder,
                        "build_inflation_growth_rule_inputs_by_date",
                        timed_build_inputs,
                    )
                )
                stack.enter_context(
                    _patched_attr(
                        inflation_growth_builder,
                        "evaluate_inflation_growth_rules",
                        timed_eval,
                    )
                )
                return method(
                    context,
                    feature_store,
                    credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
                )

    return wrapped


@contextlib.contextmanager
def _patched_attr(module: Any, attr_name: str, replacement: Any):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


@contextlib.contextmanager
def install_timers(timer: Any):
    import regime_detection.axis_series as axis_series_module
    import regime_detection.engine as engine_module
    import regime_detection.feature_store as feature_store_module
    import regime_detection.timeline as timeline_module

    patches = [
        (
            engine_module,
            "build_market_context",
            _timed_wrapper(
                timer, "build_market_context", engine_module.build_market_context
            ),
        ),
        (
            engine_module,
            "build_regime_timeline",
            _timed_wrapper(
                timer,
                "build_regime_timeline_total",
                engine_module.build_regime_timeline,
            ),
        ),
        (
            timeline_module,
            "slice_context_to_recent_sessions",
            _timed_wrapper(
                timer,
                "slice_context_to_recent_sessions",
                timeline_module.slice_context_to_recent_sessions,
            ),
        ),
        (
            timeline_module,
            "build_feature_store",
            _timed_wrapper(
                timer, "build_feature_store_total", timeline_module.build_feature_store
            ),
        ),
        (
            timeline_module,
            "build_axis_series_bundle",
            _timed_wrapper(
                timer,
                "build_axis_series_bundle",
                timeline_module.build_axis_series_bundle,
            ),
        ),
        (
            timeline_module,
            "build_transition_risk_series",
            _timed_wrapper(
                timer,
                "build_transition_risk_series",
                timeline_module.build_transition_risk_series,
            ),
        ),
        (
            feature_store_module,
            "compute_network_fragility_features",
            _timed_wrapper(
                timer,
                "feature_store.network_fragility",
                feature_store_module.compute_network_fragility_features,
            ),
        ),
        (
            feature_store_module,
            "compute_trend_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.trend_direction_v2",
                feature_store_module.compute_trend_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_volatility_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.volatility_state_v2",
                feature_store_module.compute_volatility_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_breadth_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.breadth_state_v2",
                feature_store_module.compute_breadth_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_volume_liquidity_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.volume_liquidity_v2",
                feature_store_module.compute_volume_liquidity_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_monetary_pressure_features",
            _timed_wrapper(
                timer,
                "feature_store.monetary_pressure_v2",
                feature_store_module.compute_monetary_pressure_features,
            ),
        ),
        (
            feature_store_module,
            "compute_credit_funding_features",
            _timed_wrapper(
                timer,
                "feature_store.credit_funding",
                feature_store_module.compute_credit_funding_features,
            ),
        ),
        (
            feature_store_module,
            "compute_inflation_growth_features",
            _timed_wrapper(
                timer,
                "feature_store.inflation_growth",
                feature_store_module.compute_inflation_growth_features,
            ),
        ),
        (
            feature_store_module,
            "compute_hmm_features",
            _timed_wrapper(
                timer, "feature_store.hmm", feature_store_module.compute_hmm_features
            ),
        ),
        (
            feature_store_module,
            "compute_clustering_features",
            _timed_wrapper(
                timer,
                "feature_store.gmm_clustering",
                feature_store_module.compute_clustering_features,
            ),
        ),
        (
            feature_store_module,
            "compute_change_point_features",
            _timed_wrapper(
                timer,
                "feature_store.change_point",
                feature_store_module.compute_change_point_features,
            ),
        ),
        (
            axis_series_module,
            "build_trend_direction_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.trend_direction",
                axis_series_module.build_trend_direction_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_trend_character_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.trend_character",
                axis_series_module.build_trend_character_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_volatility_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.volatility_state",
                axis_series_module.build_volatility_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_breadth_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.breadth_state",
                axis_series_module.build_breadth_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_credit_funding_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.credit_funding",
                axis_series_module.build_credit_funding_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_network_fragility_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.network_fragility",
                axis_series_module.build_network_fragility_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_volume_liquidity_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.volume_liquidity_state",
                axis_series_module.build_volume_liquidity_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_monetary_pressure_axis_series",
            _timed_method_wrapper(
                timer,
                "axis_series.monetary_pressure_state",
                axis_series_module.build_monetary_pressure_axis_series,
            ),
        ),
        (
            axis_series_module,
            "build_inflation_growth_axis_series",
            _timed_inflation_growth_builder(
                timer, axis_series_module.build_inflation_growth_axis_series
            ),
        ),
        (
            axis_series_module,
            "build_event_calendar_series",
            _timed_wrapper(
                timer,
                "axis_series.event_calendar",
                axis_series_module.build_event_calendar_series,
            ),
        ),
    ]
    with contextlib.ExitStack() as stack:
        for module, attr_name, replacement in patches:
            stack.enter_context(_patched_attr(module, attr_name, replacement))
        yield
