from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import TypeAlias

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue

TRACE_ID_HEADER = "X-Trace-ID"
JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"]
_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "regime_trace_id", default=None
)
_METRICS: contextvars.ContextVar["MetricsCollector | None"] = contextvars.ContextVar(
    "regime_metrics", default=None
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _json_log(
    logger: logging.Logger, level: int, event: str, **payload: object
) -> None:
    body = {"event": event, "trace_id": current_trace_id(), **payload}
    logger.log(level, json.dumps(body, sort_keys=True, default=str))


def current_trace_id() -> str | None:
    trace_id = _TRACE_ID.get()
    if trace_id:
        return trace_id
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if not ctx.is_valid:
        return None
    return f"{ctx.trace_id:032x}"


def start_trace(trace_id: str | None = None) -> str:
    resolved = trace_id or uuid.uuid4().hex
    _TRACE_ID.set(resolved)
    return resolved


def clear_trace() -> None:
    _TRACE_ID.set(None)


@dataclass
class MetricsCollector:
    counters: dict[str, int]
    timings_ms: dict[str, list[float]]

    def __init__(self) -> None:
        self.counters = {}
        self.timings_ms = {}

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def observe_ms(self, name: str, duration_ms: float) -> None:
        self.timings_ms.setdefault(name, []).append(round(duration_ms, 3))

    def snapshot(self) -> dict[str, JsonValue]:
        summary: dict[str, JsonValue] = {
            "counters": dict(sorted(self.counters.items()))
        }
        timings: dict[str, JsonValue] = {}
        for name, values in sorted(self.timings_ms.items()):
            if not values:
                continue
            timings[name] = {
                "count": float(len(values)),
                "min_ms": min(values),
                "max_ms": max(values),
                "avg_ms": round(sum(values) / len(values), 3),
            }
        summary["timings_ms"] = timings
        return summary


def get_metrics_collector() -> MetricsCollector:
    existing = _METRICS.get()
    if existing is not None:
        return existing
    collector = MetricsCollector()
    _METRICS.set(collector)
    return collector


def clear_metrics() -> None:
    _METRICS.set(None)


def configure_error_tracking(
    *, logger: logging.Logger | None = None
) -> dict[str, JsonValue]:
    logger = logger or logging.getLogger(__name__)
    enabled = _env_flag("REGIME_ERROR_TRACKING_ENABLED")
    config: dict[str, JsonValue] = {
        "enabled": enabled,
        "backend": os.environ.get("REGIME_ERROR_TRACKING_BACKEND", "structured_logs"),
        "dsn": os.environ.get("REGIME_ERROR_TRACKING_DSN", "").strip() or None,
        "breadcrumbs": True,
        "user_context_env": os.environ.get("REGIME_OPERATOR_NAME", "").strip() or None,
    }
    _json_log(logger, logging.INFO, "error_tracking_configured", **config)
    return config


def configure_deployment_observability(
    *, logger: logging.Logger | None = None
) -> dict[str, JsonValue]:
    logger = logger or logging.getLogger(__name__)
    config: dict[str, JsonValue] = {
        "dashboard_url": os.environ.get("REGIME_DASHBOARD_URL", "").strip() or None,
        "deploy_log_url": os.environ.get("REGIME_DEPLOY_LOG_URL", "").strip() or None,
        "annotation_sink": os.environ.get("REGIME_DEPLOY_ANNOTATION_SINK", "").strip()
        or None,
    }
    _json_log(logger, logging.INFO, "deployment_observability_configured", **config)
    return config


def configure_product_analytics(
    *, logger: logging.Logger | None = None
) -> dict[str, JsonValue]:
    logger = logger or logging.getLogger(__name__)
    config: dict[str, JsonValue] = {
        "enabled": _env_flag("REGIME_PRODUCT_ANALYTICS_ENABLED"),
        "backend": os.environ.get(
            "REGIME_PRODUCT_ANALYTICS_BACKEND", "structured_logs"
        ),
        "project": os.environ.get("REGIME_PRODUCT_ANALYTICS_PROJECT", "").strip()
        or None,
        "sample_rate": os.environ.get("REGIME_PRODUCT_ANALYTICS_SAMPLE_RATE", "1.0"),
    }
    _json_log(logger, logging.INFO, "product_analytics_configured", **config)
    return config


def load_feature_flags(*, logger: logging.Logger | None = None) -> dict[str, bool]:
    logger = logger or logging.getLogger(__name__)
    prefix = "REGIME_FEATURE_FLAG_"
    flags = {
        name.removeprefix(prefix).lower(): _env_flag(name)
        for name in sorted(os.environ)
        if name.startswith(prefix)
    }
    _json_log(logger, logging.INFO, "feature_flags_loaded", flags=flags)
    return flags


def capture_exception(
    error: BaseException,
    *,
    logger: logging.Logger | None = None,
    component: str,
    extra: dict[str, JsonValue] | None = None,
) -> None:
    logger = logger or logging.getLogger(__name__)
    payload: dict[str, JsonValue] = {
        "component": component,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if extra:
        payload["extra"] = extra
    _json_log(logger, logging.ERROR, "captured_exception", **payload)
    get_metrics_collector().increment("exceptions_total")


def tracer(name: str) -> Tracer:
    return trace.get_tracer(name)


def start_span(
    tracer_instance: Tracer,
    name: str,
    *,
    attributes: dict[str, AttributeValue] | None = None,
) -> Span:
    span = tracer_instance.start_span(name)
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    trace_id = current_trace_id()
    if trace_id:
        span.set_attribute("regime.trace_id", trace_id)
    return span


def record_timing(name: str, start_time: float) -> None:
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    get_metrics_collector().observe_ms(name, elapsed_ms)
