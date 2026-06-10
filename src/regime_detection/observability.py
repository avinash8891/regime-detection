from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeAlias, cast

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue


def _load_sentry_bindings() -> tuple[Any | None, type[Any] | None]:
    try:
        import sentry_sdk as sentry_sdk_module  # pyright: ignore[reportMissingImports]
        from sentry_sdk.integrations.logging import (  # pyright: ignore[reportMissingImports]
            LoggingIntegration as logging_integration,
        )
    except ImportError:  # pragma: no cover - exercised in runtime packaging
        return None, None
    return sentry_sdk_module, logging_integration


_SENTRY_SDK, _SENTRY_LOGGING_INTEGRATION = _load_sentry_bindings()

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


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float between 0.0 and 1.0") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


def _release_name() -> str:
    explicit = os.environ.get("REGIME_RELEASE", "").strip()
    if explicit:
        return explicit
    try:
        return f"regime-detection@{version('regime-detection')}"
    except PackageNotFoundError:
        return "regime-detection@unknown"


def _operator_name() -> str:
    return os.environ.get("REGIME_OPERATOR_NAME", "").strip() or "unknown"


def _before_send_sentry_event(
    event: dict[str, object], _hint: dict[str, object]
) -> dict[str, object]:
    request_obj = event.get("request")
    if isinstance(request_obj, dict):
        request = cast(dict[str, Any], request_obj)
        headers = request.get("headers")
        if isinstance(headers, dict):
            header_map = cast(dict[str, object], headers)
            sanitized: dict[str, object] = {
                key: (
                    "[redacted]"
                    if key.lower() in {"authorization", "cookie", "x-api-key"}
                    else value
                )
                for key, value in header_map.items()
            }
            request["headers"] = sanitized
    return event


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
    backend = os.environ.get("REGIME_ERROR_TRACKING_BACKEND", "structured_logs")
    dsn = os.environ.get("REGIME_ERROR_TRACKING_DSN", "").strip() or None
    environment = os.environ.get("REGIME_ENVIRONMENT", "").strip() or "development"
    release = _release_name()
    sample_rate = 1.0
    traces_sample_rate = 0.0
    user_context_env = _operator_name()
    sentry_sdk = _SENTRY_SDK
    logging_integration_cls = _SENTRY_LOGGING_INTEGRATION
    should_initialize_sentry = (
        enabled
        and backend == "sentry"
        and dsn is not None
        and sentry_sdk is not None
        and logging_integration_cls is not None
    )
    if should_initialize_sentry:
        sample_rate = _env_float("REGIME_ERROR_TRACKING_SAMPLE_RATE", sample_rate)
        traces_sample_rate = _env_float(
            "REGIME_ERROR_TRACKING_TRACES_SAMPLE_RATE",
            traces_sample_rate,
        )
    config: dict[str, JsonValue] = {
        "enabled": enabled,
        "backend": backend,
        "dsn": dsn,
        "environment": environment,
        "release": release,
        "sample_rate": sample_rate,
        "traces_sample_rate": traces_sample_rate,
        "breadcrumbs": True,
        "user_context_env": user_context_env,
        "initialized": False,
    }
    if should_initialize_sentry:
        if sentry_sdk is None or logging_integration_cls is None:
            raise RuntimeError("sentry backend selected without sentry bindings")
        logging_integration = logging_integration_cls(
            level=logging.INFO,
            event_level=logging.ERROR,
        )
        sentry_sdk.init(
            dsn=dsn,
            release=release,
            environment=environment,
            sample_rate=sample_rate,
            traces_sample_rate=traces_sample_rate,
            send_default_pii=False,
            max_breadcrumbs=50,
            attach_stacktrace=True,
            include_local_variables=False,
            enable_tracing=bool(traces_sample_rate > 0.0),
            integrations=[logging_integration],
            before_send=_before_send_sentry_event,
        )
        if hasattr(sentry_sdk, "set_user"):
            sentry_sdk.set_user({"username": user_context_env})
        trace_id = current_trace_id()
        if trace_id and hasattr(sentry_sdk, "set_tag"):
            sentry_sdk.set_tag("trace_id", trace_id)
        config["initialized"] = True
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
    if _SENTRY_SDK is not None and _SENTRY_SDK.is_initialized():
        if hasattr(_SENTRY_SDK, "set_tag"):
            _SENTRY_SDK.set_tag("component", component)
        trace_id = current_trace_id()
        if trace_id and hasattr(_SENTRY_SDK, "set_tag"):
            _SENTRY_SDK.set_tag("trace_id", trace_id)
        if hasattr(_SENTRY_SDK, "set_user"):
            _SENTRY_SDK.set_user({"username": _operator_name()})
        if extra and hasattr(_SENTRY_SDK, "set_context"):
            _SENTRY_SDK.set_context("regime_extra", extra)
        if hasattr(_SENTRY_SDK, "set_context"):
            _SENTRY_SDK.set_context(
                "regime_error",
                {
                    "component": component,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        _SENTRY_SDK.capture_exception(error)


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
