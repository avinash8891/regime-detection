from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import regime_detection.observability as observability_module
from regime_detection.loaders import load_event_calendar
from regime_detection.observability import (
    TRACE_ID_HEADER,
    capture_exception,
    clear_metrics,
    clear_trace,
    configure_deployment_observability,
    configure_error_tracking,
    configure_product_analytics,
    current_trace_id,
    get_metrics_collector,
    load_feature_flags,
    start_trace,
)


@pytest.fixture(autouse=True)
def _reset_observability() -> None:
    clear_trace()
    clear_metrics()


def test_start_trace_sets_current_trace_id() -> None:
    trace_id = start_trace("abc123")

    assert trace_id == "abc123"
    assert current_trace_id() == "abc123"


def test_load_event_calendar_records_metrics() -> None:
    start_trace("trace-load-events")
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "market": ["US"],
            "type": ["FOMC"],
            "importance": ["high"],
        }
    )

    loaded = load_event_calendar(frame)
    metrics = get_metrics_collector().snapshot()

    assert len(loaded) == 1
    assert "load_event_calendar" in metrics["timings_ms"]


def test_capture_exception_emits_structured_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    start_trace("trace-capture")
    logger = logging.getLogger("regime_detection.tests")

    with caplog.at_level(logging.ERROR):
        capture_exception(
            ValueError("boom"),
            logger=logger,
            component="unit_test",
            extra={"phase": "exercise"},
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "captured_exception"
    assert payload["trace_id"] == "trace-capture"
    assert payload["component"] == "unit_test"
    assert payload["extra"] == {"phase": "exercise"}


def test_configure_error_tracking_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIME_ERROR_TRACKING_ENABLED", "true")
    monkeypatch.setenv("REGIME_ERROR_TRACKING_BACKEND", "sentry")
    monkeypatch.setenv("REGIME_ERROR_TRACKING_DSN", "https://dsn.example")
    monkeypatch.setenv("REGIME_OPERATOR_NAME", "operator-a")
    monkeypatch.setenv("REGIME_ENVIRONMENT", "test")
    monkeypatch.setenv("REGIME_RELEASE", "regime-detection@test")

    sentry_init_calls: list[dict[str, object]] = []

    class FakeLoggingIntegration:
        def __init__(self, *, level: int, event_level: int) -> None:
            self.level = level
            self.event_level = event_level

    fake_sentry = SimpleNamespace(
        init=lambda **kwargs: sentry_init_calls.append(kwargs),
        is_initialized=lambda: True,
    )
    monkeypatch.setattr(observability_module, "_SENTRY_SDK", fake_sentry, raising=False)
    monkeypatch.setattr(
        observability_module,
        "_SENTRY_LOGGING_INTEGRATION",
        FakeLoggingIntegration,
        raising=False,
    )

    config = configure_error_tracking()

    assert config["enabled"] is True
    assert config["backend"] == "sentry"
    assert config["dsn"] == "https://dsn.example"
    assert config["user_context_env"] == "operator-a"
    assert config["initialized"] is True
    assert sentry_init_calls[0]["dsn"] == "https://dsn.example"
    assert sentry_init_calls[0]["environment"] == "test"
    assert sentry_init_calls[0]["release"] == "regime-detection@test"
    assert sentry_init_calls[0]["include_local_variables"] is False


def test_configure_error_tracking_ignores_bad_sentry_rates_when_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIME_ERROR_TRACKING_ENABLED", "false")
    monkeypatch.setenv("REGIME_ERROR_TRACKING_BACKEND", "structured_logs")
    monkeypatch.setenv("REGIME_ERROR_TRACKING_SAMPLE_RATE", "not-a-float")
    monkeypatch.setenv("REGIME_ERROR_TRACKING_TRACES_SAMPLE_RATE", "also-bad")

    config = configure_error_tracking()

    assert config["initialized"] is False
    assert config["sample_rate"] == 1.0
    assert config["traces_sample_rate"] == 0.0


def test_capture_exception_forwards_context_to_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry_calls: list[tuple[str, object]] = []

    fake_sentry = SimpleNamespace(
        is_initialized=lambda: True,
        set_tag=lambda key, value: sentry_calls.append(("tag", (key, value))),
        set_user=lambda user: sentry_calls.append(("user", user)),
        set_context=lambda key, value: sentry_calls.append(("context", (key, value))),
        capture_exception=lambda error: sentry_calls.append(("exception", error)),
    )
    monkeypatch.setattr(observability_module, "_SENTRY_SDK", fake_sentry, raising=False)

    start_trace("trace-sentry")
    capture_exception(
        RuntimeError("boom"),
        component="profile_engine",
        extra={"manifest": "manifests/runs/regime.yaml"},
    )

    assert ("tag", ("component", "profile_engine")) in sentry_calls
    assert ("tag", ("trace_id", "trace-sentry")) in sentry_calls
    assert ("user", {"username": "unknown"}) in sentry_calls
    assert (
        "context",
        (
            "regime_extra",
            {"manifest": "manifests/runs/regime.yaml"},
        ),
    ) in sentry_calls
    assert any(kind == "exception" for kind, _ in sentry_calls)


def test_configure_deployment_observability_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIME_DASHBOARD_URL", "https://grafana.example/dash")
    monkeypatch.setenv("REGIME_DEPLOY_LOG_URL", "https://deploy.example/run/1")
    monkeypatch.setenv("REGIME_DEPLOY_ANNOTATION_SINK", "grafana")

    config = configure_deployment_observability()

    assert config["dashboard_url"] == "https://grafana.example/dash"
    assert config["deploy_log_url"] == "https://deploy.example/run/1"
    assert config["annotation_sink"] == "grafana"


def test_configure_product_analytics_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIME_PRODUCT_ANALYTICS_ENABLED", "true")
    monkeypatch.setenv("REGIME_PRODUCT_ANALYTICS_BACKEND", "posthog")
    monkeypatch.setenv("REGIME_PRODUCT_ANALYTICS_PROJECT", "regime-prod")
    monkeypatch.setenv("REGIME_PRODUCT_ANALYTICS_SAMPLE_RATE", "0.25")

    config = configure_product_analytics()

    assert config["enabled"] is True
    assert config["backend"] == "posthog"
    assert config["project"] == "regime-prod"
    assert config["sample_rate"] == "0.25"


def test_load_feature_flags_reads_prefixed_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIME_FEATURE_FLAG_SHADOW_MODE", "true")
    monkeypatch.setenv("REGIME_FEATURE_FLAG_EXPERIMENTAL_RISK", "0")

    flags = load_feature_flags()

    assert flags == {"experimental_risk": False, "shadow_mode": True}


def test_trace_header_constant_is_stable() -> None:
    assert TRACE_ID_HEADER == "X-Trace-ID"
