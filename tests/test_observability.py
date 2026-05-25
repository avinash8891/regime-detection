from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from regime_detection.loaders import load_event_calendar
from regime_detection.observability import (
    TRACE_ID_HEADER,
    capture_exception,
    clear_metrics,
    clear_trace,
    configure_deployment_observability,
    configure_error_tracking,
    current_trace_id,
    get_metrics_collector,
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

    config = configure_error_tracking()

    assert config["enabled"] is True
    assert config["backend"] == "sentry"
    assert config["dsn"] == "https://dsn.example"
    assert config["user_context_env"] == "operator-a"


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


def test_trace_header_constant_is_stable() -> None:
    assert TRACE_ID_HEADER == "X-Trace-ID"
