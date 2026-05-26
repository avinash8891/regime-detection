from __future__ import annotations

from importlib.metadata import entry_points

from regime_detection.pytest_runtime_policy import _integration_only_markexpr


def test_integration_only_markexpr_rejects_mixed_marker_expressions() -> None:
    assert _integration_only_markexpr("integration")
    assert _integration_only_markexpr("(integration)")
    assert _integration_only_markexpr("integration and not slow")

    assert not _integration_only_markexpr("integration or unit")
    assert not _integration_only_markexpr("not integration")
    assert not _integration_only_markexpr("integration and unit")


def test_runtime_policy_is_registered_as_pytest_plugin() -> None:
    pytest_plugins = entry_points(group="pytest11")
    assert any(
        plugin.name == "santo_domingo_runtime_policy"
        and plugin.value == "regime_detection.pytest_runtime_policy"
        for plugin in pytest_plugins
    )
