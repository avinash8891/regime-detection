"""V1 and shared regime configuration import surface.

This module intentionally re-exports objects from ``regime_detection.config`` so
the long-standing compatibility surface remains the single source of truth while
callers that only need the V1/shared schema can import a smaller namespace.
"""

from __future__ import annotations

from regime_detection.config import (
    DataQualityConfig,
    EarningsSeasonConfig,
    ETFProxyConfig,
    EventCalendarConfig,
    ExpiryRulesConfig,
    HysteresisConfig,
    MonthlyOptionsExpiryRuleConfig,
    RegimeConfig,
    _default_config_resource_name_for_version,
    load_default_regime_config,
    load_regime_config,
)

__all__ = (
    "HysteresisConfig",
    "DataQualityConfig",
    "EventCalendarConfig",
    "ETFProxyConfig",
    "MonthlyOptionsExpiryRuleConfig",
    "ExpiryRulesConfig",
    "EarningsSeasonConfig",
    "RegimeConfig",
    "load_regime_config",
    "_default_config_resource_name_for_version",
    "load_default_regime_config",
)
