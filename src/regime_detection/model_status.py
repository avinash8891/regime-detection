from __future__ import annotations

from typing import Literal

DataQualityStatus = Literal[
    "ok", "degraded", "insufficient_data", "insufficient_history", "stale_data"
]
ClassificationStatus = Literal[
    "classified",
    "no_rule_fired",
    "no_rule_fired_hysteresis",
    "no_rule_fired_missing_feature",
    "data_unavailable",
    "stale_data",
    "insufficient_history",
    "not_wired",
]
