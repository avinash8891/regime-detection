from __future__ import annotations

import math

import pandas as pd

from regime_detection._rule_helpers import scalar_at


def test_scalar_at_duplicate_index_returns_nan_instead_of_crashing() -> None:
    dt = pd.Timestamp("2024-01-02")
    series = pd.Series([1.0, 2.0], index=[dt, dt])

    assert math.isnan(scalar_at(series, dt))
