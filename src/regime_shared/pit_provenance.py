"""Point-in-time provenance schemas: bias-warning columns and source identifiers.

Single home for the canonical 4-column bias-warning DataFrame schema used
cross-axis (breadth, inflation_growth, credit_funding) to record data-quality
caveats alongside emitted features.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
SOURCE_NAME = "fja05680/sp500"
# Downstream quality reports grep this exact token; change it only with a
# matching migration for stored bias-warning artifacts.
BIAS_WARNING = "survivorship_biased_constituent_universe"

# Canonical column order for bias-warnings DataFrames. Consumed by the
# `make_bias_warnings_frame` builder below and used to validate rows from
# every cross-axis caller.
_BIAS_WARNING_COLUMNS = ("warning_code", "feature_name", "source", "source_url")


def make_bias_warnings_frame(rows: Iterable[Mapping[str, str]]) -> pd.DataFrame:
    """Build the canonical 4-column bias-warnings frame.

    Each row must have exactly the keys: warning_code, feature_name,
    source, source_url. Raises ValueError on key mismatch (missing or extra).
    Empty input returns a 4-column DataFrame with 0 rows.

    Cross-axis: consumed by breadth_state_v2, inflation_growth, and
    credit_funding to record provenance caveats alongside features.
    """
    expected = set(_BIAS_WARNING_COLUMNS)
    materialized = []
    for idx, row in enumerate(rows):
        got = set(row)
        if got != expected:
            missing = expected - got
            extra = got - expected
            raise ValueError(
                f"bias_warnings row {idx} key mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        materialized.append({k: row[k] for k in _BIAS_WARNING_COLUMNS})
    if not materialized:
        return pd.DataFrame(
            {col: pd.Series(dtype=object) for col in _BIAS_WARNING_COLUMNS}
        )
    return pd.DataFrame(materialized, columns=list(_BIAS_WARNING_COLUMNS))
