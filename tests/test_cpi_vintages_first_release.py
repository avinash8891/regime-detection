"""Tests for v2 §2A first-release CPI for historical replay (audit M2).

Spec authority: docs/regime_engine_v2_spec.md §2A lines 2587-2593.
Resolution: docs/spec_code_data_audit_2026_05_15.md §3.2.

Per CLAUDE.md: use real FRED series id (CPIAUCSL) and real column names
that ``regime_data_fetch.fred`` writes when ``--include-cpi-vintages``
is enabled. Test the integration path (loader → MarketContext →
inflation_growth.compute_inflation_growth_features), not just the unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.config import InflationGrowthRulesConfig
from regime_detection.inflation_growth import (
    FIRST_RELEASE_CPI_PROVENANCE_CODE,
    compute_inflation_growth_features,
)
from regime_detection.loaders import load_cpi_vintages_first_release


def _build_vintage_frame() -> pd.DataFrame:
    """Build a realistic mini vintage frame: two reference months, each
    with a first release and a later revision.

    Schema mirrors ``regime_data_fetch.fred`` output for CPIAUCSL with
    ``--include-cpi-vintages``:
        - ``date``: reference date (1st of the reference month)
        - ``value``: published level for that vintage
        - ``realtime_start``: date the value first became public
        - ``realtime_end``: date the value was superseded (or NaT)
        - ``series_id``: "CPIAUCSL"
    """
    return pd.DataFrame(
        [
            # January 2024 CPI — first release on 2024-02-13, revised on 2024-03-12.
            {
                "date": "2024-01-01",
                "value": 308.5,
                "realtime_start": "2024-02-13",
                "realtime_end": "2024-03-11",
                "series_id": "CPIAUCSL",
            },
            {
                "date": "2024-01-01",
                "value": 308.7,
                "realtime_start": "2024-03-12",
                "realtime_end": pd.NaT,
                "series_id": "CPIAUCSL",
            },
            # February 2024 CPI — single release (no revision yet).
            {
                "date": "2024-02-01",
                "value": 309.7,
                "realtime_start": "2024-03-12",
                "realtime_end": pd.NaT,
                "series_id": "CPIAUCSL",
            },
        ]
    )


def test_load_cpi_vintages_first_release_picks_earliest_realtime_start() -> None:
    """The first-release loader must pick the EARLIEST realtime_start per
    reference date, not the most recent."""
    out = load_cpi_vintages_first_release(_build_vintage_frame())
    # Series keyed by RELEASE DATE (realtime_start), not reference date.
    assert pd.Timestamp("2024-02-13") in out.index
    assert pd.Timestamp("2024-03-12") in out.index
    # 2024-01 reference: first release was 2024-02-13 with value 308.5
    # (NOT the revised 308.7 published on 2024-03-12).
    assert out.loc[pd.Timestamp("2024-02-13")] == 308.5
    # 2024-02 reference: only release was 2024-03-12 with value 309.7.
    assert out.loc[pd.Timestamp("2024-03-12")] == 309.7


def test_load_cpi_vintages_first_release_deduplicates_release_dates() -> None:
    """FRED/ALFRED history can contain multiple reference months whose first
    available vintage shares the same realtime_start. The as-of replay series
    must still have a unique release-date index so pandas reindex/ffill works."""
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "value": 308.5,
                "realtime_start": "2024-02-13",
                "realtime_end": pd.NaT,
                "series_id": "CPIAUCSL",
            },
            {
                "date": "2024-02-01",
                "value": 309.7,
                "realtime_start": "2024-02-13",
                "realtime_end": pd.NaT,
                "series_id": "CPIAUCSL",
            },
        ]
    )

    out = load_cpi_vintages_first_release(frame)

    assert out.index.tolist() == [pd.Timestamp("2024-02-13")]
    assert out.tolist() == [309.7]


def test_load_cpi_vintages_first_release_empty_returns_empty_series() -> None:
    empty = pd.DataFrame(columns=["date", "value", "realtime_start"])
    out = load_cpi_vintages_first_release(empty)
    assert out.empty


def test_load_cpi_vintages_first_release_missing_columns_raises() -> None:
    bad = pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]})
    try:
        load_cpi_vintages_first_release(bad)
    except ValueError as exc:
        assert "realtime_start" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing realtime_start")


def _spy_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=periods, freq="B")


def test_inflation_growth_features_uses_first_release_when_flag_set() -> None:
    """End-to-end smoke: the bias-warnings frame must surface the
    `cpi_first_release_vintage_replay` provenance row when the
    first-release substitution is in effect."""
    spy_index = _spy_index(200)
    # Build the standard inputs at realistic levels.
    cpi_revised = pd.Series(
        np.linspace(300.0, 310.0, len(spy_index)), index=spy_index, name="cpi_all_items"
    )
    cpi_first_release = pd.Series(
        np.linspace(299.5, 309.0, len(spy_index)),
        index=spy_index,
        name="cpi_first_release",
    )
    pmi = pd.Series(52.0, index=spy_index, name="pmi_manufacturing")
    dgs10 = pd.Series(4.0, index=spy_index, name="10y_yield")
    dbc_close = pd.Series(
        np.linspace(20.0, 22.0, len(spy_index)), index=spy_index, name="DBC"
    )
    spy_close = pd.Series(
        np.linspace(450.0, 470.0, len(spy_index)), index=spy_index, name="SPY"
    )
    tlt_close = pd.Series(
        np.linspace(95.0, 92.0, len(spy_index)), index=spy_index, name="TLT"
    )
    xly_close = pd.Series(180.0, index=spy_index, name="XLY")
    xli_close = pd.Series(120.0, index=spy_index, name="XLI")
    xlp_close = pd.Series(80.0, index=spy_index, name="XLP")
    xlu_close = pd.Series(70.0, index=spy_index, name="XLU")

    config = InflationGrowthRulesConfig()
    out = compute_inflation_growth_features(
        cpi_all_items=cpi_revised,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc_close,
        spy_close=spy_close,
        tlt_close=tlt_close,
        xly_close=xly_close,
        xli_close=xli_close,
        xlp_close=xlp_close,
        xlu_close=xlu_close,
        config=config,
        cpi_first_release=cpi_first_release,
        use_first_release_cpi_when_available=True,
    )
    # Bias-warning row must be present for the three CPI-derived
    # features (cpi_3m_change_pct, cpi_6m_change_pct, inflation_surprise_zscore).
    warning_codes = set(out.bias_warnings["warning_code"].tolist())
    assert FIRST_RELEASE_CPI_PROVENANCE_CODE in warning_codes
    cpi_feature_rows = out.bias_warnings[
        out.bias_warnings["warning_code"] == FIRST_RELEASE_CPI_PROVENANCE_CODE
    ]
    assert set(cpi_feature_rows["feature_name"]) == {
        "cpi_3m_change_pct",
        "cpi_6m_change_pct",
        "inflation_surprise_zscore",
    }


def test_inflation_growth_features_preserves_revised_path_when_flag_off() -> None:
    """V1 byte-identity: when the flag is False the existing revised
    CPIAUCSL path is preserved unchanged (no bias warning emitted)."""
    spy_index = _spy_index(200)
    cpi_revised = pd.Series(
        np.linspace(300.0, 310.0, len(spy_index)), index=spy_index, name="cpi_all_items"
    )
    cpi_first_release = pd.Series(
        np.linspace(299.5, 309.0, len(spy_index)),
        index=spy_index,
        name="cpi_first_release",
    )
    pmi = pd.Series(52.0, index=spy_index, name="pmi_manufacturing")
    dgs10 = pd.Series(4.0, index=spy_index, name="10y_yield")
    dbc_close = pd.Series(
        np.linspace(20.0, 22.0, len(spy_index)), index=spy_index, name="DBC"
    )
    spy_close = pd.Series(
        np.linspace(450.0, 470.0, len(spy_index)), index=spy_index, name="SPY"
    )
    tlt_close = pd.Series(
        np.linspace(95.0, 92.0, len(spy_index)), index=spy_index, name="TLT"
    )
    xly_close = pd.Series(180.0, index=spy_index, name="XLY")
    xli_close = pd.Series(120.0, index=spy_index, name="XLI")
    xlp_close = pd.Series(80.0, index=spy_index, name="XLP")
    xlu_close = pd.Series(70.0, index=spy_index, name="XLU")

    out = compute_inflation_growth_features(
        cpi_all_items=cpi_revised,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc_close,
        spy_close=spy_close,
        tlt_close=tlt_close,
        xly_close=xly_close,
        xli_close=xli_close,
        xlp_close=xlp_close,
        xlu_close=xlu_close,
        config=InflationGrowthRulesConfig(),
        cpi_first_release=cpi_first_release,
        use_first_release_cpi_when_available=False,
    )
    warning_codes = set(out.bias_warnings["warning_code"].tolist())
    assert FIRST_RELEASE_CPI_PROVENANCE_CODE not in warning_codes


def test_inflation_growth_features_no_vintage_supplied_uses_revised() -> None:
    """When cpi_first_release is None, behavior is the existing revised
    path — V1/V2 byte-identity preserved for callers that don't wire vintages."""
    spy_index = _spy_index(200)
    cpi_revised = pd.Series(
        np.linspace(300.0, 310.0, len(spy_index)), index=spy_index, name="cpi_all_items"
    )
    pmi = pd.Series(52.0, index=spy_index, name="pmi_manufacturing")
    dgs10 = pd.Series(4.0, index=spy_index, name="10y_yield")
    dbc_close = pd.Series(
        np.linspace(20.0, 22.0, len(spy_index)), index=spy_index, name="DBC"
    )
    spy_close = pd.Series(
        np.linspace(450.0, 470.0, len(spy_index)), index=spy_index, name="SPY"
    )
    tlt_close = pd.Series(
        np.linspace(95.0, 92.0, len(spy_index)), index=spy_index, name="TLT"
    )
    xly_close = pd.Series(180.0, index=spy_index, name="XLY")
    xli_close = pd.Series(120.0, index=spy_index, name="XLI")
    xlp_close = pd.Series(80.0, index=spy_index, name="XLP")
    xlu_close = pd.Series(70.0, index=spy_index, name="XLU")

    out = compute_inflation_growth_features(
        cpi_all_items=cpi_revised,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc_close,
        spy_close=spy_close,
        tlt_close=tlt_close,
        xly_close=xly_close,
        xli_close=xli_close,
        xlp_close=xlp_close,
        xlu_close=xlu_close,
        config=InflationGrowthRulesConfig(),
    )
    warning_codes = set(out.bias_warnings["warning_code"].tolist())
    assert FIRST_RELEASE_CPI_PROVENANCE_CODE not in warning_codes
