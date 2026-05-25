"""Tests for V2 §1D BreadthV2Features.bias_warnings.

Pins the optional ``bias_warnings`` metadata field on ``BreadthV2Features`` and
the public ``make_bias_warnings_frame`` schema helper. The sector-breadth path
returns ``bias_warnings=None``; PIT-derived breadth features emit the
``survivorship_biased_constituent_universe`` warning.

Spec refs:
    docs/regime_engine_v2_spec.md §1D PIT breadth bias-warning seam.
    Implementation Ambiguity Log #54–#59.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.breadth_state_v2 import (
    BreadthV2Features,
    compute_breadth_v2_features,
    make_bias_warnings_frame,
)
from regime_detection.config import BreadthV2Config
from regime_detection.fragility_universe import SECTOR_ETFS

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _sector_closes_all_positive(n: int = 60) -> dict[str, pd.Series]:
    """Real 11-sector input with monotone-rising closes (all returns > 0)."""
    index = pd.bdate_range(end="2024-12-31", periods=n)
    arr = 100.0 * (1.0 + np.arange(n) * 0.001)
    return {sym: pd.Series(arr, index=index, name=sym) for sym in SECTOR_ETFS}


def _canonical_warning_row(
    *,
    feature_name: str = "pct_above_50dma",
) -> dict:
    return {
        "warning_code": "survivorship_biased_constituent_universe",
        "feature_name": feature_name,
        "source": "fja05680/sp500",
        "source_url": "https://github.com/fja05680/sp500",
    }


# -----------------------------------------------------------------------------
# BreadthV2Features.bias_warnings field
# -----------------------------------------------------------------------------


def test_breadth_v2_features_defaults_bias_warnings_to_none() -> None:
    sector_breadth = pd.Series(
        [0.5, 0.6, 0.7],
        index=pd.bdate_range("2024-12-27", periods=3),
        name="sector_breadth",
    )
    out = BreadthV2Features(sector_breadth=sector_breadth)
    assert out.bias_warnings is None


def test_breadth_v2_features_accepts_bias_warnings_df() -> None:
    sector_breadth = pd.Series(
        [0.5, 0.6, 0.7],
        index=pd.bdate_range("2024-12-27", periods=3),
        name="sector_breadth",
    )
    bias_df = pd.DataFrame(
        [_canonical_warning_row(feature_name="pct_above_50dma")],
        columns=["warning_code", "feature_name", "source", "source_url"],
    )
    out = BreadthV2Features(sector_breadth=sector_breadth, bias_warnings=bias_df)
    # Stored verbatim — no copy/reindex.
    assert out.bias_warnings is bias_df
    pd.testing.assert_frame_equal(out.bias_warnings, bias_df)


def test_feature_names_excludes_bias_warnings() -> None:
    sector_breadth = pd.Series(
        [0.5, 0.6],
        index=pd.bdate_range("2024-12-30", periods=2),
        name="sector_breadth",
    )
    bias_df = pd.DataFrame(
        [_canonical_warning_row()],
        columns=["warning_code", "feature_name", "source", "source_url"],
    )
    out_none = BreadthV2Features(sector_breadth=sector_breadth)
    out_with = BreadthV2Features(sector_breadth=sector_breadth, bias_warnings=bias_df)
    assert out_none.feature_names == ("sector_breadth",)
    assert out_with.feature_names == out_none.feature_names


def test_to_frame_omits_bias_warnings_column() -> None:
    sector_breadth = pd.Series(
        [0.5, 0.6],
        index=pd.bdate_range("2024-12-30", periods=2),
        name="sector_breadth",
    )
    bias_df = pd.DataFrame(
        [_canonical_warning_row()],
        columns=["warning_code", "feature_name", "source", "source_url"],
    )
    out = BreadthV2Features(sector_breadth=sector_breadth, bias_warnings=bias_df)
    assert out.to_frame().columns.tolist() == list(out.feature_names)


def test_compute_breadth_v2_features_emits_available_sector_proxy_warning() -> None:
    """The available-denominator backtest proxy is explicit metadata; it does
    not replace strict sector_breadth."""
    config = BreadthV2Config(sector_breadth_lookback_days=21)
    out = compute_breadth_v2_features(
        sector_etf_closes=_sector_closes_all_positive(n=60),
        config=config,
    )
    assert out.bias_warnings is not None
    assert out.bias_warnings["warning_code"].tolist() == [
        "available_sector_breadth_proxy"
    ]
    assert out.bias_warnings["feature_name"].tolist() == ["available_sector_breadth"]


# -----------------------------------------------------------------------------
# make_bias_warnings_frame helper
# -----------------------------------------------------------------------------


def test_make_bias_warnings_frame_builds_canonical_4_column_df() -> None:
    row = _canonical_warning_row(feature_name="pct_above_50dma")
    df = make_bias_warnings_frame([row])
    assert df.columns.tolist() == [
        "warning_code",
        "feature_name",
        "source",
        "source_url",
    ]
    assert len(df) == 1
    assert df.iloc[0]["warning_code"] == "survivorship_biased_constituent_universe"
    assert df.iloc[0]["feature_name"] == "pct_above_50dma"
    assert df.iloc[0]["source"] == "fja05680/sp500"
    assert df.iloc[0]["source_url"] == "https://github.com/fja05680/sp500"


def test_make_bias_warnings_frame_accepts_multiple_rows() -> None:
    rows = [
        _canonical_warning_row(feature_name="pct_above_50dma"),
        _canonical_warning_row(feature_name="ad_line"),
    ]
    df = make_bias_warnings_frame(rows)
    assert len(df) == 2
    assert df["feature_name"].tolist() == ["pct_above_50dma", "ad_line"]


def test_make_bias_warnings_frame_raises_on_missing_key() -> None:
    bad = {
        "warning_code": "survivorship_biased_constituent_universe",
        "feature_name": "pct_above_50dma",
        "source": "fja05680/sp500",
        # source_url intentionally omitted
    }
    with pytest.raises(ValueError, match="source_url"):
        make_bias_warnings_frame([bad])


def test_make_bias_warnings_frame_raises_on_extra_key() -> None:
    bad = {
        **_canonical_warning_row(),
        "severity": "high",  # unexpected key
    }
    with pytest.raises(ValueError, match="severity"):
        make_bias_warnings_frame([bad])


def test_make_bias_warnings_frame_empty_input_returns_empty_frame() -> None:
    df = make_bias_warnings_frame([])
    assert len(df) == 0
    assert df.columns.tolist() == [
        "warning_code",
        "feature_name",
        "source",
        "source_url",
    ]
