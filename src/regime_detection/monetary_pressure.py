"""v2 §2A Layer 2A Monetary / Liquidity axis — feature compute.

Five z-score series:
  - yield_change_zscore_2y_63d
  - yield_change_zscore_10y_63d
  - broad_usd_index_zscore_63d
  - yield_change_zscore_21d_2y
  - yield_change_zscore_21d_10y

The classify layer (labels, risk rank, rule inputs, precedence walker)
lives in ``monetary_pressure_rules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_detection._rolling_stats import (
    rolling_change_zscore as _rolling_change_zscore,
)
from regime_detection.config import MonetaryPressureV2FeaturesConfig


@dataclass(frozen=True)
class MonetaryPressureV2Features:
    """v2 §2A — per-session continuous monetary-pressure features.

    Five z-score series aligned to the input DatetimeIndex. NaN cold-start
    at the head of each series until the corresponding (lookback +
    normalizer) window fills.
    """

    yield_change_zscore_2y_63d: pd.Series
    yield_change_zscore_10y_63d: pd.Series
    broad_usd_index_zscore_63d: pd.Series
    yield_change_zscore_21d_2y: pd.Series
    yield_change_zscore_21d_10y: pd.Series
    # v2 §2A central-bank-text evidence (implementation decision; §2A "Central Bank Text / Sentiment" subsection). Daily
    # forward-filled, smoothed net_score (hawkish - dovish) / total in
    # [-1, +1]. Evidence-only — never consumed by §2A rule predicates.
    # None when no central-bank-text release frame was supplied to
    # build_feature_store (V1 byte-identity preserved). See source-data verification.
    # TODO(monetary-calibration): decide, with calibration evidence, whether
    # central_bank_text_score should become a configured confirmation gate for
    # tightening/easing labels or remain reporting-only monetary evidence.
    central_bank_text_score: pd.Series | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "yield_change_zscore_2y_63d",
            "yield_change_zscore_10y_63d",
            "broad_usd_index_zscore_63d",
            "yield_change_zscore_21d_2y",
            "yield_change_zscore_21d_10y",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


def _yield_change_zscore(
    *,
    yield_series: pd.Series,
    lookback: int,
    normalizer_window: int,
    output_name: str,
) -> pd.Series:
    """Thin §2A wrapper over the shared `rolling_change_zscore` helper.

    One home per concept lives in `_rolling_stats.py`; §2A and §2C only
    differ in their change_window / normalizer_window defaults.
    """
    return _rolling_change_zscore(
        _carry_forward_observations(yield_series),
        change_window=lookback,
        normalizer_window=normalizer_window,
        output_name=output_name,
    )


def _carry_forward_observations(series: pd.Series) -> pd.Series:
    """Align sparse macro observations to NYSE sessions without lookahead.

    FRED daily rates can miss NYSE sessions because of source publication
    calendars. Rolling 5y normalizers should consume the latest observation
    available as of the session; staleness remains a separate data-quality
    concern at the classifier boundary.
    """
    return series.astype(float).ffill()


def compute_monetary_pressure_features(
    *,
    dgs2: pd.Series,
    dgs10: pd.Series,
    broad_usd_index: pd.Series,
    central_bank_text_score: pd.Series | None = None,
    config: MonetaryPressureV2FeaturesConfig,
) -> MonetaryPressureV2Features:
    """Compute the v2 §2A yield + USD z-score features.

    Parameters
    ----------
    dgs2
        FRED ``DGS2`` (2y constant-maturity Treasury yield) series.
    dgs10
        FRED ``DGS10`` (10y constant-maturity Treasury yield) series.
    broad_usd_index
        Required FRED broad USD index level (e.g. ``DTWEXBGS``).
    config
        ``MonetaryPressureV2FeaturesConfig`` — supplies all four window
        lengths (yield 63d, normalizer 1260d, rate-shock 21d, broad-USD 63d).
    """
    if broad_usd_index is None:
        raise ValueError("broad_usd_index is required for monetary pressure features")
    z_2y = _yield_change_zscore(
        yield_series=dgs2,
        lookback=config.yield_change_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_2y_63d",
    )
    z_10y = _yield_change_zscore(
        yield_series=dgs10,
        lookback=config.yield_change_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_10y_63d",
    )
    z_21d_2y = _yield_change_zscore(
        yield_series=dgs2,
        lookback=config.rate_shock_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_21d_2y",
    )
    z_21d_10y = _yield_change_zscore(
        yield_series=dgs10,
        lookback=config.rate_shock_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_21d_10y",
    )
    usd_z = _rolling_change_zscore(
        _carry_forward_observations(broad_usd_index),
        change_window=config.broad_usd_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="broad_usd_index_zscore_63d",
    )
    # v2 §2A central-bank-text seam (source-data verification). Pure pass-through onto
    # the features dataclass — the rule engine never reads this field.
    # Reindexed to the yield series' DatetimeIndex so downstream
    # consumers get a single coherent calendar.
    if central_bank_text_score is not None:
        aligned_cb_score = central_bank_text_score.reindex(dgs2.index)
        aligned_cb_score.name = "central_bank_text_score"
    else:
        aligned_cb_score = None
    return MonetaryPressureV2Features(
        yield_change_zscore_2y_63d=z_2y,
        yield_change_zscore_10y_63d=z_10y,
        broad_usd_index_zscore_63d=usd_z,
        yield_change_zscore_21d_2y=z_21d_2y,
        yield_change_zscore_21d_10y=z_21d_10y,
        central_bank_text_score=aligned_cb_score,
    )
