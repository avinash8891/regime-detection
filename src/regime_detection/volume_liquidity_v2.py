"""v2 §1E Layer 1 V2 Volume / Liquidity features.

Scope-restricted feature module: ships only the volume z-score feature. The
``gap_frequency_20d`` and ``intraday_range_percentile_252d`` features
that §1E references already live in
``regime_detection.volatility_state_v2`` and are consumed
from that seam by the §1E axis classifier — they are NOT
recomputed here. See documented implementation decision for the
feature-store layout decision.

The §1E labels (``normal_volume``, ``panic_volume``,
``liquidity_gap_behavior``), rule engine, risk-rank table, and per-label
hysteresis live in ``regime_detection.volume_liquidity_rules`` and
``VolumeLiquidityStateSeriesClassifier``.

Feature shipped here:

- ``volume_zscore_20d``  v2 §1E line 256
  ``(volume[t] - rolling_mean(volume, 20)[t]) / rolling_std(volume, 20)[t]``.

Implementation choices that resolve ambiguities:

- **Sample standard deviation (``ddof=1``)**: spec is silent on
  population vs sample. ``ddof=1`` is the pandas default
  (``Series.rolling(N).std()``) and the standard convention for
  z-scores on financial time series. Pinned via
  ``VolumeLiquidityV2Config.volume_zscore_ddof`` so future calibration
  (§9.1) can flip without code changes. See documented implementation decision.
- **Constant-volume cold-start**: ``rolling_std == 0`` ⇒ ``0 / 0 = NaN``
  (masked, not infinity). Constant series have no z-score by
  definition; surfacing NaN matches the V1 cold-start contract
  (missing input → NaN, never a synthesized value).
- **NaN cold-start window**: first non-NaN at ``t = lookback - 1``
  (``min_periods=lookback``) — i.e. ``t = 19`` for the default
  ``volume_zscore_lookback_days = 20``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_detection.config import VolumeLiquidityV2Config


@dataclass(frozen=True)
class VolumeLiquidityV2Features:
    """v2 §1E — per-session continuous volume/liquidity features.

    NOTE: §1E also names ``gap_frequency_20d`` and
    ``intraday_range_percentile_252d`` in its feature list (lines
    257–258), but those are already computed in
    ``regime_detection.volatility_state_v2.VolatilityV2Features``.
    The §1E axis classifier reads them from the
    ``FeatureStore.volatility_state_v2`` seam rather than recomputing
    them here. See documented implementation notes.
    """

    volume_zscore_20d: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("volume_zscore_20d",)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


def _volume_zscore(
    *,
    volume: pd.Series,
    lookback: int,
    ddof: int,
) -> pd.Series:
    """v2 §1E line 256: ``(volume[t] - rolling_mean) / rolling_std``.

    ``min_periods=lookback`` so the first non-NaN value lands at
    ``t = lookback - 1``. Constant-volume windows produce a zero std
    which is masked to NaN (avoids 0/0 = NaN-but-via-RuntimeWarning;
    explicit ``.where(std > 0)`` makes the intent clear).
    """
    volume = volume.astype(float)
    rolling = volume.rolling(window=lookback, min_periods=lookback)
    mean = rolling.mean()
    std = rolling.std(ddof=ddof)
    return ((volume - mean) / std.where(std > 0)).rename("volume_zscore_20d")


def compute_volume_liquidity_v2_features(
    *,
    volume: pd.Series,
    config: VolumeLiquidityV2Config,
) -> VolumeLiquidityV2Features:
    """Compute the v2 §1E volume_zscore_20d feature from a SPY-like volume series.

    Parameters
    ----------
    volume
        Daily volume series (SPY shares traded), indexed by trading-day
        ``DatetimeIndex``.
    config
        ``VolumeLiquidityV2Config`` instance — supplies
        ``volume_zscore_lookback_days`` and ``volume_zscore_ddof`` (no
        magic numbers in the function body).

    Returns
    -------
    VolumeLiquidityV2Features
        Frozen dataclass with ``volume_zscore_20d: pd.Series`` aligned
        to ``volume.index``.
    """
    if not isinstance(volume.index, pd.DatetimeIndex):
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)

    zscore = _volume_zscore(
        volume=volume,
        lookback=config.volume_zscore_lookback_days,
        ddof=config.volume_zscore_ddof,
    )
    return VolumeLiquidityV2Features(volume_zscore_20d=zscore)
