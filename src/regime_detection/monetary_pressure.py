"""v2 §2A Layer 2A Monetary / Liquidity V2 features — evidence-only compute (Slice 4.1).

Scope-restricted slice: ships ONLY the ONE feature formula that v2 §2A
pins verbatim at line 896::

    yield_change_zscore = (yield_change_63d - mean_5y) / std_5y

where ``yield_change_63d[t] = yield[t] - yield[t-63]`` and ``mean_5y`` /
``std_5y`` are the rolling mean / std (sample, ``ddof=1``) of the
change series over the prior 5y (≈ 1260 NYSE trading days).

Applied to TWO FRED series whose source contract IS spec-pinned at
§2A lines 887–889:

- ``DGS2``  → ``yield_change_zscore_2y_63d``
- ``DGS10`` → ``yield_change_zscore_10y_63d``

DEFERRED (per V2 §10 ABSOLUTE RULE; documented in Ambiguity Log #44
and #45):

- ``broad_usd_index_zscore_63d`` — §2A names the predicate but no
  formula. The 63d-change generalization to a USD index level (vs a
  yield level) is a spec invention.
- ``yield_change_zscore_21d_2y`` / ``yield_change_zscore_21d_10y`` —
  §2A names the 21d-variant rule predicates but specifies neither the
  change-window nor the mean/std normalizer window for the 21d form.
- The §2A label set (``tightening_pressure``, ``easing_pressure``,
  ``rate_shock``, neutral, unknown), precedence ordering, risk-rank,
  per-label hysteresis days.
- The ``MonetaryPressureSeriesClassifier`` axis classifier.

Mirrors the slice-2.4 precedent (``volume_zscore_20d`` shipped as
evidence-only before the §1E axis classifier landed in slice 2.7;
Ambiguity Log entry #29).

Implementation choices that resolve sub-ambiguities:

- **Sample std (``ddof=1``)**: §2A is silent on population vs sample
  std for ``std_5y``. Pinned to pandas / numpy default (``ddof=1``),
  matching the slice-2.4 ``volume_zscore_20d`` convention (Ambiguity
  Log entry #28). Constant-change windows produce ``std == 0`` which
  is masked to NaN (explicit ``.where(std > 0)``).
- **First valid index**: ``yield_change_63d`` is NaN for the first
  ``yield_change_lookback_days`` sessions (``shift(63)`` introduces
  63 NaN at the head). The 5y rolling normalizer then needs
  ``min_periods=zscore_normalizer_window_days`` non-NaN observations,
  so the first non-NaN z-score lands at
  ``t = yield_change_lookback_days + zscore_normalizer_window_days - 1``.
  With defaults (63 + 1260 - 1) the first valid index is ``t = 1322``.
- **DGS2 / DGS10 independence**: the two series are processed in
  separate compute pipelines; a NaN in DGS2 must NOT propagate to the
  DGS10 z-score.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_detection.config import MonetaryPressureV2FeaturesConfig


@dataclass(frozen=True)
class MonetaryPressureV2Features:
    """v2 §2A — per-session continuous monetary-pressure features (slice 4.1).

    Only the two spec-pinned yield z-scores ship today. The USD-index
    z-score and the 21d-variant yield z-scores referenced by the §2A
    rule predicates are deferred (Ambiguity Log #44, #45).
    """

    yield_change_zscore_2y_63d: pd.Series
    yield_change_zscore_10y_63d: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("yield_change_zscore_2y_63d", "yield_change_zscore_10y_63d")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


# v2 §2A pins std at the standard sample / pandas default. Centralised here
# so both yield series consume the same normalizer convention (and the
# Ambiguity Log can cite a single source line).
_ZSCORE_DDOF = 1


def _yield_change_zscore(
    *,
    yield_series: pd.Series,
    lookback: int,
    normalizer_window: int,
    output_name: str,
) -> pd.Series:
    """v2 §2A line 896: ``(yield_change_63d - mean_5y) / std_5y``.

    Computed independently per yield series — DGS2 NaNs cannot leak into
    the DGS10 output and vice versa because each invocation closes over
    its own input series.
    """
    yield_series = yield_series.astype(float)

    # yield_change_63d[t] = yield[t] - yield[t-63]
    yield_change = yield_series - yield_series.shift(lookback)

    # mean_5y / std_5y on the CHANGE series (per spec text — the
    # normalizer is over yield_change, not over yield level).
    rolling = yield_change.rolling(
        window=normalizer_window, min_periods=normalizer_window
    )
    mean = rolling.mean()
    std = rolling.std(ddof=_ZSCORE_DDOF)
    zscore = (yield_change - mean) / std.where(std > 0)
    return zscore.rename(output_name)


def compute_monetary_pressure_features(
    *,
    dgs2: pd.Series,
    dgs10: pd.Series,
    config: MonetaryPressureV2FeaturesConfig,
) -> MonetaryPressureV2Features:
    """Compute the v2 §2A yield-change z-score features from DGS2 and DGS10.

    Parameters
    ----------
    dgs2
        FRED ``DGS2`` (2y constant-maturity Treasury yield) series,
        indexed by trading-day ``DatetimeIndex`` aligned to the SPY
        calendar.
    dgs10
        FRED ``DGS10`` (10y constant-maturity Treasury yield) series,
        indexed by trading-day ``DatetimeIndex`` aligned to the SPY
        calendar.
    config
        ``MonetaryPressureV2FeaturesConfig`` — supplies
        ``yield_change_lookback_days`` and
        ``zscore_normalizer_window_days`` (no magic numbers in the
        function body).

    Returns
    -------
    MonetaryPressureV2Features
        Frozen dataclass with two ``pd.Series`` aligned to each input's
        index.
    """
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
    return MonetaryPressureV2Features(
        yield_change_zscore_2y_63d=z_2y,
        yield_change_zscore_10y_63d=z_10y,
    )
