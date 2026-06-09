"""v2 §3.2 Network Fragility feature compute.

Pure numpy/pandas implementation of the features specified at
docs/regime_engine_v2_spec.md §3.2 (lines 3434–3479) — five base features
plus four derived rolling percentiles:

- avg_pairwise_corr_63d            (mean of off-diagonal of 63d corr matrix)
- avg_pairwise_corr_percentile_504d
- largest_eigenvalue_share         (largest eigenvalue / sum(eigvals))
- largest_eigenvalue_share_percentile_504d
- effective_rank                   (exp(Shannon entropy of normalised eigvals))
- effective_rank_percentile_504d
- absorption_ratio_top3            (sum(top 3 eigenvalues) / sum(eigvals))
- dispersion_ratio                 (mean 21d realised vol / SPY 21d vol)
- dispersion_ratio_percentile_252d

Per-session window:
  - Take the last `correlation_lookback_days` rows ending at session t.
  - Drop columns whose non-null fraction within the window is below
    `min_window_completeness`.
  - If fewer than `min_universe_size` symbols survive, emit NaN for all
    correlation/eigen features at that session (the downstream data-quality
    layer then renders the network_fragility axis as `unknown`).

Universe order is enforced from `fragility_universe.NETWORK_FRAGILITY_UNIVERSE`.
SPY is sourced from `spy_close` (v1 path) — not `cross_asset_closes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    INDEX_SYMBOL,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class NetworkFragilityFeatures:
    """v2 spec §3.2 — per-session features for the network fragility classifier."""

    avg_pairwise_corr_63d: pd.Series
    avg_pairwise_corr_percentile_504d: pd.Series
    largest_eigenvalue_share: pd.Series
    largest_eigenvalue_share_percentile_504d: pd.Series
    effective_rank: pd.Series
    effective_rank_percentile_504d: pd.Series
    absorption_ratio_top3: pd.Series
    dispersion_ratio: pd.Series
    dispersion_ratio_percentile_252d: pd.Series
    surviving_universe_size: pd.Series | None = None
    complete_observation_count: pd.Series | None = None


def _assemble_returns_matrix(
    *,
    sector_etf_closes: dict[str, pd.Series],
    cross_asset_closes: dict[str, pd.Series],
    spy_close: pd.Series,
    universe: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Align all available closes onto the SPY index in universe order,
    return percentage changes. Missing symbols become all-NaN columns so the
    completeness filter can drop them per-session."""
    index = spy_close.index
    columns: list[str] = []
    series_map: dict[str, pd.Series] = {}

    effective_universe = (
        tuple(universe) if universe is not None else NETWORK_FRAGILITY_UNIVERSE
    )
    for symbol in effective_universe:
        if symbol == INDEX_SYMBOL:
            series = spy_close
        elif symbol in SECTOR_ETFS:
            series = sector_etf_closes.get(symbol)
        else:
            assert symbol in CROSS_ASSET_SYMBOLS, (
                f"Unreachable: symbol {symbol!r} is outside the closed "
                f"network fragility universe (INDEX_SYMBOL | SECTOR_ETFS | CROSS_ASSET_SYMBOLS)."
            )
            series = cross_asset_closes.get(symbol)

        if series is None:
            # Symbol absent — fill with NaN so completeness gate drops it.
            series = pd.Series(np.nan, index=index, dtype=float)
        else:
            series = series.reindex(index)

        columns.append(symbol)
        series_map[symbol] = series

    prices = pd.DataFrame({c: series_map[c] for c in columns}, index=index)
    # fill_method=None: do NOT forward-fill NaN prices before differencing —
    # NaNs must propagate so the completeness gate sees true missing data.
    return prices.pct_change(fill_method=None)


def _shannon_effective_rank(eigvals: np.ndarray) -> float:
    """exp(Shannon entropy of normalised eigenvalues) — natural log per spec."""
    total = eigvals.sum()
    if total <= 0:
        return float("nan")
    p = eigvals / total
    # entropy ignores zero/negative components (log undefined / 0·log(0) = 0)
    positive = p[p > 0]
    if positive.size == 0:
        return float("nan")
    entropy = -(positive * np.log(positive)).sum()
    return float(np.exp(entropy))


def _positive_correlation_eigenvalues(eigvals: np.ndarray) -> np.ndarray:
    """Clip floating-point PSD noise from a correlation-matrix eigenspectrum."""
    return np.clip(np.asarray(eigvals, dtype=float), 0.0, None)


def _per_session_corr_features(
    returns: pd.DataFrame,
    *,
    correlation_lookback_days: int,
    min_universe_size: int,
    min_window_completeness: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Walk each session t with t >= correlation_lookback_days and emit the
    four corr/eigen features."""
    index = returns.index
    n = len(index)
    avg_corr = np.full(n, np.nan)
    largest_share = np.full(n, np.nan)
    eff_rank = np.full(n, np.nan)
    absorption = np.full(n, np.nan)
    surviving_universe_size = np.full(n, np.nan)
    complete_observation_count = np.full(n, np.nan)

    arr = returns.to_numpy()

    for t in range(correlation_lookback_days, n):
        # last `correlation_lookback_days` rows ending at session t inclusive
        start = t - correlation_lookback_days + 1
        window = arr[start : t + 1, :]

        # column-wise completeness on the window
        not_null = ~np.isnan(window)
        completeness = not_null.mean(axis=0)
        keep_mask = completeness >= min_window_completeness
        surviving_universe_size[t] = keep_mask.sum()
        if keep_mask.sum() < min_universe_size:
            continue

        sub = window[:, keep_mask]
        # Drop rows with any NaN among surviving columns; correlation needs
        # complete observations.
        row_complete = ~np.isnan(sub).any(axis=1)
        complete_observation_count[t] = row_complete.sum()
        sub = sub[row_complete, :]
        if sub.shape[0] < 2:
            continue

        corr = np.corrcoef(sub, rowvar=False)
        if not np.all(np.isfinite(corr)):
            continue

        iu = np.triu_indices_from(corr, k=1)
        avg_corr[t] = corr[iu].mean()

        eigs = _positive_correlation_eigenvalues(np.linalg.eigvalsh(corr))
        total = eigs.sum()
        if total > 0:
            largest_share[t] = eigs[-1] / total
            absorption[t] = eigs[-min(3, eigs.size) :].sum() / total
        eff_rank[t] = _shannon_effective_rank(eigs)

    return (
        pd.Series(avg_corr, index=index, name="avg_pairwise_corr_63d"),
        pd.Series(largest_share, index=index, name="largest_eigenvalue_share"),
        pd.Series(eff_rank, index=index, name="effective_rank"),
        pd.Series(absorption, index=index, name="absorption_ratio_top3"),
        pd.Series(
            surviving_universe_size,
            index=index,
            name="surviving_universe_size",
            dtype="Float64",
        ),
        pd.Series(
            complete_observation_count,
            index=index,
            name="complete_observation_count",
            dtype="Float64",
        ),
    )


def _dispersion_ratio_series(
    returns: pd.DataFrame,
    *,
    realized_vol_lookback_days: int,
    spy_vol_floor: float = 1e-6,
    spy_column: str = INDEX_SYMBOL,
) -> pd.Series:
    """mean(per-symbol annualised realised vol) / SPY annualised realised vol."""
    realised_vol = returns.rolling(realized_vol_lookback_days).std(ddof=1) * np.sqrt(
        _TRADING_DAYS_PER_YEAR
    )
    mean_vol = realised_vol.mean(axis=1, skipna=True)
    spy_vol = realised_vol[spy_column]
    safe_spy = spy_vol.where(spy_vol >= spy_vol_floor)
    return mean_vol / safe_spy


def compute_features(
    *,
    sector_etf_closes: dict[str, pd.Series],
    cross_asset_closes: dict[str, pd.Series],
    spy_close: pd.Series,
    correlation_lookback_days: int = 63,
    percentile_lookback_days: int = 504,
    realized_vol_lookback_days: int = 21,
    dispersion_percentile_lookback_days: int = 252,
    dispersion_spy_vol_floor: float = 1e-6,
    min_universe_size: int = 20,
    min_window_completeness: float = 0.9,
    universe: tuple[str, ...] | list[str] | None = None,
) -> NetworkFragilityFeatures:
    """Compute v2 §3.2 features. See module docstring for contract.

    All lookback / completeness parameters are sourced from
    ``RegimeConfig.network_fragility`` in production; the kwargs here keep
    unit-test control inline. effective_rank uses natural log per v2 §3.2.
    """
    returns = _assemble_returns_matrix(
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        spy_close=spy_close,
        universe=universe,
    )

    (
        avg_corr,
        largest_share,
        eff_rank,
        absorption,
        surviving_universe_size,
        complete_observation_count,
    ) = _per_session_corr_features(
        returns,
        correlation_lookback_days=correlation_lookback_days,
        min_universe_size=min_universe_size,
        min_window_completeness=min_window_completeness,
    )

    avg_corr_pct = avg_corr.rolling(percentile_lookback_days).rank(pct=True)
    largest_share_pct = largest_share.rolling(percentile_lookback_days).rank(pct=True)
    eff_rank_pct = eff_rank.rolling(percentile_lookback_days).rank(pct=True)

    dispersion = _dispersion_ratio_series(
        returns,
        realized_vol_lookback_days=realized_vol_lookback_days,
        spy_vol_floor=dispersion_spy_vol_floor,
    )
    dispersion_pct = dispersion.rolling(dispersion_percentile_lookback_days).rank(
        pct=True
    )

    spy_index = spy_close.index
    return NetworkFragilityFeatures(
        avg_pairwise_corr_63d=avg_corr.reindex(spy_index),
        avg_pairwise_corr_percentile_504d=avg_corr_pct.reindex(spy_index),
        largest_eigenvalue_share=largest_share.reindex(spy_index),
        largest_eigenvalue_share_percentile_504d=largest_share_pct.reindex(spy_index),
        effective_rank=eff_rank.reindex(spy_index),
        effective_rank_percentile_504d=eff_rank_pct.reindex(spy_index),
        absorption_ratio_top3=absorption.reindex(spy_index),
        dispersion_ratio=dispersion.reindex(spy_index),
        dispersion_ratio_percentile_252d=dispersion_pct.reindex(spy_index),
        surviving_universe_size=surviving_universe_size.reindex(spy_index),
        complete_observation_count=complete_observation_count.reindex(spy_index),
    )
