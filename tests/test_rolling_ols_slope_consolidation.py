"""Pre-merge equivalence test for the rolling_ols_slope consolidation (C1).

Two implementations existed before consolidation:

1. ``credit_funding._rolling_ols_slope`` (centered closed-form, ``cov(x,y)/var(x)``):
   subtracts means before multiplying. Numerically stable for near-zero slopes.

2. ``network_fragility_rules._rolling_ols_slope_series`` (uncentered normal
   equations, ``(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`` via
   ``numpy.lib.stride_tricks.sliding_window_view``): vectorised across the
   full series. Faster on paper, but the algebraic form can suffer catastrophic
   cancellation when slope ≈ 0 because Σxy and Σx·Σy can be large numbers
   subtracting to a small difference.

The consolidation canonicalises on the centered form (now in
``regime_detection._rolling_stats.rolling_ols_slope``) because slope outputs
feed sign-sensitive ``> 0`` predicates in ``network_fragility_rules`` (which
is part of the V1 wire and must remain bit-stable). Even ULP-level drift at
the zero boundary could flip a label.

This test snapshots both old implementations inline as fixtures and asserts:

  a) The new shared helper agrees with credit_funding's old centered form to
     bit-identity (same body, so this is trivially true and serves as a
     regression anchor against future refactors).
  b) The new shared helper agrees with network_fragility's old uncentered form
     within ``max abs diff <= 1e-12`` on real-shaped production series.
  c) Zero sign disagreements between the two old forms at the ``> 0`` boundary
     on any of the test series, confirming the consolidation is safe for the
     downstream predicates.

If (b) or (c) ever fail, the consolidation has uncovered a real numerical
divergence and the canonical implementation must be reconsidered (or
production data shifted enough to expose the cancellation pathology).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection._rolling_stats import rolling_ols_slope

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Snapshots of the two old implementations (verbatim from pre-consolidation
# credit_funding.py:255-281 and network_fragility_rules.py:233-250).
# ---------------------------------------------------------------------------


def _credit_funding_centered_form(series: pd.Series, *, window: int) -> pd.Series:
    """credit_funding's pre-consolidation centered closed-form."""
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    series = series.astype(float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = float((x_centered**2).sum())

    def _slope(window_arr: np.ndarray) -> float:
        if np.isnan(window_arr).any():
            return float("nan")
        y_mean = window_arr.mean()
        return float((x_centered * (window_arr - y_mean)).sum() / x_var)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _network_fragility_uncentered_form(series: pd.Series, window: int) -> pd.Series:
    """network_fragility_rules' pre-consolidation sliding_window_view form."""
    from numpy.lib.stride_tricks import sliding_window_view

    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=series.index)

    windows = sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)
    if valid.any():
        x = np.arange(window, dtype=float)
        x_sum = float(x.sum())
        x_sq_sum = float(np.square(x).sum())
        denom = window * x_sq_sum - x_sum * x_sum
        valid_windows = windows[valid]
        y_sum = valid_windows.sum(axis=1)
        xy_sum = valid_windows @ x
        out[window - 1 :][valid] = (window * xy_sum - x_sum * y_sum) / denom
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Synthetic production-shaped series covering numerical edge cases.
# ---------------------------------------------------------------------------


def _date_index(n: int) -> pd.DatetimeIndex:
    """Trading-day-like index (B-frequency)."""
    return pd.date_range("2020-01-01", periods=n, freq="B")


@pytest.fixture
def credit_spread_like_series() -> pd.Series:
    """Random-walk series shaped like ``hy_oas_63d`` percentiles.

    Magnitude ~O(1), gentle drift, occasional NaN gap (cold-start).
    """
    rng = np.random.default_rng(seed=20260608)
    n = 500
    drift = 0.0001
    noise = rng.standard_normal(n) * 0.01
    levels = np.cumsum(drift + noise) + 0.5
    levels[10:15] = np.nan  # cold-start gap
    return pd.Series(levels, index=_date_index(n), name="hy_oas_63d")


@pytest.fixture
def near_zero_slope_series() -> pd.Series:
    """Mean-reverting series with slope ≈ 0 across most windows.

    This is the pathological case for the uncentered form. The centered form
    handles it without cancellation; the uncentered form may produce a
    sign-flipping rounding error.
    """
    rng = np.random.default_rng(seed=42)
    n = 400
    mean_level = 1000.0  # large baseline → Σxy and Σx·Σy are both large
    perturbation = rng.standard_normal(n) * 0.001  # tiny noise
    return pd.Series(
        mean_level + perturbation,
        index=_date_index(n),
        name="near_zero_slope",
    )


@pytest.fixture
def realized_vol_like_series() -> pd.Series:
    """Volatility-shaped series: positive, episodic spikes, log-normal tails."""
    rng = np.random.default_rng(seed=12345)
    n = 600
    base = np.abs(rng.standard_normal(n)) * 0.15 + 0.10
    # Inject vol spikes
    base[150:160] = base[150:160] * 5.0
    base[400:405] = base[400:405] * 8.0
    return pd.Series(base, index=_date_index(n), name="realized_vol_21d")


@pytest.fixture
def correlation_slope_series() -> pd.Series:
    """avg_pairwise_corr_63d-shaped series in [0, 1] with mean-reversion."""
    rng = np.random.default_rng(seed=7)
    n = 800
    mean = 0.45
    series = np.empty(n)
    series[0] = mean
    for i in range(1, n):
        series[i] = 0.97 * series[i - 1] + 0.03 * mean + rng.standard_normal() * 0.01
    return pd.Series(
        np.clip(series, 0.0, 1.0),
        index=_date_index(n),
        name="avg_pairwise_corr_63d",
    )


_PRODUCTION_WINDOWS: tuple[int, ...] = (21, 63)


@pytest.fixture(
    params=[
        "credit_spread_like_series",
        "near_zero_slope_series",
        "realized_vol_like_series",
        "correlation_slope_series",
    ]
)
def production_series(request: pytest.FixtureRequest) -> pd.Series:
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------------------
# Assertions.
# ---------------------------------------------------------------------------


class TestRollingOlsSlopeConsolidation:
    """C1 consolidation safety: the shared helper matches both old forms."""

    @pytest.mark.parametrize("window", _PRODUCTION_WINDOWS)
    def test_shared_helper_bit_identical_to_centered_form(
        self, production_series: pd.Series, window: int
    ) -> None:
        """The shared helper IS credit_funding's centered form → bit-identical."""
        new = rolling_ols_slope(production_series, window=window)
        old = _credit_funding_centered_form(production_series, window=window)
        # Both NaN-aligned on cold-start
        new_arr = new.to_numpy()
        old_arr = old.to_numpy()
        new_nan = np.isnan(new_arr)
        old_nan = np.isnan(old_arr)
        assert np.array_equal(new_nan, old_nan), "NaN positions diverge"
        # Bit-identity on finite values
        finite = ~new_nan
        assert np.array_equal(
            new_arr[finite], old_arr[finite]
        ), "Centered-form bodies must produce bit-identical results"

    @pytest.mark.parametrize("window", _PRODUCTION_WINDOWS)
    def test_shared_helper_within_tolerance_of_uncentered_form(
        self, production_series: pd.Series, window: int
    ) -> None:
        """The shared helper matches the uncentered form within 1e-12."""
        new = rolling_ols_slope(production_series, window=window).to_numpy()
        uncentered = _network_fragility_uncentered_form(
            production_series, window
        ).to_numpy()
        new_nan = np.isnan(new)
        unc_nan = np.isnan(uncentered)
        assert np.array_equal(new_nan, unc_nan), "NaN positions diverge"
        finite = ~new_nan
        if finite.any():
            max_abs_diff = float(np.max(np.abs(new[finite] - uncentered[finite])))
            assert max_abs_diff <= 1e-12, (
                f"Centered vs uncentered diverge by {max_abs_diff} > 1e-12; "
                f"window={window}, series={production_series.name}. "
                "The consolidation would change downstream values."
            )

    @pytest.mark.parametrize("window", _PRODUCTION_WINDOWS)
    def test_no_sign_disagreement_at_zero_boundary(
        self, production_series: pd.Series, window: int
    ) -> None:
        """No sign flips between the two old forms on any production series.

        Sign flips at the ``> 0`` boundary are the failure mode that motivated
        the canonical choice (centered form). If this assertion ever fails on
        production data, we have direct evidence the consolidation prevented
        a real bug, and the test should be updated to document the case
        rather than relaxed.
        """
        centered = _credit_funding_centered_form(
            production_series, window=window
        ).to_numpy()
        uncentered = _network_fragility_uncentered_form(
            production_series, window
        ).to_numpy()
        finite = ~np.isnan(centered) & ~np.isnan(uncentered)
        if not finite.any():
            pytest.skip("No finite overlap (insufficient data for this window).")
        # Sign disagreement: one strictly positive, the other not (and vice versa).
        c_pos = centered[finite] > 0.0
        u_pos = uncentered[finite] > 0.0
        disagreements = int(np.sum(c_pos != u_pos))
        assert disagreements == 0, (
            f"{disagreements} sign disagreements at > 0 boundary; "
            f"window={window}, series={production_series.name}. "
            "Centered form is canonical; uncentered form must be deleted, "
            "but production behavior must agree on these series."
        )
