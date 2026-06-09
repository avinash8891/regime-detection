"""v2 §6.3 BOCPD Change-Point Detection (evidence-only).

Library reuse: ``bayesian_changepoint_detection.online_changepoint_detection``
(documented implementation decision) — Adams-MacKay 2007 algorithm. No hand-rolled BOCPD;
~70 lines of glue.

Observation series: realized_vol_21d (#63).
Score formula: 5-session rolling max of the recent short-run BOCPD posterior
mass (#64 adapted to the realized_vol_21d observation horizon).
Break: posterior >= ``break_threshold`` (#65).

Per V2 §10 + spec §6.3 (L4252) this is EVIDENCE only.
``RegimeOutput.change_point`` is populated when the config is present and
inputs are sufficient. The ``transition_score`` consumer is now wired
via the §4.3 4-table weight system (spec L2166):
when the change_point seam is lit, ``compose_transition_score_for_session``
folds it into the ``model_instability`` component. Dynamic weighting then
uses the single configured ``transition_score.weights`` table for all
available components.

Implementation note on indexing into the algorithm's posterior matrix
``R``: with a constant hazard function, row ``R[0, t]`` collapses to the
configured hazard and is not data-conditioned. Because the observation is
``realized_vol_21d``, a market break enters the input as a rolling-window
ramp; the data-conditioned break signal spreads over short run lengths
rather than only row ``R[1, t]``. The emitted posterior therefore sums
``R[1:recent_run_length_window_days + 1, t]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from bayesian_changepoint_detection.online_changepoint_detection import (
    StudentT as _StudentT,
    constant_hazard as _constant_hazard,
    online_changepoint_detection as _online_changepoint_detection,
)

from regime_detection.config import ChangePointConfig

__all__ = ["ChangePointFeatures", "compute_change_point_features"]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangePointFeatures:
    """v2 §6.3 — per-session BOCPD posterior + derived series."""

    posterior_changepoint_prob: pd.Series  # recent short-run BOCPD posterior mass
    score: pd.Series  # 5-session rolling max of posterior (spec L2135-2150)
    days_since_last_break: (
        pd.Series
    )  # nullable Int64; sessions since last break (spec L2152-2164)
    method: Literal["BOCPD"]


def compute_change_point_features(
    *,
    realized_vol_21d: pd.Series | None,
    config: ChangePointConfig,
) -> ChangePointFeatures | None:
    """Run BOCPD over as-of ``realized_vol_21d`` history and return a
    per-session evidence triple aligned to the input index.

    Raises when:
      - ``realized_vol_21d`` is ``None``,
      - non-NaN history is shorter than ``training_window_days``,
      - the training window has zero variance (degenerate input —
        BOCPD's Student-T predictive would be singular),
      - ``bayesian_changepoint_detection`` raises (numerical instability).

    BOCPD is evaluated forward over the caller-provided history, so the
    posterior at session ``t`` depends only on observations ``<= t``.
    The first ``training_window_days - 1`` non-null observations are masked
    to NaN: they are available to seed the online posterior, but they do
    not satisfy the configured strict PIT warm-up gate for emitted evidence.

    Two distinct NaN regimes in the output series:

    1. **Pre-training-window rows** (the first
       ``training_window_days - 1`` non-null observations). All three
       series (posterior, score, days_since_last_break) are NaN /
       ``pd.NA``. The timeline consumer filters these out before emitting
       ``RegimeOutput.change_point`` so the wire is None there.

    2. **In-window rows where no break has yet occurred in trailing
       history** (cold-start within the BOCPD window). ``posterior`` and
       ``score`` are real numbers; ``days_since_last_break`` is ``pd.NA``
       per spec L2152-L2164 / V1 §2.7 cold-start contract. The timeline
       consumer maps ``pd.NA`` → ``None`` for the wire field while
       preserving the real ``score`` value. This is the load-bearing
       path — quiet markets with no detected breaks still emit a valid
       low-score evidence row.

    Determinism note: the observation series is rounded before BOCPD reads
    it so 1-ULP upstream noise (pandas rolling std accumulator path) does
    not propagate into the posterior. Appending future rows cannot change
    already-emitted historical scores because the online recursion is
    forward-only.
    """
    if realized_vol_21d is None:
        raise RuntimeError("BOCPD missing required input: realized_vol_21d")

    clean = realized_vol_21d.dropna()
    if len(clean) < config.training_window_days:
        raise RuntimeError(
            "BOCPD insufficient history: "
            f"need {config.training_window_days} rows, got {len(clean)}"
        )

    train_series = clean
    # Round trailing window to a deterministic precision so 1-ULP
    # upstream noise (pandas rolling std accumulator path) does not
    # propagate into the BOCPD posterior. The ~1e-12 epsilon is far
    # below the realised-vol signal floor (~1e-3) and far above
    # machine epsilon (~1e-16), preserving evidence fidelity while
    # eliminating ULP-level non-determinism.
    _ROUND_DECIMALS = 12
    data = np.round(train_series.to_numpy(dtype=float), decimals=_ROUND_DECIMALS)

    # Fail-loud guard on degenerate (zero-variance) input — Student-T
    # posterior is singular when the data has no spread. The std floor
    # is intentionally aligned with the rounding precision above:
    # a series with true std below ~1 * 10^-_ROUND_DECIMALS collapses
    # to a single rounded value (std == 0 exactly) and trips the guard.
    # A series with std in (10^-_ROUND_DECIMALS, ~10 * 10^-_ROUND_DECIMALS)
    # survives this guard but typically has only 1-2 distinct rounded
    # values, which drives the Student-T predictive near-singular and
    # triggers the exception branch below.
    _STD_FLOOR = 10**-_ROUND_DECIMALS
    if not (data.std() > _STD_FLOOR):
        raise RuntimeError("BOCPD degenerate input: realized_vol_21d has no variance")

    try:
        posterior_arr = _bocpd_posterior_changepoint_prob(data=data, config=config)
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        _LOGGER.warning(
            "BOCPD online_changepoint_detection failed; "
            "change_point seam fails loudly: %s",
            exc,
        )
        raise RuntimeError("BOCPD fit failed") from exc

    posterior_series = pd.Series(
        posterior_arr,
        index=train_series.index,
        name="posterior_changepoint_prob",
    )
    posterior_series.iloc[: config.training_window_days - 1] = np.nan
    posterior_aligned = posterior_series.reindex(realized_vol_21d.index)

    score_aligned = _rolling_max_changepoint_prob(
        posterior_aligned, window=config.score_window_days
    ).rename("change_point_score")

    days_since = _days_since_last_break(
        posterior_aligned, threshold=config.break_threshold
    ).rename("days_since_last_break")

    return ChangePointFeatures(
        posterior_changepoint_prob=posterior_aligned,
        score=score_aligned,
        days_since_last_break=days_since,
        method=config.method,
    )


def _bocpd_posterior_changepoint_prob(
    *,
    data: np.ndarray,
    config: ChangePointConfig,
) -> np.ndarray:
    """Adapter for the ``bayesian-changepoint-detection`` API.

    The project needs the Adams-MacKay online BOCPD implementation, exposed by
    the package as:

    - ``online_changepoint_detection(data, hazard_func, observation_likelihood)``
    - ``constant_hazard(lam, r)``, passed via ``functools.partial``
    - ``StudentT(alpha, beta, kappa, mu)`` observation likelihood

    See the module docstring for why the emitted posterior sums short
    run-length rows instead of reading ``R[0]`` or ``R[1]`` directly.
    """
    R, _maxes = _online_changepoint_detection(
        data,
        partial(_constant_hazard, config.hazard_lambda),
        _StudentT(
            alpha=config.student_t_alpha,
            beta=config.student_t_beta,
            kappa=config.student_t_kappa,
            mu=config.student_t_mu,
        ),
    )
    n = len(data)
    expected_shape = (n + 1, n + 1)
    if R.shape != expected_shape:
        raise RuntimeError(
            "BOCPD posterior matrix shape changed: "
            f"expected {expected_shape}, got {R.shape}"
        )
    window = min(config.recent_run_length_window_days, n)
    posterior = np.asarray(R[1 : window + 1, 1 : n + 1].sum(axis=0), dtype=float)
    if not np.isfinite(posterior).all():
        raise FloatingPointError("BOCPD posterior contains non-finite values")
    return posterior


def _rolling_max_changepoint_prob(posterior: pd.Series, window: int) -> pd.Series:
    """5-session rolling max per spec L2135-L2150."""
    return posterior.rolling(window=window, min_periods=1).max()


def _days_since_last_break(posterior: pd.Series, threshold: float) -> pd.Series:
    """Sessions since last posterior crossing per spec L2152-L2164.

    ``pd.NA`` when no break has occurred in available history.
    """
    is_break_arr = (posterior >= threshold).where(posterior.notna(), False).to_numpy()
    positions = np.arange(len(posterior), dtype=float)
    last_break_pos = pd.Series(
        np.where(is_break_arr.astype(bool), positions, np.nan),
        index=posterior.index,
    ).ffill()
    days_since = positions - last_break_pos.to_numpy()
    result = pd.Series(days_since, index=posterior.index, dtype="Float64")
    return result.astype("Int64")
