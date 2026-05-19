"""v2 §6.3 BOCPD Change-Point Detection (evidence-only).

Library reuse: ``bayesian_changepoint_detection.online_changepoint_detection``
(documented implementation decision) — Adams-MacKay 2007 algorithm. No hand-rolled BOCPD;
~70 lines of glue.

Observation series: realized_vol_21d (#63).
Score formula: 5-session rolling max of posterior P(change-point at t) (#64).
Break: posterior >= ``break_threshold`` (#65).

Per V2 §10 + spec §6.3 line 2887 this is EVIDENCE only.
``RegimeOutput.change_point`` is populated when the config is present and
inputs are sufficient. The ``transition_score`` consumer is V2.1
spec-amendment work (spec §4.2 doesn't yet declare a
``change_point_score`` component).

Implementation note on indexing into the algorithm's posterior matrix
``R``: in this library ``R[1, t]`` carries the per-session change-point
posterior. Adams-MacKay's renormalization step folds the hazard mass back
into row 0 such that ``R[0, t]`` collapses to ~hazard at every session;
the data-conditioned "a change just happened" signal lives in
``R[1, t]`` (P(run_length=1 at time t) = P(change-point at t-1)). The
spec/Ambiguity-Log description "P(run_length=0)" maps to this row in the
``bayesian-changepoint-detection`` implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

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

    posterior_changepoint_prob: pd.Series  # raw BOCPD per-session changepoint posterior
    score: pd.Series  # 5-session rolling max of posterior (documented implementation decision)
    days_since_last_break: pd.Series  # nullable Int64; sessions since last break (documented implementation decision)
    method: str  # "BOCPD"


def compute_change_point_features(
    *,
    realized_vol_21d: pd.Series | None,
    config: ChangePointConfig,
) -> ChangePointFeatures | None:
    """Run BOCPD on the trailing ``training_window_days`` of
    ``realized_vol_21d`` and return a per-session evidence triple aligned
    to the input index.

    Returns ``None`` when:
      - ``realized_vol_21d`` is ``None``,
      - non-NaN history is shorter than ``training_window_days``,
      - the training window has zero variance (degenerate input —
        BOCPD's Student-T predictive would be singular),
      - ``bayesian_changepoint_detection`` raises (numerical instability).

    Two distinct NaN regimes in the output series:

    1. **Pre-training-window rows** (sessions that precede the trailing
       ``training_window_days`` start). All three series (posterior,
       score, days_since_last_break) are NaN / ``pd.NA``. The timeline
       consumer filters these out before emitting ``RegimeOutput.
       change_point`` so the wire is None there.

    2. **In-window rows where no break has yet occurred in trailing
       history** (cold-start within the BOCPD window). ``posterior`` and
       ``score`` are real numbers; ``days_since_last_break`` is ``pd.NA``
       per documented implementation decision / V1 §2.7 cold-start contract. The timeline
       consumer maps ``pd.NA`` → ``None`` for the wire field while
       preserving the real ``score`` value. This is the load-bearing
       path — quiet markets with no detected breaks still emit a valid
       low-score evidence row.

    Determinism note: callers that pass extra trailing history beyond
    ``training_window_days`` can introduce 1-ULP differences in the input
    series upstream (pandas rolling std's accumulator depends on the
    starting buffer position). To preserve byte-identical output across
    lookback values, this function pinches the input to exactly the
    trailing ``training_window_days`` values BEFORE BOCPD reads them; if
    the source series had ULP-level noise farther back, the trailing
    window itself is identical (verified empirically: the 1260
    spy_close values are byte-equal across slicings).
    """
    if realized_vol_21d is None:
        return None

    clean = realized_vol_21d.dropna()
    if len(clean) < config.training_window_days:
        return None

    train_series = clean.tail(config.training_window_days)
    # Round trailing window to a deterministic precision so 1-ULP
    # upstream noise (pandas rolling std accumulator path) does not
    # propagate into the BOCPD posterior. The ~1e-12 epsilon is far
    # below the realised-vol signal floor (~1e-3) and far above
    # machine epsilon (~1e-16), preserving evidence fidelity while
    # eliminating ULP-level non-determinism.
    _ROUND_DECIMALS = 12
    data = np.round(train_series.to_numpy(dtype=float), decimals=_ROUND_DECIMALS)

    # Fail-open guard on degenerate (zero-variance) input — Student-T
    # posterior is singular when the data has no spread. The std floor
    # is intentionally aligned with the rounding precision above:
    # a series with true std below ~1 * 10^-_ROUND_DECIMALS collapses
    # to a single rounded value (std == 0 exactly) and trips the guard.
    # A series with std in (10^-_ROUND_DECIMALS, ~10 * 10^-_ROUND_DECIMALS)
    # survives this guard but typically has only 1-2 distinct rounded
    # values, which drives the Student-T predictive near-singular and
    # triggers the bare-except branch below (also returning None). Both
    # paths preserve fail-open semantics.
    _STD_FLOOR = 10 ** -_ROUND_DECIMALS
    if not (data.std() > _STD_FLOOR):
        return None

    try:
        posterior_arr = _bocpd_posterior_changepoint_prob(data=data, config=config)
    except ArithmeticError as exc:
        _LOGGER.warning(
            "BOCPD online_changepoint_detection failed; "
            "change_point seam returns None: %s",
            exc,
        )
        return None

    posterior_aligned = pd.Series(
        posterior_arr,
        index=train_series.index,
        name="posterior_changepoint_prob",
    ).reindex(realized_vol_21d.index)

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

    See the module docstring for why row 1 of ``R`` carries the per-session
    change-point posterior in this library.
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
    return np.asarray(R[1, 1 : n + 1], dtype=float)


def _rolling_max_changepoint_prob(posterior: pd.Series, window: int) -> pd.Series:
    """5-session rolling max per documented implementation decision."""
    return posterior.rolling(window=window, min_periods=1).max()


def _days_since_last_break(
    posterior: pd.Series, threshold: float
) -> pd.Series:
    """Sessions since last posterior crossing per documented implementation decision.

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
