"""v2 §6.1 Hidden Markov Model evidence layer (Slice 6).

Reuses existing FeatureStore seams as inputs (``return_1d``,
``realized_vol_21d``, ``drawdown_63d``, ``volume_zscore_20d``,
``avg_pairwise_corr_63d``). Fits a Gaussian HMM per spec line 2740 via
``hmmlearn.GaussianHMM`` — no hand-rolled Baum-Welch. Emits
``top_state_prob`` (permutation-invariant) for consumption by the
``transition_score`` §4.2 ``hmm_probability_shift_score`` 6th component.

The HMM is evidence-only per V2 §10 / spec line 2780-2783: state indices
are raw ``0``..``n-1`` integers, never auto-mapped to economic labels.
Operator mapping is deferred — see Implementation Ambiguity Log #62/#63.

Per V2 engine statelessness, ``compute_hmm_features`` re-fits ONCE per
``classify_window`` call on the trailing ``training_window_days`` rows.
The yaml ``hmm.retrain_cadence_days`` field is an ops/dev concern (when
to refresh persisted parameters in a long-running deployment), not a
per-classify gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM

from regime_detection.config import HMMConfig

__all__ = ["HMMFeatures", "compute_hmm_features"]

_LOGGER = logging.getLogger(__name__)


class _StrictConvergenceMonitor(ConvergenceMonitor):
    """Track non-monotonic EM fits without letting hmmlearn write to stderr."""

    non_monotonic: bool

    def __init__(self, tol: float, n_iter: int, verbose: bool) -> None:
        super().__init__(tol=tol, n_iter=n_iter, verbose=verbose)
        self.non_monotonic = False

    def report(self, log_prob: float) -> None:
        precision = np.finfo(float).eps ** 0.5
        if self.history and (log_prob - self.history[-1]) < -precision:
            self.non_monotonic = True
        self.history.append(log_prob)
        self.iter += 1

    @property
    def converged(self) -> bool:
        return self.non_monotonic or super().converged


@dataclass(frozen=True)
class HMMFeatures:
    """v2 §6.1 — per-session HMM posterior + top-state probability.

    Attributes:
        top_state_prob: max-of-posterior per session, NaN on dropped rows.
        state_probabilities: ``(n_sessions × n_states)`` posterior frame
            with columns labeled ``0``..``n_states-1`` (no economic
            mapping — V2 §10 ABSOLUTE RULE).
        n_states: configured number of hidden states.
    """

    top_state_prob: pd.Series
    state_probabilities: pd.DataFrame
    n_states: int
    selected_seed: int
    log_likelihood: float


def compute_hmm_features(
    *,
    return_1d: pd.Series | None,
    realized_vol_21d: pd.Series | None,
    drawdown_63d: pd.Series | None,
    volume_zscore_20d: pd.Series | None,
    avg_pairwise_corr_63d: pd.Series | None,
    config: HMMConfig,
) -> HMMFeatures | None:
    """Fit ``hmmlearn.GaussianHMM`` and return aligned posterior features.

    Aligns the five inputs to their common (intersected, non-NaN) index,
    fits ONE ``GaussianHMM`` on the trailing ``training_window_days``
    rows, then ``predict_proba`` over the full aligned index. Sessions
    with any NaN input are masked to NaN in the output (cold-start +
    missing-data safe).

    Returns ``None`` when:
      - any required input is ``None``
      - the joined non-NaN inputs have fewer than ``training_window_days``
        rows
      - the HMM fit fails (e.g. singular covariance, non-convergence) —
        per AGENTS error policy, return None (evidence absent) rather
        than crash the whole classify call.

    Permutation invariance: ``top_state_prob = posterior.max(axis=1)``
    is invariant to state-index permutation. The
    ``state_probabilities`` frame has integer column labels
    ``0``..``n_states-1`` (no economic mapping per V2 §10).
    """
    inputs: dict[str, pd.Series | None] = {
        "return_1d": return_1d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "volume_zscore_20d": volume_zscore_20d,
        "avg_pairwise_corr_63d": avg_pairwise_corr_63d,
    }
    if any(series is None for series in inputs.values()):
        return None

    # mypy/pyright: filtered above; safe to narrow.
    frame = pd.DataFrame({k: v for k, v in inputs.items()}).dropna(how="any")
    n_train = config.training_window_days
    if len(frame) < n_train:
        return None

    train_frame = frame.tail(n_train)
    fit_frame, predict_frame = _prepare_hmm_frames(
        train_frame=train_frame,
        full_frame=frame,
        standardize_inputs=config.standardize_inputs,
    )
    best: dict[str, Any] | None = None
    skipped: list[tuple[int, float | None, float | None, float | None]] = []
    try:
        for seed in config.random_seeds:
            model = GaussianHMM(
                n_components=config.n_states,
                covariance_type=config.covariance_type,
                min_covar=config.min_covar,
                n_iter=200,
                random_state=seed,
            )
            model.monitor_ = _StrictConvergenceMonitor(
                tol=model.monitor_.tol,
                n_iter=model.monitor_.n_iter,
                verbose=model.monitor_.verbose,
            )
            model.fit(fit_frame)
            history = list(model.monitor_.history)
            final_log_likelihood = float(history[-1]) if history else float("-inf")
            previous_log_likelihood = float(history[-2]) if len(history) >= 2 else None
            delta = (
                final_log_likelihood - previous_log_likelihood
                if previous_log_likelihood is not None
                else None
            )
            if getattr(model.monitor_, "non_monotonic", False):
                skipped.append((seed, final_log_likelihood, previous_log_likelihood, delta))
                continue
            posterior = model.predict_proba(predict_frame)
            candidate = {
                "posterior": posterior,
                "seed": seed,
                "log_likelihood": final_log_likelihood,
            }
            if best is None or final_log_likelihood > best["log_likelihood"]:
                best = candidate
    except Exception as exc:  # noqa: BLE001
        # Fail-open: degenerate inputs (singular covariance, etc.) should
        # not crash the engine — the seam goes None and downstream falls
        # back to the 5-component transition score.
        _LOGGER.warning(
            "GaussianHMM fit/predict failed; HMM seam returns None: %s", exc
        )
        return None
    if best is None:
        _LOGGER.warning(
            "GaussianHMM produced no monotonic fit across %d seed(s); "
            "HMM seam returns None. skipped=%s",
            len(config.random_seeds),
            skipped[:5],
        )
        return None

    # Re-align to the canonical SPY index (return_1d carries it). Sessions
    # dropped by .dropna() get NaN both in the state_probabilities frame
    # and in the top_state_prob series.
    canonical_index = return_1d.index  # type: ignore[union-attr]
    state_prob_frame = pd.DataFrame(
        best["posterior"],
        index=frame.index,
        columns=list(range(config.n_states)),
    ).reindex(canonical_index)

    # V1 §2.2 stateless-replay: the HMM is fit ONCE on frame.tail(n_train)
    # ending at frame.index[-1]. Posterior values for sessions earlier than
    # that fit-end were computed using parameters trained on data that, from
    # the earlier session's perspective, is the FUTURE. To preserve PIT
    # semantics in classify_window(lookback_days > 1), mask every session
    # before the trailing training row to NaN. The transition_score consumer
    # treats NaN as "HMM seam absent at this session" and falls back to the
    # 5-component weights_without_hmm path (V1 byte-identity preserved).
    fit_end = frame.index[-1]
    leak_mask = state_prob_frame.index < fit_end
    if leak_mask.any():
        state_prob_frame.loc[leak_mask, :] = float("nan")

    top_state_prob = state_prob_frame.max(axis=1).rename("top_state_prob")
    # state_probabilities.max on an all-NaN row returns NaN — desired
    # cold-start/missing-data propagation. No further masking needed.

    return HMMFeatures(
        top_state_prob=top_state_prob,
        state_probabilities=state_prob_frame,
        n_states=config.n_states,
        selected_seed=int(best["seed"]),
        log_likelihood=float(best["log_likelihood"]),
    )


def _prepare_hmm_frames(
    *,
    train_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    standardize_inputs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not standardize_inputs:
        return train_frame.to_numpy(dtype=float), full_frame.to_numpy(dtype=float)
    means = train_frame.mean()
    stds = train_frame.std(ddof=0).replace(0.0, 1.0)
    train = ((train_frame - means) / stds).to_numpy(dtype=float)
    full = ((full_frame - means) / stds).to_numpy(dtype=float)
    return train, full
