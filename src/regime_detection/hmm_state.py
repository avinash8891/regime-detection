"""v2 §6.1 Hidden Markov Model evidence layer.

Reuses existing FeatureStore seams as inputs (``return_1d``,
``realized_vol_21d``, ``drawdown_63d``, ``volume_zscore_20d``,
``avg_pairwise_corr_63d``). Fits a Gaussian HMM per V2 §6.1 (spec line 4076) via
``hmmlearn.GaussianHMM`` — no hand-rolled Baum-Welch. Emits
``top_state_prob`` (permutation-invariant) for consumption by the
``transition_score`` §4.2 ``model_instability`` component.

The HMM is evidence-only per V2 §10 (spec lines 4378-4385): state indices
are raw ``0``..``n-1`` integers, never auto-mapped to economic labels.
Operator mapping is deferred — see ADR 0013 (docs/decisions/0013-evidence-strategy-governance-closeouts.md) and docs/verification/hmm_state_label_map.yaml.

For multi-session output, ``compute_hmm_features`` refits at PIT-safe
checkpoints separated by ``hmm.retrain_cadence_days``. Each checkpoint
trains on the trailing ``training_window_days`` rows ending at that
checkpoint, then scores the segment until the next checkpoint.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

from regime_detection.config import HMMConfig

__all__ = [
    "HMMFeatures",
    "HMMParameterDrift",
    "compute_hmm_features",
    "compute_hmm_parameter_drift",
]

_LOGGER = logging.getLogger(__name__)

# v2 §6.1 operator calibration-review thresholds (spec lines 4434-4468).
# State-mean drift alerts at 20%; the transition-probability shift is a
# separate non-blocking review flag at 30%.
_STATE_MEAN_DRIFT_ALERT_THRESHOLD = 0.20
_TRANSITION_PROB_REVIEW_THRESHOLD = 0.30


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
        return False if self.non_monotonic else super().converged


@dataclass(frozen=True)
class HMMFeatures:
    """v2 §6.1 — per-session HMM posterior + top-state probability.

    Attributes:
        top_state_prob: max-of-posterior per session, NaN on dropped rows.
        state_probabilities: ``(n_sessions × n_states)`` posterior frame
            with columns labeled ``0``..``n_states-1`` (no economic
            mapping — V2 §10 no-auto-label rule).
        n_states: configured number of hidden states.
    """

    top_state_prob: pd.Series
    state_probabilities: pd.DataFrame
    n_states: int
    selected_seed: int
    log_likelihood: float
    # §6.1 calibration-drift report between the last two PIT refit checkpoints
    # (F-025). None when fewer than two checkpoints were fit (no prior to
    # compare against). Advisory only — never feeds the runtime transition score.
    parameter_drift: "HMMParameterDrift | None" = None


def _fit_single_seed(
    *,
    fit_frame: np.ndarray,
    predict_frame: np.ndarray,
    seed: int,
    n_states: int,
    covariance_type: str,
    min_covar: float,
    tol: float,
) -> dict[str, Any]:
    """Fit one GaussianHMM seed and return its posterior + diagnostics.

    Runs at module scope so joblib's loky backend can pickle it; mirrors the
    semantics of the prior inline seed loop (StrictConvergenceMonitor, NaN /
    non_monotonic propagation) so parallelization is output-preserving.
    """
    # Pin BLAS to one thread per worker — without this the 16 workers each
    # spawn 16 OpenBLAS threads and oversubscribe the box, defeating the
    # parallel speedup. setdefault so an operator can still override.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        min_covar=min_covar,
        n_iter=200,
        tol=tol,
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
    non_monotonic = bool(getattr(model.monitor_, "non_monotonic", False))
    posterior = None if non_monotonic else model.predict_proba(predict_frame)
    return {
        "posterior": posterior,
        "seed": seed,
        "log_likelihood": final_log_likelihood,
        "previous_log_likelihood": previous_log_likelihood,
        "delta": delta,
        "non_monotonic": non_monotonic,
        # Fitted parameters for the §6.1 calibration-drift monitor (F-025).
        # means_ are in the per-window standardized space; the caller
        # de-standardizes before comparing checkpoints. getattr keeps the
        # monitor fail-open if a fitter does not expose the parameters.
        "means": None if non_monotonic else getattr(model, "means_", None),
        "transmat": None if non_monotonic else getattr(model, "transmat_", None),
    }


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
    fits PIT-safe ``GaussianHMM`` checkpoints on trailing
    ``training_window_days`` rows, then ``predict_proba`` for each
    checkpoint's forward segment. Sessions with any NaN input are masked
    to NaN in the output (cold-start + missing-data safe).

    Returns ``None`` when:
      - any required input is ``None``
      - the joined non-NaN inputs have fewer than ``training_window_days``
        rows
      - the HMM fit fails (e.g. singular covariance, non-convergence):
        return None (evidence absent) rather than crash the whole
        classify call (fail-open seam).

    Permutation invariance: ``top_state_prob = posterior.max(axis=1)``
    is invariant to state-index permutation. The
    ``state_probabilities`` frame has integer column labels
    ``0``..``n_states-1`` (no economic mapping per V2 §10 no-auto-label rule).
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

    state_prob_frame = pd.DataFrame(
        float("nan"),
        index=frame.index,
        columns=list(range(config.n_states)),
    )
    latest_fit: dict[str, Any] | None = None
    # Findings #3/#4: align HMM state ordering across PIT refit checkpoints
    # so that state IDs are stable (mapped_label, state_persistence_days).
    reference_hmm_means: np.ndarray | None = None
    # F-025: §6.1 calibration drift between consecutive PIT refit checkpoints.
    previous_raw_means: np.ndarray | None = None
    previous_transmat: np.ndarray | None = None
    latest_drift: HMMParameterDrift | None = None
    all_skipped: list[tuple[int, int, float | None, float | None, float | None]] = []
    # joblib's loky workers re-import this module — they always see the
    # real hmmlearn.GaussianHMM, even when a test monkeypatches the
    # module-level symbol. When the symbol has been swapped (typically by
    # pytest), drop to in-process serial execution so the patch takes
    # effect; otherwise dispatch the seed sweep across processes.
    is_patched = getattr(GaussianHMM, "__module__", "") != "hmmlearn.hmm"
    n_workers = 1 if is_patched else min(len(config.random_seeds), os.cpu_count() or 1)
    try:
        train_end_positions = list(
            range(n_train - 1, len(frame), max(1, config.retrain_cadence_days))
        )
        if train_end_positions[-1] != len(frame) - 1:
            train_end_positions.append(len(frame) - 1)
        backend = "threading" if is_patched else "loky"
        with Parallel(n_jobs=n_workers, backend=backend, batch_size=1) as parallel:
            for offset, train_end_pos in enumerate(train_end_positions):
                train_frame = frame.iloc[
                    train_end_pos - n_train + 1 : train_end_pos + 1
                ]
                next_train_end_pos = (
                    train_end_positions[offset + 1]
                    if offset + 1 < len(train_end_positions)
                    else len(frame)
                )
                segment_frame = frame.iloc[train_end_pos:next_train_end_pos]
                fit_frame, predict_frame = _prepare_hmm_frames(
                    train_frame=train_frame,
                    full_frame=segment_frame,
                    standardize_inputs=config.standardize_inputs,
                )
                if not _has_sufficient_distinct_rows(
                    fit_frame, n_states=config.n_states
                ):
                    _LOGGER.warning(
                        "GaussianHMM training window has fewer distinct rows than "
                        "configured states; HMM seam skips checkpoint: "
                        "checkpoint=%s distinct_rows=%s n_states=%s",
                        train_end_pos,
                        len(np.unique(fit_frame, axis=0)),
                        config.n_states,
                    )
                    continue
                best: dict[str, Any] | None = None
                seed_results = parallel(
                    delayed(_fit_single_seed)(
                        fit_frame=fit_frame,
                        predict_frame=predict_frame,
                        seed=int(seed),
                        n_states=config.n_states,
                        covariance_type=config.covariance_type,
                        min_covar=config.min_covar,
                        tol=config.tol,
                    )
                    for seed in config.random_seeds
                )
                for result in seed_results:
                    if result["non_monotonic"]:
                        # Recoverable per-seed rejection is DEBUG-only; the
                        # no-usable-seed path below remains WARNING.
                        _LOGGER.debug(
                            "GaussianHMM skipped non-monotonic seed: "
                            "checkpoint=%s seed=%s log_likelihood=%s "
                            "previous_log_likelihood=%s delta=%s",
                            train_end_pos,
                            result["seed"],
                            result["log_likelihood"],
                            result["previous_log_likelihood"],
                            result["delta"],
                        )
                        all_skipped.append(
                            (
                                train_end_pos,
                                result["seed"],
                                result["log_likelihood"],
                                result["previous_log_likelihood"],
                                result["delta"],
                            )
                        )
                        continue
                    if (
                        best is None
                        or result["log_likelihood"] > best["log_likelihood"]
                    ):
                        best = result
                if best is None:
                    continue

                # De-standardize state means to raw feature units BEFORE
                # alignment so the Hungarian cost matrix uses a scale that
                # is comparable across windows (findings #3, #4).
                raw_means: np.ndarray | None = None
                if best["means"] is not None:
                    raw_means = best["means"]
                    if config.standardize_inputs:
                        train_means = train_frame.mean().to_numpy(dtype=float)
                        train_stds = (
                            train_frame.std(ddof=0)
                            .replace(0.0, 1.0)
                            .to_numpy(dtype=float)
                        )
                        raw_means = best["means"] * train_stds + train_means

                # Align state ordering to previous checkpoint so IDs are
                # stable across refit boundaries (findings #3, #4).
                if reference_hmm_means is not None and raw_means is not None:
                    aligned_post, aligned_raw = _align_posterior_to_reference(
                        best["posterior"], raw_means, reference_hmm_means
                    )
                    best["posterior"] = aligned_post
                    raw_means = aligned_raw
                if raw_means is not None:
                    reference_hmm_means = raw_means

                state_prob_frame.loc[segment_frame.index, :] = best["posterior"]
                latest_fit = best

                # §6.1 drift monitor (F-025): compare de-standardized,
                # aligned state means to the previous checkpoint.
                # Transition probabilities are scale-invariant.
                # Fail-open: skip silently if a fitter did not expose params.
                if raw_means is not None and best["transmat"] is not None:
                    if previous_raw_means is not None and previous_transmat is not None:
                        latest_drift = compute_hmm_parameter_drift(
                            previous_state_means=previous_raw_means,
                            current_state_means=raw_means,
                            previous_transition_matrix=previous_transmat,
                            current_transition_matrix=best["transmat"],
                        )
                        # F-039: surface the §6.1 quarterly-review alerts. The drift
                        # seam was computed and exposed on HmmOutput but never logged,
                        # so the >20% state-mean / >30% transition-prob thresholds
                        # passed silently. Emit a WARNING the moment a refit crosses
                        # either threshold so the operational review receives it.
                        if (
                            latest_drift.state_mean_drift_alert
                            or latest_drift.transition_prob_review_flag
                        ):
                            _LOGGER.warning(
                                "HMM parameter drift alert: state_mean_drift=%.4f "
                                "(alert=%s, threshold=%.2f), max_transition_prob_shift="
                                "%.4f (review=%s, threshold=%.2f)",
                                latest_drift.parameter_drift,
                                latest_drift.state_mean_drift_alert,
                                _STATE_MEAN_DRIFT_ALERT_THRESHOLD,
                                latest_drift.max_transition_prob_shift,
                                latest_drift.transition_prob_review_flag,
                                _TRANSITION_PROB_REVIEW_THRESHOLD,
                            )
                    previous_raw_means = raw_means
                    previous_transmat = best["transmat"]
    except Exception as exc:  # noqa: BLE001
        # Fail-open: degenerate inputs (singular covariance, etc.) should
        # not crash the engine — the seam goes None and downstream falls
        # back to the 5-component transition score.
        _LOGGER.warning(
            "GaussianHMM fit/predict failed; HMM seam returns None: %s", exc
        )
        return None
    if latest_fit is None:
        _LOGGER.warning(
            "GaussianHMM produced no PIT monotonic fit across %d seed(s); "
            "HMM seam returns None. skipped=%s",
            len(config.random_seeds),
            all_skipped[:5],
        )
        return None

    # Re-align to the canonical SPY index (return_1d carries it). Sessions
    # dropped by .dropna() get NaN both in the state_probabilities frame
    # and in the top_state_prob series.
    canonical_index = return_1d.index  # type: ignore[union-attr]
    state_prob_frame = state_prob_frame.reindex(canonical_index)

    top_state_prob = state_prob_frame.max(axis=1).rename("top_state_prob")
    # state_probabilities.max on an all-NaN row returns NaN — desired
    # cold-start/missing-data propagation. No further masking needed.

    return HMMFeatures(
        top_state_prob=top_state_prob,
        state_probabilities=state_prob_frame,
        n_states=config.n_states,
        selected_seed=int(latest_fit["seed"]),
        log_likelihood=float(latest_fit["log_likelihood"]),
        parameter_drift=latest_drift,
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


def _align_posterior_to_reference(
    posterior: np.ndarray,
    current_means: np.ndarray,
    reference_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align HMM state ordering to a reference checkpoint via Hungarian matching.

    HMM state IDs from ``hmmlearn`` are arbitrary after each refit. This
    finds the optimal permutation mapping the current states to the
    reference states (minimising pairwise Euclidean distance on state
    means), then reorders posterior columns and means accordingly.

    Returns ``(permuted_posterior, permuted_means)``.
    """
    cost = np.linalg.norm(
        current_means[:, np.newaxis, :] - reference_means[np.newaxis, :, :],
        axis=2,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    # col_ind[i] = which reference slot new state i maps to.
    # Invert: perm[j] = which new state fills reference slot j.
    n = len(col_ind)
    perm = np.empty(n, dtype=int)
    perm[col_ind] = row_ind

    return posterior[:, perm], current_means[perm]


def _has_sufficient_distinct_rows(frame: np.ndarray, *, n_states: int) -> bool:
    return len(np.unique(frame, axis=0)) >= n_states


@dataclass(frozen=True)
class HMMParameterDrift:
    """v2 §6.1 operator-side HMM calibration-drift report (spec lines 4434-4468).

    Non-blocking review artifact comparing a freshly refit HMM to the prior
    versioned model. "Prior versioned model" is pinned (F-047, ADR 0024) as the
    **immediately preceding in-call PIT refit checkpoint** — ``compute_hmm_features``
    walks the retrain-cadence checkpoints in one call and compares each refit to the
    one before it (``previous_raw_means`` / ``previous_transmat``). It does NOT load a
    persisted artifact from an earlier process; with a single checkpoint there is no
    prior and ``parameter_drift`` is None. Per the §6.1 operational definition,
    ``parameter_drift`` is the maximum over (state × feature) of the relative absolute
    change in state-mean parameters AFTER Hungarian-algorithm alignment of the new
    state indices to the old (so a pure relabel is not counted as drift), and
    ``state_mean_drift_alert`` fires when it exceeds 20%.

    Transition probabilities and covariances are excluded from the alert (they
    drift naturally with the refit window). A separate ``transition_prob_review_flag``
    fires when any aligned transition probability shifts by more than 30%. The §6.1
    30% threshold is pinned (F-052, ADR 0024) as an ABSOLUTE move: because transition
    entries are bounded in ``[0, 1]``, the shift is measured as the maximum absolute
    change (a "30 percentage point" move), NOT a relative change — the only stable
    reading for near-zero probabilities, where a relative ratio explodes. This flag is
    advisory and does NOT block deployment.

    Attributes:
        parameter_drift: max relative state-mean drift after alignment.
        state_mean_drift_alert: ``parameter_drift > state_mean_drift_threshold``.
        max_transition_prob_shift: max absolute aligned transition-prob change.
        transition_prob_review_flag: ``max_transition_prob_shift > transition_prob_review_threshold``.
        alignment: per old state ``s``, the matched new state index.
    """

    parameter_drift: float
    state_mean_drift_alert: bool
    max_transition_prob_shift: float
    transition_prob_review_flag: bool
    alignment: tuple[int, ...]


def compute_hmm_parameter_drift(
    *,
    previous_state_means: np.ndarray,
    current_state_means: np.ndarray,
    previous_transition_matrix: np.ndarray,
    current_transition_matrix: np.ndarray,
    state_mean_drift_threshold: float = _STATE_MEAN_DRIFT_ALERT_THRESHOLD,
    transition_prob_review_threshold: float = _TRANSITION_PROB_REVIEW_THRESHOLD,
) -> HMMParameterDrift:
    """Compare two fitted Gaussian-HMM parameter sets for §6.1 calibration drift.

    Aligns the current state indices to the previous model by Hungarian
    matching on Euclidean state-mean distance, then reports the relative
    state-mean drift alert and the absolute transition-probability review flag.
    Pure function — no model fitting, no I/O. See :class:`HMMParameterDrift`.
    """
    prev_means = np.asarray(previous_state_means, dtype=float)
    curr_means = np.asarray(current_state_means, dtype=float)
    prev_trans = np.asarray(previous_transition_matrix, dtype=float)
    curr_trans = np.asarray(current_transition_matrix, dtype=float)

    if prev_means.shape != curr_means.shape or prev_means.ndim != 2:
        raise ValueError(
            "state-mean arrays must share a (n_states, n_features) shape; got "
            f"{prev_means.shape} vs {curr_means.shape}"
        )
    n_states = prev_means.shape[0]
    for name, matrix in (
        ("previous_transition_matrix", prev_trans),
        ("current_transition_matrix", curr_trans),
    ):
        if matrix.shape != (n_states, n_states):
            raise ValueError(
                f"{name} must be ({n_states}, {n_states}); got {matrix.shape}"
            )

    # Hungarian alignment of new states to old by closest mean (§6.1: "state
    # index permutations across refits are not counted as drift").
    cost = np.linalg.norm(prev_means[:, None, :] - curr_means[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    alignment = np.empty(n_states, dtype=int)
    alignment[row_ind] = col_ind

    aligned_means = curr_means[alignment]
    aligned_trans = curr_trans[np.ix_(alignment, alignment)]

    # Relative state-mean drift: |new - old| / max(|old|, 1e-9), max over
    # (state × feature). Matches the spec's pinned operational form.
    denominator = np.maximum(np.abs(prev_means), 1e-9)
    parameter_drift = float((np.abs(aligned_means - prev_means) / denominator).max())

    # Absolute transition-probability shift (bounded [0,1]) — review only.
    max_transition_prob_shift = float(np.abs(aligned_trans - prev_trans).max())

    return HMMParameterDrift(
        parameter_drift=parameter_drift,
        state_mean_drift_alert=parameter_drift > state_mean_drift_threshold,
        max_transition_prob_shift=max_transition_prob_shift,
        transition_prob_review_flag=(
            max_transition_prob_shift > transition_prob_review_threshold
        ),
        alignment=tuple(int(x) for x in alignment),
    )
