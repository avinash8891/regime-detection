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
                state_prob_frame.loc[segment_frame.index, :] = best["posterior"]
                latest_fit = best
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
