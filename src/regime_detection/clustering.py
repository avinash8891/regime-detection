"""v2 §6.2 GMM clustering evidence layer.

Library reuse: ``sklearn.mixture.GaussianMixture`` provides fit +
``predict_proba`` + per-cluster Mahalanobis distance via the stored
``precisions_`` array. We own the input plumbing, fail-open guards, and
FeatureStore wiring. NO hand-rolled EM / k-means++ / covariance
regularization.

K-Means is documented in the spec (§6.2 line 4191) as an "acceptable
fallback when GMM convergence is unstable". GMM is the only path shipped;
K-Means fallback is a future option for a follow-up slice.

Cluster IDs are raw integers ``0..n_clusters-1``; mapping to economic
labels is operator-side per V2 §10 (line 4378) + the
``cluster_label_map.yaml`` artifact at spec §6.2 line 4233. Never auto-map.

Per V2 §6.2 refit cadence (spec line 4201), ``compute_clustering_features``
re-fits the GMM at every ``retrain_cadence_days`` checkpoint across the
joined frame; each checkpoint scores the
``[train_end_pos, next_train_end_pos)`` segment with a freshly-fit model
trained on the trailing ``training_window_days`` rows. PIT-safe by
construction: every fit's training window ends at ``t' <= t`` for every
emitted session ``t``.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingTypeStubs=false, reportOptionalSubscript=false

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from regime_detection.config import ClusteringConfig

__all__ = ["ClusteringFeatures", "compute_clustering_features"]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteringFeatures:
    """v2 §6.2 — per-session cluster assignment + Mahalanobis distance.

    Attributes:
        cluster_id: nullable-int per session (raw 0..n_clusters-1; NaN on
            sessions dropped by the join-non-NaN mask).
        distance_to_centroid: distance to the *assigned* cluster centroid
            per session, NaN on dropped sessions. Formula depends on
            ``ClusteringConfig.covariance_type``: Mahalanobis (via
            ``precisions_``) when ``covariance_type == "full"`` (the
            shipped default — makes the metric scale-invariant in the
            cluster's principal axes); Euclidean otherwise (`diag`,
            `tied`, `spherical` — ``precisions_`` shapes don't admit the
            per-cluster quadratic form). Spec §6.2 is silent on the
            formula; the default is chosen so the metric carries the
            most information.
        cluster_probabilities: ``(n_sessions × n_clusters)`` posterior
            frame with integer column labels ``0..n_clusters-1`` (no
            economic mapping — V2 §10 ABSOLUTE RULE).
        n_clusters: configured number of clusters.
        model_version: spec-pinned model version tag (e.g.
            ``"gmm_8cluster_v1.0"``).
    """

    cluster_id: pd.Series
    distance_to_centroid: pd.Series
    cluster_probabilities: pd.DataFrame
    n_clusters: int
    model_version: str


def compute_clustering_features(
    *,
    return_21d: pd.Series | None,
    return_63d: pd.Series | None,
    realized_vol_21d: pd.Series | None,
    drawdown_63d: pd.Series | None,
    adx_14: pd.Series | None,
    avg_pairwise_corr_63d: pd.Series | None,
    pct_above_50dma: pd.Series | None,
    config: ClusteringConfig,
) -> ClusteringFeatures | None:
    """Fit ``sklearn.mixture.GaussianMixture`` and return per-session
    cluster IDs + probabilities + Mahalanobis distance to the assigned
    centroid, aligned to the canonical (``return_21d``) index.

    Returns ``None`` when:
      - any required input is ``None``,
      - the joined non-NaN inputs have fewer than
        ``training_window_days`` rows,
      - the GMM fit/predict fails (singular covariance, non-convergence) —
        fail-open per AGENTS error policy.

    Permutation invariance: cluster IDs are arbitrary integers. The
    Mahalanobis distance is permutation-invariant by construction (it is
    the distance to the *assigned* centroid, not to centroid ``0``).
    Probabilities are NOT permutation-invariant — but we expose the full
    ``(n_sessions × n_clusters)`` matrix and per-session ID, so the
    operator yaml can apply post-hoc label mapping.
    """
    inputs: dict[str, pd.Series | None] = {
        "return_21d": return_21d,
        "return_63d": return_63d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "adx_14": adx_14,
        "avg_pairwise_corr_63d": avg_pairwise_corr_63d,
        "pct_above_50dma": pct_above_50dma,
    }
    if any(series is None for series in inputs.values()):
        return None

    frame = pd.DataFrame({k: v for k, v in inputs.items()}).dropna(how="any")
    n_train = config.training_window_days
    if len(frame) < n_train:
        return None

    proba_frame = pd.DataFrame(
        float("nan"),
        index=frame.index,
        columns=list(range(config.n_clusters)),
    )
    cluster_id_series = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    distance_series = pd.Series(float("nan"), index=frame.index, dtype="float64")
    successful_fit = False
    cadence = max(1, config.retrain_cadence_days)
    train_end_positions = list(range(n_train - 1, len(frame), cadence))
    if not train_end_positions or train_end_positions[-1] != len(frame) - 1:
        train_end_positions.append(len(frame) - 1)
    for offset, train_end_pos in enumerate(train_end_positions):
        train = frame.iloc[train_end_pos - n_train + 1 : train_end_pos + 1].to_numpy(
            dtype=float
        )
        if not np.any(train.std(axis=0) > 0.0):
            continue
        next_train_end_pos = (
            train_end_positions[offset + 1]
            if offset + 1 < len(train_end_positions)
            else len(frame)
        )
        # Segment is [train_end_pos, next_train_end_pos): all sessions
        # scored by this checkpoint's freshly-fit GMM. PIT-safe because
        # session train_end_pos is the last row in the training window.
        segment_frame = frame.iloc[train_end_pos:next_train_end_pos]
        model = GaussianMixture(
            n_components=config.n_clusters,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            max_iter=config.max_iter,
            reg_covar=config.reg_covar,
        )
        try:
            model.fit(train)
            segment_X = segment_frame.to_numpy(dtype=float)
            proba = model.predict_proba(segment_X)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "GaussianMixture PIT fit/predict skipped for %s: %s",
                frame.index[train_end_pos],
                exc,
            )
            continue
        ids = proba.argmax(axis=1)
        means_for_rows = model.means_[ids]
        diff = segment_X - means_for_rows
        if config.covariance_type == "full":
            precisions_for_rows = model.precisions_[ids]
            quad = np.einsum("ij,ijk,ik->i", diff, precisions_for_rows, diff)
            distances = np.sqrt(np.clip(quad, 0.0, None))
        else:
            distances = np.linalg.norm(diff, axis=1)
        idx = segment_frame.index
        proba_frame.loc[idx, :] = proba
        cluster_id_series.loc[idx] = ids.astype("int64")
        distance_series.loc[idx] = distances
        successful_fit = True
    if not successful_fit:
        return None

    canonical_index = return_21d.index  # type: ignore[union-attr]
    proba_frame = proba_frame.reindex(canonical_index)
    cluster_id_series = cluster_id_series.reindex(canonical_index)
    distance_series = distance_series.reindex(canonical_index)

    return ClusteringFeatures(
        cluster_id=cluster_id_series.rename("cluster_id"),
        distance_to_centroid=distance_series.rename("distance_to_centroid"),
        cluster_probabilities=proba_frame,
        n_clusters=config.n_clusters,
        model_version=config.model_version,
    )
