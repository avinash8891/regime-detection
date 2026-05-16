"""v2 §6.2 GMM clustering evidence layer (Slice 7).

Library reuse: ``sklearn.mixture.GaussianMixture`` provides fit +
``predict_proba`` + per-cluster Mahalanobis distance via the stored
``precisions_`` array. We own the input plumbing, fail-open guards, and
FeatureStore wiring. NO hand-rolled EM / k-means++ / covariance
regularization.

K-Means is documented in the spec (line 2835) as an "acceptable fallback
when GMM convergence is unstable". Slice 7 ships GMM only; the K-Means
fallback is a future option for a follow-up slice.

Cluster IDs are raw integers ``0..n_clusters-1``; mapping to economic
labels is operator-side per V2 §10 + spec line 2837 (``cluster_label_map.yaml``).
Never auto-map.

Per V2 engine statelessness, ``compute_clustering_features`` re-fits ONCE
per ``classify_window`` call on the trailing ``training_window_days`` rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from regime_detection.config import ClusteringConfig

__all__ = ["ClusteringFeatures", "compute_clustering_features"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteringFeatures:
    """v2 §6.2 — per-session cluster assignment + Mahalanobis distance.

    Attributes:
        cluster_id: nullable-int per session (raw 0..n_clusters-1; NaN on
            sessions dropped by the join-non-NaN mask).
        distance_to_centroid: Mahalanobis distance to the *assigned*
            cluster centroid per session, NaN on dropped sessions.
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

    train = frame.tail(n_train).to_numpy(dtype=float)
    # Fail-open guard: when the training window has zero variance across
    # every feature, GMM converges to a degenerate solution (all rows in
    # cluster 0 with epsilon spread from reg_covar). That's not evidence;
    # surface as a seam-absent signal instead — matches the
    # singular-covariance fail-open contract documented in the module
    # docstring.
    if not np.any(train.std(axis=0) > 0.0):
        return None
    model = GaussianMixture(
        n_components=config.n_clusters,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
        max_iter=200,
        reg_covar=1e-6,
    )
    try:
        model.fit(train)
        full_X = frame.to_numpy(dtype=float)
        proba = model.predict_proba(full_X)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "GaussianMixture fit/predict failed; clustering seam returns None: %s",
            exc,
        )
        return None

    cluster_id_arr = proba.argmax(axis=1)

    # Mahalanobis distance to the *assigned* cluster centroid.
    distance_arr = np.empty(len(frame), dtype=float)
    for k in range(config.n_clusters):
        mask = cluster_id_arr == k
        if not mask.any():
            continue
        diff = full_X[mask] - model.means_[k]
        if config.covariance_type == "full":
            # precisions_ shape: (n_components, n_features, n_features)
            precision = model.precisions_[k]
            # Mahalanobis: sqrt(diff @ P @ diff.T) per row
            # Clamp negative values from FP noise before sqrt.
            quad = np.einsum("ij,jk,ik->i", diff, precision, diff)
            distance_arr[mask] = np.sqrt(np.clip(quad, 0.0, None))
        else:
            # tied/diag/spherical — fall back to Euclidean (K-Means style,
            # matches the spec-line-2835 K-Means fallback case).
            distance_arr[mask] = np.linalg.norm(diff, axis=1)

    canonical_index = return_21d.index  # type: ignore[union-attr]
    proba_frame = pd.DataFrame(
        proba,
        index=frame.index,
        columns=list(range(config.n_clusters)),
    ).reindex(canonical_index)
    cluster_id_series = pd.Series(
        cluster_id_arr,
        index=frame.index,
        dtype="Int64",
    ).reindex(canonical_index)
    distance_series = pd.Series(
        distance_arr,
        index=frame.index,
        dtype="float64",
    ).reindex(canonical_index)

    # V1 §2.2 stateless-replay: GMM is fit ONCE on frame.tail(n_train) ending
    # at frame.index[-1]. Cluster assignments + Mahalanobis distances for
    # sessions earlier than that fit-end were derived from parameters
    # trained on (from that earlier session's perspective) future data. Mask
    # them to NaN/NA so classify_window(lookback_days > 1) preserves PIT
    # semantics. The clustering seam is diagnostic-only (not consumed by
    # transition_score), so masking only affects the wire surface for the
    # earlier emitted sessions; the trailing session keeps its real values.
    fit_end = frame.index[-1]
    leak_mask = canonical_index < fit_end
    if leak_mask.any():
        proba_frame.loc[leak_mask, :] = float("nan")
        cluster_id_series.loc[leak_mask] = pd.NA
        distance_series.loc[leak_mask] = float("nan")

    return ClusteringFeatures(
        cluster_id=cluster_id_series.rename("cluster_id"),
        distance_to_centroid=distance_series.rename("distance_to_centroid"),
        cluster_probabilities=proba_frame,
        n_clusters=config.n_clusters,
        model_version=config.model_version,
    )
