# Audit Findings Resolution — 10 Silent-Wrong-Answer Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, verify, and resolve all 10 findings from the silent-wrong-answer audit — 4 label-affecting bugs and 6 evidence/metadata bugs — with ideal long-term solutions, no patches or shims.

**Architecture:** Three structural fixes retire multiple findings each: (A) Hungarian alignment for GMM/HMM state IDs across PIT refit boundaries resolves findings 2–5; (B) a max-staleness guard pattern for ffilled time series resolves findings 1 and 10; (C) a PIT-safe followthrough_rate rewrite resolves finding 7. Findings 6, 8, and 9 each get independent targeted fixes.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (already in deps — `linear_sum_assignment`), sklearn, hmmlearn, pydantic, pytest.

**Ordering:** Tasks are ordered by structural impact — the Hungarian alignment fix (Tasks 1–4) is the largest and retires 4 findings. Then staleness (Tasks 5–6), lookahead (Task 7), exception scope (Task 8), NaN→False (Task 9), and denominator (Task 10). Each task is independently committable.

---

## Task 1: Hungarian alignment for GMM cluster IDs across PIT refit checkpoints

**Findings resolved:** #2 (cluster mapped_label), #5 (cluster_flip in transition_score)

**Why:** `clustering.py` refits a fresh `GaussianMixture` at every `cadence=21` checkpoint. Each fit assigns cluster IDs via `proba.argmax(axis=1)` (line 165). Since GMM component ordering is arbitrary, cluster 0 at checkpoint N may correspond to cluster 3 at checkpoint N+1. This corrupts `cluster_label_map` (finding #2) and causes spurious `cluster_flip = 1.0` at refit boundaries (finding #5).

**Fix:** After each checkpoint's GMM fit, align the new component ordering to the previous checkpoint's ordering using the Hungarian algorithm on the cost matrix of centroid distances. Permute the new model's `means_`, `covariances_`, `precisions_`, `weights_`, and the posterior columns before writing IDs.

**Files:**
- Modify: `src/regime_detection/clustering.py:125-178`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test — cluster IDs are stable across PIT refit boundaries**

Add to `tests/test_clustering.py`:

```python
def test_cluster_ids_are_stable_across_pit_refit_boundary() -> None:
    """A refit at a new checkpoint must produce IDs aligned to the previous
    checkpoint's component ordering — not arbitrary argmax of the new fit.

    Construct synthetic data where the GMM has 3 clearly separated clusters.
    Run with cadence=1 (refit every session) so every session is a refit
    boundary. Verify that the cluster ID assigned to a given centroid's
    region does not flip when the training window shifts by 1 session.
    """
    from regime_detection.clustering import compute_clustering_features
    from regime_detection._config_evidence_strategy import ClusteringConfig

    np.random.seed(42)
    n = 300
    # Three well-separated clusters in 2D feature space
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    labels_true = np.repeat([0, 1, 2], n // 3)
    data = centers[labels_true] + np.random.randn(n, 2) * 0.3
    idx = pd.bdate_range("2023-01-01", periods=n, freq="B")

    # Build Series for the required inputs (use 2 of the 7 required features,
    # fill the rest with the same data — we only care about ID stability)
    feature_series = {}
    feature_names = [
        "return_21d", "return_63d", "realized_vol_21d", "drawdown_63d",
        "adx_14", "avg_pairwise_corr_63d", "pct_above_50dma",
    ]
    for i, name in enumerate(feature_names):
        col = data[:, i % 2]
        feature_series[name] = pd.Series(col, index=idx, name=name)

    config = ClusteringConfig(
        n_clusters=3,
        training_window_days=100,
        retrain_cadence_days=21,
        random_state=42,
    )

    result = compute_clustering_features(**feature_series, config=config)
    assert result is not None

    # At a refit boundary, the cluster ID for a given data point should not
    # flip. Check: for sessions that straddle two consecutive checkpoint
    # segments, the ID is consistent with the nearest centroid.
    ids = result.cluster_id
    # No spurious flips: within a region of well-separated data, consecutive
    # sessions should have the same cluster ID (the data is constant within
    # each third of the series).
    first_third_ids = ids.iloc[100:n // 3].dropna().unique()
    assert len(first_third_ids) == 1, (
        f"Expected one cluster ID for the first region, got {first_third_ids}"
    )
    second_third_ids = ids.iloc[n // 3 : 2 * n // 3].dropna().unique()
    assert len(second_third_ids) == 1, (
        f"Expected one cluster ID for the second region, got {second_third_ids}"
    )
    third_third_ids = ids.iloc[2 * n // 3 :].dropna().unique()
    assert len(third_third_ids) == 1, (
        f"Expected one cluster ID for the third region, got {third_third_ids}"
    )
    # All three regions have distinct IDs
    all_ids = {first_third_ids[0], second_third_ids[0], third_third_ids[0]}
    assert len(all_ids) == 3, f"Expected 3 distinct cluster IDs, got {all_ids}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_clustering.py::test_cluster_ids_are_stable_across_pit_refit_boundary -xvs 2>&1 | tail -30`
Expected: FAIL — cluster IDs flip at refit boundaries because there is no alignment.

- [ ] **Step 3: Implement Hungarian alignment in `clustering.py`**

Add a helper function `_align_components_to_reference` and call it in the refit loop:

```python
# At the top of clustering.py, add import:
from scipy.optimize import linear_sum_assignment

def _align_components_to_reference(
    model: GaussianMixture,
    proba: np.ndarray,
    reference_means: np.ndarray,
) -> np.ndarray:
    """Permute GMM component ordering to match a reference checkpoint.

    Uses the Hungarian algorithm on the pairwise Euclidean distance matrix
    between the current checkpoint's means and the reference checkpoint's
    means. Returns the permuted posterior probability matrix. Also permutes
    model.means_, model.covariances_, model.weights_, and model.precisions_
    in-place so downstream Mahalanobis distance calculations use the aligned
    components.
    """
    cost = np.linalg.norm(
        model.means_[:, None, :] - reference_means[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.argsort(col_ind[np.argsort(row_ind)])
    proba = proba[:, perm]
    model.means_ = model.means_[perm]
    model.weights_ = model.weights_[perm]
    if hasattr(model, "covariances_"):
        model.covariances_ = model.covariances_[perm]
    if hasattr(model, "precisions_"):
        model.precisions_ = model.precisions_[perm]
    if hasattr(model, "precisions_cholesky_"):
        model.precisions_cholesky_ = model.precisions_cholesky_[perm]
    return proba
```

Then in the refit loop (after `model.fit(train)` and `proba = model.predict_proba(segment_X)`), add:

```python
        # --- existing code at line 157 ---
        proba = model.predict_proba(segment_X)

        # Align component ordering to the previous checkpoint's centroid
        # positions, so cluster IDs are stable across refit boundaries.
        if reference_means is not None:
            proba = _align_components_to_reference(model, proba, reference_means)
        reference_means = model.means_.copy()

        # --- rest of existing code: ids = proba.argmax(axis=1) ---
```

Initialize `reference_means = None` before the loop (around line 128).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_clustering.py::test_cluster_ids_are_stable_across_pit_refit_boundary -xvs 2>&1 | tail -30`
Expected: PASS

- [ ] **Step 5: Run the full clustering test suite to check for regressions**

Run: `python3 -m pytest tests/test_clustering.py -xvs 2>&1 | tail -40`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/regime_detection/clustering.py tests/test_clustering.py
git commit -m "fix(clustering): align GMM component ordering across PIT refit checkpoints via Hungarian algorithm"
```

---

## Task 2: Hungarian alignment for HMM state IDs across PIT refit checkpoints

**Findings resolved:** #3 (HMM mapped_label), #4 (state_persistence_days)

**Why:** Same root cause as Task 1 but in `hmm_state.py`. Each PIT checkpoint refits a fresh `GaussianHMM`, and the state ordering from `hmmlearn` is arbitrary. The `linear_sum_assignment` call at `hmm_state.py:483` only exists inside `compute_hmm_parameter_drift` for drift reporting — it does NOT align the posteriors or state IDs written to the output. This means `mapped_label` and `state_persistence_days` both consume unstable raw IDs.

**Fix:** After each checkpoint's best HMM fit, align state ordering to the previous checkpoint using Hungarian on state means. Permute the posterior columns before writing to `state_prob_frame`.

**Files:**
- Modify: `src/regime_detection/hmm_state.py:220-360`
- Test: `tests/test_hmm_state.py`

- [ ] **Step 1: Write the failing test — HMM state IDs are stable across PIT refit boundaries**

Add to `tests/test_hmm_state.py`:

```python
def test_hmm_state_ids_are_stable_across_pit_refit_boundary() -> None:
    """State IDs must be aligned across PIT refit checkpoints so that
    state_persistence_days and mapped_label consume stable identifiers.

    Construct synthetic data with 2 clearly separated regimes (low-vol and
    high-vol). Refit at cadence=21. The state assigned to the low-vol regime
    should be the same integer before and after a refit boundary.
    """
    from regime_detection.hmm_state import compute_hmm_features
    from regime_detection._config_evidence_strategy import HMMConfig

    np.random.seed(42)
    n = 400
    idx = pd.bdate_range("2022-01-01", periods=n, freq="B")

    # Two clearly separated regimes: sessions 0-199 are "calm" (low vol),
    # sessions 200-399 are "stress" (high vol).
    returns = np.concatenate([
        np.random.randn(200) * 0.005,  # calm
        np.random.randn(200) * 0.03,   # stress
    ])
    vol = np.concatenate([
        np.abs(np.random.randn(200)) * 0.01,  # calm
        np.abs(np.random.randn(200)) * 0.05,  # stress
    ])

    close = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)
    realized_vol = pd.Series(vol, index=idx)
    return_21d = close.pct_change(21)
    return_63d = close.pct_change(63)
    realized_vol_21d = realized_vol.rolling(21).mean()
    drawdown_63d = (close / close.rolling(63).max() - 1).abs()

    config = HMMConfig(
        n_states=2,
        training_window_days=100,
        retrain_cadence_days=21,
        random_state=42,
        random_seeds=(42, 101, 202),
    )

    result = compute_hmm_features(
        return_21d=return_21d,
        return_63d=return_63d,
        realized_vol_21d=realized_vol_21d,
        drawdown_63d=drawdown_63d,
        config=config,
    )
    assert result is not None

    # The top_state for sessions in the calm regime should be the same
    # integer across all refit checkpoints.
    calm_states = result.top_state.iloc[150:200].dropna().unique()
    assert len(calm_states) == 1, (
        f"Expected one state ID for calm regime, got {calm_states}"
    )

    # The top_state for sessions in the stress regime should be a
    # different, but consistent, integer.
    stress_states = result.top_state.iloc[250:350].dropna().unique()
    assert len(stress_states) == 1, (
        f"Expected one state ID for stress regime, got {stress_states}"
    )

    assert calm_states[0] != stress_states[0], (
        "Calm and stress regimes should have different state IDs"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hmm_state.py::test_hmm_state_ids_are_stable_across_pit_refit_boundary -xvs 2>&1 | tail -30`
Expected: FAIL — state IDs flip across refit boundaries.

- [ ] **Step 3: Implement Hungarian alignment in `hmm_state.py`**

Add a helper function and use it in the PIT refit loop:

```python
# At the top of hmm_state.py, add import:
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment_scipy

def _align_posterior_to_reference(
    posterior: np.ndarray,
    current_means: np.ndarray,
    reference_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute HMM state columns to align with a reference checkpoint's ordering.

    Returns (permuted_posterior, permuted_means).
    """
    cost = np.linalg.norm(
        current_means[:, None, :] - reference_means[None, :, :], axis=2
    )
    row_ind, col_ind = _linear_sum_assignment_scipy(cost)
    perm = np.argsort(col_ind[np.argsort(row_ind)])
    return posterior[:, perm], current_means[perm]
```

Then in the PIT refit loop, after `state_prob_frame.loc[segment_frame.index, :] = best["posterior"]` (line 301), insert alignment:

```python
                # Align state ordering to the previous checkpoint so IDs
                # are stable across PIT refit boundaries.
                posterior = best["posterior"]
                current_means = best["means"]
                if reference_hmm_means is not None and current_means is not None:
                    posterior, current_means = _align_posterior_to_reference(
                        posterior, current_means, reference_hmm_means,
                    )
                if current_means is not None:
                    reference_hmm_means = current_means.copy()
                state_prob_frame.loc[segment_frame.index, :] = posterior
                latest_fit = best
```

Initialize `reference_hmm_means = None` before the loop.

**Important:** The alignment of `current_means` must happen BEFORE the drift computation, so the de-standardized `raw_means` on line 310-318 use the already-aligned means. Adjust the drift block to use the aligned means.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hmm_state.py::test_hmm_state_ids_are_stable_across_pit_refit_boundary -xvs 2>&1 | tail -30`
Expected: PASS

- [ ] **Step 5: Run the full HMM test suite to check for regressions**

Run: `python3 -m pytest tests/test_hmm_state.py -xvs 2>&1 | tail -60`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/regime_detection/hmm_state.py tests/test_hmm_state.py
git commit -m "fix(hmm): align HMM state ordering across PIT refit checkpoints via Hungarian algorithm"
```

---

## Task 3: cluster_flip now uses aligned IDs (regression test only)

**Finding resolved:** #5 (cluster_flip in transition_score)

**Why:** After Task 1, cluster IDs are stable across refit boundaries. `cluster_flip` at `transition_score.py:257` compares `cluster_id_now != cluster_id_5d_ago` — with aligned IDs, this comparison is now meaningful. No code change is needed in `transition_score.py`; this task adds a regression test to lock the fix.

**Files:**
- Test: `tests/test_transition_score_v2.py`

- [ ] **Step 1: Write the regression test**

Add to `tests/test_transition_score_v2.py`:

```python
def test_cluster_flip_does_not_fire_at_aligned_refit_boundary() -> None:
    """After Hungarian alignment in clustering.py, cluster_flip should NOT
    fire at a PIT refit boundary when the underlying regime is stable.

    Feed transition_score with cluster IDs that are stable (same ID for
    now and 5d ago). Verify model_instability does not spike from a
    spurious cluster_flip.
    """
    from regime_detection.transition_score import compose_transition_score_for_session

    # Stable cluster: same ID now and 5 days ago
    result = compose_transition_score_for_session(
        hmm_state_now=0,
        hmm_state_5d_ago=0,
        change_point_score=0.1,
        cluster_id_now=2,
        cluster_id_5d_ago=2,
        transition_score_config=_default_transition_score_config(),
    )
    # cluster_flip should be 0.0
    assert result.components["cluster_flip"] == 0.0
    assert result.components["model_instability"] < 1.0

    # Genuine flip: different IDs
    result_flip = compose_transition_score_for_session(
        hmm_state_now=0,
        hmm_state_5d_ago=0,
        change_point_score=0.1,
        cluster_id_now=2,
        cluster_id_5d_ago=5,
        transition_score_config=_default_transition_score_config(),
    )
    assert result_flip.components["cluster_flip"] == 1.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_transition_score_v2.py::test_cluster_flip_does_not_fire_at_aligned_refit_boundary -xvs 2>&1 | tail -20`
Expected: PASS (the test validates correct behavior; the fix is upstream in Task 1)

- [ ] **Step 3: Commit**

```bash
git add tests/test_transition_score_v2.py
git commit -m "test(transition_score): lock regression — cluster_flip uses aligned IDs from Task 1 fix"
```

---

## Task 4: HMM state_persistence_days now uses aligned IDs (regression test only)

**Finding resolved:** #4 (state_persistence_days)

**Why:** After Task 2, HMM state IDs are stable across refit boundaries. `_hmm_state_persistence_days` at `timeline.py:265` compares `int(prev) != int(current_state)` — with aligned IDs, this comparison is now meaningful. No code change needed; add a regression test.

**Files:**
- Test: `tests/test_schema_and_timeline.py`

- [ ] **Step 1: Write the regression test**

Add to `tests/test_schema_and_timeline.py`:

```python
def test_hmm_persistence_days_stable_across_refit_boundary() -> None:
    """state_persistence_days should not spuriously reset at a PIT refit
    boundary when the underlying physical state is unchanged.

    With aligned state IDs (Task 2 fix), consecutive sessions that belong
    to the same economic regime should produce a persistence count equal to
    the actual run length, not a truncated count at the refit boundary.
    """
    from regime_detection.timeline import _hmm_state_persistence_days

    # 20 sessions, all in the same state (state 1). No refit boundary in the
    # raw data — the alignment fix ensures this is what the function sees.
    idx = pd.bdate_range("2025-01-01", periods=20, freq="B")
    stable_states = pd.Series([1] * 20, index=idx, dtype="Int64")
    target = idx[15]

    persistence = _hmm_state_persistence_days(stable_states, pd.Timestamp(target))
    assert persistence == 15, (
        f"Expected 15 days of persistence for a stable state, got {persistence}"
    )

    # A genuine state change at session 10 should reset persistence.
    changing_states = pd.Series(
        [0] * 10 + [1] * 10, index=idx, dtype="Int64"
    )
    persistence_after_change = _hmm_state_persistence_days(
        changing_states, pd.Timestamp(target)
    )
    assert persistence_after_change == 5, (
        f"Expected 5 days after state change at session 10, got {persistence_after_change}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schema_and_timeline.py::test_hmm_persistence_days_stable_across_refit_boundary -xvs 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_schema_and_timeline.py
git commit -m "test(timeline): lock regression — hmm persistence_days uses aligned IDs from Task 2 fix"
```

---

## Task 5: AAII sentiment max-staleness guard

**Finding resolved:** #1 (AAII sentiment unbounded ffill → wrong euphoria label)

**Why:** `_build_sentiment_score_series` at `_feature_specs.py:276` forward-fills AAII data indefinitely via `aligned.reindex(session_index, method="ffill")`. If AAII stops publishing, the last reading persists forever, potentially keeping `euphoria` firing (or suppressing it) based on stale data. The `CentralBankTextConfig.max_release_age_days` pattern is the existing solution for this class of bug.

**Fix:** Create a `SentimentScoreConfig` with `max_staleness_sessions` (default 40 = ~8 weeks, matching the AAII 8-week MA smoothing window). After the ffill, NaN-out any session whose last real AAII reading is older than the staleness cap. This makes stale sentiment_score NaN, which the euphoria gate at `trend_direction_rules.py:341-345` already handles (returns False on NaN).

**Files:**
- Modify: `src/regime_detection/_config_layer2.py` (add `SentimentScoreConfig`)
- Modify: `src/regime_detection/_feature_specs.py:216-292` (consume the config)
- Modify: `src/regime_detection/config.py` (wire `SentimentScoreConfig` into `RegimeConfig`)
- Test: `tests/test_trend_direction.py` or new `tests/test_sentiment_staleness.py`

- [ ] **Step 1: Write the failing test**

```python
def test_aaii_sentiment_goes_nan_after_max_staleness() -> None:
    """If the last AAII reading is older than max_staleness_sessions,
    sentiment_score should be NaN — not a stale forward-fill.
    """
    from regime_detection._feature_specs import _build_sentiment_score_series
    from regime_detection._config_layer2 import SentimentScoreConfig

    idx = pd.bdate_range("2025-01-01", periods=100, freq="B")
    # 5 weekly AAII readings in the first 35 sessions, then nothing.
    aaii_dates = idx[[0, 5, 10, 15, 20]]
    aaii_df = pd.DataFrame({
        "bull_bear_spread": [10.0, 15.0, 20.0, 25.0, 30.0],
    }, index=aaii_dates)

    config = SentimentScoreConfig(max_staleness_sessions=40)
    result = _build_sentiment_score_series(
        aaii_sentiment=aaii_df,
        session_index=idx,
        config=config,
    )
    assert result is not None

    # Session 50 (30 sessions after last reading) — within staleness window
    assert not pd.isna(result.iloc[50]), "Should still be valid at session 50"

    # Session 70 (50 sessions after last reading) — beyond 40-session cap
    assert pd.isna(result.iloc[70]), (
        f"Should be NaN at session 70 (50 sessions stale), got {result.iloc[70]}"
    )

    # Session 99 — still NaN
    assert pd.isna(result.iloc[99]), "Should be NaN at end of series"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sentiment_staleness.py::test_aaii_sentiment_goes_nan_after_max_staleness -xvs 2>&1 | tail -20`
Expected: FAIL — `_build_sentiment_score_series` does not accept `config` yet.

- [ ] **Step 3: Create `SentimentScoreConfig` in `_config_layer2.py`**

Add after `NewsSentimentConfig` (around line 91):

```python
class SentimentScoreConfig(StrictBaseModel):
    """AAII bull-bear sentiment staleness guard.

    max_staleness_sessions: after this many NYSE sessions without a fresh
    AAII reading, the forward-filled value is replaced with NaN. Default 40
    (~8 weeks) matches the 8-week MA smoothing the spec prescribes.
    """

    max_staleness_sessions: int = Field(default=40, gt=0)
```

- [ ] **Step 4: Add staleness enforcement to `_build_sentiment_score_series`**

Modify `_feature_specs.py`. Add `config: SentimentScoreConfig` parameter. After the ffill at line 276, add:

```python
    # Staleness cap: NaN-out sessions whose last real reading is older than
    # max_staleness_sessions.
    raw_dates = aligned.dropna().index
    if len(raw_dates) > 0:
        last_real = aligned.reindex(session_index).notna()
        sessions_since_real = (~last_real).astype(int).groupby(
            last_real.cumsum()
        ).cumsum()
        result = result.where(sessions_since_real <= config.max_staleness_sessions)
```

- [ ] **Step 5: Wire `SentimentScoreConfig` into `RegimeConfig` and callers**

Update `config.py` to include the new config. Update the caller in `_feature_specs.py` that invokes `_build_sentiment_score_series` to pass the config through.

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sentiment_staleness.py -xvs 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 7: Run the full trend_direction test suite**

Run: `python3 -m pytest tests/test_trend_direction*.py -xvs 2>&1 | tail -40`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/regime_detection/_config_layer2.py src/regime_detection/_feature_specs.py src/regime_detection/config.py tests/test_sentiment_staleness.py
git commit -m "fix(sentiment): add max-staleness guard for AAII sentiment ffill — prevents stale euphoria"
```

---

## Task 6: News sentiment max-staleness guard

**Finding resolved:** #10 (news_sentiment_score unbounded staleness)

**Why:** Same pattern as Task 5 but for SF Fed news sentiment. Currently evidence-only (not label-affecting), but the TODO at `trend_direction.py:74-76` flags promotion to a confidence gate. Fix it now so promotion doesn't introduce a staleness bug.

**Fix:** Add `max_staleness_sessions` to `NewsSentimentConfig`. Apply the same staleness enforcement pattern after the ffill.

**Files:**
- Modify: `src/regime_detection/_config_layer2.py:73-91` (add field to `NewsSentimentConfig`)
- Modify: `src/regime_detection/_feature_specs.py:295-307` (enforce staleness)
- Test: `tests/test_sentiment_staleness.py`

- [ ] **Step 1: Write the failing test**

```python
def test_news_sentiment_goes_nan_after_max_staleness() -> None:
    """news_sentiment_score should go NaN when the last SF Fed reading
    is older than max_staleness_sessions."""
    from regime_detection._feature_specs import _build_news_sentiment_score_series
    from regime_detection._config_layer2 import NewsSentimentConfig

    idx = pd.bdate_range("2025-01-01", periods=100, freq="B")
    # Daily readings for 20 sessions, then nothing.
    news = pd.Series(0.5, index=idx[:20], name="news_sentiment")

    config = NewsSentimentConfig(
        smoothing_window_sessions=5,
        max_staleness_sessions=30,
    )
    result = _build_news_sentiment_score_series(
        news_sentiment=news,
        session_index=idx,
        config=config,
    )

    # Session 40 (20 sessions stale) — within staleness window
    assert not pd.isna(result.iloc[40]), "Should still be valid at session 40"

    # Session 60 (40 sessions stale) — beyond 30-session cap
    assert pd.isna(result.iloc[60]), (
        f"Should be NaN at session 60, got {result.iloc[60]}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `NewsSentimentConfig` does not have `max_staleness_sessions` yet.

- [ ] **Step 3: Add `max_staleness_sessions` to `NewsSentimentConfig`**

```python
class NewsSentimentConfig(StrictBaseModel):
    smoothing_window_sessions: int = Field(default=21, gt=0)
    max_staleness_sessions: int = Field(default=63, gt=0)
```

Default 63 = ~3 months of NYSE sessions — conservative for a daily publication.

- [ ] **Step 4: Enforce staleness in `_build_news_sentiment_score_series`**

After the rolling mean at `_feature_specs.py:302-306`, add the same staleness cap logic from Task 5.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sentiment_staleness.py -xvs 2>&1 | tail -20`

- [ ] **Step 6: Commit**

```bash
git add src/regime_detection/_config_layer2.py src/regime_detection/_feature_specs.py tests/test_sentiment_staleness.py
git commit -m "fix(news_sentiment): add max-staleness guard for SF Fed news sentiment ffill"
```

---

## Task 7: PIT-safe followthrough_rate — eliminate lookahead bias

**Finding resolved:** #7 (followthrough_rate uses future closes)

**Why:** `_compute_followthrough_rate` at `trend_character.py:159-172` precomputes `held[b]` using `close_arr[b+1..b+hold_sessions]` over the entire working context. For a session `t`, if a breakout `b` happened at `t-2`, `held[b]` was computed using `close_arr[b+1..b+hold_sessions]` which includes closes at `t+1..t+hold_sessions-2` — future data relative to `t`. This breaks the classify_series PIT replay contract.

**Fix:** Rewrite `_compute_followthrough_rate` so that `held[b]` is evaluated **at query time `t`**, only using closes up to `t`. For each session `t`, a breakout `b` is "held as of `t`" iff `close[b+1..min(b+hold_sessions, t)] > breakout_level[b]` AND `min(b+hold_sessions, t) - b >= hold_sessions` (i.e., the full hold window has elapsed). If the hold window hasn't fully elapsed yet, the breakout is "pending" — neither held nor not-held — and should be excluded from the ft_rate denominator.

**Files:**
- Modify: `src/regime_detection/trend_character.py:116-200`
- Test: `tests/test_trend_character.py`

- [ ] **Step 1: Write the failing test — PIT equivalence**

```python
def test_followthrough_rate_is_pit_safe() -> None:
    """followthrough_rate[t] must produce the same value whether computed
    from the full series or from close[:t+1] (the PIT truncation).

    This catches lookahead bias: if held[b] uses close[b+hold..] that
    extends past t, the full-series and truncated-series results diverge.
    """
    from regime_detection.trend_character import _compute_followthrough_rate, _compute_breakout_20d_or_50d

    np.random.seed(42)
    n = 200
    idx = pd.bdate_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)), index=idx
    )
    breakout = _compute_breakout_20d_or_50d(close)

    full_ft = _compute_followthrough_rate(close, breakout)

    # For each session t from 100..150, compute ft from truncated series
    # and compare to the full-series value.
    mismatches = []
    for t in range(100, 150):
        trunc_close = close.iloc[: t + 1]
        trunc_breakout = _compute_breakout_20d_or_50d(trunc_close)
        trunc_ft = _compute_followthrough_rate(trunc_close, trunc_breakout)

        full_val = full_ft.iloc[t]
        trunc_val = trunc_ft.iloc[t]

        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        if pd.isna(full_val) != pd.isna(trunc_val):
            mismatches.append((t, full_val, trunc_val))
            continue
        if abs(full_val - trunc_val) > 1e-12:
            mismatches.append((t, full_val, trunc_val))

    assert not mismatches, (
        f"followthrough_rate diverged at {len(mismatches)} sessions "
        f"(first 5: {mismatches[:5]}). This indicates lookahead bias."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trend_character.py::test_followthrough_rate_is_pit_safe -xvs 2>&1 | tail -30`
Expected: FAIL — mismatches found.

- [ ] **Step 3: Rewrite `_compute_followthrough_rate` to be PIT-safe**

The key change: `held[b]` must be evaluated relative to query time `t`, not precomputed over the full series. A breakout `b` is "held as of `t`" iff:
1. `t >= b + hold_sessions` (the full hold window has elapsed by time `t`)
2. `close[b+1..b+hold_sessions] > breakout_level[b]` (using only data available at `b+hold_sessions <= t`)

Since condition 1 means we only look at `close[b+1..b+hold_sessions]` and `b+hold_sessions <= t`, there is no lookahead. Breakouts where `t < b + hold_sessions` are excluded from the ft_rate computation entirely (not yet resolved).

```python
def _compute_followthrough_rate(
    close: pd.Series,
    breakout_20d_or_50d: pd.Series,
    *,
    lookback_sessions: int = _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    window_count: int = _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
    hold_sessions: int = _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
) -> pd.Series:
    """PIT-safe followthrough rate.

    For each session t, look back at the most recent `window_count` breakouts
    that are fully *resolved* as of t (i.e., b + hold_sessions <= t). A
    resolved breakout is "held" iff close stayed above breakout_level for the
    full hold window. Unresolved breakouts (too recent) are excluded.
    """
    n = len(close)
    close_arr = close.to_numpy(dtype=float)
    is_breakout = breakout_20d_or_50d.fillna(False).to_numpy(dtype=bool)

    prior_max_20 = close.shift(1).rolling(20).max().to_numpy(dtype=float)
    prior_max_50 = close.shift(1).rolling(50).max().to_numpy(dtype=float)

    breakout_level = np.full(n, np.nan, dtype=float)
    for b in range(n):
        if not is_breakout[b]:
            continue
        m20 = prior_max_20[b]
        m50 = prior_max_50[b]
        if not np.isnan(m20) and close_arr[b] > m20:
            breakout_level[b] = m20
        elif not np.isnan(m50) and close_arr[b] > m50:
            breakout_level[b] = m50

    # held[b]: True iff close[b+1..b+hold_sessions] > breakout_level[b].
    # Only computed for breakouts where b + hold_sessions < n. This is the
    # same computation as before, but now the result is ONLY consumed for
    # sessions t >= b + hold_sessions (enforced below), so there is no
    # lookahead.
    held = np.zeros(n, dtype=bool)
    resolved_at = np.full(n, -1, dtype=np.int64)  # session when this breakout resolves
    for b in range(n):
        if np.isnan(breakout_level[b]):
            continue
        end = b + hold_sessions
        if end >= n:
            continue
        resolved_at[b] = end
        level = breakout_level[b]
        if np.all(close_arr[b + 1 : end + 1] > level):
            held[b] = True

    # For each session t, collect the most recent `window_count` RESOLVED
    # breakouts (resolved_at[b] <= t) within lookback_sessions of t.
    out = np.full(n, np.nan, dtype=float)
    breakout_indices = np.flatnonzero(~np.isnan(breakout_level))

    for t in range(n):
        # Collect resolved breakouts in [t - lookback_sessions, t] window
        count = 0
        held_count = 0
        for bi in range(len(breakout_indices) - 1, -1, -1):
            b = breakout_indices[bi]
            if b > t - 1:
                continue
            if b < t - lookback_sessions:
                break
            if resolved_at[b] < 0 or resolved_at[b] > t:
                continue  # not yet resolved as of t
            count += 1
            if held[b]:
                held_count += 1
            if count >= window_count:
                break
        if count >= window_count:
            out[t] = held_count / window_count

    return pd.Series(out, index=close.index)
```

- [ ] **Step 4: Run PIT-safety test**

Run: `python3 -m pytest tests/test_trend_character.py::test_followthrough_rate_is_pit_safe -xvs 2>&1 | tail -30`
Expected: PASS

- [ ] **Step 5: Run the full trend_character test suite**

Run: `python3 -m pytest tests/test_trend_character.py tests/test_trend_character_v2_labels.py -xvs 2>&1 | tail -60`
Expected: All tests PASS. Some pinned-output tests may need fixture updates since the PIT-safe computation produces different (correct) values. Update fixtures if needed — the old fixtures encoded the lookahead bug.

- [ ] **Step 6: Commit**

```bash
git add src/regime_detection/trend_character.py tests/test_trend_character.py
git commit -m "fix(trend_character): rewrite followthrough_rate to be PIT-safe — eliminate lookahead bias"
```

---

## Task 8: Narrow the HMM `except Exception` to documented failure modes

**Finding resolved:** #6 (HMM except Exception masks programming bugs)

**Why:** The `except Exception` at `hmm_state.py:348` wraps 125 lines of business logic (PIT loop, joblib dispatch, drift computation). The docstring says it should only catch singular covariance and non-convergence. But it catches `KeyError`, `TypeError`, `IndexError`, `ValueError` from programming bugs too, mapping them to `reason="not_configured"` — indistinguishable from "HMM disabled."

**Fix:** Replace the single broad `except Exception` with targeted exception handling:
1. Move the `try/except` inside the per-checkpoint loop body (same scope as the `clustering.py:154-164` pattern).
2. Catch only the documented failure modes: `np.linalg.LinAlgError` (singular covariance), `ValueError` (non-convergence / degenerate input from hmmlearn).
3. Let programming bugs (`KeyError`, `TypeError`, `IndexError`, `AttributeError`) propagate.

**Files:**
- Modify: `src/regime_detection/hmm_state.py:222-355`
- Test: `tests/test_hmm_state.py`

- [ ] **Step 1: Write the failing test — programming bugs must not be swallowed**

```python
def test_compute_hmm_features_raises_on_programming_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """A programming bug (e.g. KeyError) inside the HMM PIT loop should
    propagate, not be swallowed as 'not_configured'.
    """
    from regime_detection.hmm_state import compute_hmm_features, _fit_single_seed
    from regime_detection._config_evidence_strategy import HMMConfig

    # Monkeypatch _fit_single_seed to raise a KeyError (simulating a bug)
    def _buggy_fit(*args, **kwargs):
        raise KeyError("this is a programming bug, not a degenerate input")

    monkeypatch.setattr("regime_detection.hmm_state._fit_single_seed", _buggy_fit)

    # Build minimal valid inputs
    idx = pd.bdate_range("2020-01-01", periods=300, freq="B")
    config = HMMConfig(
        n_states=2,
        training_window_days=100,
        retrain_cadence_days=21,
        random_seeds=(42,),
    )
    series = pd.Series(np.random.randn(300).cumsum(), index=idx)

    with pytest.raises(KeyError, match="programming bug"):
        compute_hmm_features(
            return_21d=series,
            return_63d=series,
            realized_vol_21d=series.abs(),
            drawdown_63d=series.abs(),
            config=config,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hmm_state.py::test_compute_hmm_features_raises_on_programming_bug -xvs 2>&1 | tail -20`
Expected: FAIL — KeyError is caught by `except Exception` and returns None.

- [ ] **Step 3: Narrow the exception handler**

Replace the outer `try/except Exception` (lines 222-355) with per-checkpoint try/except inside the loop:

```python
    # Remove the outer try/except at line 222 and 348-355.
    # Inside the per-checkpoint loop (around line 258-302), wrap only the
    # fit/predict section:

                try:
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
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _LOGGER.warning(
                        "GaussianHMM checkpoint skipped (degenerate input): "
                        "checkpoint=%s error=%s",
                        train_end_pos,
                        exc,
                    )
                    continue
```

Keep the loop structure and drift computation outside the try/except so programming bugs in those sections propagate normally.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hmm_state.py::test_compute_hmm_features_raises_on_programming_bug -xvs 2>&1 | tail -20`
Expected: PASS — KeyError propagates.

- [ ] **Step 5: Verify documented failure modes still return None gracefully**

Run: `python3 -m pytest tests/test_hmm_state.py::test_compute_hmm_features_returns_none_when_hmm_fit_fails -xvs 2>&1 | tail -20`
Expected: PASS — singular covariance still returns None.

- [ ] **Step 6: Run the full HMM test suite**

Run: `python3 -m pytest tests/test_hmm_state.py -xvs 2>&1 | tail -60`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/regime_detection/hmm_state.py tests/test_hmm_state.py
git commit -m "fix(hmm): narrow except Exception to documented failure modes — let programming bugs propagate"
```

---

## Task 9: Volume NaN → data_quality evidence instead of silent False

**Finding resolved:** #8 (NaN→False volume suppresses breakout_expansion)

**Why:** When volume data is NaN (missing), `trend_character_rules.py:105` does `fillna(False)`, silently suppressing `breakout_expansion`. The axis builder's `required_inputs` omits volume, so there's no data-quality signal that the label was degraded.

**Fix:** Two changes:
1. Add `volume` to `required_inputs` in `axis_builders/trend_character.py` so missing volume is surfaced in `DataQuality`.
2. When volume is partially NaN (present column but missing values), preserve NaN in `vol_above` and exclude volume from the `breakout_expansion` predicate for those sessions. Instead of `valid & breakout_flag & bb_expanding & vol_above`, use `valid & breakout_flag & bb_expanding & (vol_above | vol_unknown)` where `vol_unknown` means "volume data was NaN, so we can't confirm but we don't want to falsify."

Actually, the simpler and more correct fix: when volume is absent, `breakout_expansion` should not be gated on volume at all — the predicate should only include `vol_above` when volume data exists.

**Files:**
- Modify: `src/regime_detection/axis_builders/trend_character.py:59`
- Modify: `src/regime_detection/trend_character_rules.py:100-130`
- Test: `tests/test_trend_character_v2_labels.py`

- [ ] **Step 1: Write the failing test**

```python
def test_breakout_expansion_fires_when_volume_is_missing() -> None:
    """When volume data is NaN/missing, breakout_expansion should not be
    suppressed — the vol_above gate should be bypassed, not falsified.
    """
    from regime_detection.trend_character_rules import build_raw_outputs
    from regime_detection.trend_character import TrendCharacterFeatures

    # Construct features where breakout, bb_expanding, and ft_rate all pass,
    # but volume_above_20d_average is NaN.
    idx = pd.bdate_range("2025-01-01", periods=5, freq="B")
    features = TrendCharacterFeatures(
        breakout_20d_or_50d=pd.Series([False, False, False, True, True], index=idx),
        bb_expanding=pd.Series([False, False, False, True, True], index=idx),
        volume_above_20d_average=pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=idx),
        followthrough_rate=pd.Series([np.nan, np.nan, np.nan, 0.8, 0.8], index=idx),
        # ... fill other required fields with valid data
    )
    config = _default_trend_character_rules_config()
    result = build_raw_outputs(features, config)

    # breakout_expansion should fire at sessions 3 and 4 — vol_above is
    # unknown (NaN), not False.
    assert result.raw_label.iloc[3] == "breakout_expansion", (
        f"Expected breakout_expansion when volume is NaN, got {result.raw_label.iloc[3]}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `fillna(False)` kills breakout_expansion.

- [ ] **Step 3: Fix the predicate**

In `trend_character_rules.py`, replace `vol_above = f.volume_above_20d_average.fillna(False).astype(bool)` with:

```python
    vol_available = f.volume_above_20d_average.notna()
    vol_above = f.volume_above_20d_average.fillna(False).astype(bool)
    # When volume data is unavailable, do not falsify breakout_expansion.
    vol_gate = vol_above | ~vol_available
```

Then in the `breakout_expansion` predicate, use `vol_gate` instead of `vol_above`:

```python
    breakout_expansion = (
        valid
        & breakout_flag
        & bb_expanding
        & vol_gate
        & ft_rate.notna()
        & ft_rate.ge(followthrough_rate_threshold)
    )
```

- [ ] **Step 4: Add volume to `required_inputs` in axis builder**

In `axis_builders/trend_character.py:59`, add the volume series to `required_inputs` so missing volume is surfaced in `DataQuality`:

```python
        required_inputs=[close, context.spy_ohlcv["high"], context.spy_ohlcv["low"], context.spy_ohlcv["volume"]],
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trend_character_v2_labels.py::test_breakout_expansion_fires_when_volume_is_missing -xvs 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 6: Run the full trend_character test suite**

Run: `python3 -m pytest tests/test_trend_character.py tests/test_trend_character_v2_labels.py -xvs 2>&1 | tail -60`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/regime_detection/trend_character_rules.py src/regime_detection/axis_builders/trend_character.py tests/test_trend_character_v2_labels.py
git commit -m "fix(trend_character): stop falsifying breakout_expansion when volume data is missing"
```

---

## Task 10: Fix upvol_downvol_ratio denominator

**Finding resolved:** #9 (upvol_downvol_ratio denominator substitution)

**Why:** `_compute_upvol_downvol_ratio` at `breadth_state.py:548-550` does `downvol.where(downvol > 0, other=1.0)` — when no stocks declined, the ratio becomes `upvol / 1.0 = upvol` (raw volume in millions). The sibling `_compute_nh_nl_ratio` handles this correctly by using `(numerator + denominator)` as the denominator and masking to NaN when no members exist.

**Fix:** Use the same pattern as `_compute_nh_nl_ratio`: denominator is `(upvol + downvol)`, result is NaN when no members have volume.

**Files:**
- Modify: `src/regime_detection/breadth_state.py:534-552`
- Test: `tests/test_breadth_state_v2_pit_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_upvol_downvol_ratio_returns_nan_when_no_decliners() -> None:
    """When downvol is zero (no declining stocks), upvol_downvol_ratio
    should be NaN, not the raw upvol value.
    """
    from regime_detection.breadth_state import _compute_upvol_downvol_ratio

    idx = pd.bdate_range("2025-01-01", periods=3, freq="B")
    symbols = ["AAPL", "MSFT", "GOOG"]

    # All stocks advance — no decliners
    advance_mask = pd.DataFrame(True, index=idx, columns=symbols)
    decline_mask = pd.DataFrame(False, index=idx, columns=symbols)
    membership_mask = pd.DataFrame(True, index=idx, columns=symbols)
    volume_frame = pd.DataFrame(
        [[1e6, 2e6, 3e6]] * 3, index=idx, columns=symbols, dtype=float
    )

    ratio = _compute_upvol_downvol_ratio(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        membership_mask=membership_mask,
        volume_frame=volume_frame,
    )

    # Should be NaN, not 6e6 (the raw upvol sum)
    assert ratio.isna().all(), (
        f"Expected NaN when no decliners, got {ratio.tolist()}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_breadth_state_v2_pit_features.py::test_upvol_downvol_ratio_returns_nan_when_no_decliners -xvs 2>&1 | tail -20`
Expected: FAIL — returns raw upvol value.

- [ ] **Step 3: Fix the denominator**

In `breadth_state.py`, replace the ratio computation:

```python
    total = upvol + downvol
    ratio = upvol / total.where(total > 0)
    ratio.name = "upvol_downvol_ratio"
    return ratio
```

This produces NaN when `total == 0` (no volume at all), and a bounded 0-1 ratio otherwise. When all volume is advancing, ratio = 1.0 (not millions). When all declining, ratio = 0.0.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_breadth_state_v2_pit_features.py::test_upvol_downvol_ratio_returns_nan_when_no_decliners -xvs 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Run the full breadth test suite**

Run: `python3 -m pytest tests/test_breadth_state*.py -xvs 2>&1 | tail -60`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/regime_detection/breadth_state.py tests/test_breadth_state_v2_pit_features.py
git commit -m "fix(breadth): use (upvol+downvol) denominator — no raw-volume blowup when no decliners"
```

---

## Final Verification

After all 10 tasks, run the full test suite:

```bash
python3 -m pytest tests/ -x --timeout=300 2>&1 | tail -60
```

Then run black:

```bash
python3 -m black src tests
```

All 10 findings are resolved. No backward-compatibility shims. No patches.
