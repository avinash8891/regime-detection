"""V2 Slice 6 — failing tests for the HMM evidence layer (TDD RED phase).

Spec pins: ``docs/regime_engine_v2_spec.md`` §6.1 (lines 2723–2804). The HMM
module reuses existing FeatureStore seams as inputs and emits a
permutation-invariant ``top_state_prob`` consumable by the §4.2 transition
score 6th component.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import HMMConfig, load_default_regime_config
from regime_detection.hmm_state import (
    HMMFeatures,
    HMMParameterDrift,
    _StrictConvergenceMonitor,
    compute_hmm_features,
    compute_hmm_parameter_drift,
)
from regime_shared.pandas_compat import cow_safe_assign

# ---------------------------------------------------------------------------
# Helpers — synthetic but deterministic inputs that mimic the 5 spec seams.
# ---------------------------------------------------------------------------


def _synthetic_inputs(n_sessions: int = 1500, *, seed: int = 0) -> dict[str, pd.Series]:
    """Build five synthetic series with two distinct regime patterns.

    First half ~calm (low vol, positive drift), second half ~volatile (high
    vol, negative drift, elevated correlation). Designed so a 4-state
    Gaussian HMM can separate the regimes given enough data.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n_sessions)
    half = n_sessions // 2

    # return_1d: calm then volatile
    calm_ret = rng.normal(loc=0.0005, scale=0.005, size=half)
    vol_ret = rng.normal(loc=-0.0005, scale=0.020, size=n_sessions - half)
    returns = np.concatenate([calm_ret, vol_ret])
    return_1d = pd.Series(returns, index=index, name="return_1d")

    # realized_vol_21d: rolling std × sqrt(252)
    realized_vol_21d = return_1d.rolling(21).std() * np.sqrt(252)
    realized_vol_21d.name = "realized_vol_21d"

    # drawdown_63d derived from a synthetic cumulative price path
    price = (1.0 + return_1d).cumprod() * 100.0
    peak = price.rolling(63, min_periods=63).max()
    drawdown_63d = (price / peak - 1.0).rename("drawdown_63d")

    # volume z-score
    base_vol = rng.normal(loc=0.0, scale=1.0, size=n_sessions)
    base_vol[half:] += 1.5  # elevated in regime 2
    volume_zscore_20d = pd.Series(base_vol, index=index, name="volume_zscore_20d")

    # avg pairwise correlation: low in calm regime, high in volatile regime
    corr_calm = rng.normal(loc=0.30, scale=0.05, size=half).clip(0.0, 0.95)
    corr_vol = rng.normal(loc=0.65, scale=0.05, size=n_sessions - half).clip(0.0, 0.95)
    avg_pairwise_corr_63d = pd.Series(
        np.concatenate([corr_calm, corr_vol]),
        index=index,
        name="avg_pairwise_corr_63d",
    )

    return {
        "return_1d": return_1d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "volume_zscore_20d": volume_zscore_20d,
        "avg_pairwise_corr_63d": avg_pairwise_corr_63d,
    }


def _default_hmm_config(training_window_days: int = 1260) -> HMMConfig:
    """Build a realistic HMMConfig using production-pinned values."""
    return HMMConfig(
        n_states=4,
        training_window_days=training_window_days,
        retrain_cadence_days=21,
        random_state=42,
        standardize_inputs=True,
        covariance_type="full",
        min_covar=0.001,
        random_seeds=(42, 101, 202),
    )


@pytest.fixture(scope="module")
def _computed_default_hmm() -> HMMFeatures:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    result = compute_hmm_features(config=cfg, **inputs)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Group A — compute_hmm_features unit tests
# ---------------------------------------------------------------------------


def test_compute_hmm_features_raises_when_any_input_is_none() -> None:
    inputs = _synthetic_inputs()
    cfg = _default_hmm_config()
    for missing in (
        "return_1d",
        "realized_vol_21d",
        "drawdown_63d",
        "volume_zscore_20d",
        "avg_pairwise_corr_63d",
    ):
        call_kwargs = {k: (None if k == missing else v) for k, v in inputs.items()}
        with pytest.raises(RuntimeError, match="HMM missing required inputs"):
            compute_hmm_features(config=cfg, **call_kwargs)


def test_compute_hmm_features_raises_when_insufficient_history() -> None:
    inputs = _synthetic_inputs(n_sessions=100)
    cfg = _default_hmm_config(training_window_days=1260)
    with pytest.raises(RuntimeError, match="HMM insufficient history"):
        compute_hmm_features(config=cfg, **inputs)


def test_compute_hmm_features_succeeds_on_synthetic_inputs_with_full_history(
    _computed_default_hmm: HMMFeatures,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    result = _computed_default_hmm
    assert result is not None
    assert isinstance(result, HMMFeatures)
    assert result.n_states == 4
    # top_state_prob is reindexed to the SPY (return_1d) index.
    assert len(result.top_state_prob) == 1500
    # State probabilities are in [0, 1] wherever non-NaN.
    non_null = result.top_state_prob.dropna()
    assert (non_null >= 0.0).all()
    assert (non_null <= 1.0).all()
    # V1 §2.2 PIT semantics: HMM evidence is populated from models trained
    # on data available at or before each emitted session. With a 1260-session
    # training window and the default 21-session retrain cadence, the warmed
    # tail should be populated instead of blanking every pre-final row.
    assert len(non_null) > 30
    assert non_null.index.max() == inputs["return_1d"].dropna().index[-1]


def test_top_state_prob_permutation_invariant_under_fixed_seed() -> None:
    inputs = _synthetic_inputs(n_sessions=500)
    cfg = _default_hmm_config(training_window_days=252)
    first = compute_hmm_features(config=cfg, **inputs)
    second = compute_hmm_features(config=cfg, **inputs)
    assert first is not None and second is not None
    pd.testing.assert_series_equal(first.top_state_prob, second.top_state_prob)


def test_state_probabilities_sum_to_one_per_session(
    _computed_default_hmm: HMMFeatures,
) -> None:
    result = _computed_default_hmm
    assert result is not None
    valid_rows = result.state_probabilities.dropna(how="any")
    assert len(valid_rows) > 0
    row_sums = valid_rows.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-6)


def test_compute_hmm_features_raises_when_hmm_fit_fails() -> None:
    """Constant (zero-variance) inputs force a singular covariance failure."""
    index = pd.bdate_range("2010-01-04", periods=1500)
    constant_series = pd.Series(np.zeros(1500), index=index)
    inputs = {
        "return_1d": constant_series.copy(),
        "realized_vol_21d": constant_series.copy(),
        "drawdown_63d": constant_series.copy(),
        "volume_zscore_20d": constant_series.copy(),
        "avg_pairwise_corr_63d": constant_series.copy(),
    }
    cfg = _default_hmm_config()
    with pytest.raises(RuntimeError, match="HMM fit failed"):
        compute_hmm_features(config=cfg, **inputs)


def test_compute_hmm_features_raises_on_programming_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programming bugs (KeyError, TypeError, etc.) must propagate, not be swallowed."""
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()

    def _exploding_seed(*args, **kwargs):
        raise KeyError("programming bug")

    monkeypatch.setattr("regime_detection.hmm_state._fit_single_seed", _exploding_seed)

    with pytest.raises(KeyError, match="programming bug"):
        compute_hmm_features(config=cfg, **inputs)


def test_compute_hmm_features_raises_when_hmm_fit_is_non_monotonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()

    class FakeMonitor:
        tol = 0.01
        n_iter = 200
        verbose = False
        non_monotonic = True

    class FakeGaussianHMM:
        def __init__(self, *args, **kwargs) -> None:
            self.monitor_ = FakeMonitor()

        def fit(self, _train):
            self.monitor_.non_monotonic = True
            return self

        def predict_proba(self, frame):
            return np.full((len(frame), cfg.n_states), 1.0 / cfg.n_states)

    monkeypatch.setattr("regime_detection.hmm_state.GaussianHMM", FakeGaussianHMM)

    with pytest.raises(RuntimeError, match="HMM fit failed"):
        compute_hmm_features(config=cfg, **inputs)


def test_strict_convergence_monitor_does_not_treat_non_monotonic_fit_as_converged() -> (
    None
):
    monitor = _StrictConvergenceMonitor(tol=0.01, n_iter=200, verbose=False)

    monitor.report(10.0)
    monitor.report(9.0)

    assert monitor.non_monotonic is True
    assert monitor.converged is False


def test_compute_hmm_features_uses_best_monotonic_seed_after_standardizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config().model_copy(
        update={
            "covariance_type": "diag",
            "min_covar": 0.123,
            "standardize_inputs": True,
            "random_seeds": (42, 101),
        }
    )
    seen: list[dict[str, object]] = []

    class FakeMonitor:
        tol = 0.01
        n_iter = 200
        verbose = False

    class FakeGaussianHMM:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.random_state = kwargs["random_state"]
            self.monitor_ = FakeMonitor()

        def fit(self, train):
            seen.append(
                {
                    "random_state": self.random_state,
                    "covariance_type": self.kwargs["covariance_type"],
                    "min_covar": self.kwargs["min_covar"],
                    "mean": float(train.mean()),
                    "std": float(train.std()),
                }
            )
            if self.random_state == 42:
                self.monitor_.report(10.0)
                self.monitor_.report(9.0)
            else:
                self.monitor_.report(10.0)
                self.monitor_.report(12.0)
            return self

        def predict_proba(self, frame):
            return np.tile(np.array([[0.7, 0.1, 0.1, 0.1]]), (len(frame), 1))

    monkeypatch.setattr("regime_detection.hmm_state.GaussianHMM", FakeGaussianHMM)

    result = compute_hmm_features(config=cfg, **inputs)

    assert result is not None
    assert result.selected_seed == 101
    assert result.log_likelihood == 12.0
    assert [item["random_state"] for item in seen[:2]] == [42, 101]
    assert {item["random_state"] for item in seen} == {42, 101}
    assert all(item["covariance_type"] == "diag" for item in seen)
    assert all(item["min_covar"] == 0.123 for item in seen)
    assert abs(float(seen[0]["mean"])) < 1e-12
    assert float(seen[0]["std"]) == pytest.approx(1.0)


def test_compute_hmm_features_does_not_warn_for_recoverable_non_monotonic_seed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config().model_copy(update={"random_seeds": (42, 101)})

    class FakeMonitor:
        tol = 0.01
        n_iter = 200
        verbose = False

    class FakeGaussianHMM:
        def __init__(self, **kwargs) -> None:
            self.random_state = kwargs["random_state"]
            self.monitor_ = FakeMonitor()

        def fit(self, _train):
            if self.random_state == 42:
                self.monitor_.report(10.0)
                self.monitor_.report(9.0)
            else:
                self.monitor_.report(10.0)
                self.monitor_.report(12.0)
            return self

        def predict_proba(self, frame):
            return np.tile(np.array([[0.7, 0.1, 0.1, 0.1]]), (len(frame), 1))

    monkeypatch.setattr("regime_detection.hmm_state.GaussianHMM", FakeGaussianHMM)

    with caplog.at_level("WARNING", logger="regime_detection.hmm_state"):
        result = compute_hmm_features(config=cfg, **inputs)

    assert result is not None
    assert result.selected_seed == 101
    assert "GaussianHMM skipped non-monotonic seed" not in caplog.text


def test_compute_hmm_features_debug_logs_recoverable_non_monotonic_seed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config().model_copy(update={"random_seeds": (42, 101)})

    class FakeMonitor:
        tol = 0.01
        n_iter = 200
        verbose = False

    class FakeGaussianHMM:
        def __init__(self, **kwargs) -> None:
            self.random_state = kwargs["random_state"]
            self.monitor_ = FakeMonitor()

        def fit(self, _train):
            if self.random_state == 42:
                self.monitor_.report(10.0)
                self.monitor_.report(9.0)
            else:
                self.monitor_.report(10.0)
                self.monitor_.report(12.0)
            return self

        def predict_proba(self, frame):
            return np.tile(np.array([[0.7, 0.1, 0.1, 0.1]]), (len(frame), 1))

    monkeypatch.setattr("regime_detection.hmm_state.GaussianHMM", FakeGaussianHMM)

    with caplog.at_level("DEBUG", logger="regime_detection.hmm_state"):
        result = compute_hmm_features(config=cfg, **inputs)

    assert result is not None
    assert result.selected_seed == 101
    assert "GaussianHMM skipped non-monotonic seed" in caplog.text
    assert "seed=42" in caplog.text


def test_top_state_prob_is_at_least_one_over_n_states(
    _computed_default_hmm: HMMFeatures,
) -> None:
    cfg = _default_hmm_config()
    result = _computed_default_hmm
    assert result is not None
    non_null = result.top_state_prob.dropna()
    # argmax probability ≥ 1/n_states by construction
    assert (non_null >= (1.0 / cfg.n_states) - 1e-9).all()


def test_hmm_uses_real_default_config_n_states() -> None:
    cfg = load_default_regime_config().hmm
    assert cfg is not None
    assert cfg.n_states == 4
    assert cfg.standardize_inputs is True
    assert cfg.covariance_type == "full"
    assert cfg.min_covar == pytest.approx(0.001)
    assert cfg.random_seeds == (42, 101, 202, 303, 404, 505, 606, 707, 808, 909)
    assert cfg.training_window_days == 1260
    assert cfg.random_state == 42


# ---------------------------------------------------------------------------
# Group B — FeatureStore seam wiring
# ---------------------------------------------------------------------------


def test_feature_store_hmm_seam_lit_when_all_inputs_present(
    golden_rows: list[dict[str, object]],
) -> None:
    """Sanity wire-in: the engine's golden-date classify path must light the
    HMM seam — verified indirectly by inspecting one classify call's
    feature_store via the engine."""
    from regime_detection.engine import RegimeEngine
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    # Use the latest golden date — that is far enough out to have >= 1260
    # sessions of trailing history under the bundled fixture.
    latest = max(date.fromisoformat(str(row["as_of_date"])) for row in golden_rows)
    engine = RegimeEngine()
    # Pull the matching market_data slice from the fixture (re-load via
    # conftest helpers is heavy; instead read the parquet directly).
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "tests" / "fixtures" / "raw"
    market_parquet = raw_dir / "market_data.parquet"
    if market_parquet.exists():
        raw = pd.read_parquet(market_parquet)
    else:
        parts = [
            pd.read_csv(raw_dir / f"{symbol}.csv") for symbol in ("SPY", "RSP", "VIXY")
        ]
        raw = pd.concat(parts, ignore_index=True)
    raw = cow_safe_assign(raw, {"date": pd.to_datetime(raw["date"]).dt.date})
    market_data = raw[raw["date"] <= latest].copy().reset_index(drop=True)
    context = build_market_context(
        end_date=latest,
        market_data=market_data,
        config=engine.config,
    )
    feature_store = build_feature_store(
        context,
        network_fragility_config=engine.config.network_fragility,
        trend_direction_v2_config=engine.config.trend_direction_v2,
        volatility_state_v2_config=engine.config.volatility_state_v2,
        breadth_state_v2_config=engine.config.breadth_state_v2,
        volume_liquidity_v2_config=engine.config.volume_liquidity_v2,
        monetary_pressure_v2_config=engine.config.monetary_pressure_v2,
    )
    # network_fragility is None in this fixture (no sector ETFs), so HMM
    # seam should be None — accept either outcome but assert behavior is
    # gated on input availability per the predicate.
    if (
        feature_store.network_fragility is None
        or feature_store.volume_liquidity_v2 is None
    ):
        assert feature_store.hmm is None
    else:
        assert feature_store.hmm is not None


def test_feature_store_hmm_seam_none_when_hmm_config_absent(
    raw_market_frames: dict[str, pd.DataFrame],
) -> None:
    from regime_detection.config import load_default_regime_config
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    cfg = load_default_regime_config().model_copy(update={"hmm": None})
    # Use the last available date in the fixture.
    spy = raw_market_frames["SPY"]
    rsp = raw_market_frames["RSP"]
    vix = raw_market_frames["VIX"]
    raw = pd.concat([spy, rsp, vix], ignore_index=True)
    raw = cow_safe_assign(raw, {"date": pd.to_datetime(raw["date"]).dt.date})

    last_session = max(d for d in raw["date"].unique())
    # Walk back to a valid NYSE session if needed.
    while True:
        try:
            from regime_detection.calendar import require_nyse_trading_day

            require_nyse_trading_day(last_session)
            break
        except Exception:
            last_session = last_session.fromordinal(last_session.toordinal() - 1)
    market_data = raw[raw["date"] <= last_session].copy().reset_index(drop=True)
    context = build_market_context(
        end_date=last_session,
        market_data=market_data,
        config=cfg,
    )
    feature_store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
        volatility_state_v2_config=cfg.volatility_state_v2,
        breadth_state_v2_config=cfg.breadth_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
    )
    assert feature_store.hmm is None


# ---------------------------------------------------------------------------
# F-025 — v2 §6.1 HMM parameter-drift monitor (operator calibration review).
# Spec: docs/regime_engine_v2_spec.md lines 4434-4468. State-mean drift is the
# max over (state x feature) relative change after Hungarian alignment, alert
# at 20%. Transition-probability shift is a separate non-blocking review flag
# at 30% (absolute, since transition entries are bounded [0,1]).
# ---------------------------------------------------------------------------

# Realistic fitted-state means over the 5 §6.1 HMM seams, in the order
# compute_hmm_features feeds them: return_1d, realized_vol_21d, drawdown_63d,
# volume_zscore_20d, avg_pairwise_corr_63d. State 0 = calm, State 1 = stress.
_CALM_STATE_MEAN = [0.0005, 0.10, -0.02, -0.10, 0.30]
_STRESS_STATE_MEAN = [-0.010, 0.45, -0.20, 2.00, 0.80]
_PREV_STATE_MEANS = np.array([_CALM_STATE_MEAN, _STRESS_STATE_MEAN])
# calm is sticky (0.95 self), stress decays faster.
_PREV_TRANSITION_MATRIX = np.array([[0.95, 0.05], [0.20, 0.80]])


def test_hmm_parameter_drift_reports_no_drift_when_parameters_unchanged() -> None:
    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=_PREV_STATE_MEANS.copy(),
        previous_transition_matrix=_PREV_TRANSITION_MATRIX,
        current_transition_matrix=_PREV_TRANSITION_MATRIX.copy(),
    )

    assert isinstance(drift, HMMParameterDrift)
    assert drift.parameter_drift == 0.0
    assert drift.state_mean_drift_alert is False
    assert drift.max_transition_prob_shift == 0.0
    assert drift.transition_prob_review_flag is False
    assert drift.alignment == (0, 1)


def test_hmm_parameter_drift_ignores_state_index_permutation() -> None:
    # New refit relabels the same two states in swapped index order. Hungarian
    # alignment must recover the match so a pure permutation is NOT drift.
    swap = [1, 0]
    current_means = _PREV_STATE_MEANS[swap]
    current_transition = _PREV_TRANSITION_MATRIX[np.ix_(swap, swap)]

    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=current_means,
        previous_transition_matrix=_PREV_TRANSITION_MATRIX,
        current_transition_matrix=current_transition,
    )

    assert drift.alignment == (1, 0)
    assert drift.parameter_drift == 0.0
    assert drift.state_mean_drift_alert is False
    assert drift.max_transition_prob_shift == 0.0
    assert drift.transition_prob_review_flag is False


def test_hmm_parameter_drift_flags_state_mean_drift_above_twenty_percent() -> None:
    # Stress-state realized_vol_21d jumps 0.45 -> 0.5625 (a 25% relative move);
    # every other parameter is unchanged. Drift metric = 0.25 > 0.20 alert.
    current_means = _PREV_STATE_MEANS.copy()
    current_means[1][1] = 0.45 * 1.25

    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=current_means,
        previous_transition_matrix=_PREV_TRANSITION_MATRIX,
        current_transition_matrix=_PREV_TRANSITION_MATRIX.copy(),
    )

    assert drift.parameter_drift == pytest.approx(0.25)
    assert drift.state_mean_drift_alert is True
    # Means moved but transitions did not — the review flag stays independent.
    assert drift.transition_prob_review_flag is False


def test_hmm_parameter_drift_transition_review_flag_is_independent() -> None:
    # Identical means; the stress-state self-transition shifts 0.80 -> 0.42
    # (a 0.38 absolute move > 0.30). Mean alert must stay False, review True.
    current_transition = np.array([[0.95, 0.05], [0.58, 0.42]])

    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=_PREV_STATE_MEANS.copy(),
        previous_transition_matrix=_PREV_TRANSITION_MATRIX,
        current_transition_matrix=current_transition,
    )

    assert drift.state_mean_drift_alert is False
    assert drift.max_transition_prob_shift == pytest.approx(0.38)
    assert drift.transition_prob_review_flag is True


def test_hmm_parameter_drift_transition_flag_is_absolute_not_relative() -> None:
    # F-052 / ADR 0024: the 30% transition-prob flag is an ABSOLUTE move, not a
    # relative one. A near-zero entry shifts 0.01 -> 0.05: +400% RELATIVE (would trip
    # a relative threshold) but only 0.04 ABSOLUTE (< 0.30). The flag must stay False,
    # and max_transition_prob_shift must read the absolute 0.04 — proving the absolute
    # definition is what ships.
    previous_transition = np.array([[0.99, 0.01], [0.20, 0.80]])
    current_transition = np.array([[0.95, 0.05], [0.20, 0.80]])

    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=_PREV_STATE_MEANS.copy(),
        previous_transition_matrix=previous_transition,
        current_transition_matrix=current_transition,
    )

    assert drift.max_transition_prob_shift == pytest.approx(0.04)
    assert drift.transition_prob_review_flag is False


def test_hmm_parameter_drift_below_thresholds_raises_no_alert() -> None:
    # 10% mean move and a 0.10 transition shift — both under their thresholds.
    current_means = _PREV_STATE_MEANS.copy()
    current_means[0][1] = 0.10 * 1.10
    current_transition = np.array([[0.85, 0.15], [0.20, 0.80]])

    drift = compute_hmm_parameter_drift(
        previous_state_means=_PREV_STATE_MEANS,
        current_state_means=current_means,
        previous_transition_matrix=_PREV_TRANSITION_MATRIX,
        current_transition_matrix=current_transition,
    )

    assert drift.parameter_drift == pytest.approx(0.10)
    assert drift.state_mean_drift_alert is False
    assert drift.max_transition_prob_shift == pytest.approx(0.10)
    assert drift.transition_prob_review_flag is False


# ---------------------------------------------------------------------------
# F-025 (ideal) — the §6.1 drift monitor RUNS inside compute_hmm_features,
# comparing consecutive PIT refit checkpoints (de-standardized to raw units).
# ---------------------------------------------------------------------------


def test_compute_hmm_features_reports_parameter_drift_across_refit_checkpoints(
    _computed_default_hmm: HMMFeatures,
) -> None:
    result = _computed_default_hmm  # 1500 sessions → many 21-day refit checkpoints
    assert result.parameter_drift is not None
    assert isinstance(result.parameter_drift, HMMParameterDrift)
    # Alignment covers every state; drift metrics are well-formed and finite.
    assert len(result.parameter_drift.alignment) == result.n_states
    assert result.parameter_drift.parameter_drift >= 0.0
    assert result.parameter_drift.max_transition_prob_shift >= 0.0
    # De-standardized to raw feature units: relative drift must stay finite
    # (a standardized-space comparison would explode near zero-mean states).
    assert np.isfinite(result.parameter_drift.parameter_drift)


def test_compute_hmm_features_warns_when_drift_alert_fires(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # F-039: the §6.1 drift seam is computed and exposed, but the >20% state-mean /
    # >30% transition-prob alerts must reach the quarterly operator review via a
    # WARNING. Force an alerting drift at every refit-checkpoint comparison and assert
    # the WARNING is emitted (with the magnitudes) so the review cannot miss it.
    alerting_drift = HMMParameterDrift(
        parameter_drift=0.42,
        state_mean_drift_alert=True,
        max_transition_prob_shift=0.37,
        transition_prob_review_flag=True,
        alignment=(0, 1, 2, 3),
    )
    monkeypatch.setattr(
        "regime_detection.hmm_state.compute_hmm_parameter_drift",
        lambda **_kwargs: alerting_drift,
    )

    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    with caplog.at_level("WARNING", logger="regime_detection.hmm_state"):
        result = compute_hmm_features(config=cfg, **inputs)

    assert result is not None
    assert result.parameter_drift is alerting_drift
    assert "HMM parameter drift alert" in caplog.text
    assert "0.4200" in caplog.text  # state_mean_drift magnitude surfaced
    assert "0.3700" in caplog.text  # max_transition_prob_shift surfaced


def test_compute_hmm_features_does_not_warn_when_drift_below_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # F-039: a quiet refit (both flags False) must NOT emit the alert WARNING — the
    # quarterly review channel stays silent unless a real threshold is crossed.
    quiet_drift = HMMParameterDrift(
        parameter_drift=0.05,
        state_mean_drift_alert=False,
        max_transition_prob_shift=0.04,
        transition_prob_review_flag=False,
        alignment=(0, 1, 2, 3),
    )
    monkeypatch.setattr(
        "regime_detection.hmm_state.compute_hmm_parameter_drift",
        lambda **_kwargs: quiet_drift,
    )

    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    with caplog.at_level("WARNING", logger="regime_detection.hmm_state"):
        result = compute_hmm_features(config=cfg, **inputs)

    assert result is not None
    assert "HMM parameter drift alert" not in caplog.text


def test_compute_hmm_features_parameter_drift_is_none_with_single_checkpoint() -> None:
    # frame length = n_sessions - 62 warm-up (drawdown_63d is binding); pick a
    # training window equal to that so only ONE refit checkpoint fits → no prior
    # to compare against → drift is None.
    inputs = _synthetic_inputs(n_sessions=314)
    cfg = _default_hmm_config(training_window_days=252)
    result = compute_hmm_features(config=cfg, **inputs)
    assert result is not None
    assert result.parameter_drift is None


# ---------------------------------------------------------------------------
# Finding #3 / #4 — HMM state IDs must be stable across PIT refit boundaries
# ---------------------------------------------------------------------------


def test_hmm_state_ids_are_stable_across_pit_refit_boundary() -> None:
    """State IDs assigned by consecutive PIT refit checkpoints must be
    consistent: the same physical regime must keep the same integer label
    across the refit boundary. Without Hungarian alignment of HMM state
    ordering, hmmlearn's arbitrary ID assignment can swap state labels at
    each checkpoint, corrupting mapped_label and state_persistence_days.

    Strategy: build 800 sessions split into alternating calm/stress blocks
    (each 200 sessions), with a 252-day training window that always spans
    both regimes. The window ensures the HMM learns both states in every
    refit, and the 21-day cadence creates several checkpoints per block.
    We verify state IDs stay consistent deep within each block.
    """
    rng = np.random.default_rng(42)
    n_sessions = 800
    index = pd.bdate_range("2012-01-03", periods=n_sessions)

    # Two alternating regime blocks: calm (0-199, 400-599), stress (200-399, 600-799)
    returns = np.empty(n_sessions)
    vol_z = np.empty(n_sessions)
    corr_vals = np.empty(n_sessions)
    for block_start in range(0, n_sessions, 200):
        block_end = block_start + 200
        is_stress = (block_start // 200) % 2 == 1
        if is_stress:
            returns[block_start:block_end] = rng.normal(
                loc=-0.003, scale=0.030, size=200
            )
            vol_z[block_start:block_end] = rng.normal(loc=2.0, scale=0.3, size=200)
            corr_vals[block_start:block_end] = rng.normal(
                loc=0.75, scale=0.03, size=200
            ).clip(0.0, 0.95)
        else:
            returns[block_start:block_end] = rng.normal(
                loc=0.001, scale=0.005, size=200
            )
            vol_z[block_start:block_end] = rng.normal(loc=-0.5, scale=0.3, size=200)
            corr_vals[block_start:block_end] = rng.normal(
                loc=0.20, scale=0.03, size=200
            ).clip(0.0, 0.95)

    return_1d = pd.Series(returns, index=index, name="return_1d")
    realized_vol_21d = return_1d.rolling(21).std() * np.sqrt(252)
    realized_vol_21d.name = "realized_vol_21d"
    price = (1.0 + return_1d).cumprod() * 100.0
    peak = price.rolling(63, min_periods=63).max()
    drawdown_63d = (price / peak - 1.0).rename("drawdown_63d")
    volume_zscore_20d = pd.Series(vol_z, index=index, name="volume_zscore_20d")
    avg_pairwise_corr_63d = pd.Series(
        corr_vals, index=index, name="avg_pairwise_corr_63d"
    )

    cfg = HMMConfig(
        n_states=2,
        training_window_days=252,
        retrain_cadence_days=21,
        random_state=42,
        standardize_inputs=True,
        covariance_type="full",
        min_covar=0.001,
        random_seeds=(42, 101, 202),
    )

    result = compute_hmm_features(
        config=cfg,
        return_1d=return_1d,
        realized_vol_21d=realized_vol_21d,
        drawdown_63d=drawdown_63d,
        volume_zscore_20d=volume_zscore_20d,
        avg_pairwise_corr_63d=avg_pairwise_corr_63d,
    )
    assert result is not None

    # Extract the top state (argmax of posterior) for each session.
    top_state = result.state_probabilities.idxmax(axis=1)

    # Second calm block: sessions 420-580 (well inside, past warm-up).
    calm_states = top_state.iloc[420:580].dropna()
    assert len(calm_states) > 0, "calm block should have assigned states"
    calm_id = calm_states.mode().iloc[0]
    calm_agreement = (calm_states == calm_id).mean()
    assert (
        calm_agreement >= 0.90
    ), f"calm block should be consistently one state, got {calm_agreement:.0%} agreement"

    # Second stress block: sessions 620-780 (well inside stress).
    stress_states = top_state.iloc[620:780].dropna()
    assert len(stress_states) > 0, "stress block should have assigned states"
    stress_id = stress_states.mode().iloc[0]
    stress_agreement = (stress_states == stress_id).mean()
    assert (
        stress_agreement >= 0.90
    ), f"stress block should be consistently one state, got {stress_agreement:.0%} agreement"

    # The two regimes must have DIFFERENT state IDs.
    assert (
        calm_id != stress_id
    ), f"calm and stress regimes must have different state IDs, both got {calm_id}"
