"""V2 Slice 8 — BOCPD change-point detection evidence layer.

Spec: docs/regime_engine_v2_spec.md §4.6 (lines 2457-2472) + §6.3
(lines 2861-2887). Implementation library:
``bayesian-changepoint-detection`` per Ambiguity Log #62 (Adams-MacKay
2007). Observation series ``realized_vol_21d`` per Log #63. Score
formula = 5-session rolling max of posterior per Log #64. Break
threshold = posterior >= 0.5 per Log #65.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from regime_detection.change_point import (
    ChangePointFeatures,
    compute_change_point_features,
)
from regime_detection.config import (
    ChangePointConfig,
    load_default_regime_config,
)
from regime_shared.pandas_compat import cow_safe_assign

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_change_point_config(
    training_window_days: int = 1260,
) -> ChangePointConfig:
    return ChangePointConfig(
        hazard_lambda=250.0,
        score_window_days=5,
        break_threshold=0.5,
        training_window_days=training_window_days,
        student_t_alpha=0.1,
        student_t_beta=0.01,
        student_t_kappa=1.0,
        student_t_mu=0.0,
        recent_run_length_window_days=21,
        method="BOCPD",
    )


def _synthetic_two_regime_realized_vol(
    n_sessions: int = 1500,
    *,
    shift_index: int = 750,
    seed: int = 0,
) -> pd.Series:
    """Build a synthetic realized_vol_21d series with a clear regime shift.

    Pre-shift: low volatility (mean 0.10). Post-shift: high volatility
    (mean 0.35). A spread-of-5x in mean gives BOCPD ample signal to
    detect the break — sufficient for the >0.5 posterior assertion.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n_sessions)
    pre = rng.normal(loc=0.10, scale=0.01, size=shift_index)
    post = rng.normal(loc=0.35, scale=0.01, size=n_sessions - shift_index)
    values = np.concatenate([pre, post])
    return pd.Series(values, index=index, name="realized_vol_21d")


# ---------------------------------------------------------------------------
# Group A — compute_change_point_features unit tests
# ---------------------------------------------------------------------------


def test_compute_change_point_features_raises_when_input_is_none() -> None:
    cfg = _default_change_point_config()
    with pytest.raises(RuntimeError, match="BOCPD missing required input"):
        compute_change_point_features(realized_vol_21d=None, config=cfg)


def test_compute_change_point_features_raises_when_insufficient_history() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    short = pd.Series(np.linspace(0.10, 0.20, 100), index=index)
    cfg = _default_change_point_config(training_window_days=1260)
    with pytest.raises(RuntimeError, match="BOCPD insufficient history"):
        compute_change_point_features(realized_vol_21d=short, config=cfg)


def test_compute_change_point_features_succeeds_on_synthetic_two_regime_data() -> None:
    shift_index = 750
    series = _synthetic_two_regime_realized_vol(
        n_sessions=1500, shift_index=shift_index, seed=0
    )
    cfg = _default_change_point_config(training_window_days=100)
    result = compute_change_point_features(realized_vol_21d=series, config=cfg)
    assert result is not None
    assert isinstance(result, ChangePointFeatures)

    # Spike should appear within ±30 sessions of the engineered shift.
    posterior = result.posterior_changepoint_prob
    window_start = max(0, shift_index - 30)
    window_end = min(len(posterior), shift_index + 30)
    window = posterior.iloc[window_start:window_end].dropna()
    assert (window > 0.5).any(), (
        f"Expected a BOCPD posterior > 0.5 within ±30 sessions of session "
        f"{shift_index}; max in window = {window.max()}"
    )


def test_change_point_masks_rows_before_strict_pit_warmup() -> None:
    series = _synthetic_two_regime_realized_vol(n_sessions=140, shift_index=110, seed=1)
    cfg = _default_change_point_config(training_window_days=100)

    result = compute_change_point_features(realized_vol_21d=series, config=cfg)

    assert result is not None
    assert result.posterior_changepoint_prob.iloc[:99].isna().all()
    assert result.score.iloc[:99].isna().all()
    assert result.days_since_last_break.iloc[:99].isna().all()
    assert result.posterior_changepoint_prob.iloc[99:].notna().all()
    assert result.score.iloc[99:].notna().all()


def test_change_point_historical_scores_do_not_change_when_future_rows_append() -> None:
    base = _synthetic_two_regime_realized_vol(n_sessions=160, shift_index=120, seed=2)
    future = _synthetic_two_regime_realized_vol(n_sessions=40, shift_index=20, seed=3)
    future.index = pd.bdate_range(base.index[-1] + pd.offsets.BDay(), periods=40)
    extended = pd.concat([base, future])
    cfg = _default_change_point_config(training_window_days=100)
    historical_date = base.index[130]

    base_result = compute_change_point_features(realized_vol_21d=base, config=cfg)
    extended_result = compute_change_point_features(
        realized_vol_21d=extended, config=cfg
    )

    assert base_result is not None
    assert extended_result is not None
    assert pd.notna(base_result.score.loc[historical_date])
    assert extended_result.score.loc[historical_date] == pytest.approx(
        base_result.score.loc[historical_date]
    )
    assert extended_result.posterior_changepoint_prob.loc[
        historical_date
    ] == pytest.approx(base_result.posterior_changepoint_prob.loc[historical_date])


def test_change_point_detects_gradual_21_session_volatility_ramp() -> None:
    index = pd.bdate_range("2020-01-02", periods=300)
    values = np.concatenate(
        [
            np.full(140, 0.10),
            np.linspace(0.10, 0.70, 21),
            np.full(139, 0.70),
        ]
    )
    series = pd.Series(values, index=index, name="realized_vol_21d")
    cfg = _default_change_point_config(training_window_days=100)

    result = compute_change_point_features(realized_vol_21d=series, config=cfg)

    assert result is not None
    ramp_window = result.posterior_changepoint_prob.iloc[140:170].dropna()
    assert ramp_window.max() >= cfg.break_threshold
    assert result.days_since_last_break.iloc[140:170].notna().any()


def test_score_is_5_session_rolling_max_of_posterior() -> None:
    series = _synthetic_two_regime_realized_vol(
        n_sessions=1500, shift_index=750, seed=0
    )
    cfg = _default_change_point_config(training_window_days=100)
    result = compute_change_point_features(realized_vol_21d=series, config=cfg)
    assert result is not None
    posterior = result.posterior_changepoint_prob
    score = result.score
    # Pick a session in the middle where rolling window is fully populated.
    t = 800
    expected = posterior.iloc[t - 4 : t + 1].max()
    actual = score.iloc[t]
    assert actual == pytest.approx(expected, nan_ok=False)


def test_days_since_last_break_zero_at_break_session() -> None:
    from regime_detection.change_point import _days_since_last_break

    index = pd.bdate_range("2020-01-02", periods=20)
    posterior_vals = np.full(20, 0.05)
    posterior_vals[10] = 0.9  # single break at index 10
    posterior = pd.Series(posterior_vals, index=index)
    days_since = _days_since_last_break(posterior, threshold=0.5)

    assert days_since.iloc[10] == 0
    assert days_since.iloc[11] == 1
    assert days_since.iloc[15] == 5


def test_days_since_last_break_none_when_no_break_in_history() -> None:
    from regime_detection.change_point import _days_since_last_break

    index = pd.bdate_range("2020-01-02", periods=20)
    posterior = pd.Series(np.full(20, 0.1), index=index)
    days_since = _days_since_last_break(posterior, threshold=0.5)
    assert days_since.isna().all()


def test_method_field_is_BOCPD() -> None:
    series = _synthetic_two_regime_realized_vol(
        n_sessions=1500, shift_index=750, seed=0
    )
    cfg = _default_change_point_config(training_window_days=1500)
    result = compute_change_point_features(realized_vol_21d=series, config=cfg)
    assert result is not None
    assert result.method == "BOCPD"


def test_compute_change_point_features_raises_on_zero_variance_input() -> None:
    """Constant-vol input is degenerate for the Student-T predictive —
    fail loudly per the module's documented contract."""
    index = pd.bdate_range("2010-01-04", periods=1500)
    constant = pd.Series(np.full(1500, 0.12), index=index)
    cfg = _default_change_point_config(training_window_days=1260)
    with pytest.raises(RuntimeError, match="BOCPD degenerate input"):
        compute_change_point_features(realized_vol_21d=constant, config=cfg)


def test_compute_change_point_features_raises_on_numeric_instability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.change_point as change_point

    series = _synthetic_two_regime_realized_vol(
        n_sessions=1500, shift_index=750, seed=0
    )
    cfg = _default_change_point_config(training_window_days=1260)

    def raise_floating_point_error(
        *, data: np.ndarray, config: ChangePointConfig
    ) -> np.ndarray:
        del data, config
        raise FloatingPointError("singular predictive")

    monkeypatch.setattr(
        change_point,
        "_bocpd_posterior_changepoint_prob",
        raise_floating_point_error,
    )

    with pytest.raises(RuntimeError, match="BOCPD fit failed"):
        compute_change_point_features(realized_vol_21d=series, config=cfg)


def test_compute_change_point_features_raises_when_bocpd_dependency_changes_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.change_point as change_point

    series = _synthetic_two_regime_realized_vol(
        n_sessions=1500, shift_index=750, seed=0
    )
    cfg = _default_change_point_config(training_window_days=1260)

    def raise_value_error(*, data: np.ndarray, config: ChangePointConfig) -> np.ndarray:
        del data, config
        raise ValueError("dependency contract changed")

    monkeypatch.setattr(
        change_point,
        "_bocpd_posterior_changepoint_prob",
        raise_value_error,
    )

    with pytest.raises(RuntimeError, match="BOCPD fit failed"):
        compute_change_point_features(realized_vol_21d=series, config=cfg)


def test_bocpd_adapter_rejects_unexpected_posterior_matrix_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.change_point as change_point

    def fake_online_changepoint_detection(
        data: np.ndarray,
        hazard_func,
        observation_likelihood: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del hazard_func, observation_likelihood
        return np.zeros((len(data), len(data)), dtype=float), np.arange(len(data))

    monkeypatch.setattr(
        change_point,
        "_online_changepoint_detection",
        fake_online_changepoint_detection,
    )

    cfg = _default_change_point_config()
    data = np.array([0.11, 0.12, 0.25], dtype=float)

    with pytest.raises(RuntimeError, match="BOCPD posterior matrix shape"):
        change_point._bocpd_posterior_changepoint_prob(data=data, config=cfg)


def test_bocpd_adapter_rejects_nonfinite_posterior_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.change_point as change_point

    def fake_online_changepoint_detection(
        data: np.ndarray,
        hazard_func,
        observation_likelihood: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del hazard_func, observation_likelihood
        R = np.zeros((len(data) + 1, len(data) + 1), dtype=float)
        R[1, 1:] = np.array([0.1, np.nan, 0.2], dtype=float)
        return R, np.arange(len(data))

    monkeypatch.setattr(
        change_point,
        "_online_changepoint_detection",
        fake_online_changepoint_detection,
    )

    cfg = _default_change_point_config()
    data = np.array([0.11, 0.12, 0.25], dtype=float)

    with pytest.raises(FloatingPointError, match="non-finite"):
        change_point._bocpd_posterior_changepoint_prob(data=data, config=cfg)


def test_bocpd_adapter_calls_expected_dependency_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.change_point as change_point

    calls: dict[str, object] = {}

    class FakeStudentT:
        def __init__(
            self, *, alpha: float, beta: float, kappa: float, mu: float
        ) -> None:
            calls["student_t"] = {
                "alpha": alpha,
                "beta": beta,
                "kappa": kappa,
                "mu": mu,
            }

    def fake_constant_hazard(lam: float, r: np.ndarray) -> np.ndarray:
        return np.full_like(r, fill_value=1.0 / lam, dtype=float)

    def fake_online_changepoint_detection(
        data: np.ndarray,
        hazard_func,
        observation_likelihood: FakeStudentT,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls["data"] = data.copy()
        calls["hazard_func"] = hazard_func
        calls["observation_likelihood"] = observation_likelihood
        R = np.zeros((len(data) + 1, len(data) + 1), dtype=float)
        R[1, 1:] = np.array([0.1, 0.2, 0.7], dtype=float)
        R[2, 1:] = np.array([0.3, 0.1, 0.1], dtype=float)
        R[3, 1:] = np.array([0.4, 0.0, 0.1], dtype=float)
        return R, np.arange(len(data))

    monkeypatch.setattr(change_point, "_StudentT", FakeStudentT)
    monkeypatch.setattr(change_point, "_constant_hazard", fake_constant_hazard)
    monkeypatch.setattr(
        change_point,
        "_online_changepoint_detection",
        fake_online_changepoint_detection,
    )

    cfg = _default_change_point_config()
    data = np.array([0.11, 0.12, 0.25], dtype=float)

    posterior = change_point._bocpd_posterior_changepoint_prob(
        data=data,
        config=cfg,
    )

    np.testing.assert_allclose(posterior, np.array([0.8, 0.3, 0.9]))
    np.testing.assert_allclose(calls["data"], data)
    assert calls["student_t"] == {
        "alpha": cfg.student_t_alpha,
        "beta": cfg.student_t_beta,
        "kappa": cfg.student_t_kappa,
        "mu": cfg.student_t_mu,
    }
    hazard_func = calls["hazard_func"]
    assert hazard_func.func is fake_constant_hazard
    assert hazard_func.args == (cfg.hazard_lambda,)
    assert isinstance(calls["observation_likelihood"], FakeStudentT)


# ---------------------------------------------------------------------------
# Group B — FeatureStore seam wiring + default config
# ---------------------------------------------------------------------------


def test_real_default_config_carries_change_point_block() -> None:
    assert ChangePointConfig().training_window_days == 2705
    assert ChangePointConfig().recent_run_length_window_days == 21

    cfg = load_default_regime_config()
    assert cfg.change_point is not None
    assert cfg.change_point.hazard_lambda == 250.0
    assert cfg.change_point.method == "BOCPD"
    assert cfg.change_point.score_window_days == 5
    assert cfg.change_point.break_threshold == 0.5
    assert cfg.change_point.training_window_days == 2705
    assert cfg.change_point.recent_run_length_window_days == 21


def test_feature_store_change_point_seam_none_when_config_absent(
    raw_market_frames: dict[str, pd.DataFrame],
) -> None:
    from regime_detection.calendar import require_nyse_trading_day
    from regime_detection.market_context import build_market_context

    cfg = load_default_regime_config().model_copy(update={"change_point": None})
    spy = raw_market_frames["SPY"]
    rsp = raw_market_frames["RSP"]
    vix = raw_market_frames["VIX"]
    raw = pd.concat([spy, rsp, vix], ignore_index=True)
    raw = cow_safe_assign(raw, {"date": pd.to_datetime(raw["date"]).dt.date})
    last_session = max(d for d in raw["date"].unique())
    while True:
        try:
            require_nyse_trading_day(last_session)
            break
        except Exception:
            last_session = last_session.fromordinal(last_session.toordinal() - 1)
    market_data = raw[raw["date"] <= last_session].copy().reset_index(drop=True)
    with pytest.raises(ValueError, match="missing required V2 sections: change_point"):
        build_market_context(
            end_date=last_session,
            market_data=market_data,
            config=cfg,
        )


def test_feature_store_change_point_seam_present_with_default_config(
    raw_market_frames: dict[str, pd.DataFrame],
) -> None:
    from regime_detection.calendar import require_nyse_trading_day
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    cfg = load_default_regime_config()
    assert cfg.change_point is not None
    # Override training_window to fit the test fixture's ~650 sessions
    cfg = cfg.model_copy(
        update={
            "change_point": cfg.change_point.model_copy(
                update={"training_window_days": 500}
            ),
        }
    )
    spy = raw_market_frames["SPY"]
    rsp = raw_market_frames["RSP"]
    vix = raw_market_frames["VIX"]
    raw = pd.concat([spy, rsp, vix], ignore_index=True)
    raw = cow_safe_assign(raw, {"date": pd.to_datetime(raw["date"]).dt.date})
    last_session = max(d for d in raw["date"].unique())
    while True:
        try:
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
    with pytest.raises(RuntimeError, match="sentiment_score"):
        build_feature_store(
            context,
            network_fragility_config=cfg.network_fragility,
            trend_direction_v2_config=cfg.trend_direction_v2,
            volatility_state_v2_config=cfg.volatility_state_v2,
            breadth_state_v2_config=cfg.breadth_state_v2,
            volume_liquidity_v2_config=cfg.volume_liquidity_v2,
            monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        )


def test_regime_output_carries_change_point_when_seam_present(
    raw_market_data: pd.DataFrame,
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    from regime_detection.engine import RegimeEngine

    engine = RegimeEngine()
    assert engine.config.change_point is not None
    cfg = engine.config.model_copy(
        update={
            "change_point": engine.config.change_point.model_copy(
                update={"training_window_days": 100}
            ),
            "network_fragility": None,
            "trend_direction_v2": None,
            "volatility_state_v2": None,
            "breadth_state_v2": None,
            "volume_liquidity_v2": None,
            "volume_liquidity_state": None,
            "transition_score": None,
            "monetary_pressure_v2": None,
            "monetary_pressure_state": None,
            "central_bank_text": None,
            "news_sentiment": None,
            "inflation_growth": None,
            "credit_funding": None,
            "hmm": None,
            "clustering": None,
            "cohort_routing": None,
            "strategy_family_constraints": None,
            "strategy_event_modifiers": None,
        }
    )
    last_session = max(raw_market_data["date"].unique())
    market_data = v2_market_df_for_asof(last_session)
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    kwargs["config"] = cfg
    with pytest.raises(ValueError, match="missing required V2 sections"):
        engine.classify(
            as_of_date=last_session,
            market_data=market_data,
            **kwargs,
        )
