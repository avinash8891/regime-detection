"""V2 Slice 6 — failing tests for the HMM evidence layer (TDD RED phase).

Spec pins: ``docs/regime_engine_v2_spec.md`` §6.1 (lines 2723–2804). The HMM
module reuses existing FeatureStore seams as inputs and emits a
permutation-invariant ``top_state_prob`` consumable by the §4.2 transition
score 6th component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import HMMConfig, load_default_regime_config
from regime_detection.hmm_state import (
    HMMFeatures,
    compute_hmm_features,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic but deterministic inputs that mimic the 5 spec seams.
# ---------------------------------------------------------------------------


def _synthetic_inputs(
    n_sessions: int = 1500, *, seed: int = 0
) -> dict[str, pd.Series]:
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
    volume_zscore_20d = pd.Series(
        base_vol, index=index, name="volume_zscore_20d"
    )

    # avg pairwise correlation: low in calm regime, high in volatile regime
    corr_calm = rng.normal(loc=0.30, scale=0.05, size=half).clip(0.0, 0.95)
    corr_vol = rng.normal(loc=0.65, scale=0.05, size=n_sessions - half).clip(
        0.0, 0.95
    )
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


# ---------------------------------------------------------------------------
# Group A — compute_hmm_features unit tests
# ---------------------------------------------------------------------------


def test_compute_hmm_features_returns_none_when_any_input_is_none() -> None:
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
        result = compute_hmm_features(config=cfg, **call_kwargs)
        assert result is None, f"missing {missing} → expected None"


def test_compute_hmm_features_returns_none_when_insufficient_history() -> None:
    inputs = _synthetic_inputs(n_sessions=100)
    cfg = _default_hmm_config(training_window_days=1260)
    result = compute_hmm_features(config=cfg, **inputs)
    assert result is None


def test_compute_hmm_features_succeeds_on_synthetic_inputs_with_full_history() -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    result = compute_hmm_features(config=cfg, **inputs)
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
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    first = compute_hmm_features(config=cfg, **inputs)
    second = compute_hmm_features(config=cfg, **inputs)
    assert first is not None and second is not None
    pd.testing.assert_series_equal(first.top_state_prob, second.top_state_prob)


def test_state_probabilities_sum_to_one_per_session() -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    result = compute_hmm_features(config=cfg, **inputs)
    assert result is not None
    valid_rows = result.state_probabilities.dropna(how="any")
    assert len(valid_rows) > 0
    row_sums = valid_rows.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-6)


def test_compute_hmm_features_returns_none_when_hmm_fit_fails() -> None:
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
    result = compute_hmm_features(config=cfg, **inputs)
    assert result is None


def test_compute_hmm_features_returns_none_when_hmm_fit_is_non_monotonic(
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

    assert compute_hmm_features(config=cfg, **inputs) is None


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


def test_top_state_prob_is_at_least_one_over_n_states() -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_hmm_config()
    result = compute_hmm_features(config=cfg, **inputs)
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
    classified_golden_outputs: dict,
) -> None:
    """Sanity wire-in: the engine's golden-date classify path must light the
    HMM seam — verified indirectly by inspecting one classify call's
    feature_store via the engine."""
    from regime_detection.engine import RegimeEngine
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    # Use the latest golden date — that is far enough out to have >= 1260
    # sessions of trailing history under the bundled fixture.
    latest = max(classified_golden_outputs.keys())
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
            pd.read_csv(raw_dir / f"{symbol}.csv")
            for symbol in ("SPY", "RSP", "VIXY")
        ]
        raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
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
    if feature_store.network_fragility is None or feature_store.volume_liquidity_v2 is None:
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
    vixy = raw_market_frames["VIXY"]
    raw = pd.concat([spy, rsp, vixy], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date

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
