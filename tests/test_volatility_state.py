from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.config import load_default_regime_config
from regime_detection.volatility_state import (
    _RISK_RANK,
    VolatilityFeatures,
    build_raw_outputs,
    compute_features,
    raw_label_for_day,
)


def test_volatility_state_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert (
            out.volatility_state.active_label == row["expected"]["volatility_state"]
        ), f"{as_of}: expected {row['expected']['volatility_state']}, got {out.volatility_state.active_label}"


def test_v1_volatility_risk_rank_contract_keeps_crisis_vol_at_three() -> None:
    assert _RISK_RANK["crisis_vol"] == 3


def test_raw_label_for_day_is_single_source_of_truth_over_build_raw_outputs() -> None:
    # F-043: the per-day scalar path must be a thin wrapper over the vectorized
    # builder so the §5.5 rule predicates have ONE encoding. Guard fails if the
    # two paths ever diverge — covers v1-only mode and the v2 §1C override mode.
    idx = pd.bdate_range("2022-06-01", periods=360)
    close = pd.Series(
        [300.0 + i * 0.4 - (i % 13) * 2.0 for i in range(360)], index=idx, name="close"
    )
    high = close * 1.01
    low = close * 0.99
    open_ = close.shift(1).fillna(close.iloc[0])
    vix_proxy = pd.Series(
        [18.0 + (i % 17) * 0.5 for i in range(360)], index=idx, name="vix"
    )
    features = compute_features(close=close, vix_proxy_close=vix_proxy)

    cfg = load_default_regime_config()
    assert cfg.volatility_state_v2 is not None
    from regime_detection.volatility_state_v2 import compute_volatility_v2_features

    v2_features = compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=cfg.volatility_state_v2,
        rules_config=cfg.volatility_state_v2.rules,
    )

    def _norm(value: object) -> object:
        if isinstance(value, float) and pd.isna(value):
            return "__nan__"
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in value.items()}
        return value

    for v2_kwargs in (
        {},
        {
            "volatility_state_v2_features": v2_features,
            "volatility_state_v2_rules": cfg.volatility_state_v2.rules,
        },
    ):
        labels, evidence = build_raw_outputs(features, **v2_kwargs)
        for i, dt in enumerate(idx):
            day_label, day_evidence = raw_label_for_day(features, dt, **v2_kwargs)
            assert day_label == labels[i], f"{dt}: {day_label} != {labels[i]}"
            assert _norm(day_evidence) == _norm(evidence[i]), f"{dt}: evidence mismatch"


def _volatility_features(
    *,
    return_1d: float = 0.0,
    return_5d: float = 0.0,
    return_21d: float = 0.0,
    realized_vol_percentile_252d: float = 0.50,
    vix_percentile_252d: float | None = None,
) -> VolatilityFeatures:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    vix = (
        None
        if vix_percentile_252d is None
        else pd.Series([vix_percentile_252d], index=idx)
    )
    return VolatilityFeatures(
        close=pd.Series([100.0], index=idx),
        return_1d=pd.Series([return_1d], index=idx),
        return_5d=pd.Series([return_5d], index=idx),
        return_21d=pd.Series([return_21d], index=idx),
        realized_vol_21d=pd.Series([0.20], index=idx),
        realized_vol_percentile_252d=pd.Series(
            [realized_vol_percentile_252d], index=idx
        ),
        vix_percentile_252d=vix,
    )


def test_volatility_raw_label_thresholds_for_v1_labels() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert (
        raw_label_for_day(_volatility_features(return_1d=-0.05), dt)[0] == "crisis_vol"
    )
    assert (
        raw_label_for_day(
            _volatility_features(return_21d=-0.05, realized_vol_percentile_252d=0.90),
            dt,
        )[0]
        == "crisis_vol"
    )
    assert (
        raw_label_for_day(
            _volatility_features(realized_vol_percentile_252d=0.80),
            dt,
        )[0]
        == "high_vol"
    )
    assert (
        raw_label_for_day(
            _volatility_features(realized_vol_percentile_252d=0.30),
            dt,
        )[0]
        == "low_vol"
    )
    assert (
        raw_label_for_day(
            _volatility_features(realized_vol_percentile_252d=0.50),
            dt,
        )[0]
        == "normal_vol"
    )


def test_volatility_raw_label_uses_optional_vix_percentile_thresholds() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert (
        raw_label_for_day(
            _volatility_features(
                realized_vol_percentile_252d=0.50, vix_percentile_252d=0.80
            ),
            dt,
        )[0]
        == "high_vol"
    )
    assert (
        raw_label_for_day(
            _volatility_features(
                realized_vol_percentile_252d=0.50, vix_percentile_252d=0.95
            ),
            dt,
        )[0]
        == "crisis_vol"
    )


def test_volatility_raw_label_unknown_when_required_feature_is_nan() -> None:
    dt = pd.Timestamp("2024-01-02")

    label, evidence = raw_label_for_day(
        _volatility_features(realized_vol_percentile_252d=float("nan")),
        dt,
    )

    assert label == "unknown"
    assert evidence == {"reason": "insufficient_history"}
