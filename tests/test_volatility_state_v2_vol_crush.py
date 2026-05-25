"""TDD tests for v2 §1C `vol_crush` rule + `event_window_just_passed`
(ADR 0005 / Ambiguity Log #19+#20 closure).

Spec references (docs/regime_engine_v2_spec.md §1C):

    vol_crush:
      realized_vol_10d < realized_vol_21d * 0.75
      AND implied_vol_5d_change <= -0.20    (relative 5-session change)
      AND event_window_just_passed          (3 NYSE sessions strictly after
                                             an event window-end)

    §1C precedence (line 191):
      crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown

Per AGENTS.md rules G/L: realistic SPY-like inputs, no toy names, the
real production Pydantic config.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_detection.config import VolatilityV2RulesConfig
from regime_detection.event_calendar import compute_event_window_just_passed
from regime_detection.volatility_state_v2 import (
    VolatilityV2Features,
    evaluate_v2_volatility_label,
    evaluate_vol_crush,
)

# ADR 0005 — pinned vol_crush thresholds.
_SPEC_RV_RATIO = 0.75
_SPEC_IV_CHANGE_THRESHOLD = -0.20
_SPEC_IV_CHANGE_LOOKBACK = 5
_SPEC_EVENT_WINDOW_TRAILING = 3


@pytest.fixture
def vol_crush_rules() -> VolatilityV2RulesConfig:
    return VolatilityV2RulesConfig(
        atr_ratio_threshold=1.15,
        realized_vol_ratio_threshold=1.25,
        realized_vol_short_period=10,
        realized_vol_long_period=63,
        vol_crush_realized_vol_mid_period=21,
        vol_crush_realized_vol_ratio_threshold=_SPEC_RV_RATIO,
        vol_crush_implied_vol_change_threshold=_SPEC_IV_CHANGE_THRESHOLD,
        vol_crush_implied_vol_change_lookback_sessions=_SPEC_IV_CHANGE_LOOKBACK,
        vol_crush_event_window_trailing_sessions=_SPEC_EVENT_WINDOW_TRAILING,
    )


def _vol_crush_features_at(
    *,
    dt: pd.Timestamp,
    realized_vol_short: float,
    realized_vol_21d: float,
    implied_vol_5d_change: float | None,
    event_window_just_passed: bool | None,
) -> VolatilityV2Features:
    """Build the minimal VolatilityV2Features the vol_crush predicate
    reads at session ``dt``. Fields irrelevant to vol_crush are NaN."""
    idx = pd.DatetimeIndex([dt])
    nan = pd.Series([float("nan")], index=idx)

    iv_change_series: pd.Series | None
    if implied_vol_5d_change is None:
        iv_change_series = None
    else:
        iv_change_series = pd.Series([implied_vol_5d_change], index=idx)

    event_window_series: pd.Series | None
    if event_window_just_passed is None:
        event_window_series = None
    else:
        event_window_series = pd.Series([event_window_just_passed], index=idx)

    return VolatilityV2Features(
        atr_ratio=nan.copy(),
        gap_frequency_20d=nan.copy(),
        gap_frequency_percentile_252d=nan.copy(),
        intraday_range_percentile_252d=nan.copy(),
        realized_vol_short=pd.Series([realized_vol_short], index=idx),
        realized_vol_long=nan.copy(),
        realized_vol_21d=pd.Series([realized_vol_21d], index=idx),
        implied_vol_5d_change=iv_change_series,
        event_window_just_passed=event_window_series,
    )


# ---------------------------------------------------------------------------
# Group A — evaluate_vol_crush predicate (spec §1C three-conjunct rule).
# ---------------------------------------------------------------------------


def test_vol_crush_fires_when_all_three_conjuncts_satisfied(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,  # 0.10 < 0.20 * 0.75 = 0.15 ✓
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,  # -0.30 <= -0.20 ✓
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is True


def test_vol_crush_false_when_realized_vol_not_collapsed(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.16,  # 0.16 NOT < 0.20 * 0.75 = 0.15
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_false_when_iv_change_above_threshold(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.10,  # -0.10 NOT <= -0.20
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_fires_at_iv_change_boundary_exactly_at_threshold(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """`implied_vol_5d_change <= -0.20` is non-strict — exactly -0.20 fires."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=_SPEC_IV_CHANGE_THRESHOLD,  # exactly -0.20
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is True


def test_vol_crush_false_when_no_event_window_just_passed(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=False,  # not in a just-passed window
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_false_when_iv_features_absent(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """When implied_vol_30d was not supplied (implied_vol_5d_change is None),
    vol_crush falsifies — V2 §10 'do not invent', V1 byte-identity preserved."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=None,  # IV feature not wired
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_false_when_event_window_series_absent(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """When no event calendar was supplied (event_window_just_passed is
    None), vol_crush falsifies."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=None,  # no event calendar
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_false_on_nan_inputs(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """V1 §2.7 cold-start: NaN in any required input falsifies."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=float("nan"),
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


def test_vol_crush_false_when_dt_missing_from_feature_indices(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=pd.Timestamp("2024-02-06"),
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=True,
    )
    assert evaluate_vol_crush(features, dt=dt, rules_config=vol_crush_rules) is False


# ---------------------------------------------------------------------------
# Group B — precedence: vol_crush vs rising_vol vs crisis_vol.
# ---------------------------------------------------------------------------


def test_vol_crush_outranks_high_vol_and_rising_vol(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """vol_crush is rank 1 — only crisis_vol outranks it. A v1 `high_vol`
    day with the vol_crush predicate true resolves to vol_crush."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=True,
    )
    result = evaluate_v2_volatility_label(
        v1_label="high_vol",
        features=features,
        dt=dt,
        rules_config=vol_crush_rules,
    )
    assert result == "vol_crush"


def test_crisis_vol_outranks_vol_crush(
    vol_crush_rules: VolatilityV2RulesConfig,
) -> None:
    """crisis_vol (rank 0) outranks vol_crush (rank 1) — a v1 crisis_vol
    day keeps its label even when the vol_crush predicate fires."""
    dt = pd.Timestamp("2024-02-05")
    features = _vol_crush_features_at(
        dt=dt,
        realized_vol_short=0.10,
        realized_vol_21d=0.20,
        implied_vol_5d_change=-0.30,
        event_window_just_passed=True,
    )
    result = evaluate_v2_volatility_label(
        v1_label="crisis_vol",
        features=features,
        dt=dt,
        rules_config=vol_crush_rules,
    )
    assert result is None  # keep crisis_vol


# ---------------------------------------------------------------------------
# Group C — compute_event_window_just_passed (ADR 0005 Q3).
# ---------------------------------------------------------------------------


def test_event_window_just_passed_fires_on_trailing_3_sessions() -> None:
    """A FOMC event on 2024-01-31 has window-end E = event_date + 2 trading
    days. The 3 NYSE sessions strictly after E are `just_passed`; E itself
    and E+4 are not."""
    sessions = tuple(
        pd.bdate_range(start="2024-01-29", end="2024-02-12", freq="C").date
    )
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 31),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            }
        ]
    )
    result = compute_event_window_just_passed(
        normalized_event_calendar=events,
        sessions=sessions,
        trailing_sessions=3,
    )
    # FOMC window for fed_week is (-2, +2): window-end E = 2 trading days
    # after 2024-01-31 = 2024-02-02. The 3 just-passed sessions are
    # 2024-02-05, 02-06, 02-07 (02-03/02-04 are weekend).
    assert result.loc[pd.Timestamp("2024-02-05")]
    assert result.loc[pd.Timestamp("2024-02-06")]
    assert result.loc[pd.Timestamp("2024-02-07")]
    # E itself (2024-02-02) is still inside the window — does not fire.
    assert not result.loc[pd.Timestamp("2024-02-02")]
    # E+4 (2024-02-08) is past the trailing window — does not fire.
    assert not result.loc[pd.Timestamp("2024-02-08")]


def test_event_window_just_passed_all_false_when_no_calendar() -> None:
    """No event calendar → all-False (vol_crush then cannot fire)."""
    sessions = tuple(pd.bdate_range(start="2024-01-29", end="2024-02-12").date)
    result = compute_event_window_just_passed(
        normalized_event_calendar=None,
        sessions=sessions,
        trailing_sessions=3,
    )
    assert not result.any()
    assert len(result) == len(sessions)


def test_event_window_just_passed_respects_publication_date() -> None:
    """V1 §2.2 stateless replay: an event whose publication_date is after
    a trailing session does not mark that session."""
    sessions = tuple(
        pd.bdate_range(start="2024-01-29", end="2024-02-12", freq="C").date
    )
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 31),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                # Published AFTER the trailing window — must not mark any
                # session as just-passed (would be lookahead).
                "publication_date": date(2024, 3, 1),
            }
        ]
    )
    result = compute_event_window_just_passed(
        normalized_event_calendar=events,
        sessions=sessions,
        trailing_sessions=3,
    )
    assert not result.any()
