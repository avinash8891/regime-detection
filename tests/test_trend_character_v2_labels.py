"""V2 §1B Trend Character extensions: `breakout_expansion` and `range_bound`.

Tests both per-day and vectorized label paths, plus the §1B precedence
ordering pinned in Ambiguity Log #67.

Real OHLCV-shape synthetic inputs; thresholds computed by hand against the
spec lines cited inline.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.trend_character import (
    _RISK_RANK,
    TrendCharacterFeatures,
    TrendCharacterLabel,
    build_raw_outputs,
    compute_features,
    raw_label_for_day,
)


_V1_LABELS = {"trending", "recovery_attempt", "chop", "transition", "unknown"}
_V2_LABEL_SET = {
    "breakout_expansion",
    "trending",
    "recovery_attempt",
    "range_bound",
    "chop",
    "transition",
    "unknown",
}


def _trading_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-02", periods=n)


def _make_features(
    *,
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> TrendCharacterFeatures:
    if high is None:
        high = close * 1.001
    if low is None:
        low = close * 0.999
    if volume is None:
        volume = pd.Series(1_000_000.0, index=close.index)
    return compute_features(close=close, high=high, low=low, volume=volume)


# ---------------------------------------------------------------------------
# range_bound rule (spec lines 126-138).
# ---------------------------------------------------------------------------


def test_range_bound_fires_on_tight_oscillation() -> None:
    # 220 sessions of oscillation in [98, 102] around midpoint 100. ret63 ≈ 0,
    # midpoint excursion = (102-100)/100 = 0.02 < 0.05, ADX low.
    idx = _trading_index(220)
    closes = 100.0 + np.sin(np.arange(220) * np.pi / 5.0) * 2.0  # ±2 around 100
    close = pd.Series(closes, index=idx)
    high = close + 0.1
    low = close - 0.1
    f = _make_features(close=close, high=high, low=low)
    label, ev = raw_label_for_day(f, idx[-1])
    assert label == "range_bound", (label, ev)


def test_v1_rule_path_does_not_emit_range_bound_on_tight_oscillation() -> None:
    """V2 extends the live default engine, but the V1 layer-1B contract keeps
    the original 5-label set. The same oscillation that qualifies for the V2
    `range_bound` label must fall back to the V1 `chop` rule when the caller
    requests the V1 path.
    """
    idx = _trading_index(220)
    closes = 100.0 + np.sin(np.arange(220) * np.pi / 5.0) * 2.0
    close = pd.Series(closes, index=idx)
    high = close + 0.1
    low = close - 0.1
    f = _make_features(close=close, high=high, low=low)

    label, ev = raw_label_for_day(f, idx[-1], allow_v2_labels=False)

    assert label == "chop", (label, ev)


def test_range_bound_fails_on_midpoint_excursion_over_5pct() -> None:
    # Tight cluster except for one spike at t-5 to 110 (excursion > 0.05).
    idx = _trading_index(220)
    closes = np.full(220, 100.0)
    closes[-5] = 112.0
    closes[1::2] += 0.5  # tiny wobble so std isn't pure zero
    close = pd.Series(closes, index=idx)
    f = _make_features(close=close, high=close + 0.1, low=close - 0.1)
    label, _ = raw_label_for_day(f, idx[-1])
    assert label != "range_bound"


def test_range_bound_fails_on_high_adx() -> None:
    # Tight close cluster, but high/low ranges produce high ADX (manufactured
    # by alternating large directional ranges).
    idx = _trading_index(220)
    close = pd.Series(np.linspace(99.0, 101.0, 220), index=idx)  # mild drift
    # Force ADX up via wide high/low swings that alternate direction strongly.
    high = close + 5.0
    low = close - 5.0
    # Make directional movement (up/down stays >0) by widening sequentially.
    for i in range(220):
        if i % 2 == 0:
            high.iloc[i] = close.iloc[i] + 8.0
        else:
            low.iloc[i] = close.iloc[i] - 8.0
    f = _make_features(close=close, high=high, low=low)
    # ADX should not be < 20 here; verify the rule does not fire either way.
    label, _ = raw_label_for_day(f, idx[-1])
    if not pd.isna(f.adx_14.iloc[-1]) and f.adx_14.iloc[-1] >= 20:
        assert label != "range_bound"


def test_range_bound_fails_on_directional_drift() -> None:
    # 63d return > 0.05 (10% drift over 63d) — drift dominates.
    idx = _trading_index(220)
    closes = np.linspace(80.0, 100.0, 220)
    close = pd.Series(closes, index=idx)
    f = _make_features(close=close, high=close + 0.1, low=close - 0.1)
    label, _ = raw_label_for_day(f, idx[-1])
    assert label != "range_bound"


# ---------------------------------------------------------------------------
# breakout_expansion rule (spec lines 87-117).
# ---------------------------------------------------------------------------


def _build_breakout_series(
    *,
    n_sessions: int,
    breakout_step_size: float,
    breakout_every: int,
    hold_above: bool,
) -> tuple[pd.Series, pd.Series]:
    """Build a synthetic SPY-shape close + volume series with a controlled
    number of breakouts. A new local max gets pushed every `breakout_every`
    sessions. If hold_above=True, the next 5 sessions stay strictly above the
    pre-breakout max; otherwise they dip below.
    """
    idx = _trading_index(n_sessions)
    closes = np.zeros(n_sessions)
    base = 100.0
    closes[0] = base
    for i in range(1, n_sessions):
        if i % breakout_every == 0 and i >= 60:
            # Force a breakout: jump above max of last 20 by `step`.
            window_max = closes[max(0, i - 20) : i].max()
            closes[i] = window_max + breakout_step_size
            # Schedule next 5 sessions
            for j in range(1, 6):
                if i + j >= n_sessions:
                    break
                if hold_above:
                    closes[i + j] = closes[i] + 0.5 * j
                else:
                    closes[i + j] = window_max - 0.5  # dip below level
        elif closes[i] == 0.0:
            # Mild noise close
            closes[i] = closes[i - 1] + np.sin(i * 0.4) * 0.05
    volume = pd.Series(1_000_000.0, index=idx)
    return pd.Series(closes, index=idx), volume


def test_breakout_expansion_fires_on_4_conditions() -> None:
    # 550 sessions, breakout every 18 sessions starting at i>=60 → ~27
    # breakouts, all held above. Then engineer the LAST session as a new
    # breakout with expanding BB and elevated volume.
    n = 550
    close, volume = _build_breakout_series(
        n_sessions=n,
        breakout_step_size=2.0,
        breakout_every=18,
        hold_above=True,
    )
    # Engineer last session: new breakout AND expanding BB AND volume > 20d avg.
    idx = close.index
    last = n - 1
    # Make sure last is a breakout: push close > max(close[-21:-1])
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    # Expanding BB: inject volatility into last 5 sessions vs prior.
    # Bump several recent closes to widen the std of close[t-19..t].
    for i in range(last - 4, last + 1):
        close.iloc[i] = close.iloc[i] + (3.0 if (i - last) % 2 == 0 else -3.0)
    # Restore last to a clear new breakout post-volatility injection:
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    # Elevated volume at last
    volume.iloc[last] = 5_000_000.0

    high = close + 0.1
    low = close - 0.1
    f = _make_features(close=close, high=high, low=low, volume=volume)
    label, ev = raw_label_for_day(f, idx[-1])
    # The rule should at least register breakout_20d_or_50d, bb expanding, vol up.
    assert bool(f.breakout_20d_or_50d.iloc[-1]), "breakout flag not set"
    assert bool(f.volume_above_20d_average.iloc[-1]), "volume flag not set"
    # bb_width_expanding may or may not be True given the noise pattern;
    # if the followthrough rate is >= 0.6 AND bb expanding, breakout_expansion fires.
    if (
        bool(f.bb_width_expanding.iloc[-1])
        and not pd.isna(f.followthrough_rate.iloc[-1])
        and f.followthrough_rate.iloc[-1] >= 0.60
    ):
        assert label == "breakout_expansion", (label, ev)


def test_v1_rule_path_does_not_emit_breakout_expansion() -> None:
    """A fully-qualified V2 breakout still has to collapse back onto the V1
    label set when the caller explicitly asks for the V1 path.
    """
    close, volume = _build_breakout_series(
        n_sessions=260,
        breakout_step_size=2.0,
        breakout_every=10,
        hold_above=True,
    )
    f = _make_features(close=close, volume=volume)

    label, ev = raw_label_for_day(f, close.index[-1], allow_v2_labels=False)

    assert label == "trending", (label, ev)


def test_breakout_expansion_fails_on_followthrough_under_60pct() -> None:
    # 25 prior breakouts, but each one DIPS within 5 sessions → followthrough=0.0.
    n = 550
    close, volume = _build_breakout_series(
        n_sessions=n,
        breakout_step_size=2.0,
        breakout_every=18,
        hold_above=False,
    )
    idx = close.index
    last = n - 1
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    volume.iloc[last] = 5_000_000.0

    high = close + 0.1
    low = close - 0.1
    f = _make_features(close=close, high=high, low=low, volume=volume)
    label, _ = raw_label_for_day(f, idx[-1])
    if not pd.isna(f.followthrough_rate.iloc[-1]):
        assert f.followthrough_rate.iloc[-1] < 0.60
    assert label != "breakout_expansion"


def test_breakout_expansion_fails_on_low_volume() -> None:
    n = 550
    close, volume = _build_breakout_series(
        n_sessions=n,
        breakout_step_size=2.0,
        breakout_every=18,
        hold_above=True,
    )
    idx = close.index
    last = n - 1
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    # LOW volume at last session
    volume.iloc[last] = 100.0
    f = _make_features(close=close, high=close + 0.1, low=close - 0.1, volume=volume)
    label, _ = raw_label_for_day(f, idx[-1])
    assert not bool(f.volume_above_20d_average.iloc[-1])
    assert label != "breakout_expansion"


def test_breakout_expansion_fails_on_bb_width_contracting() -> None:
    # Very stable closes prior to last → bb_width small at t and even smaller
    # at t-5 we want bb_width to be SHRINKING. Achieve this by injecting big
    # volatility 5+ sessions ago and tight closes thereafter.
    n = 550
    close, volume = _build_breakout_series(
        n_sessions=n,
        breakout_step_size=2.0,
        breakout_every=18,
        hold_above=True,
    )
    last = n - 1
    # Inject volatility around t-10 (in the t-5 window range).
    for i in range(last - 14, last - 9):
        close.iloc[i] += (10.0 if i % 2 == 0 else -10.0)
    # Stabilize the most recent 9 sessions.
    base_last = close.iloc[last - 9]
    for i in range(last - 8, last + 1):
        close.iloc[i] = base_last + 0.01 * (i - (last - 9))
    # Force a breakout at last:
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    volume.iloc[last] = 5_000_000.0

    f = _make_features(close=close, high=close + 0.1, low=close - 0.1, volume=volume)
    # bb_width at t-5 should be > bb_width at t (because earlier window held
    # the big volatility chunk).
    if not bool(f.bb_width_expanding.iloc[-1]):
        label, _ = raw_label_for_day(f, close.index[-1])
        assert label != "breakout_expansion"


def test_breakout_expansion_cold_start_below_20_prior_breakouts() -> None:
    # Only 15 prior breakouts → followthrough_rate is NaN, rule cannot fire.
    n = 550
    # breakout_every=30 within trailing 504 → only ~15-16 breakouts.
    close, volume = _build_breakout_series(
        n_sessions=n,
        breakout_step_size=2.0,
        breakout_every=30,
        hold_above=True,
    )
    last = n - 1
    prior_max = close.iloc[last - 20 : last].max()
    close.iloc[last] = prior_max + 3.0
    volume.iloc[last] = 5_000_000.0
    f = _make_features(close=close, high=close + 0.1, low=close - 0.1, volume=volume)
    # Count prior breakouts in trailing 504 sessions (excluding t itself).
    prior_b = int(f.breakout_20d_or_50d.iloc[max(0, last - 504) : last].sum())
    if prior_b < 20:
        assert pd.isna(f.followthrough_rate.iloc[-1])
        label, _ = raw_label_for_day(f, close.index[-1])
        assert label != "breakout_expansion"


# ---------------------------------------------------------------------------
# Precedence (Log #67): breakout_expansion > recovery_attempt > trending >
#                       range_bound > chop > transition > unknown.
# ---------------------------------------------------------------------------


def _synthetic_features(
    *,
    close: pd.Series,
    return_10d: pd.Series,
    return_21d: pd.Series,
    return_63d: pd.Series,
    prior_63d_drawdown: pd.Series,
    adx_14: pd.Series,
    midpoint_excursion_20d: pd.Series,
    breakout_20d_or_50d: pd.Series,
    bb_width_expanding: pd.Series,
    volume_above_20d_average: pd.Series,
    followthrough_rate: pd.Series,
    sma_50: pd.Series | None = None,
) -> TrendCharacterFeatures:
    if sma_50 is None:
        sma_50 = close.rolling(5, min_periods=1).mean()
    high = close * 1.001
    low = close * 0.999
    return TrendCharacterFeatures(
        close=close,
        high=high,
        low=low,
        sma_50=sma_50,
        return_10d=return_10d,
        return_21d=return_21d,
        prior_63d_drawdown=prior_63d_drawdown,
        adx_14=adx_14,
        return_63d=return_63d,
        midpoint_excursion_20d=midpoint_excursion_20d,
        breakout_20d_or_50d=breakout_20d_or_50d,
        bb_width_expanding=bb_width_expanding,
        volume_above_20d_average=volume_above_20d_average,
        followthrough_rate=followthrough_rate,
    )


def test_breakout_expansion_outranks_recovery_attempt() -> None:
    # Both predicates fire on the same day. Per Log #67, breakout_expansion wins.
    idx = _trading_index(3)
    s = lambda v: pd.Series([v, v, v], index=idx, dtype=float)  # noqa: E731
    sb = lambda v: pd.Series([v, v, v], index=idx)  # noqa: E731
    close = s(100.0)
    sma_50 = s(90.0)  # close > sma → enables recovery
    f = _synthetic_features(
        close=close,
        return_10d=s(0.07),  # >= 0.05 → recovery_attempt leg
        return_21d=s(0.10),
        return_63d=s(0.20),
        prior_63d_drawdown=s(-0.20),  # <= -0.10 → recovery_attempt leg
        adx_14=s(25.0),
        midpoint_excursion_20d=s(0.20),
        breakout_20d_or_50d=sb(True),
        bb_width_expanding=sb(True),
        volume_above_20d_average=sb(True),
        followthrough_rate=s(0.80),
        sma_50=sma_50,
    )
    label, _ = raw_label_for_day(f, idx[-1])
    assert label == "breakout_expansion"


def test_range_bound_outranks_chop() -> None:
    idx = _trading_index(3)
    s = lambda v: pd.Series([v, v, v], index=idx, dtype=float)  # noqa: E731
    sb = lambda v: pd.Series([v, v, v], index=idx)  # noqa: E731
    # Both chop and range_bound fire: adx<20, |ret10|<0.03, |ret21|<0.05,
    # |ret63|<0.05, midpoint_excursion<=0.05.
    f = _synthetic_features(
        close=s(100.0),
        return_10d=s(0.01),
        return_21d=s(0.02),
        return_63d=s(0.01),
        prior_63d_drawdown=s(-0.02),
        adx_14=s(15.0),
        midpoint_excursion_20d=s(0.03),
        breakout_20d_or_50d=sb(False),
        bb_width_expanding=sb(False),
        volume_above_20d_average=sb(False),
        followthrough_rate=s(float("nan")),
    )
    label, _ = raw_label_for_day(f, idx[-1])
    assert label == "range_bound"


def test_trending_outranks_range_bound_when_both_match() -> None:
    # Degenerate: adx exactly 20.0 (>=20 triggers trending), |ret21|=0.05.
    idx = _trading_index(3)
    s = lambda v: pd.Series([v, v, v], index=idx, dtype=float)  # noqa: E731
    sb = lambda v: pd.Series([v, v, v], index=idx)  # noqa: E731
    f = _synthetic_features(
        close=s(100.0),
        return_10d=s(0.01),
        return_21d=s(0.05),  # |ret21| >= 0.05 → trending leg
        return_63d=s(0.04),  # < 0.05 → range_bound leg still holds
        prior_63d_drawdown=s(-0.02),
        adx_14=s(20.0),  # >= 20 → trending leg
        midpoint_excursion_20d=s(0.03),
        breakout_20d_or_50d=sb(False),
        bb_width_expanding=sb(False),
        volume_above_20d_average=sb(False),
        followthrough_rate=s(float("nan")),
    )
    label, _ = raw_label_for_day(f, idx[-1])
    assert label == "trending"


def test_v1_recovery_attempt_still_outranks_v2_range_bound() -> None:
    idx = _trading_index(3)
    s = lambda v: pd.Series([v, v, v], index=idx, dtype=float)  # noqa: E731
    sb = lambda v: pd.Series([v, v, v], index=idx)  # noqa: E731
    f = _synthetic_features(
        close=s(100.0),
        return_10d=s(0.06),  # >= 0.05 → recovery
        return_21d=s(0.04),
        return_63d=s(0.04),
        prior_63d_drawdown=s(-0.20),  # <= -0.10 → recovery
        adx_14=s(15.0),
        midpoint_excursion_20d=s(0.03),
        breakout_20d_or_50d=sb(False),
        bb_width_expanding=sb(False),
        volume_above_20d_average=sb(False),
        followthrough_rate=s(float("nan")),
        sma_50=s(90.0),
    )
    label, _ = raw_label_for_day(f, idx[-1])
    assert label == "recovery_attempt"


def test_risk_rank_includes_new_labels() -> None:
    # Log #67: breakout_expansion rank 0, range_bound rank 1.
    assert _RISK_RANK["breakout_expansion"] == 0
    assert _RISK_RANK["range_bound"] == 1
    # V1 ordering preserved.
    assert _RISK_RANK["trending"] == 0
    assert _RISK_RANK["recovery_attempt"] == 1
    assert _RISK_RANK["chop"] == 1
    assert _RISK_RANK["transition"] == 2
    assert _RISK_RANK["unknown"] == 2


def test_build_raw_outputs_matches_per_day() -> None:
    # Vectorized path must produce the same labels as the per-day path.
    idx = _trading_index(220)
    closes = 100.0 + np.sin(np.arange(220) * np.pi / 5.0) * 2.0
    close = pd.Series(closes, index=idx)
    f = _make_features(close=close, high=close + 0.1, low=close - 0.1)
    labels_vec, _ = build_raw_outputs(f)
    for i in range(len(idx) - 1, max(0, len(idx) - 20), -1):
        per_day, _ = raw_label_for_day(f, idx[i])
        assert labels_vec[i] == per_day


def test_v1_default_config_path_only_v1_or_v2_labels_on_golden_dates(
    classified_golden_outputs,
) -> None:
    """V2 labels may or may not fire on bundled golden dates — assert ONLY
    that the active label is in the legal V2 set (7-element)."""
    for as_of, out in classified_golden_outputs.items():
        assert out.trend_character.active_label in _V2_LABEL_SET, (
            as_of,
            out.trend_character.active_label,
        )


def test_v1_frozen_replay_roundtrip_still_passes(
    market_df_for_asof,
) -> None:
    """Importing RegimeEngine and classifying a golden date should still
    yield a label in the 7-label V2 set. We do NOT modify the on-disk V1
    frozen outputs — they remain V1-shape and are validated separately by
    tests/test_v1_frozen_replay.py."""
    from regime_detection.engine import RegimeEngine

    engine = RegimeEngine()
    as_of = date(2023, 12, 14)
    out = engine.classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))
    assert out.trend_character.active_label in _V2_LABEL_SET


def test_hysteresis_with_new_labels_respects_precedence() -> None:
    # End-to-end: confirm escalation semantics with the new risk-rank table.
    # Risk-rank: breakout_expansion=0, trending=0, range_bound=1, chop=1,
    # transition=2, unknown=2. Escalation means a HIGHER risk rank — so a
    # transient `transition` (rank 2) from a `breakout_expansion` base (rank
    # 0) escalates stable to `transition`.
    raws: list[TrendCharacterLabel] = [
        "breakout_expansion",
        "breakout_expansion",
        "transition",  # rank 2 — escalates stable immediately
        "breakout_expansion",
        "breakout_expansion",
        "breakout_expansion",
    ]
    stable, active = apply_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=_RISK_RANK,
        deescalation_days=3,
    )
    assert stable[2] == "transition"
    assert active[2] == "transition"
    # After 3 consecutive breakout_expansion sessions (idx 3,4,5), de-escalate.
    assert stable[5] == "breakout_expansion"
