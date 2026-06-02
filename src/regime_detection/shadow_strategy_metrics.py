"""§10 reproducible shadow strategy success metrics (F-014, ADR 0025).

A small, deterministic reducer over the shadow ledger — the ``runs`` table, the
per-date classification output JSON, and the archived ``market_data.parquet`` — that
computes the six §10 strategy success metrics under a pinned defensive-overlay
strategy and a no-regime baseline. Pure: re-running on the same ``output_root``
yields identical metrics. See ADR 0025 for every definition; this module is the
single source of truth for the strategy mapping and metric formulas.

This is NOT a backtest platform (shadow_runner_spec §11) — it reads already-persisted
ledger artifacts and applies one documented position rule.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

# Risk-off (flat) predicate — ADR 0025. Mirrors the §3 emergency-override semantics.
DEFENSIVE_TRANSITION_RISK_STATES = frozenset({"crisis"})
DEFENSIVE_VOLATILITY_LABELS = frozenset({"crisis_vol"})
FALSE_SWITCH_HORIZON_SESSIONS = 3
TRADING_DAYS_PER_YEAR = 252
DEFAULT_UNIVERSE_SYMBOL = "SPY"

# Canonical configured crash windows (single source of truth, shared with the §8
# walk-forward red-flag check, F-050). Drawn from the spec §9.4 stress dates /
# golden-date crisis rows (Volmageddon, the Q4-2018 selloff, the COVID crash, and the
# Jun-2022 bear capitulation). Each window is the canonical multi-session episode, not
# a single date. (name, start_iso, end_iso).
CRASH_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("volmageddon_2018", "2018-02-05", "2018-02-12"),
    ("q4_2018_selloff", "2018-12-10", "2018-12-26"),
    ("covid_crash_2020", "2020-02-24", "2020-03-23"),
    ("jun_2022_capitulation", "2022-06-13", "2022-06-17"),
)


@dataclass(frozen=True)
class ShadowStrategyMetrics:
    """The six §10 success metrics plus the no-regime baseline comparison."""

    strategy_return: float
    max_drawdown: float
    sharpe: float
    false_switch_rate: float
    average_detection_lag: float | None
    wrong_environment_trades_avoided: int
    baseline_return: float
    baseline_max_drawdown: float
    baseline_sharpe: float
    session_count: int
    covered_crash_windows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _success_dates(db_path: Path) -> list[date]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT as_of_date FROM runs WHERE status = 'success' ORDER BY as_of_date"
        ).fetchall()
    return [date.fromisoformat(str(row[0])) for row in rows]


def _defensive(label_volatility: str | None, label_transition: str | None) -> bool:
    return (
        label_transition in DEFENSIVE_TRANSITION_RISK_STATES
        or label_volatility in DEFENSIVE_VOLATILITY_LABELS
    )


def _read_exposures(outputs_dir: Path, dates: list[date]) -> list[float]:
    """Defensive-overlay exposure (0.0 flat / 1.0 invested) per session from the
    per-date classification output JSON. ADR 0025."""
    exposures: list[float] = []
    for as_of in dates:
        payload = json.loads(
            (outputs_dir / f"{as_of.isoformat()}.json").read_text(encoding="utf-8")
        )
        volatility = payload.get("volatility_state") or {}
        transition = payload.get("transition_risk") or {}
        flat = _defensive(
            volatility.get("active_label"),
            transition.get("state"),
        )
        exposures.append(0.0 if flat else 1.0)
    return exposures


def _universe_close_by_date(
    archive_market_path: Path, *, symbol: str
) -> dict[date, float]:
    frame = pd.read_parquet(archive_market_path)
    frame = frame.assign(date=pd.to_datetime(frame["date"]).dt.date)
    if "symbol" in frame.columns and (frame["symbol"] == symbol).any():
        frame = frame[frame["symbol"] == symbol]
    return {row["date"]: float(row["close"]) for _, row in frame.iterrows()}


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    series = pd.Series(returns)
    std = float(series.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(series.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _false_switch_rate(exposures: list[float]) -> float:
    """Fraction of exposure switches that revert to the pre-switch value within
    FALSE_SWITCH_HORIZON_SESSIONS. ADR 0025."""
    switches = 0
    false_switches = 0
    for i in range(1, len(exposures)):
        if exposures[i] == exposures[i - 1]:
            continue
        switches += 1
        pre_switch = exposures[i - 1]
        window = exposures[i + 1 : i + 1 + FALSE_SWITCH_HORIZON_SESSIONS]
        if pre_switch in window:
            false_switches += 1
    return false_switches / switches if switches else 0.0


def _average_detection_lag(
    dates: list[date], exposures: list[float]
) -> tuple[float | None, int]:
    """Average session lag from each covered crash window's first covered session to
    the first flat session within the window (full length if never flat). ADR 0025."""
    lags: list[int] = []
    for _name, start_iso, end_iso in CRASH_WINDOWS:
        start = date.fromisoformat(start_iso)
        end = date.fromisoformat(end_iso)
        covered = [
            (idx, as_of) for idx, as_of in enumerate(dates) if start <= as_of <= end
        ]
        if not covered:
            continue
        defensive_offsets = [
            pos for pos, (idx, _as_of) in enumerate(covered) if exposures[idx] == 0.0
        ]
        lags.append(defensive_offsets[0] if defensive_offsets else len(covered))
    if not lags:
        return None, 0
    return sum(lags) / len(lags), len(lags)


def compute_shadow_strategy_metrics(
    output_root: Path, *, universe_symbol: str = DEFAULT_UNIVERSE_SYMBOL
) -> ShadowStrategyMetrics:
    """Compute the §10 metrics deterministically from a shadow ledger. ADR 0025."""
    output_root = Path(output_root)
    db_path = output_root / "regime_shadow.db"
    dates = _success_dates(db_path)
    if len(dates) < 2:
        raise ValueError(
            "shadow strategy metrics require >= 2 successful sessions; "
            f"found {len(dates)}"
        )

    exposures = _read_exposures(output_root / "outputs", dates)
    archive_market_path = (
        output_root / "input_archives" / dates[-1].isoformat() / "market_data.parquet"
    )
    close_by_date = _universe_close_by_date(archive_market_path, symbol=universe_symbol)

    strat_returns: list[float] = []
    base_returns: list[float] = []
    wrong_env_avoided = 0
    for i in range(1, len(dates)):
        prev_close = close_by_date.get(dates[i - 1])
        curr_close = close_by_date.get(dates[i])
        if prev_close is None or curr_close is None or prev_close == 0.0:
            raise ValueError(
                f"missing/zero universe close for {dates[i - 1]} or {dates[i]} "
                f"({universe_symbol}); ledger market archive is incomplete"
            )
        market_return = curr_close / prev_close - 1.0
        signal = exposures[i - 1]  # no lookahead: prior session's regime
        strat_returns.append(signal * market_return)
        base_returns.append(market_return)
        if signal == 0.0 and market_return < 0.0:
            wrong_env_avoided += 1

    strat_equity = _equity_curve(strat_returns)
    base_equity = _equity_curve(base_returns)
    average_lag, covered_windows = _average_detection_lag(dates, exposures)

    return ShadowStrategyMetrics(
        strategy_return=strat_equity[-1] - 1.0,
        max_drawdown=_max_drawdown(strat_equity),
        sharpe=_sharpe(strat_returns),
        false_switch_rate=_false_switch_rate(exposures),
        average_detection_lag=average_lag,
        wrong_environment_trades_avoided=wrong_env_avoided,
        baseline_return=base_equity[-1] - 1.0,
        baseline_max_drawdown=_max_drawdown(base_equity),
        baseline_sharpe=_sharpe(base_returns),
        session_count=len(dates),
        covered_crash_windows=covered_windows,
    )


def _equity_curve(returns: list[float]) -> list[float]:
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return equity
