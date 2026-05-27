"""Generate the pinned followthrough_rate YAML fixture.

Run once to capture the CURRENT (slow) implementation's output against the
SPY market_data fixture. The committed YAML fixture is what
``test_followthrough_rate_matches_pinned_output_on_realistic_close_series``
asserts against, both before and after the algorithmic change.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import conftest  # noqa: E402

from regime_detection.trend_character import (  # noqa: E402
    _compute_breakout_20d_or_50d,
    _compute_followthrough_rate,
    _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
)


def main() -> None:
    market_data = conftest._load_market_data()
    close = conftest._close_series_from_market_data(market_data, "SPY")
    breakout = _compute_breakout_20d_or_50d(close)
    ft_rate = _compute_followthrough_rate(
        close,
        breakout,
        lookback_sessions=_DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
        window_count=_DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
        hold_sessions=_DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
    )

    rows: list[dict[str, object]] = []
    for ts, value in ft_rate.items():
        if isinstance(value, float) and math.isnan(value):
            rows.append({"date": ts.date().isoformat(), "value": None})
        else:
            rows.append({"date": ts.date().isoformat(), "value": float(value)})

    fixture_path = (
        REPO_ROOT / "tests" / "fixtures" / "derived" / "followthrough_rate_pinned.yaml"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "symbol": "SPY",
                "lookback_sessions": _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
                "window_count": _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
                "hold_sessions": _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
                "rows": rows,
            },
            sort_keys=False,
        )
    )
    print(f"Wrote {fixture_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
