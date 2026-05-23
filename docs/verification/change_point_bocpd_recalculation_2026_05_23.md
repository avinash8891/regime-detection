# Change-Point BOCPD Recalculation

- Generated: 2026-05-23T19:56:07+00:00
- Data end date: 2026-05-15
- Method: BOCPD
- Observation: `realized_vol_21d`
- Posterior emitted by runtime: `sum(R[1:22, t])`
- `score_window_days`: 5
- `recent_run_length_window_days`: 21
- `break_threshold`: 0.5
- `training_window_days`: 2705

This supersedes the older verification notes that reported
`change_point >= 0.5` as 0.0%. That zero activation came from reading only an
exact one-step BOCPD posterior row; the current implementation uses recent
short-run posterior mass, matching the 21-session realized-volatility
observation horizon.

## Activation Summary

| window | sessions | score non-null | score >= 0.5 | activation rate | posterior >= 0.5 | days_since_last_break non-null |
|---|---:|---:|---:|---:|---:|---:|
| 2016-01-04 to 2026-05-15 | 2607 | 2607 | 391 | 15.0% | 294 | 2544 |
| 2025-02-07 to 2026-05-08 | 314 | 314 | 73 | 23.2% | 60 | 314 |
| 2026-02-12 to 2026-05-08 | 60 | 60 | 19 | 31.7% | 15 | 60 |
| 2026-02-20 to 2026-05-15 | 60 | 60 | 19 | 31.7% | 15 | 60 |

## Top Score Dates

### 2016+ Full Replay

| date | score | posterior |
|---|---:|---:|
| 2018-02-12 | 0.996000 | 0.996000 |
| 2018-02-15 | 0.996000 | 0.996000 |
| 2018-02-14 | 0.996000 | 0.996000 |
| 2018-02-13 | 0.996000 | 0.996000 |
| 2018-02-09 | 0.996000 | 0.996000 |

### 2025-02-07 to 2026-05-08

| date | score | posterior |
|---|---:|---:|
| 2025-06-09 | 0.996000 | 0.995331 |
| 2025-06-06 | 0.996000 | 0.995980 |
| 2025-06-03 | 0.996000 | 0.996000 |
| 2025-06-04 | 0.996000 | 0.996000 |
| 2025-06-05 | 0.996000 | 0.995997 |

### 2026-02-12 to 2026-05-08

| date | score | posterior |
|---|---:|---:|
| 2026-04-13 | 0.995480 | 0.995480 |
| 2026-04-14 | 0.995480 | 0.995322 |
| 2026-04-15 | 0.995480 | 0.994626 |
| 2026-04-16 | 0.995480 | 0.992789 |
| 2026-04-17 | 0.995480 | 0.987643 |

## Reproducibility

Focused verification command:

```bash
PYTHONPATH=src:. python3 - <<'PY'
from pathlib import Path
import json
import pandas as pd
from regime_detection.config import load_default_regime_config
from regime_detection.volatility_state import realized_vol
from regime_detection.change_point import compute_change_point_features

cfg = load_default_regime_config().change_point
spy = pd.read_parquet(Path("data/raw/daily_ohlcv_762/symbol=SPY/ohlcv.parquet"))
spy["date"] = pd.to_datetime(spy["date"])
close = spy.sort_values("date").set_index("date")["close"].astype(float)
cp = compute_change_point_features(realized_vol_21d=realized_vol(close, 21), config=cfg)
print(json.dumps({
    "data_end_date": str(close.index.max().date()),
    "score_non_null_2016_plus": int(cp.score.loc["2016-01-04":].notna().sum()),
    "score_ge_0_5_2016_plus": int((cp.score.loc["2016-01-04":] >= cfg.break_threshold).sum()),
    "days_since_non_null_2016_plus": int(cp.days_since_last_break.loc["2016-01-04":].notna().sum()),
}, indent=2))
PY
```
