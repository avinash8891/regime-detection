# V1 Frozen Output Fixtures — Provenance

The `*.json` files in this directory are byte-frozen snapshots of `RegimeOutput.model_dump_json(exclude_none=True)` captured at the V1 wire-shape baseline. They are the regression anchor for `tests/test_v1_frozen_replay.py`; any drift in their JSON content (modulo `engine_version`) signals a V1 wire-contract change and must be investigated, not silently regenerated.

**Do not modify these JSON files in-place.** Doing so will mask exactly the regression they exist to catch. If a deliberate V1 wire change is required, replace the file in a dedicated commit that:

1. Names the spec line that authorized the change.
2. Updates `tests/_v1_frozen_models.py` accordingly.
3. Updates this PROVENANCE.md with the new capture record.

## Current snapshots

| File | as_of_date | Originating commit (capture) |
|---|---|---|
| `2023-12-14.json` | 2023-12-14 | `F-004 live replay recapture` — `RegimeEngine.classify(..., config=core3-v1.0.0.yaml)` using `tests/conftest.py::market_df_for_asof` and `event_calendar_df` |
| `2024-02-15.json` | 2024-02-15 | `F-004 live replay recapture` — same command/input path as `2023-12-14.json` |
| `2024-04-15.json` | 2024-04-15 | `F-004 live replay recapture` — same command/input path as `2023-12-14.json` |

## Regeneration command (only when authorized)

```
python3 - <<'PY'
from datetime import date
from pathlib import Path
from regime_detection.engine import RegimeEngine
from regime_detection.config import load_regime_config
# Replace market_data/event_calendar with the fixture loaders used by the
# rest of the suite: tests/conftest.py::market_df_for_asof and event_calendar_df.
config = load_regime_config("src/regime_detection/configs/core3-v1.0.0.yaml")
out = RegimeEngine().classify(
    as_of_date=date(2023, 12, 14),
    market_data=...,
    event_calendar=...,
    config=config,
)
Path("tests/fixtures/v1_frozen_outputs/2023-12-14.json").write_text(
    out.model_dump_json(exclude_none=True, indent=2)
)
PY
```

## Why no provenance metadata is embedded in the JSON itself

The frozen JSON is parsed back through `_v1_frozen_models.RegimeOutputV1Frozen` for round-trip equality (`tests/test_v1_frozen_replay.py:34`). Adding a `provenance` field at the top level would either:

- Force `extra="ignore"` on the frozen model shim (weakening the regression check), or
- Break the round-trip equality (since the frozen model would not preserve an unknown field).

The provenance must therefore live next to the JSON, not inside it. This file is that location.
