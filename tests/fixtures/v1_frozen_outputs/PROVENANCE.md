# V1 Frozen Output Fixtures — Provenance

The `*.json` files in this directory are byte-frozen snapshots of `RegimeOutput.model_dump_json(exclude_none=True)` captured at the V1 wire-shape baseline. They are the regression anchor for `tests/test_v1_frozen_replay.py`; any drift in their JSON content (modulo `engine_version`) signals a V1 wire-contract change and must be investigated, not silently regenerated.

**Do not modify these JSON files in-place.** Doing so will mask exactly the regression they exist to catch. If a deliberate V1 wire change is required, replace the file in a dedicated commit that:

1. Names the spec line that authorized the change.
2. Updates `tests/_v1_frozen_models.py` accordingly.
3. Updates this PROVENANCE.md with the new capture record.

## Current snapshots

| File | as_of_date | Originating commit (capture) |
|---|---|---|
| `2023-12-14.json` | 2023-12-14 | `482e44b` (Phase B snapshot baseline — see `_v1_frozen_models.py` header) |
| `2024-02-15.json` | 2024-02-15 | `482e44b` (Phase B snapshot baseline — captured in the same wire-shape pass as `2023-12-14.json`) |
| `2024-04-15.json` | 2024-04-15 | `482e44b` (Phase B snapshot baseline — captured in the same wire-shape pass as `2023-12-14.json`) |

## Regeneration command (only when authorized)

```
python3 - <<'PY'
from datetime import date
from pathlib import Path
from regime_detection.engine import RegimeEngine
# Replace market_data with the fixture loader used by the rest of the suite
# (tests/conftest.py::market_df_for_asof) and pin the engine_version + config
# at the V1 baseline commit before re-running.
out = RegimeEngine().classify(
    as_of_date=date(2023, 12, 14),
    market_data=...,
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
