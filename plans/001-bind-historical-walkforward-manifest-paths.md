# Plan 001: Bind historical walk-forward to manifest-resolved paths

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving on. If a STOP condition occurs, stop and report.
>
> **Drift check (run first)**: `git diff --stat 7f8608fe..HEAD -- scripts/run_historical_walkforward.py scripts/_v2_calibration_helpers.py tests/test_runner_manifest_materialization.py tests/test_historical_walkforward.py`
> If any in-scope file changed, compare the current excerpts below to live code before editing.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: MED
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `7f8608fe`, 2026-06-18
- **Status**: DONE

## Why this matters

ADR 0010 says production runners use the manifest input resolver so manifest `local_path` is the source of truth. `scripts/run_historical_walkforward.py` currently materializes the manifest but keeps default/manual paths, so a manifest whose artifact paths differ from defaults can download valid inputs and then classify with different or missing files.

## Current state

- `docs/decisions/0010-per-label-hysteresis-and-audit-hardening.md:34` states: "All production runners use the manifest input resolver."
- `scripts/run_historical_walkforward.py:53-57` imports `apply_manifest_input_defaults` and `register_manifest_input_args`, but not `apply_manifest_input_paths`.
- `scripts/run_historical_walkforward.py:543-555` fills default paths before materialization.
- `scripts/run_historical_walkforward.py:561-567` calls `materialize_if_requested(...)`.
- `scripts/run_historical_walkforward.py:568-583` passes `args.event_calendar`, `args.macro_parquet`, `args.pmi_path`, `args.cpi_nowcast_parquet`, and `args.aggregate_forward_eps_weekly_history_parquet` to `run_walkforward` without manifest-resolution binding.
- `tests/test_runner_manifest_materialization.py:171-227` only asserts that materialization happened; it does not assert that the runner consumed the manifest-resolved path.

Use the existing runner pattern in `scripts/run_v2_walkforward_gate.py:400-410`: defaults, materialize, then `apply_manifest_input_paths(...)`.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Targeted tests | `python3 -m pytest tests/test_runner_manifest_materialization.py tests/test_historical_walkforward.py -q ; echo "EXIT:$?"` | `EXIT:0` |
| Lint touched files | `python3 -m ruff check scripts/run_historical_walkforward.py tests/test_runner_manifest_materialization.py tests/test_historical_walkforward.py` | exit 0 |

## Scope

**In scope**
- `scripts/run_historical_walkforward.py`
- `tests/test_runner_manifest_materialization.py`
- `tests/test_historical_walkforward.py` if a smaller unit test fits better there

**Out of scope**
- `scripts/profile_engine.py`, `scripts/audit_layer2_30d.py`, V2 gate scripts
- Any change to manifest schema or artifact materialization semantics

## Steps

### Step 1: Add a failing path-binding test

Add a test that builds a manifest with `required_for: ["historical_walkforward"]` and non-default `local_path` values for at least `event_calendar_us` and `fred_macro_series`.

Patch the runner so `run_walkforward` is replaced with a fake that records the `event_calendar_path` and `macro_parquet_path` it receives, then call `main()` with `--manifest`, `--data-root`, and required CLI args. Assert the fake received the manifest-resolved paths, not the default paths under `data/raw/event_calendar/us_events.yaml` and `data/raw/macro/fred_macro_series.parquet`.

**Verify**: `python3 -m pytest tests/test_runner_manifest_materialization.py -q ; echo "EXIT:$?"` -> initially fails before the production fix and passes after Step 2.

### Step 2: Bind historical walk-forward args through the resolver

In `scripts/run_historical_walkforward.py`, import `apply_manifest_input_paths` from `scripts._v2_calibration_helpers`.

After `materialize_if_requested(...)` in `main()`, call:

```python
apply_manifest_input_paths(
    args,
    runner_name="historical_walkforward",
    repo_root=REPO_ROOT,
    required_fields=frozenset({"macro_parquet", "event_calendar"}),
)
```

Keep `market_data`, `v2_daily_ohlcv`, and `pit_constituent_intervals` unchanged; they are not currently declared required manifest inputs for this runner.

**Verify**: `python3 -m pytest tests/test_runner_manifest_materialization.py -q ; echo "EXIT:$?"` -> `EXIT:0`.

## Done criteria

- [x] Historical walk-forward calls the manifest resolver after materialization.
- [x] A test fails on the old default-path behavior and passes with manifest-resolved paths.
- [x] `python3 -m pytest tests/test_runner_manifest_materialization.py tests/test_historical_walkforward.py -q ; echo "EXIT:$?"` returns `EXIT:0`.
- [x] `python3 -m ruff check scripts/run_historical_walkforward.py tests/test_runner_manifest_materialization.py tests/test_historical_walkforward.py` exits 0.

## STOP conditions

- The live code already calls `apply_manifest_input_paths` in `run_historical_walkforward.py`.
- The resolver requires fields that historical walk-forward cannot honestly supply from a manifest.
- Fixing this requires changing manifest schema or shared materialization behavior.

## Maintenance notes

Reviewers should check CLI overrides still win over manifest resolution. This is an input-routing fix only; do not change classification behavior.
