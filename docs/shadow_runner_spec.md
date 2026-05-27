# Shadow Runner Spec

Operational qualification contract for V1 forward shadow mode.

This document defines what counts as a qualifying shadow run for the V2 activation gate. It is intentionally stricter than a historical replay because it must prove both engine correctness and operational discipline.

## 0. Purpose

Forward shadow mode answers a different question than historical walk-forward.

- Historical walk-forward asks: "Does the frozen engine logic behave sensibly on unseen historical data when fed only as-of inputs?"
- Forward shadow asks: "Can the system run daily for real NYSE sessions without operational gaps, data corruption, silent failures, or unreproducible outputs?"

V2 activation requires both:

1. Historical walk-forward passes first.
2. Forward shadow then runs for 252 consecutive NYSE trading sessions.

These modes are not interchangeable.

Historical walk-forward qualification details live in `docs/historical_walkforward_spec.md`.

## 1. Qualification Standard

The forward shadow run qualifies only if all of the following hold:

- The runner executes for 252 consecutive NYSE trading sessions.
- Each session archives the exact inputs used before classification begins.
- Each session emits exactly one immutable output for the tuple `(as_of_date, engine_version, config_version)`.
- Replays of archived historical sessions reproduce the stored output exactly, or any mismatch is logged as a qualification-breaking incident.
- No classification bug is fixed during the active qualification window without restarting the window.

## 2. Storage Model

Use local VPS storage in V1. Do not add S3 or a network database for the first qualifying shadow run.

Layout:

```text
shadow_run/
├── regime_shadow.db
├── outputs/
│   └── YYYY-MM-DD.json
└── input_archives/
    └── YYYY-MM-DD/
        ├── market_data.parquet
        ├── events.yaml
        └── checksums.json
```

Rules:

- `regime_shadow.db` is the authoritative run ledger and replay-check index.
- JSON output files are the canonical human-readable output artifacts.
- Archived input files are the canonical replay inputs.

## 3. SQLite Schema

Minimum schema:

```sql
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    as_of_date DATE NOT NULL,
    engine_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    input_archive_path TEXT NOT NULL,
    output_path TEXT,
    output_sha256 TEXT,
    UNIQUE (as_of_date, engine_version, config_version)
);

CREATE TABLE replay_checks (
    check_id INTEGER PRIMARY KEY,
    check_timestamp TIMESTAMP NOT NULL,
    original_run_id INTEGER REFERENCES runs(run_id),
    matches BOOLEAN NOT NULL,
    diff TEXT
);

CREATE TABLE incidents (
    incident_id INTEGER PRIMARY KEY,
    incident_date DATE NOT NULL,
    description TEXT NOT NULL,
    resolution TEXT,
    breaks_qualification BOOLEAN NOT NULL
);
```

The `UNIQUE (as_of_date, engine_version, config_version)` constraint is mandatory. It prevents silent overwrites and makes the ledger self-enforce output immutability for a given versioned run.

## 4. Data Source Policy

V1 shadow-mode source of truth:

- Market data: Stooq daily OHLCV for `SPY`, `RSP`, and the VIX proxy used by V1.
- Event calendar: the exact YAML/CSV snapshot supplied to the runner for that date.

Rules:

- The runner archives inputs before calling the engine.
- Historical replay reads only from archived inputs, never from a live refetch.
- If Stooq has a quality incident, document it in `incidents` and upgrade the source only via an explicit versioned operational change.

## 5. Runner Execution Contract

Daily flow:

1. Fetch market data and event calendar for the target `as_of_date`.
2. Archive all inputs to `input_archives/YYYY-MM-DD/`.
3. Compute SHA-256 checksums for every archived input and write `checksums.json`.
4. Insert a `runs` row with `status='in_progress'`.
5. Call `RegimeEngine.classify(...)` or `classify_window(...)` using only the archived inputs.
6. Write the output JSON artifact.
7. Update the `runs` row to `status='success'` with `output_path` and `output_sha256`.
8. On failure, update the `runs` row to `status='failure'` with a deterministic `failure_reason`.

Archive inputs before classification. If classification crashes, the archived inputs must still exist for diagnosis and replay.

Shadow JSON artifacts also include `v2_dependency_payload_contracts`. This field
records the active cross-axis payload shape, for example label-only edges into
network fragility and inflation/growth. It is part of replay equality: if the
model output is byte-stable but dependency payload semantics drift, replay must
report a mismatch.

## 6. Freeze Policy

Config and classification logic are fluid during V1 implementation and historical walk-forward preparation. They become frozen only when shadow qualification begins.

Required sequence:

1. V1 implementation completes.
2. Historical walk-forward passes on frozen V1 logic/config.
3. Tag the repo for shadow start.
4. Start forward shadow on the next NYSE trading day.

During the qualification window:

- Cosmetic or logging-only fixes may continue without restarting the window if they do not affect classification.
- Any classification bug fix or threshold/config change is qualification-breaking and restarts the 252-session window.

Every qualification-breaking change must be recorded in `incidents` with `breaks_qualification=true`.

## 7. Replay Verification

Replay verification is a separate process, not part of the daily classification run.

Rules:

- Select archived historical sessions for spot-check replay.
- Re-run the engine against the archived inputs.
- Compare the replayed output to the stored JSON output.
- Record the result in `replay_checks`.
- Any mismatch is treated as a serious incident. If it changes classification behavior, it breaks qualification.

## 8. Historical Walk-Forward Gate

Historical walk-forward is mandatory before forward shadow starts.

Requirements:

- At least one full out-of-sample year of frozen V1 logic/config.
- As-of historical inputs only; no future leakage.
- No engine crashes.
- No NaN leakage beyond the explicit V1 `unknown`/`insufficient_history` contract.
- All 10 golden test dates still pass.
- Label distributions and transitions are reviewed and deemed economically defensible.

This gate is fast and logic-focused. It does not replace forward shadow.

## 9. Operational Monitoring

The shadow runner must include a dead-man's switch.

Minimum behavior:

- Daily runner executes after market close on each NYSE trading day.
- A separate monitor checks the next morning that the previous NYSE session produced a `runs` row.
- If the run is missing, the monitor alerts immediately.

No silent gaps are allowed in the qualifying window.

## 10. Success Metrics Beyond Classification

The forward shadow run should also accumulate the evidence needed for the separate V2 prerequisite of measurable strategy improvement versus a no-regime baseline.

Track at minimum:

- strategy return
- max drawdown
- Sharpe
- false switch rate
- average detection lag
- wrong-environment trades avoided

The exact strategy-routing comparison may live outside this repo, but the prerequisite is not satisfied until those metrics exist in a reproducible report.

## 11. Non-Goals for V1

Do not add these to the first qualifying shadow runner:

- S3 archival
- Postgres
- multi-writer infrastructure
- strategy-backtest-specific orchestration copied from another repo

V1 needs a small, durable, reproducible runner with strict archives and a trustworthy ledger, not a large deployment platform.
