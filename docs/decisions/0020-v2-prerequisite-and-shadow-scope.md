# ADR 0020 - V2 Prerequisite and Shadow Scope

**Status:** Accepted
**Date:** 2026-05-31
**Context:** The spec-compliance review found several low-risk mismatches where
the implementation was already deterministic, but the written contract implied
a different process or source boundary. This ADR records those decisions so the
V2 extension remains one engine with explicit activation gates, not a second
system with implicit assumptions.

## Decision

- **F-019 — 9-slice prerequisite.** The 9-slice prerequisite is a process gate,
  not an engine runtime assertion. V2 activation remains blocked until the
  slice checklist, walk-forward gate, golden-date gate, and forward-shadow gate
  are all complete. The repository enforces this through
  `docs/v2_slice_gate_checklist.md`, `docs/historical_walkforward_spec.md`,
  `docs/shadow_runner_spec.md`, and their gate tests rather than by adding a
  classifier branch.
- **F-021 — PIT membership row schema.** The shipped PIT constituent artifact
  uses a ticker / start_date / end_date interval schema. That is the canonical
  row-level representation for membership validity because every trading-date
  row can be derived deterministically from the interval. Replacing the free
  `fja05680/sp500` source with CRSP / Compustat / FactSet / Norgate should keep
  the same interval shape unless the loader contract is explicitly amended.
- **F-025 — HMM parameter drift.** HMM parameter-drift flags are not part of the
  capital-protection transition score contract. The current transition score
  consumes point-in-time HMM probability movement, GMM cluster flips, and
  change-point evidence. A future calibration-review tool may add Hungarian
  alignment over HMM parameters, including the V2 §6.1 20% state-mean parameter-drift alert
  and the separate non-blocking 30% transition-probability review flag, but
  absence of that helper must not be confused with missing runtime model
  evidence.
- **F-053 — Vol-crush exposure response.** V2 §5.3 vol-crush exposure response
  is a downstream strategy-layer contract, not `regime_detection` runtime
  logic. The engine's responsibility is to emit the `vol_crush` volatility
  label and evidence correctly. The 50% long-vol exposure reduction over the
  5-day cooldown belongs to the position-management layer that consumes engine
  outputs, not to this classifier package.
- **F-045 — CPI vintage scope.** The V2 §2A dual-vintage implementation is a
  CPI-only dual-vintage store for first-release historical replay. The current
  code loads `CPIAUCSL` realtime observations into
  `cpi_all_items_vintages.parquet`, derives first-release CPI through
  `loaders.load_cpi_vintages_first_release`, and preserves the latest-revision
  path when the vintage seam is absent. There is no generic all-macro vintage
  store in this slice.
- **F-049 — Shadow source of truth.** For this repository's current forward
  shadow implementation, local/Alpaca archived parquet is the shadow source of truth.
  The earlier Stooq wording is superseded by the May 2026 data-source
  plan because the engine-facing ETF and constituent OHLCV artifacts were
  re-fetched and verified from Alpaca/local parquet. Replays read archived
  inputs only.
- **F-050 — Daily fetch boundary.** The daily fetch is upstream of the runner.
  The runner consumes already-fetched inputs, archives them before
  classification, writes checksums, and then calls the engine. This keeps the
  runner small and replayable; fetch failures belong to the acquisition layer
  before a shadow `runs` row is promoted to classification.

## Consequences

- V2 remains an extension of the same engine and contracts; V1 replay and V2
  activation are separated by gates, not by separate codebases.
- Runtime failures still stay loud where capital protection depends on the
  evidence, especially transition-score model evidence.
- Documentation that refers to shadow input sources or fetch ownership must
  cite this ADR and `docs/shadow_runner_spec.md` rather than resurrecting Stooq
  as the current source-of-truth.
