# Regime Engine Runtime Contracts

This document records the runtime contracts added after the cross-axis dependency
audit. It complements the V1/V2 specs by naming the code surfaces that now own
dependency semantics, absence policy, request validation, typed evidence, runner
diagnostics, and pyright ratcheting.

## Cross-Axis Dependency Contracts

`src/regime_detection/axis_series.py` owns `AXIS_DEPENDENCY_CONTRACTS`.

Each edge declares:

- upstream axis;
- downstream consumer;
- payload fields crossing the edge;
- behavior for absent, stale, unknown, degraded, and invalid states.

`AXIS_DEPENDENCIES` is derived from those contracts and remains the build-order
graph. It is not the source of semantic truth.

Current label-only edges intentionally pass only labels:

- `breadth_state`, `volatility_state`, and `credit_funding_effective` into
  `network_fragility`;
- `credit_funding_effective` into `inflation_growth`;
- V1 axis labels into transition-risk history and selection.

Changing a downstream consumer to read upstream evidence, stable labels, or
data-quality fields requires updating the dependency contract before changing
the wire shape.

## Absence Policy Registry

`src/regime_detection/boundary_policies.py` owns `BOUNDARY_ABSENCE_POLICIES`.

The registry declares which behavior is intentional at each boundary:

- `raise` for deterministic caller or required-input errors;
- `none` for optional V2 seams that are not built;
- `unknown` for present-but-unusable inputs that should remain visible in
  output evidence;
- `degraded` when a boundary can truthfully classify with degraded quality.

The goal is declaration, not forced uniformity. Missing `event_calendar` should
raise; an unlit optional V2 seam may stay `None`; stale series may force an
`unknown` classification.

## Request Contract

`src/regime_detection/engine.py` owns `ClassifyRequest`.

`RegimeEngine.classify()` and `RegimeEngine.classify_window()` are wrappers over
`RegimeEngine.classify_request()`. New required inputs, request-source metadata,
and invalid-combination validation should be added to `ClassifyRequest` first.

Current hard boundaries:

- `event_calendar` is required;
- `lookback_days` must be positive;
- `as_of_date` / `end_date` must be an NYSE trading day;
- `request_source="direct"` cannot carry manifest metadata;
- `request_source="profile_manifest"` must identify manifest-backed required
  inputs through `manifest_resolved_inputs` or `manifest_cli_overrides`;
- legacy `breadth_data` is removed from the API and must fail loudly if passed.

## Typed Evidence Payloads

`src/regime_detection/models.py` owns typed axis evidence payloads.

Typed payloads preserve dict-like report behavior while forbidding undeclared
keys. They currently cover:

- `TransitionRiskEvidencePayload`;
- `CreditFundingEvidencePayload`;
- `NetworkFragilityEvidencePayload`;
- `InflationGrowthEvidencePayload`;
- `MonetaryPressureEvidencePayload`;
- `VolumeLiquidityEvidencePayload`.

Typing evidence should follow dependency semantics, not lead it. Add or change a
payload field only after the owning dependency or boundary contract says the
field is part of the runtime behavior.

## Operator Diagnostics

Profile, shadow, and walk-forward artifacts expose V2 dependency payload
contracts so operators can see whether a run used label-only or richer
cross-axis payloads.

Current artifact fields:

- profile compact timeline: `dependency_payload_contracts`;
- historical walk-forward summary: `v2_dependency_payload_contracts`;
- shadow output and replay diff payloads: `v2_dependency_payload_contracts`.

Replay comparison includes the diagnostic contract, so a payload-contract drift
is a replay mismatch instead of a silent report-only change.

## Pyright Ratchet

`pyproject.toml` includes `src/regime_detection/engine.py`,
`src/regime_detection/models.py`, and `src/regime_detection/axis_series.py` in
the strict-check set.

`docs/pyright_pandas_stub_policy.md` defines the narrow rule for pandas-stub
noise and Pydantic compatibility suppressions. Suppressions may isolate framework
typing gaps, but they must not hide classifier branches, thresholds, dependency
semantics, date alignment, or unvalidated input handling.
