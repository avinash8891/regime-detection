# ADR 0010: Per-label hysteresis and audit hardening

**Date:** 2026-05-19
**Status:** Accepted

## Context

A forensic audit of the regime-detection engine revealed several classes of issues:

1. **Stale source-file comments** — 9 docstrings/comments claiming features were "deferred" or "placeholder" when fully implemented, plus ~50 "(implementation phase)" phase labels adding no value.

2. **Data integrity gaps** — CPI October 2025 missing due to US government shutdown (BLS footnote: "Data unavailable due to the 2025 lapse in appropriations"). CPI vintages parquet retained a NaN row from a previous fetch due to asymmetric `_drop_null_fred_observations` ordering.

3. **Event calendar thin coverage** — ECB/BoE/BoJ decisions only covered 2025-2026; the spec expects 2016+ historical archive coverage. The `HTTP_USER_AGENT` in event_sources was a bot-like string that central banks blocked.

4. **Missing data wiring in production runners** — `run_v2_calibration.py` silently ran without event_calendar, aaii_sentiment, and implied_vol_30d. It also constructed paths manually instead of using the manifest input resolver.

5. **Inconsistent hysteresis** — L2 axes (network_fragility, credit_funding, inflation_growth, monetary_pressure, volume_liquidity) used per-label asymmetric hysteresis. L1 axes (trend_direction, trend_character, volatility, breadth) used flat hysteresis (same days for all labels). The `TrendCharacterV2Config` class existed with per-label hysteresis but was never wired.

6. **Dead schema entries** — `monthly_options_expiry` in EventType was never consumed from YAML rows (expiry_week is computed from trading calendar). `strategy_family_constraints` and `agent_routing` were computed but not surfaced in profile reporting.

## Decisions

### D1: Per-label hysteresis is mandatory for all 9 label axes

All axes now use `apply_per_label_asymmetric_hysteresis` with `deescalation_days_by_label` from the axis-level config section. High-risk labels are sticky (crisis_vol=5d, bear=5d, divergent_fragile=5d), low-risk labels transition fast (normal_vol=0d, bull=0d, healthy_breadth=0d). Missing config raises immediately — no silent flat fallback.

Both `core3-v1.0.0.yaml` and `core3-v2.0.0.yaml` ship per-label hysteresis under neutral axis names (`trend_direction`, `trend_character`, `volatility_state`, `breadth_state`, etc.). V2 feature/rule sections such as `trend_direction_v2` do not own hysteresis for V1-origin raw labels.

### D2: Event calendar is an S3 artifact, not git-tracked config

`us_events.yaml` follows the same S3/manifest pattern as all other data sources. The manifest materializer downloads it; no special-case git resolution. Provenance artifacts (event_candidates, event_validations, event_quarantine) are audit-only manifest entries.

### D3: All production runners use the manifest input resolver

`run_v2_calibration.py` now uses `register_manifest_input_args` + `apply_manifest_input_paths` + `apply_manifest_input_defaults` — the same pattern as `profile_engine.py`. Adding a new data source to `MANIFEST_INPUT_SPECS` auto-wires into every runner.

### D4: CPI October 2025 is a permanent BLS gap

The US government shutdown prevented BLS from collecting CPI data for October 2025. The engine handles this via forward-fill (Sep value carries through October) and the 60-day staleness gate. This gap cannot be resolved by re-fetching.

### D5: Hysteresis does not apply to evidence/score outputs

Event_calendar (precedence-based), transition_risk (final state from named
rules plus continuous score), cluster (raw GMM assignment), change_point
(posterior probability), and hmm (state probabilities) are NOT label axes and
do NOT use hysteresis. Hysteresis applies only to the 9 axes that produce a
raw→stable→active label triple.

## Axes with per-label hysteresis (all 9)

| Axis | Labels | Stickiest | Fastest |
|------|--------|-----------|---------|
| trend_direction | bull, bear, sideways, transition, recovery, euphoria | bear=5d | bull=0d |
| trend_character | breakout_expansion, trending, range_bound, chop, transition, recovery_attempt | breakout_expansion=3d | trending=0d, chop=0d |
| volatility | crisis_vol, vol_crush, high_vol, rising_vol, low_vol, normal_vol | crisis_vol=5d | normal_vol=0d, low_vol=0d |
| breadth | divergent_fragile, weak_breadth, narrowing_breadth, broadening_breadth, healthy_breadth | divergent_fragile=5d | healthy_breadth=0d |
| network_fragility | systemic_stress, correlation_to_one, correlation_concentration, rising_fragility, ... | systemic_stress=5d | diversified_normal=0d, unknown=0d |
| volume_liquidity | panic_volume, liquidity_gap_behavior, normal_volume | panic_volume=2d | normal_volume=0d |
| monetary_pressure | rate_shock, tightening_pressure, easing_pressure, neutral_monetary | rate_shock=5d | neutral_monetary=0d |
| inflation_growth | inflation_shock, recession_scare, disinflation, goldilocks, ... | inflation_shock=5d | goldilocks=0d, unknown=0d |
| credit_funding | deleveraging, funding_squeeze, credit_stress, spread_widening, credit_calm | deleveraging=5d | credit_calm=0d, unknown=0d |

`unknown` is absence of signal rather than a market regime. It must not hold
back recovery into a valid classified label. Transient drops from high-risk
classified labels remain debounced by the high-risk label's own threshold
because the per-label algorithm keys de-escalation on the stable label being
left.

## Outputs WITHOUT hysteresis (by design)

| Output | Why |
|--------|-----|
| event_calendar | Precedence-based: highest-priority event wins per session |
| transition_risk | Final state selected from named warnings plus continuous score band; no raw→stable→active hysteresis |
| cluster | Raw GMM cluster_id — evidence, not a regime label |
| change_point | BOCPD posterior probability — evidence, not a regime label |
| hmm | HMM state probabilities — evidence, not a regime label |
