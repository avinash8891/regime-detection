# Decision 0007: V2 §2C credit-spread — TLT-vs-HYG/LQD proxy as a parallel metric

**Status:** accepted — the real ICE BofA OAS metric is kept at its FRED-available depth (2023-05-15+), and the TLT-vs-HYG/LQD total-return-differential proxy is reintroduced as a **separate parallel metric** producing its own `credit_funding_state_proxy` label. Pinned in `docs/regime_engine_v2_spec.md` Ambiguity Log #71; this ADR is the standalone decision record.

## Context

V2 §2C credit/funding sources its HY/IG credit-spread signal from the real ICE BofA Option-Adjusted Spread series on FRED — `BAMLH0A0HYM2` (HY) and `BAMLC0A4CBBB` (BBB IG). Ambiguity Log #49 closed §2C onto those, and commit `9cad7e7` deleted a prior TLT-vs-HYG/LQD total-return-differential proxy *fallback*, on the reasoning that the fallback was unreachable: "any operator able to build the §2C seam at all already has the FRED key that fetches the OAS series."

A 2026-05 macro re-fetch invalidated that reasoning. FRED now exposes only a **trailing ~3-year window** of these ICE BofA OAS series — both `BAMLH0A0HYM2` and `BAMLC0A4CBBB` start **2023-05-15** (confirmed against FRED's `/series` metadata: `observation_start = 2023-05-15`; ICE Data Indices tightened redistribution licensing — the series IDs are unchanged but the public history is truncated). The previously-"impossible" state is now real: the FRED key is present and the OAS fetch *succeeds*, but the series is empty before 2023-05-15. §2C therefore has no real-OAS signal for ~70% of the available backtest history (~2016–2023).

## Decision

Three pins (Ambiguity Log #71):

1. **Accept the 2023+ depth for the real-OAS metric.** No splicing, no backfill. Where OAS has no data the §2C real-OAS label (`credit_funding_state`) is NaN/`unknown` — V1 §2.7 cold-start behavior, "use the feed when it is available."

2. **Reintroduce the TLT-vs-HYG/LQD proxy as a SEPARATE, parallel metric** that produces its own §2C label (`RegimeOutput.credit_funding_state_proxy`), covering the longer history. The §2C rule schema is scale-invariant (percentile + slope predicates), so the *same* `CreditFundingSeriesClassifier` logic runs a second time on the proxy series — one rule schema, two input series, two outputs. The proxy output always carries the `credit_spread_proxy_total_return_differential` bias-warning row.

3. **Rename the misleadingly-named legacy fields.** `hy_spread_proxy_*` / `ig_spread_proxy_*` held the *real* OAS values (since #49) but were named "proxy" — backwards. Renamed to `hy_oas_*` / `ig_oas_*`. The new proxy metric's fields are `hy_tr_differential_*` / `ig_tr_differential_*`. The `CreditFundingRuleInputs` spread fields became source-neutral (`hy_spread_*`) so one rule-input builder serves both runs.

## Why this is NOT the dual-sourcing commit `9cad7e7` removed

`9cad7e7` removed **dual-sourcing**: one column fed by *either* the real OAS *or* the proxy depending on availability — mixing two genuinely-different measurements into one series. This decision does the opposite: **two distinct metrics, two distinct label outputs, never blended.** A consumer always knows which it is reading; the proxy carries a permanent bias-warning row, and the real-OAS and proxy labels surface on separate `RegimeOutput` fields.

## Consequences

- §2C real-OAS backtest depth is capped at ~2023-05. The proxy covers ~2018→current (the `_percentile_504d` 504-session warm-up from the 2016-01-04 data start), so the ~2018→2023 window — otherwise fully dark for §2C — now has a credit read.
- The two metrics measure a *similar* thing (credit-spread direction); the proxy exists because FRED's OAS series lack pre-2023 history. They are parallel and independent — never spliced.
- V1 byte-identity preserved: `RegimeOutput.credit_funding_state_proxy` defaults `None`, omitted from the wire via `exclude_none=True`.

## Decision

Accepted as above. The spec amendment (Ambiguity Log #71 + §2C Features/Rules text) and the code-wiring slice (`credit_funding.py` rename + proxy compute, parallel classifier in `axis_series.py`, `RegimeOutput.credit_funding_state_proxy`, timeline wiring, tests) land in the same cycle.
