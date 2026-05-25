# ADR 0018 - Verified Regime Issue Closeout

**Status:** Accepted
**Date:** 2026-05-25
**Context:** A verified issue pass found several classifier-contract gaps where
the code either conflated evidence states, hid rule provenance, or allowed
tests to encode stale assumptions. This ADR records the closeout boundary and
the verification gates for the May 2026 long-term fixes.

## Decision

The fixes are accepted as semantic contract changes, not temporary guards:

- Breadth data gaps are reported through `DataQuality` while alternate
  point-in-time evidence may still classify the session.
- Strategy-family constraints resolve through one effective inheritance path,
  with explicit collision checks instead of local fallback rules.
- Cold-start severe labels require explicit fallback predicates and evidence
  reasons rather than generic `unknown` behavior.
- Network fragility can surface credit-unavailable systemic stress through an
  explicit label, preserving rule precedence and avoiding false
  `correlation_to_one` output.
- Inflation/growth disinflation authority is reconciled in the rules rather
  than split across ad hoc predicate assumptions.
- HMM and cluster evidence-label mappings are explicit operator-reviewed
  artifacts; unmapped state ids fail before classification.
- Shared temporal parsing returns normalized NYSE session indexes accepted by
  `calendar.as_date`, while non-midnight naive datetimes still fail loudly.
- Central-bank text remains evidence-only and does not masquerade as a direct
  monetary-pressure rule input.
- Volume/liquidity exposes rule provenance at the top evidence level while
  keeping live feature values under `rule_evidence`.

## Verification

Commands run on 2026-05-25:

```text
python3.14 -m pytest -o addopts='' tests/test_v1_frozen_replay.py -q; echo "EXIT:$?"
1 passed in 0.32s
EXIT:0

python3.14 -m pytest -o addopts='' tests/test_schema_and_timeline.py tests/test_v2_config.py -q; echo "EXIT:$?"
53 passed in 87.50s (0:01:27)
EXIT:0

python3.14 -m pytest -o addopts='' tests/test_temporal_normalization.py -q; echo "EXIT:$?"
6 passed in 0.56s
EXIT:0

python3.14 -m ruff check src/regime_detection/temporal.py src/regime_detection/calendar.py src/regime_detection/axis_builders/volume_liquidity.py tests/test_temporal_normalization.py tests/test_network_fragility_rules_precedence_series.py tests/test_volume_liquidity_classifier.py tests/test_axis_builders.py tests/test_reconciliation_contracts.py; echo "EXIT:$?"
All checks passed!
EXIT:0

git diff --check; echo "EXIT:$?"
EXIT:0

python3.14 -m pytest; echo "EXIT:$?"
1524 passed, 1 skipped, 2 warnings in 876.72s (0:14:36)
EXIT:0

rtk pytest; echo "EXIT:$?"
Pytest: No tests collected
EXIT:0
```

RTK is configured in failures-only mode for this repository; exit `0` with
`Pytest: No tests collected` means no pytest failures were found.

## Consequences

- The closeout preserves V1 frozen replay compatibility.
- New or changed tests assert exact behavior at the classifier contract
  boundaries instead of accepting broad "not None" output.
- Future changes that alter these behaviors should update the corresponding
  rule tests and this decision trail rather than adding local exception
  branches.
