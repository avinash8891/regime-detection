# Complete Business Contracts Design

## Goal

Resolve the two remaining business-logic gaps completely:

- calibration / threshold provenance must be mechanical at scalar-field level;
- `ClassifyRequest` must enforce configured V2 input-family requirements before classification logic can silently degrade.

## Design

`src/regime_detection/rule_provenance.py` remains the owner of rule provenance, but the registry expands from section rows into scalar rows. The registry declares business-logic config roots and mechanically expands them against the default `RegimeConfig`, producing one `RuleProvenance` row per scalar threshold, weight, window, staleness gate, hysteresis value, and strategy/routing knob. Static non-config contracts such as rule precedence and risk-rank tables stay as explicit rows. Tests compare the scalar paths emitted by the registry against an independent config traversal so newly added tunables fail until provenance is declared.

`src/regime_detection/engine.py` owns request-boundary validation. A first-class V2 input contract matrix declares which configured sections require which source families. `RegimeEngine.classify_request()` validates this matrix after canonical market context construction and before feature/timeline construction. Required source families fail loudly; evidence-only seams stay optional but declared so absence is intentional rather than implicit.

## Testing

Tests are behavior-first:

- provenance tests require unique keys, traceability, and complete scalar coverage;
- request tests exercise missing sector, volume, macro, cross-asset, HMM, clustering, and optional-evidence contracts;
- existing V1 wrapper behavior remains covered by the foundation tests.

