# regime-detection

Regime detection engine (V1 + V2 unified). Classifies market regimes across
multiple axes — trend direction, trend character, volatility state, breadth,
credit/funding, inflation/growth, monetary pressure, network fragility, and
volume/liquidity — using rule-based classifiers backed by market data.

## Architecture

- **V1** (frozen): `core3-v1.0.0` config produces byte-identical output for
  archive replay. See `docs/regime_engine_v1_final_spec.md`.
- **V2** (in progress): extends V1 with new axes and a transition-risk score.
  See `docs/regime_engine_v2_spec.md`. V2 slices ship behind config-version
  guards; V1 byte-identity is preserved when V2 config is absent.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

## Key docs

- V1 spec: `docs/regime_engine_v1_final_spec.md`
- V2 spec: `docs/regime_engine_v2_spec.md`
- Data requirements: `docs/regime_engine_v1_data_requirements.md`
- Shadow runner: `docs/shadow_runner_spec.md`
- Historical walk-forward: `docs/historical_walkforward_spec.md`
- Agent operating rules: `AGENTS.md`
