# Decision 0001: VPS Runner

**Status:** accepted

## Decision

Do not copy `agents-auto-research/vps_runner.py` into this repo.

V1 does not need a VPS deployment runner during implementation.

## Why

- The autoresearch runner is strategy-backtest specific.
- It imports autoresearch strategy registries and trace helpers.
- It syncs `backtest/`, `configs/`, and `strategies/`.
- It runs `python3 backtest/runner.py --strategy ... --config ...`.
- This repo is a regime-engine package, not a strategy backtest runner.

Copying that script would create dead or misleading deployment code.

## V1 Implementation Rule

Do not add `vps_runner.py` in V1 implementation slices.

Focus V1 on:

- package layout;
- fixture verification;
- local tests;
- deterministic `RegimeEngine.classify(...)`;
- golden-date regression.

## Future Runner

Add a regime-specific runner only after V1 can run locally end-to-end and daily shadow mode is needed.

That future runner must:

- install or sync the `regime_detection` package;
- load daily market data;
- run `RegimeEngine.classify(...)` or `classify_window(...)`;
- write versioned JSON outputs;
- preserve `engine_version`, `config_version`, and `as_of_date`;
- avoid strategy-backtest imports and autoresearch-specific environment variables.

Candidate names:

- `scripts/run_daily_shadow.py`
- `scripts/deploy_shadow_runner.py`

## Agent Instruction

If asked to add VPS deployment before V1 implementation is complete, do not copy the autoresearch runner. Add or update this decision record instead, unless the user explicitly approves a regime-specific operational runner.
