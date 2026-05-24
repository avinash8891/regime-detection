# V2 Walk-forward Performance Gate (§9.1)

- Window: 2025-02-07 → 2026-05-08 (314 NYSE sessions)
- Engine versions: v1=regime-engine-v2.0.0, v2=regime-engine-v2.0.0
- v1-mode errors (sessions): 0
- v2-mode errors (sessions): 0
- Generated: 2026-05-13T13:34:02.585211+00:00

## Reproducibility

- Generator script: `scripts/run_v2_walkforward_gate.py`
- Generator commit: `22cd943`
- Regenerate:
  ```
  python3 scripts/run_v2_walkforward_gate.py \
      --start-date 2025-02-07 --end-date 2026-05-08 \
      --out docs/verification/v2_walkforward_perf_gate.md
  ```
- Input data SHA-256 (full):
  - `data/raw/pit_constituents/sp500_ticker_intervals.parquet`
    → `a56e14fffc9a690b9335e21f9d5ec0a986871ee74f6adb2faf1b209e67c6a494`
  - `data/raw/macro/fred_macro_series.parquet`
    → `3004cc6b9e7513095670dd0edd7e34445d7ecdba6a95ad6753cd7a54b80e674f`
  - `data/raw/daily_ohlcv_762/` aggregate
    → `06a0f82ffeed48db952886ad63d0c951a1e58114e07d4d949f3688094a014115`
  - Per-symbol manifest: `data/raw/daily_ohlcv_762/MANIFEST.sha256.json`
    (gitignored along with `data/raw/`).

## Wire-level metrics

| metric | v1 | v2 | delta |
|---|---|---|---|
| sessions classified | 314 | 314 | 0 |
| sessions with transition_risk.state = crisis | 37 | 37 | 0 |
| sessions with bear_stress fired | 0 | 0 | 0 |
| sessions with bull_fragile fired | 0 | 0 | 0 |
| sessions with recovery_attempt | 0 | 0 | 0 |
| sessions with score_components dict | 0 | 0 | 0 |
| sessions with agent_routing field | 314 | 314 | 0 |
| sessions with change_point.score | 314 | 314 | 0 |
| sessions with hmm evidence on score | 0 | 0 | 0 |
| sessions with credit_funding_state | 0 | 314 | 314 |
| sessions with inflation_growth_state | 0 | 314 | 314 |
| sessions with cluster output | 0 | 0 | 0 |

## §9.1 Gate Conditions (v2 §9.1 + docs/v2_slice_gate_checklist.md item 6)

Per the spec, AT LEAST ONE of:

- LOWER_DRAWDOWN
- HIGHER_SHARPE
- EARLIER_CRISIS_DETECTION
- LOWER_FALSE_SWITCH_RATE

must show v2 improvement. Note: this script ships the per-session wire
comparison; the strategy-PnL metrics (drawdown/sharpe/false-switch) are
operator concerns when v2 outputs route into a backtester (e.g.
vectorbt). The gate currently asserts the wire-level lit-vs-unlit
deltas as a precondition to the strategy gate.

## Per-axis activation rate (v2 mode only)

| axis | sessions lit | activation rate |
|---|---|---|
| network_fragility (non-unknown) | 225 | 71.7% |
| credit_funding (non-unknown) | 235 | 74.8% |
| inflation_growth (non-unknown) | 4 | 1.3% |
| monetary_pressure_v2 (non-unknown) | 0 | 0.0% |
| volume_liquidity_state (non-unknown) | 314 | 100.0% |
| agent_routing != default | 314 | 100.0% |
| change_point >= 0.5 | 73 | 23.2% |

Note: the change-point activation row was regenerated after the BOCPD
posterior mapping fix. Current runtime emits recent short-run posterior mass
over the 21-session realized-volatility horizon; see
`docs/verification/change_point_bocpd_recalculation_2026_05_23.md`.
