# V2 Calibration Summary

- Fitted at: 2026-05-13T12:21:59.388460+00:00
- Data end date: 2026-05-08
- SPY sessions: 2099
- Sector ETFs: ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
- Cross-asset: ['DBC', 'EEM', 'EFA', 'GLD', 'HYG', 'IWM', 'KRE', 'LQD', 'QQQ', 'TLT', 'USO', 'UUP', 'XLI', 'XLP', 'XLU', 'XLY']
- Macro series: ['10y_yield', '2y_yield', 'CPIAUCSL', 'DGS10', 'DGS2', 'DTWEXBGS', 'GDPNOW', 'IORB', 'NFCI', 'SOFR', 'broad_usd_index', 'cpi_all_items', 'dgs10', 'dgs2', 'gdp_nowcast', 'iorb', 'nfci', 'pmi_manufacturing', 'sofr']

## Feature-store seams

- `feature_store.network_fragility` lit: **True**
- `feature_store.volatility_state_v2` lit: **True**
- `feature_store.breadth_state_v2` lit: **True**
- `feature_store.volume_liquidity_v2` lit: **True**
- `feature_store.monetary` lit: **True**
- `feature_store.hmm` lit: **True**
- `feature_store.clustering` lit: **True**
- `feature_store.change_point` lit: **True**
- `feature_store.credit_funding` lit: **True**
- `feature_store.inflation_growth` lit: **True**

## Candidate artifacts (require operator review per V2 §10)

- docs/verification/hmm_state_label_map.candidate.yaml
- docs/verification/cluster_label_map.candidate.yaml

## Reproducibility

- Generator script: `scripts/run_v2_calibration.py`
- Generator commit: `22cd943`
- Regenerate:
  ```
  python3 scripts/run_v2_calibration.py \
      --training-end-date 2026-05-08 \
      --training-window-days 1260 \
      --random-state 42 \
      --emit-summary docs/verification/v2_calibration_summary.md \
      --emit-hmm-candidate docs/verification/hmm_state_label_map.candidate.yaml \
      --emit-cluster-candidate docs/verification/cluster_label_map.candidate.yaml
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
