# V2 Calibration Summary

- Fitted at: 2026-05-13T12:18:12.567659+00:00
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
