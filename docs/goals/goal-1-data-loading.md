# Codex Goal: Profile Engine Data Loading Optimization

## Goal prompt (paste into `/goal`)

```
Reduce profile_engine data loading time from 10 minutes to under 60 seconds wall clock, verified by adding time.perf_counter instrumentation to each stage of _load_profile_inputs in scripts/profile_engine.py and running on a 16-core EC2 c7i.4xlarge with data pre-materialized from S3.

Current profiled breakdown (c7i.4xlarge, 16 vCPU, 32GB):
- Total script: 19 min wall clock
- classify_window: 551s (9.2 min, already optimized separately)
- Data loading (_load_profile_inputs): ~10 min — THIS IS THE TARGET
- 733 constituent parquets loaded from data/raw/daily_ohlcv_762/
- 16 macro series reindexed via _reindex_macro_to_sessions (union + sort per series)
- build_market_context called twice (once in _load_profile_inputs, once inside classify_window)

Preserve: byte-identical profile JSON output, all 1312 tests green, --lookback-days semantics, PIT constituent filtering, manifest SHA verification.

Use only: scripts/profile_engine.py, scripts/_v2_calibration_helpers.py, src/regime_detection/market_context.py. Do not change feature computation or rule evaluation.

Between iterations: instrument the slowest unmetered stage, measure, fix, re-measure. Prioritize eliminating duplicate work (build_market_context called twice), then I/O parallelism (concurrent.futures for parquet reads), then reducing copies (reindex in-place vs creating new Series).

Done when: each sub-stage of _load_profile_inputs is instrumented and total loading < 60s on c7i.4xlarge. Stop if bottleneck is EBS I/O throughput — report measured bandwidth and recommend provisioned IOPS.
```

## Context for Codex
- Repo: regime-detection (manila-v2 workspace)
- Branch: avinash8891/regime-detection-audit
- Key files: scripts/profile_engine.py (line 305-380), src/regime_detection/market_context.py
- Test command: `python -m pytest tests/ --ignore=tests/test_pit_constituents.py -q`
- Profile command: `python scripts/profile_engine.py --manifest manifests/runs/regime_engine_2026-05-17.yaml --data-root data/raw --lookback-days 2705 --run-timeout-seconds 0 --json-output data/raw/profile_test.json`
