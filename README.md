# regime-detection

V1 regime detection engine. See:

- `docs/regime_engine_v1_final_spec.md`
- `docs/regime_engine_v1_data_requirements.md`

## Historical Walk-Forward Runner

Run the historical walk-forward script directly; it uses the repo shebang for
`python3`:

```bash
./scripts/run_historical_walkforward.py \
  --market-data tests/fixtures/raw/market_data.parquet \
  --event-calendar configs/events/us_events.yaml \
  --start-date 2026-03-24 \
  --end-date 2026-05-05 \
  --output-root .context/regime_30d_walkforward_2026-03-24_2026-05-05
```

The runner tests are marked `slow`, so execute them explicitly:

```bash
python3 -m pytest tests/test_historical_walkforward.py -m slow -q
```
