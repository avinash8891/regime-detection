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

## Data Artifacts

Production data is not stored in Git. `data/raw/` is a local materialized cache
rebuilt from a manifest that points at durable artifacts in object storage.
When `--acquisition-db` and `--artifact-store` are both set, acquisition records
persist raw captures and derived outputs to the same artifact store.
Use `pip install ".[s3]"` before passing an `s3://...` artifact store.

```bash
python3 scripts/fetch_regime_engine_v1_data.py \
  --fetch sentiment \
  --out-dir data/raw \
  --acquisition-db data/raw/acquisition/acquisition.db \
  --artifact-store /path/to/regime-data-store \
  --emit-manifest data/manifests/regime_engine_latest.yaml

python3 scripts/materialize_regime_data.py \
  --manifest data/manifests/regime_engine_latest.yaml \
  --local-root data/raw
```

See `docs/market_data_fetch_plan.md` section 0 for the storage contract.
