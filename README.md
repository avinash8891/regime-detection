# regime-detection

Unified V1+V2 regime detection engine checkout.

Primary docs:

- `docs/regime_engine_v1_final_spec.md`
- `docs/regime_engine_v1_data_requirements.md`
- `docs/regime_engine_v2_spec.md`
- `docs/market_data_fetch_plan.md`

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
rebuilt from a tracked manifest lockfile under `manifests/` that points at
durable artifacts in object storage. Keep raw Parquet/SQLite caches ignored;
commit only small, non-secret manifest lockfiles. When `--acquisition-db` and
`--artifact-store` are both set, acquisition records persist raw captures and
derived outputs to the same artifact store.
Use `pip install ".[s3]"` before passing an `s3://...` artifact store.

## Operator Env

Do not commit API keys. Fetch and materialization runners automatically load a
non-secret pointer file from the first available location:

1. `--operator-env-file /path/to/pointer.env`
2. `REGIME_OPERATOR_ENV_FILE=/path/to/pointer.env`
3. repo-local `.regime-operator.env` (gitignored)
4. `~/.config/regime-detection/operator.env`

The pointer file lists credential env files, not secret values. See
`.regime-operator.env.example` for the full repo-known key list, including
Alpaca, FRED, TinyFish, ACLED, UCDP, HDX operator identity, and Investing
browser/session keys. HDX HAPI does not use an auth secret in this repo path;
the runner sends an `app_identifier` query parameter derived from
`HDX_HAPI_APP_IDENTIFIER` or `HDX_HAPI_APP_NAME` plus `HDX_HAPI_APP_EMAIL`.
Use the home-level file for Conductor workspaces so new workspaces stop
rediscovering the same local key locations.

```bash
python3 scripts/fetch_regime_engine_v1_data.py \
  --fetch sentiment \
  --out-dir data/raw \
  --acquisition-db data/raw/acquisition/acquisition.db \
  --artifact-store /path/to/regime-data-store \
  --emit-manifest

python3 scripts/materialize_regime_data.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --local-root data/raw

python3 scripts/materialize_constituent_ohlcv_tree.py \
  --source-tree data/raw/daily_ohlcv \
  --out-tree data/raw/daily_ohlcv_762 \
  --pit-parquet data/raw/pit_constituents/sp500_ticker_intervals.parquet \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD

python3 scripts/profile_engine_30d.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --data-root data/raw

python3 scripts/run_v2_walkforward_gate.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --data-root data/raw
```

For a fresh workspace, pass the approved manifest lockfile explicitly:

```bash
make regime-data MANIFEST=manifests/runs/profile_ready_daily_ohlcv_762_2016_20260515.yaml
make profile-30d MANIFEST=manifests/runs/profile_ready_daily_ohlcv_762_2016_20260515.yaml
```

The approved constituent OHLCV lockfile currently points at
`s3://autoresearch-platform/regime-detection/artifacts/zurich-v1/profile-ready-daily-ohlcv-762-2016-20260515/canonical/daily_ohlcv_762/`.
It pins 1085 canonical symbol files through SHA-256 metadata and intentionally
keeps raw data out of Git.

See `docs/market_data_fetch_plan.md` section 0 for the storage contract.
