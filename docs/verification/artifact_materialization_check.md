# Artifact Materialization Check

This verification note documents the expected operator path for portable regime
engine data.

1. Fetch or import source data into local `data/raw/`.
2. Record acquisition metadata in SQLite with `--acquisition-db`.
3. Upload raw captures and outputs to the artifact store with `--artifact-store`
   and `--acquisition-db`. S3 roots require the optional `s3` extra.
4. Emit a manifest with `--emit-manifest`.
5. On another environment, run `scripts/materialize_regime_data.py` with that
   manifest before running profile, calibration, or V2 gate scripts.

Report `paths` may be either a string path or an object with `path` and
`local_path`. Use the object form when a source artifact lives outside its
eventual runner location, such as restoring archived `daily_ohlcv_762` files to
`data/raw/daily_ohlcv_762/` or manual PMI TSVs to `data/manual_inputs/pmi/`.

Focused verification commands:

```bash
python3 -m pytest \
  tests/test_artifact_store.py \
  tests/test_artifact_manifest.py \
  tests/test_materialize_regime_data.py \
  tests/test_artifact_export.py \
  tests/test_runner_manifest_materialization.py \
  -q
```

Expected result: all tests pass, and corrupt or missing artifacts fail before a
regime runner reads partially materialized inputs.
