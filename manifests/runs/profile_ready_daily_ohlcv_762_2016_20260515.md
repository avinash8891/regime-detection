# profile_ready_daily_ohlcv_762_2016_20260515

The materializable YAML lockfile for this artifact set lives next to this
document at
[`profile_ready_daily_ohlcv_762_2016_20260515.yaml`](./profile_ready_daily_ohlcv_762_2016_20260515.yaml).
It was regenerated from S3 metadata after the original 15,367-line lockfile
was removed from git.

> **Note:** As of 2026-05-18 this artifact set is also bundled into the
> merged engine manifest at
> [`regime_engine_2026-05-17.yaml`](./regime_engine_2026-05-17.yaml) so a
> single `--manifest manifests/runs/regime_engine_2026-05-17.yaml` pulls
> both the 11 canonical engine artifacts and these 1086 OHLCV per-symbol
> parquets in one shot. This standalone manifest remains in the tree for
> OHLCV-only callers that do not need the engine's macro / sentiment /
> events bundle.

## Summary

- artifact set: `profile_ready_daily_ohlcv_762_2016_20260515`
- created at UTC: see `created_at_utc` field in the YAML
- storage root: `s3://autoresearch-platform/regime-detection/artifacts/zurich-v1/profile-ready-daily-ohlcv-762-2016-20260515`
- local path prefix: `data/raw/daily_ohlcv_762/`
- artifact count: **1086** symbol parquets (the historical "762" label in the
  artifact-set name is stale; the live bucket prefix now holds 1086 symbols)
- date range: `2016-01-04` to `2026-05-15`
- required for: *(none)* — OHLCV-only manifest. The engine runners
  (`profile_engine`, `v2_calibration`, `historical_walkforward`,
  `audit_layer2_30d`) require macro, PIT, event-calendar, sentiment, CPI,
  FOMC/Powell, and EPS artifacts that this lockfile intentionally does not
  enumerate; point those runners at
  [`regime_engine_2026-05-17.yaml`](./regime_engine_2026-05-17.yaml) instead.
  This standalone manifest is retained only for OHLCV-only consumers
  (universe enumeration, ad-hoc per-symbol re-materialization).

## sha256 provenance

All 1086 entries in the YAML carry a real sha256 digest sourced from the
canonical S3 object's `Metadata.sha256` (the upload path stamps this on every
put). The bulk-rehash pass is automated by
`scripts/backfill_ohlcv_manifest_sha256.py`, which HEADs each object and
rewrites the lockfile in place — no full downloads required. Run it whenever
the canonical S3 prefix is repopulated (e.g. a re-ingest after a corporate
action) to keep the lockfile honest. `materialize_manifest` therefore
verifies every per-symbol parquet against its digest at fetch time.

## Provenance

- Original (deleted) manifest sha256:
  `1e46ede94cc0dbd06faf500330839489e6b5ebb09c63bd99dfe43cc17eb90808`
- The regenerated YAML's sha256 will not match the historical value because
  the regeneration draws from current S3 metadata (1086 symbols, fresh sha
  sentinels, fresh `created_at_utc`) rather than reproducing the original
  byte stream.
