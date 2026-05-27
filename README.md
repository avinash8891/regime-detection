# regime-detection

Unified V1+V2 regime detection engine checkout.

Primary docs:

- `docs/regime_engine_v1_final_spec.md`
- `docs/regime_engine_v1_data_requirements.md`
- `docs/regime_engine_v2_spec.md`
- `docs/market_data_fetch_plan.md`
- `docs/decisions/0019-valid-data-rule-partitions.md`

## Main Profile Runner

`scripts/profile_engine.py` is the main operator runner for regime detection.
It materializes the approved manifest when `--manifest` is supplied, resolves
all runtime inputs from that manifest, and runs the engine over the requested
lookback window.

```bash
python3 scripts/profile_engine.py \
  --manifest manifests/runs/regime_engine_2026-05-17.yaml \
  --data-root data/raw \
  --lookback-days 2607 \
  --run-timeout-seconds 0 \
  --json-output .context/profile_engine_2016_to_latest.json
```

For the approved `regime_engine_2026-05-17.yaml` manifest, `2607` NYSE
sessions is the exact emitted/profiled window from 2016-01-04 through the
manifest's latest OHLCV date, 2026-05-15. This does **not** drop warmup data:
the runner loads the manifest's earlier SPY/constituent history back to
2009-12-31 and computes `working_window_start` from the configured V2 training
windows before slicing the emitted `lookback-days` window. Use a larger
lookback only when you intentionally want pre-2016 output rows included in the
profile report.

For the standard 30-session operator profile, use:

```bash
make profile-30d MANIFEST=manifests/runs/regime_engine_2026-05-17.yaml
```

The historical walk-forward runner is a separate qualification-gate tool, not
the default regime-detection runner. Its tests are marked `slow`, so execute
them explicitly when changing that gate:

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

python3 scripts/profile_engine.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --data-root data/raw

python3 scripts/run_v2_walkforward_gate.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --data-root data/raw
```

For a fresh workspace, pass the approved manifest lockfile explicitly:

```bash
make regime-data MANIFEST=manifests/runs/regime_engine_YYYY-MM-DD.yaml
make profile-30d MANIFEST=manifests/runs/regime_engine_YYYY-MM-DD.yaml
```

When `--manifest` is supplied, profile, audit, and V2 gate runner data inputs
are resolved from manifest artifact names after materialization. Per-file flags
such as `--macro-parquet`, `--pmi-path`, `--event-calendar`, or
`--news-sentiment-parquet` are manual debug overrides; they are not needed for
a fresh workspace using the approved manifest lockfile.

That approved profile-ready manifest is the portable data contract for the 30d
operator run. The committed lockfile at
`manifests/runs/regime_engine_2026-05-17.yaml` is the **only valid
`--manifest` argument for `profile_engine.py`, `run_v2_walkforward_gate.py`,
and the `audit_layer2_30d` / `historical_walkforward` runners**. The same
merged manifest is also the source for universe enumeration and ad-hoc
per-symbol OHLCV re-materialization; do not use a separate OHLCV
manifest. It enumerates every input the runners read — the fixed 762-symbol
OHLCV tree root, macro, PMI, PIT constituents, CPI vintages, FOMC/Powell text
inputs, the US event calendar, and the Layer 1 sentiment extensions
(`aaii_sentiment`,
`sf_fed_news_sentiment`). A relative `storage_root` in that manifest is
anchored to the manifest file's parent directory by
`regime_data_fetch.materialization._resolve_store_root`, so the same lockfile
materializes correctly across checkouts (vaduz, nicosia, CI) without rewriting
absolute paths.

Operator caveats for the 2026-05-17 cut:

- Per-symbol OHLCV artifacts are fully enumerated in the merged manifest
  (`manifests/runs/regime_engine_2026-05-17.yaml`): 1193 `daily_ohlcv_762_<TICKER>`
  entries, each carrying a full SHA-256 digest. No placeholder remains; the
  lockfile was regenerated and merged as of the 2026-05-17 cut.
- The Cleveland Fed CPI nowcast (`cleveland_fed_cpi_nowcast`, see
  `docs/decisions/0006-inflation-surprise-cleveland-fed-nowcast-substitute.md`)
  is fetched, canonical, and wired into `profile_engine` as of 2026-05-18
  (154 monthly vintages, latest 2026-05-14). The single-signal
  `inflation_shock` limb reads real `inflation_surprise_zscore` values
  from this artifact.
- The weekly aggregate forward-EPS history (`sp500_eps_weekly_history`)
  has been seeded from the Wayback Machine (12 weekly rows, latest
  2026-01-22) but is **not wired into `profile_engine`** because the
  vintage is more than 90 days old. The `earnings_expansion` /
  `earnings_contraction` labels stay dark until an operator refreshes
  the artifact via `eps-spglobal-auto` with a Playwright browser session.
  Manifest gates EPS to `required_for: [audit_layer2_30d]` only —
  this is intentional, not a bug.
- `aaii_sentiment`, `sf_fed_news_sentiment`, `fomc_minutes`,
  `powell_speeches`, and `event_calendar_us` are now fully promoted to the
  canonical artifact store on the 2026-05-17 vintage and resolve through the
  manifest router on a fresh workspace; no operator fetch step is required
  before materializing.
- All optional manifest-routed inputs are declared in a single registry
  (`MANIFEST_INPUT_SPECS` in `src/regime_data_fetch/manifest_inputs.py`).
  Each entry pins the canonical artifact name, the runner CLI flag, and the
  default relpath under `data_root`; the runner-side helpers
  `register_manifest_input_args(parser)` and
  `apply_manifest_input_defaults(args, data_root)` derive everything from
  that registry, so adding a new manifest input is a single-entry edit and
  cannot drift away from any runner.

The approved runtime manifest currently points OHLCV artifacts at
`s3://autoresearch-platform/regime-detection/artifacts/zurich-v1/profile-ready-daily-ohlcv-762-2016-20260515/canonical/daily_ohlcv_762/`.
It pins 1193 canonical symbol files through SHA-256 metadata and intentionally
keeps raw data out of Git.

Daily `--fetch all` includes the full routine layer event-calendar surface by
default: FOMC, CPI, NFP, election, budget, ECB, BOE, and BOJ rows.
`geopolitical_event` remains approval-gated and is rendered only from the
operator approval overlay.

See `docs/market_data_fetch_plan.md` section 0 for the storage contract.
