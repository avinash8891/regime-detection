# Artifact Storage and Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make regime-engine data portable and reproducible by storing artifacts in S3-compatible object storage, tracking metadata in SQLite, and materializing local `data/raw/` from manifest files.

**Architecture:** `data/raw/` remains a gitignored local cache. Fetchers write raw, normalized, and canonical artifacts to an object-store layout, record provenance/checkpoints in SQLite, and emit manifest files that pin exact artifacts for a regime run. Runners validate and materialize the manifest before classification.

**Tech Stack:** Python stdlib, SQLite, pandas/pyarrow, existing fetcher modules, optional AWS CLI or boto3-compatible client behind a narrow adapter.

---

## File Structure

- Create: `src/regime_data_fetch/artifact_store.py` — object-store abstraction, local filesystem fallback, hash verification.
- Create: `src/regime_data_fetch/artifact_manifest.py` — manifest models, YAML/JSON serialization, validation.
- Modify: `src/regime_data_fetch/acquisition_store.py` — add durable artifact registry, checkpoints, lineage, and canonical version rows.
- Modify: `scripts/fetch_regime_engine_v1_data.py` — add object-store and manifest flags, write artifact metadata after successful fetches.
- Create: `scripts/materialize_regime_data.py` — download/copy manifest-pinned artifacts into local `data/raw/`.
- Modify: `scripts/profile_engine_30d.py` and V2 gate scripts — accept `--manifest` or `--data-root` and verify required artifacts before running.
- Create: `tests/test_artifact_store.py` — file-backed object-store behavior, hash verification, no silent overwrite.
- Create: `tests/test_artifact_manifest.py` — manifest validation and required field coverage.
- Create: `tests/test_materialize_regime_data.py` — end-to-end local object-store to `data/raw` materialization.
- Modify: fetch workflow tests for one representative source first, then broaden source-by-source.

## Task 1: Artifact Store Abstraction

**Files:**
- Create: `src/regime_data_fetch/artifact_store.py`
- Test: `tests/test_artifact_store.py`

- [ ] **Step 1: Write failing tests for put/get/hash behavior**

Test a local filesystem-backed store first so the behavior is deterministic without cloud credentials. Assert that `put_file` records size and SHA-256, `get_file` verifies the expected hash, and an existing key is not overwritten unless the bytes are identical.

Run:

```bash
python3 -m pytest tests/test_artifact_store.py -q
```

Expected: fail because `regime_data_fetch.artifact_store` does not exist.

- [ ] **Step 2: Implement minimal local object-store adapter**

Add `ArtifactStore`, `StoredArtifact`, `LocalArtifactStore`, and `sha256_file`. Keep the interface URI-based even when backed by local files so S3/R2 support can be added without changing fetchers.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m pytest tests/test_artifact_store.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/regime_data_fetch/artifact_store.py tests/test_artifact_store.py
git commit -m "feat: add artifact store abstraction"
```

## Task 2: SQLite Artifact Ledger

**Files:**
- Modify: `src/regime_data_fetch/acquisition_store.py`
- Test: existing acquisition-store tests plus a new focused test if needed

- [ ] **Step 1: Write failing tests for ledger rows**

Assert that the acquisition store can record:

- fetch run status
- source checkpoint
- raw artifact URI/hash
- canonical artifact URI/hash
- lineage from canonical artifact back to raw artifacts

Run:

```bash
python3 -m pytest tests/test_fetch_workflow.py tests/test_event_calendar.py -q
```

Expected: fail on missing ledger APIs or columns.

- [ ] **Step 2: Add additive SQLite tables**

Add tables without removing existing tables:

```text
artifact_records
source_checkpoints
canonical_versions
artifact_lineage
```

Use UTC timestamps, content hashes, size bytes, row counts, min/max dates, schema version, local materialization path, and object-store URI.

- [ ] **Step 3: Verify migrations on empty and existing DBs**

Run:

```bash
python3 -m pytest tests/test_fetch_workflow.py tests/test_event_calendar.py -q
```

Expected: pass; existing acquisition behavior remains compatible.

- [ ] **Step 4: Commit**

```bash
git add src/regime_data_fetch/acquisition_store.py tests
git commit -m "feat: record durable artifact ledger metadata"
```

## Task 3: Manifest Contract

**Files:**
- Create: `src/regime_data_fetch/artifact_manifest.py`
- Create: `tests/test_artifact_manifest.py`
- Create: `manifests/runs/.gitkeep`

- [ ] **Step 1: Write failing tests for manifest validation**

Tests must reject missing URI, missing SHA-256, duplicate local paths, unknown stage, and absolute local paths outside the requested data root.

Run:

```bash
python3 -m pytest tests/test_artifact_manifest.py -q
```

Expected: fail because the manifest module does not exist.

- [ ] **Step 2: Implement manifest model and validation**

Use stdlib dataclasses plus the repo's existing YAML dependency if available. Keep fields explicit: `artifact_set`, `created_at_utc`, `storage_root`, `artifacts`, `name`, `stage`, `uri`, `local_path`, `sha256`, `schema_version`, `rows`, `min_date`, `max_date`, `required_for`.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m pytest tests/test_artifact_manifest.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/regime_data_fetch/artifact_manifest.py tests/test_artifact_manifest.py manifests/runs/.gitkeep
git commit -m "feat: define regime data artifact manifests"
```

## Task 4: Materialization CLI

**Files:**
- Create: `scripts/materialize_regime_data.py`
- Create: `tests/test_materialize_regime_data.py`

- [ ] **Step 1: Write failing end-to-end materialization test**

Create a temporary local object-store root with two artifacts, write a manifest pointing to them, run the CLI into a temporary `data/raw`, and assert exact bytes and hash verification. Add a corrupt-hash case that fails loudly before engine execution.

Run:

```bash
python3 -m pytest tests/test_materialize_regime_data.py -q
```

Expected: fail because the CLI does not exist.

- [ ] **Step 2: Implement the CLI**

Support:

```bash
python scripts/materialize_regime_data.py \
  --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml \
  --local-root data/raw \
  --store-root s3://regime-data
```

For the first implementation, support `file://` and plain local paths. Add S3-compatible support only behind the same `ArtifactStore` interface.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m pytest tests/test_materialize_regime_data.py tests/test_artifact_store.py tests/test_artifact_manifest.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/materialize_regime_data.py tests/test_materialize_regime_data.py
git commit -m "feat: materialize regime data from manifest"
```

## Task 5: Source Fetcher Migration

**Files:**
- Modify: `scripts/fetch_regime_engine_v1_data.py`
- Modify: one source module at a time under `src/regime_data_fetch/`
- Modify: matching tests

- [ ] **Step 1: Migrate one representative small source first**

Start with AAII sentiment because it directly unblocks the previously missing `euphoria` input and has a clear raw-to-canonical chain.

Expected artifact chain:

```text
raw_capture/aaii/<run_id>/sentiment.cfb
normalized/aaii/<run_id>/aaii_sentiment.parquet
canonical/sentiment/aaii_sentiment/as_of=<date>/aaii_sentiment.parquet
```

- [ ] **Step 2: Add tests around raw capture, canonical output, and ledger rows**

Run the existing AAII fetch tests plus ledger assertions.

Run:

```bash
python3 -m pytest tests -k "aaii or artifact" -q
```

Expected: pass.

- [ ] **Step 3: Repeat source-by-source**

Migrate FRED macro, Alpaca daily OHLCV, event-calendar candidates, PIT constituents, central-bank text, EPS workbook, and manual CSV/XLSX inputs. Each source gets its own commit and must preserve existing local output behavior until all runners consume manifests.

- [ ] **Step 4: Commit each source migration**

Example:

```bash
git add scripts/fetch_regime_engine_v1_data.py src/regime_data_fetch tests
git commit -m "feat: persist aaii artifacts to durable store"
```

## Task 6: Runner Integration

**Files:**
- Modify: `scripts/profile_engine_30d.py`
- Modify: `scripts/run_v2_calibration.py`
- Modify: `scripts/run_v2_walkforward_gate.py`
- Modify: `scripts/run_v2_shadow_ab_gate.py`
- Test: focused CLI tests or existing script tests

- [ ] **Step 1: Add failing tests for missing manifest artifacts**

Assert that a runner with `--manifest` refuses to start if required artifacts are absent, hashes fail, or a required logical input has no manifest entry.

- [ ] **Step 2: Add manifest/data-root startup validation**

Runner startup order must be:

1. Load manifest if supplied.
2. Materialize artifacts if requested.
3. Validate required paths exist.
4. Only then call engine code.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m pytest tests/test_historical_walkforward.py tests/test_shadow_runner.py tests -k "manifest or materialize" -q
```

Expected: pass with no silent skips.

- [ ] **Step 4: Commit**

```bash
git add scripts tests
git commit -m "feat: run regime scripts from artifact manifests"
```

## Task 7: Cutover Docs and Operational Checks

**Files:**
- Modify: `README.md`
- Modify: `docs/market_data_fetch_plan.md`
- Create or update: `docs/verification/artifact_materialization_check.md`

- [ ] **Step 1: Document operator workflow**

Add the three-command path:

```bash
python scripts/fetch_regime_engine_v1_data.py --fetch all --artifact-store s3://regime-data --emit-manifest
python scripts/materialize_regime_data.py --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml --local-root data/raw
python scripts/profile_engine_30d.py --manifest manifests/runs/regime_engine_YYYY-MM-DD.yaml
```

- [ ] **Step 2: Run verification**

Run:

```bash
python3 -m pytest tests/test_artifact_store.py tests/test_artifact_manifest.py tests/test_materialize_regime_data.py -q
python3 -m ruff check src/regime_data_fetch scripts tests
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: document artifact materialization workflow"
```

## Self-Review

- Spec coverage: The plan covers durable S3-compatible artifact storage, SQLite metadata ledger, manifest-pinned materialization, incremental source migration, and runner validation.
- Placeholder scan: No implementation task depends on an unspecified future API; every new file has a responsibility and verification command.
- Type consistency: The same artifact terms are used throughout: raw capture, normalized, canonical, run inputs, ledger, manifest, local cache.
