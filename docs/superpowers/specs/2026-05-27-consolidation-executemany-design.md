# Consolidation `executemany` Optimization — Design Spec

**Date:** 2026-05-27
**Status:** Draft
**Scope:** `src/regime_data_fetch/acquisition_consolidation.py::_import_one_source`

## Problem

`_import_one_source` (in `src/regime_data_fetch/acquisition_consolidation.py`) imports rows from a source SQLite database into the consolidated target database. Four insert loops run per source:

1. `fetch_runs` (line ~150) — captures `cursor.lastrowid` into `fetch_run_id_map`
2. `artifacts` (line ~181) — captures `cursor.lastrowid` into `artifact_id_map`
3. `artifact_blobs` (line ~232) — uses `artifact_id_map` to remap the foreign key; no lastrowid needed
4. `derived_outputs` (line ~249) — does NOT need lastrowid; calls `_import_normalized_output` after each insert

Loops 1 and 2 must remain single-row because they capture `lastrowid` to build the ID maps used downstream. Loops 3 and 4 can be batched with `executemany` for significant SQLite performance gains. Loop 4 currently interleaves the INSERT with `_import_normalized_output` side effects; the spec separates these into two passes so the INSERT can be batched.

## Goal

Convert loops 3 and 4 to `executemany`-style batched inserts while preserving all observable behavior:

- Final row counts unchanged
- Final row content unchanged (including `notes` augmentation)
- `_import_normalized_output` still called once per `derived_outputs` source row
- ID remapping correctness preserved for `artifact_blobs.artifact_id`

## Non-Goals

- Changing `fetch_runs` or `artifacts` insert loops — they need `lastrowid`.
- Changing `_import_normalized_output` or any function in `acquisition_consolidation_normalized.py` — those normalized importers already use `executemany`.
- Changing SQLite pragmas (`journal_mode`, `synchronous`, etc.) — out of scope.
- Adding explicit `BEGIN`/`COMMIT` — Python's `sqlite3` already wraps these inserts in an implicit transaction committed at line ~326.

## Changes

### `artifact_blobs` loop

Replace lines ~232-247:

```python
if _table_exists(src_conn, ARTIFACT_BLOBS_TABLE):
    for row in src_conn.execute(
        "SELECT * FROM artifact_blobs ORDER BY artifact_id"
    ):
        old_artifact_id = int(row["artifact_id"])
        if old_artifact_id not in artifact_id_map:
            continue
        dst_conn.execute(
            """INSERT INTO artifact_blobs (artifact_id, content_bytes) VALUES (?, ?)""",
            (artifact_id_map[old_artifact_id], row["content_bytes"]),
        )
```

with:

```python
if _table_exists(src_conn, ARTIFACT_BLOBS_TABLE):
    blob_rows = [
        (artifact_id_map[int(row["artifact_id"])], row["content_bytes"])
        for row in src_conn.execute(
            "SELECT * FROM artifact_blobs ORDER BY artifact_id"
        )
        if int(row["artifact_id"]) in artifact_id_map
    ]
    if blob_rows:
        dst_conn.executemany(
            """INSERT INTO artifact_blobs (artifact_id, content_bytes) VALUES (?, ?)""",
            blob_rows,
        )
```

### `derived_outputs` loop

Split into two passes: collect+batch-insert first, then loop again for `_import_normalized_output` side effects.

Replace lines ~249-287 with a two-pass form:

```python
derived_source_rows = list(
    src_conn.execute("SELECT * FROM derived_outputs ORDER BY output_id")
)
if derived_source_rows:
    derived_insert_rows = [
        (
            fetch_run_id_map[int(row["run_id"])],
            row["output_kind"],
            row["path"],
            row["content_sha256"],
            row["row_count"],
            row["min_date"],
            row["max_date"],
            row["recorded_at_utc"],
            _merge_notes(row["notes"], f"imported_from={source.label}:{source.db_path}"),
        )
        for row in derived_source_rows
    ]
    dst_conn.executemany(
        """
        INSERT INTO derived_outputs (
            run_id, output_kind, path, content_sha256, row_count,
            min_date, max_date, recorded_at_utc, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        derived_insert_rows,
    )
    for row in derived_source_rows:
        imported = _import_normalized_output(
            dst_conn=dst_conn,
            run_id=fetch_run_id_map[int(row["run_id"])],
            output_kind=row["output_kind"],
            path=Path(row["path"]),
        )
        if imported is not None:
            normalized_counts[imported] += int(row["row_count"] or 0)
```

**Ordering invariant:** SQLite's `executemany` preserves input order, and the two passes use the same `ORDER BY output_id` source query (materialized once via `.fetchall()`). The end state is identical to per-row interleaved processing.

## Testing Strategy

The existing `test_consolidate_acquisition_dbs_merges_runs_artifacts_outputs_and_ohlcv` in `tests/test_acquisition_consolidation.py` already asserts:

- Final counts per table (including `artifact_blobs: 1`, `derived_outputs: 3`)
- Specific row content for `fetch_runs`, `artifacts`, `derived_outputs`, ohlcv, events, pmi
- The `imported_from=` and `consolidated_from_label=` augmentations

This is a strong end-to-end regression guard. The current scenario has 1 `artifact_blob` and 3 `derived_outputs` across 2 sources.

**New test to add** (regression guard for batched insert ordering and ID remap):

`test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source`

Builds a source DB with 3 artifact_blobs across 2 artifacts in a single source, runs consolidation, and asserts:

- All 3 blob rows land in target
- Each blob's `artifact_id` correctly points to its remapped artifact in the target
- Blob content (`content_bytes`) matches source

This explicitly exercises the multi-row batched insert path which the existing single-blob test does not.

`derived_outputs` is already exercised with 3 rows in the existing test — that's sufficient multi-row coverage.

## Files Touched

- `src/regime_data_fetch/acquisition_consolidation.py` — modify `_import_one_source` body (lines ~232-287). No signature changes.
- `tests/test_acquisition_consolidation.py` — add one new test for multi-blob path.

## Risk Assessment

Low. The change is localized to one function. The existing integration test asserts row counts AND content for every consolidated table. A behavior change would fail loudly.

The only correctness subtlety is ordering of side effects: `_import_normalized_output` writes into normalized tables (`event_calendar_rows`, `pmi_rows`, etc.). The new two-pass form does ALL `derived_outputs` inserts before ANY `_import_normalized_output` calls. Since these write to disjoint tables, the order doesn't affect final state — but verify by running `test_acquisition_consolidation_normalized.py` end-to-end.

## Performance Expectation

The current consolidation report logs counts in the thousands for daily_ohlcv (already batched). `derived_outputs` is typically dozens to hundreds of rows per source — `executemany` removes per-row Python↔C overhead. `artifact_blobs` is similar magnitude. Expected wall-clock improvement: 2-10× on the relevant loops; small in absolute terms but free.

No measurable speedup is required to consider this change successful. The primary benefit is code uniformity (matching the existing `daily_ohlcv_rows` `executemany` pattern in the same function at line ~295).

## Rollout

Single PR, single commit. No feature flag.
