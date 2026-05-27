# Consolidation `executemany` Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch `artifact_blobs` and `derived_outputs` inserts in `_import_one_source` using `executemany`, while preserving all observable behavior asserted by existing integration tests.

**Architecture:** Two-task TDD cycle. (1) Add a multi-blob regression test that passes against current implementation. (2) Convert both loops to `executemany`; verify all tests stay green.

**Tech Stack:** Python 3.12, sqlite3 stdlib, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-27-consolidation-executemany-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/regime_data_fetch/acquisition_consolidation.py` | `_import_one_source` — batch the `artifact_blobs` and `derived_outputs` loops | Modify |
| `tests/test_acquisition_consolidation.py` | Add one test: `test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source` | Modify |

---

## Task 1: Add multi-blob regression test (passes against current implementation)

**Files:**
- Modify: `tests/test_acquisition_consolidation.py` (add one test + one helper builder)

- [ ] **Step 1.1: Read the existing test file to understand the helper builder pattern**

```bash
python3 -c "import ast, pathlib; tree = ast.parse(pathlib.Path('tests/test_acquisition_consolidation.py').read_text()); print([n.name for n in tree.body if isinstance(n, ast.FunctionDef)])"
```
Expected output includes `_build_source_db_one`, `_build_source_db_two`, `_write_binary_fixture`. The new test will follow the same pattern.

- [ ] **Step 1.2: Add the new test and a helper builder**

Append the following to `tests/test_acquisition_consolidation.py` (after the existing tests, before the helper functions block at `_build_source_db_one`):

```python
def test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src_multi_blob.db"
    _build_source_db_with_multiple_blobs(src)

    target = tmp_path / "canonical.db"
    consolidate_acquisition_dbs(
        target_db_path=target,
        sources=[ConsolidationSource("multi", src)],
    )

    with closing(sqlite3.connect(target)) as conn:
        target_blobs = conn.execute(
            """
            SELECT artifacts.source_identifier, artifact_blobs.content_bytes
            FROM artifact_blobs
            JOIN artifacts ON artifacts.artifact_id = artifact_blobs.artifact_id
            ORDER BY artifacts.source_identifier
            """
        ).fetchall()

    assert target_blobs == [
        ("ident-A", b"blob-bytes-A"),
        ("ident-B", b"blob-bytes-B"),
        ("ident-C", b"blob-bytes-C"),
    ]
```

Then append the helper builder (alongside `_build_source_db_one` / `_build_source_db_two`):

```python
def _build_source_db_with_multiple_blobs(path: Path) -> None:
    store = AcquisitionStore(path)
    run = store.start_fetch_run(fetch_type="binary_multi", params={"n": 3})
    blob_a = _write_blob_fixture(path.parent / "a.bin", b"blob-bytes-A")
    blob_b = _write_blob_fixture(path.parent / "b.bin", b"blob-bytes-B")
    blob_c = _write_blob_fixture(path.parent / "c.bin", b"blob-bytes-C")
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:A",
        artifact_kind="binary",
        source_identifier="ident-A",
        file_path=blob_a,
        notes="blob A",
    )
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:B",
        artifact_kind="binary",
        source_identifier="ident-B",
        file_path=blob_b,
        notes="blob B",
    )
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:C",
        artifact_kind="binary",
        source_identifier="ident-C",
        file_path=blob_c,
        notes="blob C",
    )
    store.finish_fetch_run(run_id=run.run_id, status="ok", notes="done-multi")


def _write_blob_fixture(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path
```

**Note:** If `AcquisitionStore.record_file_artifact` does not in fact write the binary content into the source DB's `artifact_blobs` table (verify by reading `src/regime_data_fetch/acquisition_store.py` — the field-level mechanics depend on its implementation), you may need to adjust the helper. Specifically, check whether `record_file_artifact` reads the file and stores its bytes in `artifact_blobs`, or just stores the local_path. If it stores `local_path` only (no blob), the test setup needs to use `AcquisitionStore` API for storing binary blobs OR write directly to the source DB's `artifact_blobs` table after creating artifacts. Investigate before writing the helper; pick whichever path actually populates `artifact_blobs` rows in the source DB.

- [ ] **Step 1.3: Run the new test against the CURRENT implementation**

```bash
python3 -m pytest tests/test_acquisition_consolidation.py::test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source -v
```
Expected: **PASS**. The current single-row loop already handles multiple blobs correctly; this test pins that behavior before we refactor it to `executemany`.

If it FAILS:
- If it's an `AssertionError: target_blobs == ...`, check whether the source DB actually contains rows in `artifact_blobs`. The helper may need to use a different `AcquisitionStore` API method that writes blob bytes. Adjust and re-run.
- If it's an import or setup error, debug and fix.

Do NOT proceed to Task 2 until this test passes against the current code.

- [ ] **Step 1.4: Run the full file**

```bash
python3 -m pytest tests/test_acquisition_consolidation.py -v
```
All tests must pass.

- [ ] **Step 1.5: Commit**

```bash
git add tests/test_acquisition_consolidation.py
git commit -m "$(cat <<'EOF'
test: pin multi-row artifact_blobs consolidation behavior

Adds a regression test that consolidates a source with 3 artifact_blobs
and asserts each blob lands in the target with its content intact and
correctly joined to the remapped artifact_id. Locks in behavior before
the upcoming executemany refactor.
EOF
)"
```

Verify only one file in the commit:
```bash
git show --stat HEAD
```
Expected: only `tests/test_acquisition_consolidation.py` in the changed-files list.

---

## Task 2: Convert `artifact_blobs` and `derived_outputs` to `executemany`

**Files:**
- Modify: `src/regime_data_fetch/acquisition_consolidation.py` (function body of `_import_one_source`)

- [ ] **Step 2.1: Read the current function**

```bash
python3 -c "
import pathlib
src = pathlib.Path('src/regime_data_fetch/acquisition_consolidation.py').read_text().splitlines()
in_fn = False
for i, line in enumerate(src, 1):
    if line.startswith('def _import_one_source'):
        in_fn = True
    if in_fn:
        print(f'{i}: {line}')
        if line.strip().startswith('return {') and 'len(fetch_run_id_map)' not in line:
            break
" | head -80
```
This locates the function in case line numbers have shifted. Identify:
- The `artifact_blobs` loop (starts with `if _table_exists(src_conn, ARTIFACT_BLOBS_TABLE):`)
- The `derived_outputs` loop (starts with `for row in src_conn.execute("SELECT * FROM derived_outputs ORDER BY output_id"):`)

- [ ] **Step 2.2: Replace the `artifact_blobs` loop**

Find this block in `src/regime_data_fetch/acquisition_consolidation.py`:

```python
        if _table_exists(src_conn, ARTIFACT_BLOBS_TABLE):
            for row in src_conn.execute(
                "SELECT * FROM artifact_blobs ORDER BY artifact_id"
            ):
                old_artifact_id = int(row["artifact_id"])
                if old_artifact_id not in artifact_id_map:
                    continue
                dst_conn.execute(
                    """
                    INSERT INTO artifact_blobs (
                        artifact_id,
                        content_bytes
                    ) VALUES (?, ?)
                    """,
                    (artifact_id_map[old_artifact_id], row["content_bytes"]),
                )
```

Replace with:

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
                    """
                    INSERT INTO artifact_blobs (
                        artifact_id,
                        content_bytes
                    ) VALUES (?, ?)
                    """,
                    blob_rows,
                )
```

Preserve the leading 8-space indent (this is inside the `with` block inside `_import_one_source`).

- [ ] **Step 2.3: Replace the `derived_outputs` loop**

Find this block:

```python
        for row in src_conn.execute("SELECT * FROM derived_outputs ORDER BY output_id"):
            new_run_id = fetch_run_id_map[int(row["run_id"])]
            notes = _merge_notes(
                row["notes"], f"imported_from={source.label}:{source.db_path}"
            )
            dst_conn.execute(
                """
                INSERT INTO derived_outputs (
                    run_id,
                    output_kind,
                    path,
                    content_sha256,
                    row_count,
                    min_date,
                    max_date,
                    recorded_at_utc,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_run_id,
                    row["output_kind"],
                    row["path"],
                    row["content_sha256"],
                    row["row_count"],
                    row["min_date"],
                    row["max_date"],
                    row["recorded_at_utc"],
                    notes,
                ),
            )
            imported = _import_normalized_output(
                dst_conn=dst_conn,
                run_id=new_run_id,
                output_kind=row["output_kind"],
                path=Path(row["path"]),
            )
            if imported is not None:
                normalized_counts[imported] += int(row["row_count"] or 0)
```

Replace with:

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
                    _merge_notes(
                        row["notes"], f"imported_from={source.label}:{source.db_path}"
                    ),
                )
                for row in derived_source_rows
            ]
            dst_conn.executemany(
                """
                INSERT INTO derived_outputs (
                    run_id,
                    output_kind,
                    path,
                    content_sha256,
                    row_count,
                    min_date,
                    max_date,
                    recorded_at_utc,
                    notes
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

Preserve the leading 8-space indent.

- [ ] **Step 2.4: Run the multi-blob test added in Task 1**

```bash
python3 -m pytest tests/test_acquisition_consolidation.py::test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source -v
```
Expected: **PASS**. If FAIL, the batched insert is producing different output — debug before continuing.

- [ ] **Step 2.5: Run the full consolidation test files**

```bash
python3 -m pytest tests/test_acquisition_consolidation.py tests/test_acquisition_consolidation_normalized.py -v
```
Expected: all pass.

- [ ] **Step 2.6: Run the full test suite (final safety net)**

```bash
python3 -m pytest tests/ -x --tb=short
```
Expected: all pass.

- [ ] **Step 2.7: Commit**

Verify only one file is staged before committing:
```bash
git diff --cached --name-only
```
Should be empty at this point (nothing staged yet). Then:

```bash
git add src/regime_data_fetch/acquisition_consolidation.py
git diff --cached --name-only
```
Must show exactly:
```
src/regime_data_fetch/acquisition_consolidation.py
```

If anything else is staged, run `git restore --staged <file>` to unstage it before committing.

```bash
git commit -m "$(cat <<'EOF'
perf: batch artifact_blobs and derived_outputs inserts via executemany

Converts the two single-row insert loops in _import_one_source to
executemany, matching the daily_ohlcv_rows batching pattern already
used in the same function. fetch_runs and artifacts remain single-row
because they capture cursor.lastrowid for the ID remap. The
derived_outputs loop is split into two passes (batched insert, then
_import_normalized_output side effects) because the latter writes to
disjoint normalized tables and can run after all derived_outputs
rows are inserted.

Behavior preserved: row counts, content, and notes augmentation are
unchanged. Verified by the existing end-to-end consolidation test
plus the new multi-blob regression test.
EOF
)"
```

---

## Done Criteria

- `test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source` passes against the new implementation.
- All of `test_acquisition_consolidation.py` and `test_acquisition_consolidation_normalized.py` pass.
- Full test suite passes.
- Two commits on the branch: one for the new test, one for the executemany refactor.
- Only the intended files are in each commit.
