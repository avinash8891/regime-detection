# Complete Business Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scalar calibration provenance and V2 request input requirements complete and mechanically enforced.

**Architecture:** Expand provenance from section rows to scalar rows generated from declared business config roots. Add a request input contract matrix to `engine.py` and validate configured V2 source families before timeline construction.

**Tech Stack:** Python 3.14, Pydantic v2 models, pandas, pytest, pyright strict slice.

---

### Task 1: Scalar Provenance Coverage

**Files:**
- Modify: `src/regime_detection/rule_provenance.py`
- Modify: `tests/test_rule_provenance.py`

- [ ] Write failing tests that require each scalar business config path to have provenance.
- [ ] Implement scalar provenance expansion from declared config roots.
- [ ] Keep static precedence and risk-rank provenance explicit.
- [ ] Run `python3 -m pytest tests/test_rule_provenance.py -q`.

### Task 2: V2 Request Input Contracts

**Files:**
- Modify: `src/regime_detection/engine.py`
- Modify: `tests/test_foundation.py`
- Modify: `tests/conftest.py`

- [ ] Write failing tests for each configured V2 input family.
- [ ] Add request input contract declarations and enforcement.
- [ ] Add missing synthetic fixture macro keys so tests using full V2 config represent valid configured input.
- [ ] Run `python3 -m pytest tests/test_foundation.py -q`.

### Task 3: Documentation and Verification

**Files:**
- Modify: `docs/regime_engine_contracts.md`

- [ ] Document scalar-level provenance and request input matrix ownership.
- [ ] Run `python3 -m black --check src tests scripts`.
- [ ] Run `python3 -m ruff check .`.
- [ ] Run `python3 -m pyright`.
- [ ] Run targeted pytest files for changed behavior.

