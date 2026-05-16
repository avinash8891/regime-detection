# Tech Debt Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 30 findings from TECH_DEBT_AUDIT.md in priority order, from trivial quick wins through architectural refactors and test coverage gaps.

**Architecture:** Tasks are grouped by risk/effort: Group A = trivial fixes (<5 min), Group B = consistency/CI, Group C = type safety refactor, Group D = architectural decomposition, Group E = test coverage. Each group is independently mergeable.

**Tech Stack:** Python 3.11, pydantic v2, pandas, pytest, ruff, mypy

---

## Group A — Quick Wins (no behavior change, no tests needed)

### Task 1: F020 — Remove f-prefix from 5 literal strings in run_v2_calibration.py

**Files:**
- Modify: `scripts/run_v2_calibration.py:98-104,111-113`

- [ ] **Step 1: Make the fix**

  In `scripts/run_v2_calibration.py`, replace lines 98-104 and 111-113.

  ```python
  # BEFORE (lines 97-113):
  lines = [
      f"- `feature_store.monetary.central_bank_text_score` lit: **True** "
      f"(n={len(series)} sessions)",
      f"- Smoothing window: **{cb_cfg.smoothing_window_sessions}** NYSE "
      f"sessions (CentralBankTextConfig.smoothing_window_sessions; "
      f"v2 §9.1 walk-forward calibration placeholder).",
      f"- max_release_age_days: **{cb_cfg.max_release_age_days}**.",
      f"- Score distribution after smoothing:",
      ...
      f"- Bias-warning code emitted on feature output: "
      f"`central_bank_text_deterministic_lexicon_substitute` (audit M1 / "
      f"docs/spec_code_data_audit_2026_05_15.md §3.1).",
  ]

  # AFTER — remove f-prefix only from the lines with NO {placeholders}:
  lines = [
      f"- `feature_store.monetary.central_bank_text_score` lit: **True** "
      f"(n={len(series)} sessions)",
      f"- Smoothing window: **{cb_cfg.smoothing_window_sessions}** NYSE "
      "sessions (CentralBankTextConfig.smoothing_window_sessions; "
      "v2 §9.1 walk-forward calibration placeholder).",
      f"- max_release_age_days: **{cb_cfg.max_release_age_days}**.",
      "- Score distribution after smoothing:",
      ...
      f"- Bias-warning code emitted on feature output: "
      "`central_bank_text_deterministic_lexicon_substitute` (audit M1 / "
      "docs/spec_code_data_audit_2026_05_15.md §3.1).",
  ]
  ```

  The exact lines to fix (F541 = f-string with no `{...}` placeholder):
  - Line 101: `f"sessions (CentralBankTextConfig..."` → remove `f`
  - Line 102: `f"v2 §9.1 walk-forward..."` → remove `f`
  - Line 104: `f"- Score distribution after smoothing:"` → remove `f`
  - Line 112: `f"`central_bank_text_deterministic_lexicon_substitute`..."` → remove `f`
  - Line 113: `f"docs/spec_code_data_audit_2026_05_15.md §3.1)."` → remove `f`

- [ ] **Step 2: Verify ruff passes**

  ```bash
  ruff check scripts/run_v2_calibration.py
  ```
  Expected: 5 fewer F541 violations (other issues in the file are E402/F401, not F541).

- [ ] **Step 3: Commit**

  ```bash
  git add scripts/run_v2_calibration.py
  git commit -m "fix: remove f-prefix from 5 literal strings (F541)"
  ```

---

### Task 2: F013 — Add warning log before bare except in validators_hf_central_bank.py

**Files:**
- Modify: `src/regime_data_fetch/event_sources/validators_hf_central_bank.py:1-15,51-55`

- [ ] **Step 1: Add import and log**

  At the top of `validators_hf_central_bank.py`, `logging` is not yet imported. Add it. Then add the log call.

  ```python
  # After existing imports, add:
  import logging

  # Add near top of file (after PARQUET_URL):
  _LOG = logging.getLogger(__name__)
  ```

  Then change lines 51-55:
  ```python
  # BEFORE:
  try:
      parquet_bytes = self.parquet_fetcher()
      frame = pd.read_parquet(BytesIO(parquet_bytes))
  except Exception:
      return [_unknown(candidate) for candidate in central_bank_candidates]

  # AFTER:
  try:
      parquet_bytes = self.parquet_fetcher()
      frame = pd.read_parquet(BytesIO(parquet_bytes))
  except Exception:
      _LOG.warning(
          "HF central bank parquet fetch/parse failed — returning unknown for %d candidates",
          len(central_bank_candidates),
          exc_info=True,
      )
      return [_unknown(candidate) for candidate in central_bank_candidates]
  ```

- [ ] **Step 2: Verify no ruff errors introduced**

  ```bash
  ruff check src/regime_data_fetch/event_sources/validators_hf_central_bank.py
  ```
  Expected: no new violations.

- [ ] **Step 3: Commit**

  ```bash
  git add src/regime_data_fetch/event_sources/validators_hf_central_bank.py
  git commit -m "fix: log warning before silent except fallback in hf_central_bank validator"
  ```

---

### Task 3: F014 — Add warning log before bare except in acquisition_consolidation.py

**Files:**
- Modify: `src/regime_data_fetch/acquisition_consolidation.py:826-833`

- [ ] **Step 1: Add log call**

  Find the `_augment_params_json` function. The file already has a `LOG` or logger — check line 1 for logging setup. The file imports `logging` and uses a module logger (grep shows it does).

  ```python
  # BEFORE (lines 827-830):
  try:
      payload = json.loads(params_json)
  except Exception:
      payload = {"raw_params_json": params_json}

  # AFTER:
  try:
      payload = json.loads(params_json)
  except Exception:
      _LOG.warning(
          "params_json unparseable in _augment_params_json, using raw fallback (first 200 chars): %s",
          params_json[:200],
      )
      payload = {"raw_params_json": params_json}
  ```

  First confirm the logger variable name in this file:
  ```bash
  grep -n "^_LOG\|^LOG\b" src/regime_data_fetch/acquisition_consolidation.py | head -3
  ```
  Use whatever name is already defined (likely `_LOG` or `LOG`).

- [ ] **Step 2: Verify**

  ```bash
  ruff check src/regime_data_fetch/acquisition_consolidation.py
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add src/regime_data_fetch/acquisition_consolidation.py
  git commit -m "fix: log warning on unparseable params_json in acquisition_consolidation"
  ```

---

### Task 4: F015 — Add debug log before Excel→CSV fallback in validators_gpr_gdelt.py

**Files:**
- Modify: `src/regime_data_fetch/event_sources/validators_gpr_gdelt.py:253-258`

- [ ] **Step 1: Add log**

  The file already has `LOGGER = logging.getLogger(__name__)` at line 18.

  ```python
  # BEFORE (lines 255-258):
  try:
      df = pd.read_excel(io.BytesIO(payload))
  except Exception:
      df = pd.read_csv(io.BytesIO(payload))

  # AFTER:
  try:
      df = pd.read_excel(io.BytesIO(payload))
  except Exception:
      LOGGER.debug("GPR payload not Excel, falling back to CSV", exc_info=True)
      df = pd.read_csv(io.BytesIO(payload))
  ```

- [ ] **Step 2: Verify**

  ```bash
  ruff check src/regime_data_fetch/event_sources/validators_gpr_gdelt.py
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add src/regime_data_fetch/event_sources/validators_gpr_gdelt.py
  git commit -m "fix: log debug before Excel→CSV fallback in parse_gpr_table"
  ```

---

### Task 5: F017 — Remove redundant list comprehension in axis_series.py

**Files:**
- Modify: `src/regime_detection/axis_series.py:1387`

- [ ] **Step 1: Fix the line**

  ```python
  # BEFORE (line 1387):
  input_by_date = [series for series in required_inputs]

  # AFTER:
  input_by_date = list(required_inputs)
  ```

- [ ] **Step 2: Run the axis_series tests**

  ```bash
  python3 -m pytest tests/test_axis_series.py tests/test_axis_series_cleanup.py -v
  ```
  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add src/regime_detection/axis_series.py
  git commit -m "fix: replace no-op list comprehension with list() in _build_axis_outputs"
  ```

---

### Task 6: F021 — Update README.md to reflect V2

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README**

  ```markdown
  # regime-detection

  Regime detection engine (V1 + V2 unified). Classifies market regimes across
  multiple axes — trend direction, trend character, volatility state, breadth,
  credit/funding, inflation/growth, monetary pressure, network fragility, and
  volume/liquidity — using rule-based classifiers backed by market data.

  ## Architecture

  - **V1** (frozen): `core3-v1.0.0` config produces byte-identical output for
    archive replay. See `docs/regime_engine_v1_final_spec.md`.
  - **V2** (in progress): extends V1 with new axes and a transition-risk score.
    See `docs/regime_engine_v2_spec.md`. V2 slices ship behind config-version
    guards; V1 byte-identity is preserved when V2 config is absent.

  ## Quick start

  ```bash
  pip install -e ".[dev]"
  pytest
  ```

  ## Key docs

  - V1 spec: `docs/regime_engine_v1_final_spec.md`
  - V2 spec: `docs/regime_engine_v2_spec.md`
  - Data requirements: `docs/regime_engine_v1_data_requirements.md`
  - Shadow runner: `docs/shadow_runner_spec.md`
  - Historical walk-forward: `docs/historical_walkforward_spec.md`
  - Agent operating rules: `AGENTS.md`
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add README.md
  git commit -m "docs: update README to reflect V2 architecture and quick start"
  ```

---

## Group B — Consistency & CI

### Task 7: F012 — Standardize logger variable name to `_LOG` across regime_detection

**Files:**
- Modify: `src/regime_detection/central_bank_text.py:60`
- Modify: `src/regime_detection/loaders.py:11`
- Modify: `src/regime_detection/event_calendar.py:18`
- Modify: `src/regime_detection/hmm_state.py:33`
- Modify: `src/regime_detection/clustering.py:34`
- Modify: `src/regime_detection/change_point.py:45`

The goal: every module uses `_LOG = logging.getLogger(__name__)`. Currently three names are in use: `LOG`, `_LOGGER`, `LOGGER`.

- [ ] **Step 1: Rename in each file**

  For each file, replace the logger declaration and all usages:

  **`central_bank_text.py`:** `LOG =` → `_LOG =`. Then find all `LOG.` usages (there is only 1 at line 60, search the file for any others).

  ```bash
  grep -n "LOG\." src/regime_detection/central_bank_text.py
  ```
  Replace each `LOG.` → `_LOG.`.

  **`loaders.py`:** Same — `LOG =` → `_LOG =`, `LOG.` → `_LOG.`

  **`event_calendar.py`:** Same — `LOG =` → `_LOG =`, `LOG.` → `_LOG.`

  **`hmm_state.py`:** `_LOGGER =` → `_LOG =`, `_LOGGER.` → `_LOG.`

  **`clustering.py`:** `_LOGGER =` → `_LOG =`, `_LOGGER.` → `_LOG.`

  **`change_point.py`:** `_LOGGER =` → `_LOG =`, `_LOGGER.` → `_LOG.`

- [ ] **Step 2: Verify no broken references**

  ```bash
  grep -rn "\bLOG\b\|\b_LOGGER\b" src/regime_detection/ --include="*.py" | grep -v "_LOG\b"
  ```
  Expected: no output (all renamed).

- [ ] **Step 3: Run affected tests**

  ```bash
  python3 -m pytest tests/test_hmm_state.py tests/test_clustering.py tests/test_change_point.py tests/test_loaders.py -v
  ```
  Expected: all pass.

- [ ] **Step 4: Commit**

  ```bash
  git add src/regime_detection/central_bank_text.py src/regime_detection/loaders.py \
    src/regime_detection/event_calendar.py src/regime_detection/hmm_state.py \
    src/regime_detection/clustering.py src/regime_detection/change_point.py
  git commit -m "refactor: standardize logger variable to _LOG across regime_detection"
  ```

---

### Task 8: F018 — Extract magic `required_trading_days` numbers to named constants

**Files:**
- Modify: `src/regime_detection/axis_series.py:85-92` (near top, after imports)

- [ ] **Step 1: Add constants near top of axis_series.py**

  After the imports section and before the first class definition (line ~92), add:

  ```python
  # Lookback windows for data-quality sufficiency gates.
  # Each constant cites the spec section that mandates the window.
  _TREND_DIRECTION_MIN_SESSIONS = 200   # §1A: 200-day SMA lookback
  _TREND_CHARACTER_MIN_SESSIONS = 63    # §1B: 63-day drawdown lookback
  _VOLATILITY_MIN_SESSIONS = 252        # §1C: 252-day realized-vol window
  _BREADTH_MIN_SESSIONS = 50            # §1D: breadth 50-period minimum
  ```

- [ ] **Step 2: Replace the magic numbers at call sites**

  In `TrendDirectionSeriesClassifier.build` (line 165):
  ```python
  required_trading_days=_TREND_DIRECTION_MIN_SESSIONS,
  ```

  In `TrendCharacterSeriesClassifier.build` (line 194):
  ```python
  required_trading_days=_TREND_CHARACTER_MIN_SESSIONS,
  ```

  In `VolatilitySeriesClassifier.build` (line 231):
  ```python
  required_trading_days=_VOLATILITY_MIN_SESSIONS,
  ```

  In `BreadthSeriesClassifier.build` (line 323):
  ```python
  required_trading_days=_BREADTH_MIN_SESSIONS,
  ```

- [ ] **Step 3: Run tests**

  ```bash
  python3 -m pytest tests/test_axis_series.py tests/test_breadth_state.py tests/test_trend_direction.py tests/test_volatility_state.py -v
  ```
  Expected: all pass.

- [ ] **Step 4: Commit**

  ```bash
  git add src/regime_detection/axis_series.py
  git commit -m "refactor: extract required_trading_days magic numbers to named constants"
  ```

---

### Task 9: F016 — Add ruff, mypy, and coverage gate to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read the current CI file**

  ```bash
  cat .github/workflows/ci.yml
  ```

- [ ] **Step 2: Add lint, type-check, and coverage steps**

  After the existing `- name: Python test discovery` step, append (or replace the pytest step):

  ```yaml
  - name: Lint
    run: ruff check .

  - name: Type check
    run: python -m mypy src/ --ignore-missing-imports

  - name: Test with coverage
    run: pytest --cov=src --cov-report=term-missing --cov-fail-under=55
  ```

  Note: start `--cov-fail-under=55` to match current measured baseline. The audit recommends 70% as the target — raise it as coverage improves in Group E tasks.

- [ ] **Step 3: Verify the workflow parses**

  ```bash
  python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
  ```
  Expected: no exception.

- [ ] **Step 4: Commit**

  ```bash
  git add .github/workflows/ci.yml
  git commit -m "ci: add ruff, mypy, and coverage gate (--cov-fail-under=55)"
  ```

---

### Task 10: F019 — Fix E402 violations in scripts (extract sys.path bootstrap)

**Files:**
- Create: `scripts/_bootstrap.py`
- Modify: `scripts/fetch_regime_engine_v1_data.py`, `scripts/run_shadow_regime.py`, `scripts/profile_engine_30d.py`, `scripts/run_historical_walkforward.py`, `scripts/run_shadow_replay_check.py`, `scripts/run_shadow_deadman_check.py`, `scripts/consolidate_regime_acquisition.py`, `scripts/fetch_aaii_sentiment.py`, `scripts/approve_group_b_candidate.py`, `scripts/run_v2_calibration.py`

- [ ] **Step 1: Create `scripts/_bootstrap.py`**

  ```python
  """Adds repo src/ to sys.path so scripts can be run without installation."""
  from __future__ import annotations

  import sys
  from pathlib import Path

  _SRC = Path(__file__).resolve().parents[1] / "src"
  if str(_SRC) not in sys.path:
      sys.path.insert(0, str(_SRC))
  ```

- [ ] **Step 2: Replace sys.path block in each script**

  Each affected script currently has this block (possibly with slight variations):
  ```python
  import sys
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  SRC_DIR = REPO_ROOT / "src"
  if str(SRC_DIR) not in sys.path:
      sys.path.insert(0, str(SRC_DIR))
  ```

  Replace it with:
  ```python
  import scripts._bootstrap  # noqa: F401
  ```

  Or, since the scripts are meant to run standalone, use relative import:
  ```python
  from pathlib import Path as _Path
  import sys as _sys
  _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
  # (keep as is — the issue is only that downstream imports must follow immediately)
  ```

  Actually the E402 violations are because imports come AFTER the sys.path manipulation. The correct fix is to keep the sys.path block at the very top and move all `import` statements for library code immediately below it (no blank lines between). Ruff E402 fires when stdlib imports are not at file top, before any code. The pattern that's violating E402 is likely:

  ```python
  #!/usr/bin/env python3
  import argparse       ← stdlib, fine
  import sys            ← stdlib, fine

  REPO_ROOT = ...       ← code (not import) — triggers E402 for everything below
  SRC_DIR = ...
  sys.path.insert(...)

  import pandas as pd   ← E402 because code ran above
  from regime_data_fetch import ...  ← E402
  ```

  The correct approach: add `# noqa: E402` to each post-path-manipulation import that ruff flags. This is the pragmatic fix for CLI scripts where the sys.path manipulation is unavoidable:

  ```python
  #!/usr/bin/env python3
  from __future__ import annotations
  import sys
  from pathlib import Path

  _SRC = Path(__file__).resolve().parents[1] / "src"
  if str(_SRC) not in sys.path:
      sys.path.insert(0, str(_SRC))

  import pandas as pd  # noqa: E402
  from regime_data_fetch.cli_common import load_env_file  # noqa: E402
  ```

  Apply `# noqa: E402` to the first post-manipulation import line in each affected script. This makes ruff pass without restructuring the scripts.

  Run to find all affected lines:
  ```bash
  ruff check scripts/ --select E402 --output-format=concise
  ```

- [ ] **Step 3: Verify ruff passes on scripts**

  ```bash
  ruff check scripts/
  ```
  Expected: no E402 violations remain.

- [ ] **Step 4: Commit**

  ```bash
  git add scripts/
  git commit -m "fix: suppress E402 on post-sys.path imports in CLI scripts (noqa)"
  ```

---

## Group C — Type Safety

### Task 11: F004 — Replace `dict[str, float | str]` with `TransitionScoreInputs` dataclass

This eliminates all 14 `# type: ignore` comments in `transition_risk_series.py`.

**Files:**
- Create: `src/regime_detection/transition_score_inputs.py`
- Modify: `src/regime_detection/transition_risk_series.py`

- [ ] **Step 1: Write the failing test**

  In `tests/test_transition_score_v2.py`, add:

  ```python
  from regime_detection.transition_score_inputs import TransitionScoreInputs
  from datetime import date

  def test_transition_score_inputs_is_frozen_dataclass() -> None:
      inputs = TransitionScoreInputs(
          realized_vol_short=0.15,
          realized_vol_long=0.12,
          pct_above_50dma=0.60,
          avg_pairwise_corr_percentile_504d=0.45,
          drawdown_252d=-0.08,
          event_calendar_label="neutral",
          hmm_top_state_prob_now=float("nan"),
          hmm_top_state_prob_5d_ago=float("nan"),
          change_point_score=float("nan"),
      )
      assert inputs.realized_vol_short == 0.15
      assert inputs.event_calendar_label == "neutral"

  def test_transition_score_inputs_immutable() -> None:
      import pytest
      inputs = TransitionScoreInputs(
          realized_vol_short=0.15,
          realized_vol_long=0.12,
          pct_above_50dma=0.60,
          avg_pairwise_corr_percentile_504d=0.45,
          drawdown_252d=-0.08,
          event_calendar_label="neutral",
          hmm_top_state_prob_now=float("nan"),
          hmm_top_state_prob_5d_ago=float("nan"),
          change_point_score=float("nan"),
      )
      with pytest.raises(Exception):  # frozen=True raises FrozenInstanceError
          inputs.realized_vol_short = 0.99  # type: ignore[misc]
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  python3 -m pytest tests/test_transition_score_v2.py::test_transition_score_inputs_is_frozen_dataclass -v
  ```
  Expected: ImportError — `transition_score_inputs` not found.

- [ ] **Step 3: Create `src/regime_detection/transition_score_inputs.py`**

  ```python
  """Typed input container for the v2 §4 transition score composer.

  Replaces the loose dict[str, float | str] that forced 14 type: ignore
  suppressions in transition_risk_series.py.
  """
  from __future__ import annotations

  from dataclasses import dataclass


  @dataclass(frozen=True)
  class TransitionScoreInputs:
      """Per-session inputs for compose_transition_score_for_session (v2 §4.2).

      All float fields use NaN (not None) to signal missing data so that
      the cold-start guard in compose_transition_score_for_session can use
      math.isnan() uniformly. event_calendar_label is always a str.
      """

      realized_vol_short: float
      realized_vol_long: float
      pct_above_50dma: float
      avg_pairwise_corr_percentile_504d: float
      drawdown_252d: float
      event_calendar_label: str
      hmm_top_state_prob_now: float
      hmm_top_state_prob_5d_ago: float
      change_point_score: float
  ```

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  python3 -m pytest tests/test_transition_score_v2.py::test_transition_score_inputs_is_frozen_dataclass tests/test_transition_score_v2.py::test_transition_score_inputs_immutable -v
  ```
  Expected: PASS.

- [ ] **Step 5: Update `_build_transition_score_inputs_by_date` in transition_risk_series.py**

  At the top, add the import:
  ```python
  from regime_detection.transition_score_inputs import TransitionScoreInputs
  ```

  Change the function signature return type (line 123):
  ```python
  # BEFORE:
  ) -> dict[date, dict[str, float | str]]:

  # AFTER:
  ) -> dict[date, TransitionScoreInputs]:
  ```

  Change the builder loop (lines 166-178):
  ```python
  # BEFORE:
  out: dict[date, dict[str, float | str]] = {}
  for i, day in enumerate(sessions):
      out[day] = {
          "realized_vol_short": float(rvs[i]),
          "realized_vol_long": float(rvl[i]),
          "pct_above_50dma": float(pct50[i]),
          "avg_pairwise_corr_percentile_504d": float(corr[i]),
          "drawdown_252d": float(dd252[i]),
          "event_calendar_label": event_calendar[day].active_label,
          "hmm_top_state_prob_now": float(hmm_now[i]),
          "hmm_top_state_prob_5d_ago": float(hmm_5d_ago[i]),
          "change_point_score": float(cp[i]),
      }
  return out

  # AFTER:
  out: dict[date, TransitionScoreInputs] = {}
  for i, day in enumerate(sessions):
      out[day] = TransitionScoreInputs(
          realized_vol_short=float(rvs[i]),
          realized_vol_long=float(rvl[i]),
          pct_above_50dma=float(pct50[i]),
          avg_pairwise_corr_percentile_504d=float(corr[i]),
          drawdown_252d=float(dd252[i]),
          event_calendar_label=event_calendar[day].active_label,
          hmm_top_state_prob_now=float(hmm_now[i]),
          hmm_top_state_prob_5d_ago=float(hmm_5d_ago[i]),
          change_point_score=float(cp[i]),
      )
  return out
  ```

  Change the field type annotation (line 59):
  ```python
  # BEFORE:
  transition_score_inputs_by_date: dict[date, dict[str, float | str]] | None = None

  # AFTER:
  transition_score_inputs_by_date: dict[date, TransitionScoreInputs] | None = None
  ```

  Also update the parameter type in `build_transition_risk_outputs_by_date` (line 192):
  ```python
  # BEFORE:
  transition_score_inputs_by_date: dict[date, dict[str, float | str]] | None = None,

  # AFTER:
  transition_score_inputs_by_date: dict[date, TransitionScoreInputs] | None = None,
  ```

- [ ] **Step 6: Replace all 14 `# type: ignore` usages in the consumer loop (lines 265-303)**

  The consumer loop at lines 264-313 calls `inputs.get(...)` and `inputs[...]` on what was a dict. Replace with direct field access:

  ```python
  # BEFORE (lines 265-303, abridged with type: ignore):
  if compose_score:
      inputs = transition_score_inputs_by_date[day]  # type: ignore[index]
      hmm_now_val = inputs.get("hmm_top_state_prob_now")  # type: ignore[union-attr]
      hmm_5d_val = inputs.get("hmm_top_state_prob_5d_ago")  # type: ignore[union-attr]
      hmm_now_arg = (
          None
          if hmm_now_val is None
          or (isinstance(hmm_now_val, float) and pd.isna(hmm_now_val))
          else float(hmm_now_val)  # type: ignore[arg-type]
      )
      hmm_5d_arg = (
          None
          if hmm_5d_val is None
          or (isinstance(hmm_5d_val, float) and pd.isna(hmm_5d_val))
          else float(hmm_5d_val)  # type: ignore[arg-type]
      )
      cp_val = inputs.get("change_point_score")  # type: ignore[union-attr]
      cp_arg = (
          None
          if cp_val is None
          or (isinstance(cp_val, float) and pd.isna(cp_val))
          else float(cp_val)  # type: ignore[arg-type]
      )
      composed = compose_transition_score_for_session(
          realized_vol_short=inputs["realized_vol_short"],  # type: ignore[arg-type]
          realized_vol_long=inputs["realized_vol_long"],  # type: ignore[arg-type]
          pct_above_50dma=inputs["pct_above_50dma"],  # type: ignore[arg-type]
          avg_pairwise_corr_percentile_504d=inputs[
              "avg_pairwise_corr_percentile_504d"
          ],  # type: ignore[arg-type]
          drawdown_252d=inputs["drawdown_252d"],  # type: ignore[arg-type]
          event_calendar_label=inputs["event_calendar_label"],  # type: ignore[arg-type]
          hmm_top_state_prob_now=hmm_now_arg,
          hmm_top_state_prob_5d_ago=hmm_5d_arg,
          change_point_score=cp_arg,
          config=transition_score_config,  # type: ignore[arg-type]
      )

  # AFTER (no type: ignore):
  if compose_score:
      inputs = transition_score_inputs_by_date[day]
      hmm_now_arg = (
          None
          if pd.isna(inputs.hmm_top_state_prob_now)
          else inputs.hmm_top_state_prob_now
      )
      hmm_5d_arg = (
          None
          if pd.isna(inputs.hmm_top_state_prob_5d_ago)
          else inputs.hmm_top_state_prob_5d_ago
      )
      cp_arg = (
          None
          if pd.isna(inputs.change_point_score)
          else inputs.change_point_score
      )
      composed = compose_transition_score_for_session(
          realized_vol_short=inputs.realized_vol_short,
          realized_vol_long=inputs.realized_vol_long,
          pct_above_50dma=inputs.pct_above_50dma,
          avg_pairwise_corr_percentile_504d=inputs.avg_pairwise_corr_percentile_504d,
          drawdown_252d=inputs.drawdown_252d,
          event_calendar_label=inputs.event_calendar_label,
          hmm_top_state_prob_now=hmm_now_arg,
          hmm_top_state_prob_5d_ago=hmm_5d_arg,
          change_point_score=cp_arg,
          config=transition_score_config,
      )
  ```

  Note: `transition_score_config` no longer needs `# type: ignore[arg-type]` because `compose_score` is True only when both `transition_score_inputs_by_date is not None` and `transition_score_config is not None`, but mypy doesn't narrow through the `compose_score` boolean. Add an explicit assert if mypy complains: `assert transition_score_config is not None`.

- [ ] **Step 7: Verify no type: ignore left in the file**

  ```bash
  grep -c "type: ignore" src/regime_detection/transition_risk_series.py
  ```
  Expected: `0`

- [ ] **Step 8: Run mypy and tests**

  ```bash
  python3 -m mypy src/regime_detection/transition_risk_series.py src/regime_detection/transition_score_inputs.py --ignore-missing-imports
  python3 -m pytest tests/test_transition_risk.py tests/test_transition_score_v2.py -v
  ```
  Expected: mypy clean, all tests pass.

- [ ] **Step 9: Commit**

  ```bash
  git add src/regime_detection/transition_score_inputs.py src/regime_detection/transition_risk_series.py tests/test_transition_score_v2.py
  git commit -m "refactor: replace loose dict with TransitionScoreInputs dataclass, eliminate 14 type: ignore"
  ```

---

## Group D — Architectural

### Task 12: F002 — Consolidate sma_50 / return_63d into _rolling_stats.py

**Files:**
- Modify: `src/regime_detection/_rolling_stats.py`
- Modify: `src/regime_detection/feature_store.py:292`
- Modify: `src/regime_detection/trend_direction.py:58-61`
- Modify: `src/regime_detection/trend_character.py:235-238`

- [ ] **Step 1: Write failing test for the new helpers**

  In `tests/test_rolling_stats.py`, add:

  ```python
  import pandas as pd
  import numpy as np
  from regime_detection._rolling_stats import sma, period_return

  def test_sma_matches_pandas_rolling_mean() -> None:
      close = pd.Series([100.0, 102.0, 98.0, 105.0, 101.0], dtype=float)
      result = sma(close, window=3)
      expected = close.rolling(3).mean()
      pd.testing.assert_series_equal(result, expected)

  def test_sma_window_1_is_identity() -> None:
      close = pd.Series([10.0, 20.0, 30.0], dtype=float)
      result = sma(close, window=1)
      pd.testing.assert_series_equal(result, close.astype(float))

  def test_period_return_matches_formula() -> None:
      close = pd.Series([100.0, 105.0, 110.0, 115.0], dtype=float)
      result = period_return(close, window=2)
      expected = close / close.shift(2) - 1
      pd.testing.assert_series_equal(result, expected)

  def test_period_return_first_window_is_nan() -> None:
      close = pd.Series([100.0, 105.0, 110.0], dtype=float)
      result = period_return(close, window=2)
      assert pd.isna(result.iloc[0])
      assert pd.isna(result.iloc[1])
      assert not pd.isna(result.iloc[2])
  ```

- [ ] **Step 2: Run to verify they fail**

  ```bash
  python3 -m pytest tests/test_rolling_stats.py::test_sma_matches_pandas_rolling_mean -v
  ```
  Expected: ImportError — `sma` not found in `_rolling_stats`.

- [ ] **Step 3: Add `sma` and `period_return` to `_rolling_stats.py`**

  Append to `src/regime_detection/_rolling_stats.py`:

  ```python
  def sma(series: pd.Series, *, window: int) -> pd.Series:
      """Simple moving average with the standard pandas rolling convention.

      Used by trend_direction (§1A: 50d, 200d) and trend_character (§1B: 50d)
      so all three callers share one computation path.
      """
      return series.rolling(window).mean()


  def period_return(series: pd.Series, *, window: int) -> pd.Series:
      """Fractional price return over ``window`` periods: (p_t / p_{t-window}) - 1.

      Used by trend_direction (§1A: 63d) and trend_character (§1B: 63d).
      """
      return series / series.shift(window) - 1
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  python3 -m pytest tests/test_rolling_stats.py -v
  ```
  Expected: 4 new tests pass.

- [ ] **Step 5: Update `trend_direction.compute_features`**

  Add import at top of `trend_direction.py`:
  ```python
  from regime_detection._rolling_stats import sma, period_return
  ```

  Replace lines 58-60:
  ```python
  # BEFORE:
  sma_50 = close.rolling(50).mean()
  sma_200 = close.rolling(200).mean()
  return_63d = close / close.shift(63) - 1

  # AFTER:
  sma_50 = sma(close, window=50)
  sma_200 = sma(close, window=200)
  return_63d = period_return(close, window=63)
  ```

- [ ] **Step 6: Update `trend_character.compute_features`**

  Add import at top of `trend_character.py`:
  ```python
  from regime_detection._rolling_stats import sma, period_return
  ```

  Replace lines 235-238:
  ```python
  # BEFORE:
  sma_50 = close.rolling(50).mean()
  ...
  return_63d = close / close.shift(63) - 1

  # AFTER:
  sma_50 = sma(close, window=50)
  ...
  return_63d = period_return(close, window=63)
  ```

- [ ] **Step 7: Update `feature_store.py` line 292**

  Add import at top of `feature_store.py`:
  ```python
  from regime_detection._rolling_stats import sma
  ```

  Replace line 292:
  ```python
  # BEFORE:
  sma_50 = spy_close.rolling(50).mean()

  # AFTER:
  sma_50 = sma(spy_close, window=50)
  ```

- [ ] **Step 8: Run full affected tests**

  ```bash
  python3 -m pytest tests/test_trend_direction.py tests/test_trend_character.py tests/test_feature_store_refactors.py tests/test_v1_frozen_replay.py tests/test_rolling_stats.py -v
  ```
  Expected: all pass (V1 frozen replay must pass byte-identical — the rolling math is unchanged).

- [ ] **Step 9: Commit**

  ```bash
  git add src/regime_detection/_rolling_stats.py src/regime_detection/trend_direction.py \
    src/regime_detection/trend_character.py src/regime_detection/feature_store.py \
    tests/test_rolling_stats.py
  git commit -m "refactor: consolidate sma/period_return into _rolling_stats, remove 3 duplicate computations"
  ```

---

### Task 13: F001 — Decompose axis_series.py: extract classifier classes as module functions

This is the largest single refactor. The 11 single-method classes become free functions in their respective axis modules.

**Files:**
- Modify: `src/regime_detection/axis_series.py` (remove 11 classes, add 11 module-level `build_*_axis_series` call-throughs)
- Modify: `src/regime_detection/trend_direction.py` (add `build_axis_series`)
- Modify: `src/regime_detection/trend_character.py` (add `build_axis_series`)
- Modify: `src/regime_detection/volatility_state.py` (add `build_axis_series`)
- Modify: `src/regime_detection/breadth_state.py` (add `build_axis_series`)
- Modify: `src/regime_detection/network_fragility_rules.py` (add `build_axis_series`)
- Modify: `src/regime_detection/volume_liquidity_rules.py` (add `build_axis_series`)
- Modify: `src/regime_detection/credit_funding.py` (add `build_axis_series`)
- Modify: `src/regime_detection/inflation_growth.py` (add `build_axis_series`)
- Modify: `src/regime_detection/monetary_pressure.py` (add `build_axis_series`)

Strategy: **copy** the `build()` method body from each class in `axis_series.py` into the corresponding axis module as a free function, then replace the class in `axis_series.py` with a thin delegation call. This approach keeps the diff reviewable and preserves V1 byte-identity (the math doesn't change, only which file it lives in).

- [ ] **Step 1: Write a regression test before touching anything**

  In `tests/test_axis_series.py`, add a golden-output test:

  ```python
  def test_build_axis_series_bundle_labels_are_stable_across_refactor(
      market_df_for_asof,
  ) -> None:
      """Snapshot test: decomposing axis_series.py must not change any label."""
      from datetime import date
      from regime_detection.axis_series import build_axis_series_bundle
      from regime_detection.engine import RegimeEngine
      from regime_detection.feature_store import build_feature_store
      from regime_detection.market_context import build_market_context

      as_of = date(2023, 12, 14)
      context = build_market_context(
          end_date=as_of,
          market_data=market_df_for_asof(as_of),
          config=RegimeEngine().config,
      )
      fs = build_feature_store(context)
      bundle = build_axis_series_bundle(context=context, feature_store=fs)

      # Spot-check core axes on a known date — values confirmed by test_v1_frozen_replay
      td_label = bundle.trend_direction.stable_labels_by_date[as_of]
      tc_label = bundle.trend_character.stable_labels_by_date[as_of]
      vs_label = bundle.volatility_state.stable_labels_by_date[as_of]
      bs_label = bundle.breadth_state.stable_labels_by_date[as_of]

      # All must be valid non-empty strings (not "unknown" for this well-data date)
      assert td_label != ""
      assert tc_label != ""
      assert vs_label != ""
      assert bs_label != ""
      # Store values before the refactor and assert equality after
      # (run once before, record, then assert equality after each move)
      assert td_label in {"bull", "sideways", "bear"}
      assert tc_label in {"trending_up", "trending_down", "recovery_attempt",
                          "choppy", "breakout_expansion", "unknown"}
      assert vs_label in {"low_vol", "normal_vol", "high_vol", "crisis_vol", "unknown"}
      assert bs_label in {"healthy_breadth", "recovery_breadth", "weak_breadth",
                          "divergent_fragile", "unknown"}
  ```

- [ ] **Step 2: Run this test BEFORE any code changes**

  ```bash
  python3 -m pytest tests/test_axis_series.py::test_build_axis_series_bundle_labels_are_stable_across_refactor -v
  ```
  Expected: PASS. Record the actual label values in the test's comments.

- [ ] **Step 3: Move `TrendDirectionSeriesClassifier.build` → `trend_direction.build_axis_series`**

  In `src/regime_detection/trend_direction.py`, add at the bottom:

  ```python
  # Import here to avoid circular: axis_series imports trend_direction at top-level
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from regime_detection.axis_series import AxisSeriesResult
      from regime_detection.feature_store import FeatureStore
      from regime_detection.market_context import MarketContext


  def build_axis_series(
      context: "MarketContext",
      feature_store: "FeatureStore",
  ) -> "AxisSeriesResult":
      """Build the trend-direction AxisSeriesResult for the given context."""
      # (paste the body of TrendDirectionSeriesClassifier.build here verbatim)
      from regime_detection.axis_series import _build_axis_outputs  # local import to avoid circular
      close = context.spy_ohlcv["close"]
      features = feature_store.trend_direction
      trend_v2_features = feature_store.trend_direction_v2
      trend_v2_config = context.config.trend_direction_v2
      trend_v2_rules = trend_v2_config.rules if trend_v2_config is not None else None
      raw_labels, raw_evidence = build_raw_outputs(
          features,
          trend_direction_v2_features=trend_v2_features,
          trend_direction_v2_rules=trend_v2_rules,
      )
      stable_labels, active_labels = apply_hysteresis(
          dates=close.index,
          raw_labels=raw_labels,
          escalation_days=context.config.hysteresis.trend_direction_escalation_days,
          deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
      )
      return _build_axis_outputs(
          dates=close.index.date,
          raw_labels=raw_labels,
          stable_labels=stable_labels,
          active_labels=active_labels,
          raw_evidence=raw_evidence,
          risk_rank=_RISK_RANK,
          deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
          required_inputs=[close],
          required_trading_days=_TREND_DIRECTION_MIN_SESSIONS,  # use constant from axis_series
          max_freshness_days=context.config.data_quality.max_freshness_days,
          min_completeness=context.config.data_quality.min_completeness,
      )
  ```

  In `axis_series.py`, replace `TrendDirectionSeriesClassifier` class with:
  ```python
  from regime_detection.trend_direction import build_axis_series as _build_trend_direction_axis_series
  ```
  And in `build_axis_series_bundle`, replace:
  ```python
  trend_direction = TrendDirectionSeriesClassifier().build(context, feature_store)
  # becomes:
  trend_direction = _build_trend_direction_axis_series(context, feature_store)
  ```

- [ ] **Step 4: Run the regression test after each class moved**

  ```bash
  python3 -m pytest tests/test_axis_series.py -v
  ```
  Expected: PASS after each of the 11 moves. Stop and revert if any fail.

- [ ] **Step 5: Repeat for remaining 10 classifiers**

  Apply the same pattern to each remaining classifier class in `axis_series.py`:
  - `TrendCharacterSeriesClassifier` → `trend_character.build_axis_series`
  - `VolatilitySeriesClassifier` → `volatility_state.build_axis_series`
  - `BreadthSeriesClassifier` → `breadth_state.build_axis_series`
  - `NetworkFragilitySeriesClassifier` → `network_fragility_rules.build_axis_series`
  - `VolumeLiquidityStateSeriesClassifier` → `volume_liquidity_rules.build_axis_series`
  - `CreditFundingSeriesClassifier` → `credit_funding.build_axis_series`
  - `InflationGrowthSeriesClassifier` → `inflation_growth.build_axis_series`
  - `MonetaryPressureV2SeriesClassifier` → `monetary_pressure.build_axis_series`

  Move one at a time, running tests between each.

- [ ] **Step 6: Verify axis_series.py line count dropped**

  ```bash
  wc -l src/regime_detection/axis_series.py
  ```
  Expected: < 400 lines (vs 1,426 before).

- [ ] **Step 7: Run full test suite**

  ```bash
  python3 -m pytest tests/test_axis_series.py tests/test_v1_frozen_replay.py tests/test_breadth_state.py tests/test_trend_direction.py tests/test_volatility_state.py tests/test_credit_funding.py tests/test_inflation_growth.py -v
  ```
  Expected: all pass.

- [ ] **Step 8: Commit**

  ```bash
  git add src/regime_detection/
  git commit -m "refactor: extract axis classifier classes to module functions, shrink axis_series.py"
  ```

---

### Task 14: F026 — Move mid-file import in credit_funding.py to top

**Files:**
- Modify: `src/regime_detection/credit_funding.py:247`

- [ ] **Step 1: Move the import**

  Find line 247: `from regime_detection._rolling_stats import rolling_change_zscore as _change_zscore  # noqa: E402`

  Move this import to the top of the file with the other `from regime_detection` imports. Remove the `# noqa: E402`.

- [ ] **Step 2: Verify**

  ```bash
  ruff check src/regime_detection/credit_funding.py
  python3 -m pytest tests/test_credit_funding.py -v
  ```
  Expected: no E402, all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add src/regime_detection/credit_funding.py
  git commit -m "fix: move mid-file _rolling_stats import to top of credit_funding.py"
  ```

---

## Group E — Test Coverage

### Task 15: F007 — Cover transition_score.py (22% → ≥80%)

**Files:**
- Test: `tests/test_transition_score_v2.py`

- [ ] **Step 1: Add tests for `compute_transition_score`**

  ```python
  import math
  import pytest
  from regime_detection.transition_score import (
      compute_transition_score,
      interpret_transition_score,
      compose_transition_score_for_session,
      ComposedTransitionScore,
  )
  from regime_detection.config import RegimeConfig
  from regime_detection.engine import RegimeEngine

  def _v2_config() -> RegimeConfig:
      """Return the default V2 config which has transition_score configured."""
      return RegimeEngine().config  # uses the default YAML config

  def test_compute_transition_score_simple_weighted_sum() -> None:
      weights = {
          "volatility_acceleration": 0.4,
          "breadth_deterioration": 0.6,
      }
      score = compute_transition_score(
          volatility_acceleration_score=1.0,
          breadth_deterioration_score=0.5,
          correlation_concentration_score=0.0,
          trend_break_score=0.0,
          macro_event_score=0.0,
          weights=weights,
      )
      assert math.isclose(score, 0.4 * 1.0 + 0.6 * 0.5)

  def test_compute_transition_score_rejects_unknown_weight_key() -> None:
      with pytest.raises(ValueError, match="Unknown component"):
          compute_transition_score(
              volatility_acceleration_score=0.5,
              breadth_deterioration_score=0.5,
              correlation_concentration_score=0.0,
              trend_break_score=0.0,
              macro_event_score=0.0,
              weights={"not_a_real_component": 1.0},
          )

  def test_compute_transition_score_rejects_missing_optional_component() -> None:
      with pytest.raises(ValueError, match="no value was provided"):
          compute_transition_score(
              volatility_acceleration_score=0.5,
              breadth_deterioration_score=0.5,
              correlation_concentration_score=0.0,
              trend_break_score=0.0,
              macro_event_score=0.0,
              weights={"hmm_probability_shift": 0.3, "volatility_acceleration": 0.7},
              # hmm_probability_shift_score not supplied → None → should raise
          )

  def test_interpret_transition_score_stable_band() -> None:
      bands = {"weakening": (0.2, 0.4), "transition_warning": (0.4, 0.7), "high": (0.7, 1.0)}
      assert interpret_transition_score(0.1, bands) == "stable"

  def test_interpret_transition_score_weakening_band() -> None:
      bands = {"weakening": (0.2, 0.4), "transition_warning": (0.4, 0.7), "high": (0.7, 1.0)}
      assert interpret_transition_score(0.3, bands) == "weakening"

  def test_interpret_transition_score_high_band() -> None:
      bands = {"weakening": (0.2, 0.4), "transition_warning": (0.4, 0.7), "high": (0.7, 1.0)}
      assert interpret_transition_score(0.85, bands) == "high"

  def test_interpret_transition_score_rejects_out_of_range() -> None:
      bands = {"weakening": (0.2, 0.4), "transition_warning": (0.4, 0.7), "high": (0.7, 1.0)}
      with pytest.raises(ValueError):
          interpret_transition_score(1.5, bands)

  def test_compose_transition_score_returns_none_on_nan_input() -> None:
      config = _v2_config().transition_score
      if config is None:
          pytest.skip("V2 transition_score config not present in default config")
      result = compose_transition_score_for_session(
          realized_vol_short=float("nan"),
          realized_vol_long=0.12,
          pct_above_50dma=0.60,
          avg_pairwise_corr_percentile_504d=0.45,
          drawdown_252d=-0.08,
          event_calendar_label="neutral",
          config=config,
      )
      assert result.score is None
      assert result.interpretation is None
      assert result.components is None

  def test_compose_transition_score_returns_score_in_range() -> None:
      config = _v2_config().transition_score
      if config is None:
          pytest.skip("V2 transition_score config not present in default config")
      result = compose_transition_score_for_session(
          realized_vol_short=0.20,
          realized_vol_long=0.15,
          pct_above_50dma=0.40,
          avg_pairwise_corr_percentile_504d=0.70,
          drawdown_252d=-0.15,
          event_calendar_label="fed_week",
          config=config,
      )
      assert result.score is not None
      assert 0.0 <= result.score <= 1.0
      assert result.interpretation in {"stable", "weakening", "transition_warning", "high"}
  ```

- [ ] **Step 2: Run tests**

  ```bash
  python3 -m pytest tests/test_transition_score_v2.py -v
  ```
  Expected: all pass.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_transition_score_v2.py
  git commit -m "test: cover transition_score.py compute/interpret/compose paths"
  ```

---

### Task 16: F006 — Cover shadow_storage.py (0% → ≥70%)

**Files:**
- Read: `src/regime_detection/shadow_storage.py` (read the full file first)
- Test: `tests/test_shadow_runner.py` or new `tests/test_shadow_storage.py`

- [ ] **Step 1: Read the file**

  ```bash
  cat src/regime_detection/shadow_storage.py
  ```
  Understand the schema: what does it write, what does it read, what are the key classes/functions.

- [ ] **Step 2: Write tests covering write/read round-trip**

  In `tests/test_shadow_storage.py` (new file), using `tmp_path` pytest fixture for an in-memory/temp SQLite path:

  ```python
  import pytest
  from pathlib import Path
  from datetime import date
  # Import whatever the main classes/functions are from shadow_storage
  # (read the file first and fill in exact names)
  from regime_detection.shadow_storage import ShadowStorage  # adjust to actual API

  @pytest.fixture
  def storage(tmp_path: Path):
      return ShadowStorage(db_path=tmp_path / "shadow.db")

  def test_write_and_read_shadow_row(storage) -> None:
      # Fill in based on actual API after reading the file
      ...

  def test_idempotent_write_does_not_duplicate(storage) -> None:
      ...
  ```

  **Note:** Read `src/regime_detection/shadow_storage.py` fully before writing tests. The exact API is in that file.

- [ ] **Step 3: Run coverage check**

  ```bash
  python3 -m pytest tests/test_shadow_storage.py --cov=src/regime_detection/shadow_storage --cov-report=term-missing -v
  ```
  Expected: ≥70% coverage.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_shadow_storage.py
  git commit -m "test: cover shadow_storage.py write/read round-trips (was 0%)"
  ```

---

### Task 17: F005 — Cover loaders.py (13% → ≥60%)

**Files:**
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Read the file**

  ```bash
  cat src/regime_detection/loaders.py
  ```
  Identify what each loader function expects as input (file path, DataFrame format).

- [ ] **Step 2: Write tests for each loader with fixture data**

  Use `tmp_path` to write fixture parquet/SQLite files:

  ```python
  import pytest
  import pandas as pd
  import numpy as np
  from pathlib import Path
  from datetime import date
  # Import the specific loader functions — fill in after reading the file
  from regime_detection.loaders import load_spy_ohlcv  # adjust to actual exports

  @pytest.fixture
  def spy_ohlcv_parquet(tmp_path: Path) -> Path:
      df = pd.DataFrame({
          "date": pd.date_range("2023-01-03", periods=5, freq="B"),
          "open": [380.0, 381.0, 382.0, 379.0, 383.0],
          "high": [385.0, 386.0, 387.0, 384.0, 388.0],
          "low": [378.0, 379.0, 380.0, 377.0, 381.0],
          "close": [382.0, 383.0, 381.0, 382.0, 385.0],
          "volume": [5000000, 4800000, 5200000, 4900000, 5100000],
      })
      path = tmp_path / "spy_ohlcv.parquet"
      df.to_parquet(path)
      return path

  def test_load_spy_ohlcv_returns_expected_columns(spy_ohlcv_parquet: Path) -> None:
      result = load_spy_ohlcv(spy_ohlcv_parquet)
      assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

  def test_load_spy_ohlcv_missing_file_raises() -> None:
      with pytest.raises(FileNotFoundError):
          load_spy_ohlcv(Path("/nonexistent/spy.parquet"))
  ```

  **Note:** Adjust function names based on reading `loaders.py` first.

- [ ] **Step 3: Run coverage**

  ```bash
  python3 -m pytest tests/test_loaders.py --cov=src/regime_detection/loaders --cov-report=term-missing -v
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_loaders.py
  git commit -m "test: cover loaders.py key paths (was 13%)"
  ```

---

### Task 18: F022 + F023 + F027 — Cover network_fragility_rules, inflation_growth, hmm_state

These are grouped because they follow the same pattern: parameterized tests against rule predicates.

**Files:**
- Test: `tests/test_network_fragility_rules.py`
- Test: `tests/test_inflation_growth.py`
- Test: `tests/test_hmm_state.py`

- [ ] **Step 1: network_fragility_rules — add boundary-value tests**

  In `tests/test_network_fragility_rules.py`, add tests for each rule predicate. Read the rule-input dataclasses first:

  ```bash
  grep -n "class.*RuleInputs\|def evaluate_rules" src/regime_detection/network_fragility_rules.py | head -20
  ```

  Then write tests that exercise at-threshold / above / below for each predicate.

- [ ] **Step 2: inflation_growth — add rule evaluation tests**

  Read the evaluate_rules signature:
  ```bash
  grep -n "def evaluate_rules\|def build_rule_inputs" src/regime_detection/inflation_growth.py
  ```

  Write parameterized tests for `hot_inflation`, `stagflation`, `goldilocks`, `deflation_risk` etc. labels with boundary inputs.

- [ ] **Step 3: hmm_state — add model-fitting test**

  ```python
  import numpy as np
  import pandas as pd
  from regime_detection.hmm_state import compute_hmm_features  # adjust to actual API

  def test_hmm_state_output_probabilities_sum_to_one() -> None:
      rng = np.random.default_rng(42)
      close = pd.Series(
          100.0 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
          index=pd.date_range("2020-01-02", periods=300, freq="B"),
      )
      result = compute_hmm_features(close)  # adjust to actual function name
      assert result is not None
      # State probabilities should sum to approximately 1.0 per row
      # (adjust assertions based on what the function actually returns)
  ```

- [ ] **Step 4: Run all three**

  ```bash
  python3 -m pytest tests/test_network_fragility_rules.py tests/test_inflation_growth.py tests/test_hmm_state.py -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_network_fragility_rules.py tests/test_inflation_growth.py tests/test_hmm_state.py
  git commit -m "test: add boundary-value coverage for network_fragility_rules, inflation_growth, hmm_state"
  ```

---

## Group F — Large File Splits

### Task 19: F010 — Split config.py into config_v1.py and config_v2.py

**Files:**
- Create: `src/regime_detection/config_v1.py`
- Create: `src/regime_detection/config_v2.py`
- Modify: `src/regime_detection/config.py` (becomes thin re-export shim)

- [ ] **Step 1: Write a smoke test before any move**

  ```python
  def test_config_module_exports_unchanged_after_split() -> None:
      from regime_detection.config import (
          RegimeConfig, HysteresisConfig, DataQualityConfig,
          EventCalendarConfig, load_config,
      )
      assert RegimeConfig is not None
      assert HysteresisConfig is not None
  ```

- [ ] **Step 2: Identify which classes are V1-only vs V2-only**

  ```bash
  grep -n "^class " src/regime_detection/config.py
  ```

  V1 classes (present in `core3-v1.0.0`): `HysteresisConfig`, `DataQualityConfig`, `EventCalendarConfig`, `ETFProxyConfig`, `MonthlyOptionsExpiryRuleConfig`, `ExpiryRulesConfig`, `EarningsSeasonConfig`, `RegimeConfig` (the root).

  V2 classes: any class with `_v2` in the name or that appears only in V2 spec sections (`TrendDirectionV2Config`, `VolatilityStateV2Config`, `BreadthStateV2Config`, `NetworkFragilityRulesConfig`, `CreditFundingRulesConfig`, `InflationGrowthConfig`, `MonetaryPressureV2Config`, `TransitionScoreConfig`, `VolumeAndLiquidityV2Config`).

- [ ] **Step 3: Create `config_v1.py` with V1 classes**

  Move the V1 classes verbatim. Keep all imports they need at the top.

- [ ] **Step 4: Create `config_v2.py` with V2 classes**

  Move V2 classes. Import V1 classes they depend on from `config_v1`.

- [ ] **Step 5: Make `config.py` a re-export shim**

  ```python
  # config.py — backwards-compatible re-export shim
  from regime_detection.config_v1 import *  # noqa: F401,F403
  from regime_detection.config_v2 import *  # noqa: F401,F403
  ```

  Or enumerate explicit re-exports to avoid `*` imports if ruff is configured to reject them.

- [ ] **Step 6: Run full suite**

  ```bash
  python3 -m pytest tests/test_v2_config.py tests/test_v1_frozen_replay.py tests/ -v --tb=short
  ```
  Expected: all pass.

- [ ] **Step 7: Verify config.py shrank**

  ```bash
  wc -l src/regime_detection/config.py
  ```
  Expected: < 30 lines (shim only).

- [ ] **Step 8: Commit**

  ```bash
  git add src/regime_detection/config_v1.py src/regime_detection/config_v2.py src/regime_detection/config.py
  git commit -m "refactor: split config.py into config_v1.py + config_v2.py, keep shim for backwards compat"
  ```

---

### Task 20: F008, F009, F011 — Split aggregate_eps.py, event_calendar.py, acquisition_consolidation.py

These three follow the same extract-and-re-export pattern. Handle them one at a time in separate commits.

**For each file:**

1. Write a smoke-import test confirming public API is unchanged.
2. Identify the extraction boundary (e.g., parse helpers vs fetch orchestration).
3. Move the sub-module. Keep re-exports in the original file.
4. Run all tests, commit.

**`aggregate_eps.py` (1,102 lines):** Split into:
- `aggregate_eps_parse.py` — cell/row parsing helpers (`_parse_cell_value`, `_parse_row_for_snapshot`, `_parse_row_legacy`, `_parse_workbook`, `_extract_*`)
- `aggregate_eps_wayback.py` — Wayback-specific logic (`seed_weekly_history_from_wayback_timeline`, `run_wayback_aggregate_eps_fetch`, `_filter_wayback_snapshots`, etc.)
- `aggregate_eps.py` — retains download, `run_aggregate_eps_fetch`, `append_weekly_eps_snapshot`, and re-exports

**`event_calendar.py` (1,129 lines):** Split into:
- `event_calendar_fetch.py` — `_fetch_fomc_events`, `_fetch_bls_events`
- `event_calendar_group_ab.py` — `_build_v2_curated_candidate_events`, `_group_a_text_fetcher_from_legacy_map`, `_write_group_a_artifacts`, `_record_group_a_output`
- `event_calendar.py` — retains `run_us_event_calendar_fetch`, `resolve_event_label`, `validate_fomc_listing_integrity`, re-exports

**`acquisition_consolidation.py` (871 lines):** Split into:
- `acquisition_consolidation_db.py` — DB-merge helpers
- `acquisition_consolidation.py` — main orchestration + re-exports

Commit each split separately.

---

## Self-Review Checklist

- [x] **F001** (axis_series.py decompose) — Task 13 ✓
- [x] **F002** (sma/return_63d consolidation) — Task 12 ✓
- [x] **F003** (evidence dict) — partial: Task 11 addresses the transition_score use case; F003 full (all 11 output models) is a long-term task not in this plan since it requires touching every axis module without a single transaction
- [x] **F004** (TransitionScoreInputs) — Task 11 ✓
- [x] **F005** (loaders coverage) — Task 17 ✓
- [x] **F006** (shadow_storage coverage) — Task 16 ✓
- [x] **F007** (transition_score coverage) — Task 15 ✓
- [x] **F008** (aggregate_eps split) — Task 20 ✓
- [x] **F009** (event_calendar split) — Task 20 ✓
- [x] **F010** (config split) — Task 19 ✓
- [x] **F011** (acquisition_consolidation split) — Task 20 ✓
- [x] **F012** (logger naming) — Task 7 ✓
- [x] **F013** (hf_central_bank logging) — Task 2 ✓
- [x] **F014** (acquisition_consolidation logging) — Task 3 ✓
- [x] **F015** (gpr_gdelt logging) — Task 4 ✓
- [x] **F016** (CI gates) — Task 9 ✓
- [x] **F017** (list comprehension) — Task 5 ✓
- [x] **F018** (magic numbers) — Task 8 ✓
- [x] **F019** (E402 scripts) — Task 10 ✓
- [x] **F020** (f-strings) — Task 1 ✓
- [x] **F021** (README) — Task 6 ✓
- [x] **F022** (network_fragility_rules coverage) — Task 18 ✓
- [x] **F023** (inflation_growth coverage) — Task 18 ✓
- [x] **F024** (Alpaca env var) — not scheduled; simple KeyError is acceptable per AGENTS.md fail-loud rule
- [x] **F025** (breadth_state_v2 layering) — not scheduled; requires threading through MarketContext, deferred to dedicated PR
- [x] **F026** (credit_funding mid-file import) — Task 14 ✓
- [x] **F027** (hmm_state coverage) — Task 18 ✓
- [x] **F028** (models.py TODO) — not scheduled; evidence dict requires axis-by-axis migration, ongoing work
- [x] **F029** (triple-dict loop) — not scheduled; profiling showed no significance
- [x] **F030** (ConsolidationSummary) — not scheduled; out of scope for current V2 slice work
