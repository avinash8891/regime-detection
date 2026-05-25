---
name: repo-validation
description: Run the repository's required static analysis and test commands before claiming a change is complete.
---

# Repo validation skill

Use this skill when changing Python code, CI config, or runtime behavior in this repository.

## Required checks

1. Run `python3 -m black --check src tests scripts`.
2. Run `python3 -m ruff check .`.
3. Run `python3 -m pyright`.
4. Run `python3 -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=80`.

## Notes

- Prefer `pip install -r requirements-dev.txt && pip install -e . --no-deps` for local setup.
- Treat `tests/test_v2_gate_scripts.py` and slow tests as follow-up validation when touching V2 or gate paths.
- Do not claim success without pasting command output or exit codes.
