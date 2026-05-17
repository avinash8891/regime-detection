from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


OPERATOR_ENV_POINTER_FILE = ".regime-operator.env"
HOME_OPERATOR_ENV_POINTER_FILE = Path(".config") / "regime-detection" / "operator.env"
OPERATOR_ENV_FILE_LIST_VAR = "REGIME_ENV_FILES"
OPERATOR_ENV_POINTER_FILE_VAR = "REGIME_OPERATOR_ENV_FILE"
OPERATOR_ENV_POINTER_VARS = (
    "REGIME_ALPACA_ENV",
    "REGIME_FRED_ENV",
    "REGIME_TINYFISH_ENV",
    "REGIME_ACLED_ENV",
    "REGIME_UCDP_ENV",
    "REGIME_HDX_ENV",
    "REGIME_INVESTING_ENV",
)


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def load_env_file(path: Path) -> None:
    """Minimal dotenv loader: KEY=VALUE lines, optional quotes, no overrides."""
    expanded = path.expanduser()
    if not expanded.exists():
        raise SystemExit(f"env file not found: {path}")

    for raw_line in expanded.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or os.environ.get(key, "").strip():
            continue
        os.environ[key] = value


def _split_env_file_list(value: str) -> list[Path]:
    paths: list[Path] = []
    for chunk in value.replace(",", os.pathsep).split(os.pathsep):
        candidate = chunk.strip()
        if candidate:
            paths.append(Path(candidate).expanduser())
    return paths


def _default_operator_env_pointer(repo_root: Path) -> Path | None:
    configured = os.environ.get(OPERATOR_ENV_POINTER_FILE_VAR, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise SystemExit(f"operator env pointer file not found: {path}")
        return path

    repo_pointer = repo_root / OPERATOR_ENV_POINTER_FILE
    if repo_pointer.exists():
        return repo_pointer

    home_pointer = Path.home() / HOME_OPERATOR_ENV_POINTER_FILE
    if home_pointer.exists():
        return home_pointer
    return None


def load_operator_env_files(
    *, repo_root: Path, explicit_path: Path | None = None
) -> list[Path]:
    """Load repo operator env pointers, then each referenced secret env file.

    The pointer file is safe to keep outside Git or in a gitignored repo-local
    file because it stores file paths, not secret values. Secret env files are
    loaded with `load_env_file`, which never overwrites values already present
    in the process environment.
    """
    pointer_path = explicit_path.expanduser() if explicit_path else None
    if pointer_path is not None and not pointer_path.exists():
        raise SystemExit(f"operator env pointer file not found: {pointer_path}")
    if pointer_path is None:
        pointer_path = _default_operator_env_pointer(repo_root)
    if pointer_path is None:
        return []

    load_env_file(pointer_path)
    loaded = [pointer_path]

    target_paths = _split_env_file_list(os.environ.get(OPERATOR_ENV_FILE_LIST_VAR, ""))
    for pointer_var in OPERATOR_ENV_POINTER_VARS:
        configured = os.environ.get(pointer_var, "").strip()
        if configured:
            target_paths.append(Path(configured).expanduser())

    seen: set[Path] = set()
    for target_path in target_paths:
        resolved = target_path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(resolved)
        loaded.append(resolved)
    return loaded
