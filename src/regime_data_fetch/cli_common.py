from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def load_env_file(path: Path) -> None:
    """Minimal dotenv loader: KEY=VALUE lines, optional quotes, no overrides."""
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or os.environ.get(key, "").strip():
            continue
        os.environ[key] = value
