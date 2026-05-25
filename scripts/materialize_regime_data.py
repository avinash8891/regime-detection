#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.cli_common import (
    OPERATOR_ENV_POINTER_FILE,
    load_operator_env_files,
)
from regime_data_fetch.materialization import materialize_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize regime data artifacts from a manifest."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Manifest YAML path."
    )
    parser.add_argument(
        "--local-root",
        required=True,
        type=Path,
        help="Local data/raw root to materialize into.",
    )
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        type=Path,
        help="Repo root for repo-relative manifest paths.",
    )
    parser.add_argument(
        "--store-root", default=None, help="Override manifest storage_root."
    )
    parser.add_argument(
        "--required-for",
        default=None,
        help="Only materialize artifacts required for this use case.",
    )
    parser.add_argument(
        "--operator-env-file",
        default=None,
        type=Path,
        help=(
            "Optional non-secret pointer file listing repo credential env files. "
            f"Defaults to {OPERATOR_ENV_POINTER_FILE} or ~/.config/regime-detection/operator.env."
        ),
    )
    args = parser.parse_args(argv)

    load_operator_env_files(repo_root=REPO_ROOT, explicit_path=args.operator_env_file)
    materialized = materialize_manifest(
        manifest_path=args.manifest,
        local_root=args.local_root,
        repo_root=args.repo_root,
        store_root=args.store_root,
        required_for=args.required_for,
    )
    for artifact in materialized:
        print(f"{artifact.name}\t{artifact.sha256}\t{artifact.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
