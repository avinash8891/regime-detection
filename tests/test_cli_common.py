from __future__ import annotations

import os
from pathlib import Path

import pytest

from regime_data_fetch.cli_common import (
    OPERATOR_ENV_FILE_LIST_VAR,
    OPERATOR_ENV_POINTER_FILE_VAR,
    OPERATOR_ENV_POINTER_VARS,
    load_operator_env_files,
)


@pytest.fixture(autouse=True)
def _isolate_operator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic isolation for cli_common tests.

    Without this fixture, these tests were order-dependent: any earlier
    test on the same xdist worker that ran (directly or via subprocess
    fallback path) ``load_operator_env_files`` against the developer's
    home pointer ``~/.config/regime-detection/operator.env`` would have
    left provider env pointers such as ``REGIME_INVESTING_ENV`` /
    ``REGIME_TINYFISH_ENV`` set in ``os.environ``. The original tests only
    delenv'd the vars they personally expected, missing the leaked ones, so
    assertions like ``loaded == [pointer_file, tinyfish_env]`` failed when
    the function additionally followed those leaked pointers.

    Surfaced when experimenting with ``--dist=loadfile`` (different
    worker grouping → different chance of contamination) — bug existed
    before; the default ``--dist=load`` happened to mask it. Importing
    ``OPERATOR_ENV_POINTER_VARS`` from production keeps this fixture
    correct when new provider pointers are added.
    """
    for key in (
        OPERATOR_ENV_POINTER_FILE_VAR,
        OPERATOR_ENV_FILE_LIST_VAR,
        *OPERATOR_ENV_POINTER_VARS,
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_operator_env_files_loads_repo_pointer_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpaca_env = tmp_path / "market-data-hub.env"
    fred_env = tmp_path / "signal-engine.env"
    pointer_file = tmp_path / ".regime-operator.env"
    alpaca_env.write_text(
        "ALPACA_API_KEY_ID=alpaca-key\n" "ALPACA_API_SECRET_KEY=alpaca-secret\n",
        encoding="utf-8",
    )
    fred_env.write_text("FRED_API_KEY=fred-key\n", encoding="utf-8")
    pointer_file.write_text(
        f"REGIME_ENV_FILES={alpaca_env}:{fred_env}\n",
        encoding="utf-8",
    )
    # _isolate_operator_env clears the REGIME_* pointers. Still clear the
    # downstream API-key vars these tests assert on, since load_env_file
    # mutates os.environ directly (outside monkeypatch's tracking) and
    # earlier tests may have set them.
    for key in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "FRED_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    loaded = load_operator_env_files(repo_root=tmp_path)

    assert loaded == [pointer_file, alpaca_env, fred_env]
    assert os.environ["ALPACA_API_KEY_ID"] == "alpaca-key"
    assert os.environ["ALPACA_API_SECRET_KEY"] == "alpaca-secret"
    assert os.environ["FRED_API_KEY"] == "fred-key"


def test_load_operator_env_files_supports_named_provider_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tinyfish_env = tmp_path / "tinyfish.env"
    pointer_file = tmp_path / ".regime-operator.env"
    tinyfish_env.write_text("TINYFISH_API_KEY=tinyfish-key\n", encoding="utf-8")
    pointer_file.write_text(f"REGIME_TINYFISH_ENV={tinyfish_env}\n", encoding="utf-8")
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)

    loaded = load_operator_env_files(repo_root=tmp_path)

    assert loaded == [pointer_file, tinyfish_env]
    assert os.environ["TINYFISH_API_KEY"] == "tinyfish-key"


def test_load_operator_env_files_requires_explicit_pointer_to_exist(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.env"

    with pytest.raises(SystemExit, match="operator env pointer file not found"):
        load_operator_env_files(repo_root=tmp_path, explicit_path=missing)
