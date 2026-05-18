from __future__ import annotations

import os
from pathlib import Path

import pytest

from regime_data_fetch.cli_common import load_operator_env_files


def test_load_operator_env_files_loads_repo_pointer_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpaca_env = tmp_path / "market-data-hub.env"
    fred_env = tmp_path / "signal-engine.env"
    pointer_file = tmp_path / ".regime-operator.env"
    alpaca_env.write_text(
        "ALPACA_API_KEY_ID=alpaca-key\n"
        "ALPACA_API_SECRET_KEY=alpaca-secret\n",
        encoding="utf-8",
    )
    fred_env.write_text("FRED_API_KEY=fred-key\n", encoding="utf-8")
    pointer_file.write_text(
        f"REGIME_ENV_FILES={alpaca_env}:{fred_env}\n",
        encoding="utf-8",
    )
    for key in (
        "REGIME_OPERATOR_ENV_FILE",
        "REGIME_ENV_FILES",
        "REGIME_ALPACA_ENV",
        "REGIME_FRED_ENV",
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
    for key in (
        "REGIME_OPERATOR_ENV_FILE",
        "REGIME_ENV_FILES",
        "REGIME_TINYFISH_ENV",
        "TINYFISH_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    loaded = load_operator_env_files(repo_root=tmp_path)

    assert loaded == [pointer_file, tinyfish_env]
    assert os.environ["TINYFISH_API_KEY"] == "tinyfish-key"


def test_load_operator_env_files_requires_explicit_pointer_to_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.env"
    monkeypatch.delenv("REGIME_OPERATOR_ENV_FILE", raising=False)

    with pytest.raises(SystemExit, match="operator env pointer file not found"):
        load_operator_env_files(repo_root=tmp_path, explicit_path=missing)
