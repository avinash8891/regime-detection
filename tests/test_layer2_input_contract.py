from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.layer2_input_contract import validate_layer2_incremental_inputs


def _write_valid_layer2_inputs(data_root: Path) -> None:
    pmi_dir = data_root / "pmi"
    pmi_dir.mkdir(parents=True)
    pmi_rows = []
    for month in range(1, 13):
        for series_name in ("manufacturing", "services"):
            pmi_rows.append(
                {
                    "series_name": series_name,
                    "period": f"2025-{month:02d}",
                    "release_timestamp": f"2025-{month:02d}-05T15:00:00Z",
                    "value": 50.0 + month,
                    "source": "investing_manual",
                    "source_url": "manual://investing",
                }
            )
    for series_name in ("manufacturing", "services"):
        pmi_rows.append(
            {
                "series_name": series_name,
                "period": "2026-04",
                "release_timestamp": "2026-05-05T14:00:00Z",
                "value": 52.0,
                "source": "tradingeconomics",
                "source_url": "https://tradingeconomics.example/pmi",
            }
        )
    pd.DataFrame(pmi_rows).to_parquet(pmi_dir / "us_ism_pmi_history.parquet", index=False)

    nowcast_dir = data_root / "cleveland_fed_nowcast"
    nowcast_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": [dt.date(2026, 5, 14)],
            "cpi_nowcast": [0.27],
        }
    ).to_parquet(nowcast_dir / "cpi_nowcast.parquet", index=False)

    eps_dir = data_root / "aggregate_forward_eps"
    eps_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation_date": [
                dt.date(2026, 4, 3),
                dt.date(2026, 4, 10),
                dt.date(2026, 4, 17),
                dt.date(2026, 4, 24),
                dt.date(2026, 5, 1),
                dt.date(2026, 5, 8),
                dt.date(2026, 5, 15),
            ],
            "forward_estimate_value": [270.0, 271.0, 272.0, 273.0, 277.4, 278.0, 279.0],
        }
    ).to_parquet(eps_dir / "sp500_eps_weekly_history.parquet", index=False)


def test_layer2_input_contract_accepts_complete_fresh_inputs(tmp_path: Path) -> None:
    _write_valid_layer2_inputs(tmp_path)

    validate_layer2_incremental_inputs(
        data_root=tmp_path,
        as_of_date=dt.date(2026, 5, 15),
    )


def test_layer2_input_contract_rejects_latest_only_pmi_history(tmp_path: Path) -> None:
    _write_valid_layer2_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-04",
                "release_timestamp": "2026-05-05T14:00:00Z",
                "value": 52.0,
            },
            {
                "series_name": "services",
                "period": "2026-04",
                "release_timestamp": "2026-05-05T14:00:00Z",
                "value": 53.0,
            },
        ]
    ).to_parquet(tmp_path / "pmi" / "us_ism_pmi_history.parquet", index=False)

    with pytest.raises(ValueError, match="PMI history has 2 rows"):
        validate_layer2_incremental_inputs(
            data_root=tmp_path,
            as_of_date=dt.date(2026, 5, 15),
        )


def test_layer2_input_contract_rejects_stale_nowcast(tmp_path: Path) -> None:
    _write_valid_layer2_inputs(tmp_path)
    pd.DataFrame(
        {
            "date": [dt.date(2026, 4, 1)],
            "cpi_nowcast": [0.22],
        }
    ).to_parquet(tmp_path / "cleveland_fed_nowcast" / "cpi_nowcast.parquet", index=False)

    with pytest.raises(ValueError, match="CPI nowcast stale"):
        validate_layer2_incremental_inputs(
            data_root=tmp_path,
            as_of_date=dt.date(2026, 5, 15),
        )


def test_layer2_input_contract_rejects_eps_without_revision_signal(tmp_path: Path) -> None:
    _write_valid_layer2_inputs(tmp_path)
    pd.DataFrame(
        {
            "observation_date": [
                dt.date(2026, 5, 1),
                dt.date(2026, 5, 8),
            ],
            "forward_estimate_value": [270.0, 270.1],
        }
    ).to_parquet(
        tmp_path / "aggregate_forward_eps" / "sp500_eps_weekly_history.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="EPS weekly history has 2 rows"):
        validate_layer2_incremental_inputs(
            data_root=tmp_path,
            as_of_date=dt.date(2026, 5, 15),
        )
