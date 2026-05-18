from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts._v2_calibration_helpers import default_pmi_path, load_macro_series


def test_default_pmi_path_uses_history_parquet(tmp_path: Path) -> None:
    assert default_pmi_path(tmp_path) == tmp_path / "pmi" / "us_ism_pmi_history.parquet"


def test_load_macro_series_merges_pmi_history_with_latest_parquet(
    tmp_path: Path,
) -> None:
    macro_path = tmp_path / "macro" / "fred_macro_series.parquet"
    macro_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "value": 1.0,
                "series_id": "DGS10",
                "logical_name": "10y_yield",
            }
        ]
    ).to_parquet(macro_path, index=False)
    pmi_dir = tmp_path / "pmi"
    pmi_dir.mkdir()
    latest_path = pmi_dir / "us_ism_pmi.parquet"
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-04",
                "value": 52.7,
                "release_timestamp": "2026-05-01T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/latest",
            }
        ]
    ).to_parquet(latest_path, index=False)
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-03",
                "value": 50.3,
                "release_timestamp": "2026-04-01T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/history",
            },
            {
                "series_name": "services",
                "period": "2026-03",
                "value": 53.0,
                "release_timestamp": "2026-04-03T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/history",
            },
        ]
    ).to_parquet(pmi_dir / "us_ism_pmi_history.parquet", index=False)

    series = load_macro_series(
        macro_path,
        latest_path,
        cpi_nowcast_parquet=None,
        eps_weekly_history_parquet=None,
    )

    pmi = series["pmi_manufacturing"]
    assert list(pmi.index) == [
        pd.Timestamp("2026-04-01"),
        pd.Timestamp("2026-05-01"),
    ]
    assert list(pmi) == [50.3, 52.7]
