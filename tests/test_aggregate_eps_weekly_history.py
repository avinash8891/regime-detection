from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.aggregate_eps import (
    EPS_DIR_NAME,
    EPS_REVISION_LOOKBACK_WEEKS,
    SOURCE_NAME,
    WAYBACK_DIR_NAME,
    WAYBACK_TIMELINE_FILENAME,
    WEEKLY_HISTORY_FILENAME,
    AggregateEPSFetchError,
    AggregateEPSSnapshot,
    append_weekly_eps_snapshot,
    compute_eps_revision_direction_4w,
    seed_weekly_history_from_wayback_timeline,
)

FIXTURES = Path("tests/fixtures/raw/eps")


def _eps_snapshot(
    observation_date: dt.date, forward_eps: float
) -> AggregateEPSSnapshot:
    """Build a realistic AggregateEPSSnapshot with the two fields the
    weekly accumulator consumes populated; the rest left at None (the
    accumulator only reads observation_date / observation_label /
    forward_estimate_value)."""
    return AggregateEPSSnapshot(
        observation_date=observation_date,
        observation_label="current",
        forward_estimate_label="2026E",
        forward_estimate_value=forward_eps,
        estimate_2025e=None,
        estimate_q4_2025e=None,
        estimate_2026e=forward_eps,
        price=None,
        pe_2025e=None,
        pe_2026e=None,
        change_vs_prior_observation_2025e=None,
        change_vs_prior_observation_q4_2025e=None,
        change_vs_prior_observation_2026e=None,
        change_vs_prior_observation_price=None,
        change_vs_prior_observation_pe_2025e=None,
        change_vs_prior_observation_pe_2026e=None,
    )


def _wayback_timeline_df(rows: list[tuple[dt.date, float | None]]) -> pd.DataFrame:
    """Build a synthetic Wayback EPS timeline frame shaped like the parquet
    run_wayback_aggregate_eps_fetch materialises. The seeding bridge reads
    only workbook_as_of_date + forward_estimate_value; the rest is realistic
    filler so the fixture matches the real timeline schema."""
    return pd.DataFrame(
        [
            {
                "snapshot_date": obs_date,
                "timestamp": obs_date.strftime("%Y%m%d000000"),
                "archive_url": (
                    f"https://web.archive.org/web/{obs_date:%Y%m%d}000000/"
                    "https://www.spglobal.com/spdji/en/documents/"
                    "additional-material/sp-500-eps-est.xlsx"
                ),
                "workbook_as_of_date": obs_date,
                "forward_estimate_label": "2026E",
                "forward_estimate_value": fwd,
                "source": "wayback_machine",
            }
            for obs_date, fwd in rows
        ]
    )


def _write_wayback_timeline(tmp_path: Path, df: pd.DataFrame) -> Path:
    wayback_dir = tmp_path / WAYBACK_DIR_NAME
    wayback_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = wayback_dir / WAYBACK_TIMELINE_FILENAME
    df.to_parquet(timeline_path, index=False)
    return timeline_path


def test_seed_weekly_history_creates_accumulator_from_timeline(
    tmp_path: Path,
) -> None:
    """With no existing accumulator, the seed bridges the Wayback timeline
    straight into sp500_eps_weekly_history.parquet — one accumulator row per
    timeline row, keyed by workbook_as_of_date, sorted ascending."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 14), 272.50),
                (dt.date(2026, 1, 21), 273.10),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [
        dt.date(2026, 1, 7),
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 21),
    ]
    assert list(combined["forward_estimate_value"]) == [271.00, 272.50, 273.10]
    assert set(combined["observation_label"]) == {"wayback_backfill"}
    assert set(combined["source"]) == {"wayback_machine"}

    on_disk = pd.read_parquet(tmp_path / EPS_DIR_NAME / WEEKLY_HISTORY_FILENAME)
    assert len(on_disk) == 3


def test_seed_weekly_history_existing_live_rows_win_on_collision(
    tmp_path: Path,
) -> None:
    """A live run_aggregate_eps_fetch row is authoritative — on an
    observation_date collision the existing accumulator row is kept, not the
    Wayback-archived snapshot for the same date."""
    eps_dir = tmp_path / EPS_DIR_NAME
    eps_dir.mkdir(parents=True)
    # A live fetch already recorded 2026-01-14 with its authoritative value.
    append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 14), 272.50),
    )
    # The Wayback timeline carries a stale/different value for that date
    # plus two dates the accumulator doesn't have yet.
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 14), 999.99),  # collides with the live row
                (dt.date(2026, 1, 21), 273.10),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [
        dt.date(2026, 1, 7),
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 21),
    ]
    by_date = combined.set_index("observation_date")
    # The live row's value survived; the colliding Wayback value was dropped.
    assert by_date.loc[dt.date(2026, 1, 14), "forward_estimate_value"] == 272.50
    assert by_date.loc[dt.date(2026, 1, 14), "source"] == SOURCE_NAME
    # The two non-colliding Wayback dates were seeded in.
    assert by_date.loc[dt.date(2026, 1, 7), "source"] == "wayback_machine"


def test_seed_weekly_history_dedupes_timeline_keeping_last(
    tmp_path: Path,
) -> None:
    """Multiple Wayback snapshots can share one workbook_as_of_date — the
    last (freshest capture, timeline is snapshot-date sorted) is kept."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 7), 271.85),  # later capture, same workbook date
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [dt.date(2026, 1, 7)]
    assert list(combined["forward_estimate_value"]) == [271.85]


def test_seed_weekly_history_raises_when_timeline_missing(
    tmp_path: Path,
) -> None:
    """No Wayback timeline parquet → loud failure routing the operator to
    run the backfill first, not a silent empty seed."""
    with pytest.raises(AggregateEPSFetchError, match="No Wayback EPS timeline"):
        seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)


def test_seed_weekly_history_collapses_earnings_cold_start(
    tmp_path: Path,
) -> None:
    """The point of the bridge: a one-time Wayback backfill + seed pre-fills
    the accumulator past EPS_REVISION_LOOKBACK_WEEKS so
    compute_eps_revision_direction_4w is non-NaN immediately — no waiting for
    >4 live weekly fetches."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 270.00),
                (dt.date(2026, 1, 14), 271.00),
                (dt.date(2026, 1, 21), 272.00),
                (dt.date(2026, 1, 28), 273.00),
                (dt.date(2026, 2, 4), 277.20),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)
    revision = compute_eps_revision_direction_4w(combined)

    # Cold-start rows stay NaN; the 5th row unlocks immediately post-seed.
    assert revision.iloc[:EPS_REVISION_LOOKBACK_WEEKS].isna().all()
    # (277.20 - 270.00) / 270.00 == 0.0266...
    assert revision.iloc[4] == pytest.approx((277.20 - 270.00) / 270.00)
