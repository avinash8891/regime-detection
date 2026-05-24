from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from regime_data_fetch.aggregate_eps_wayback import (
    append_wayback_status,
    filter_wayback_snapshots,
    parse_wayback_cdx_json,
)
from regime_data_fetch.aggregate_eps_models import (
    AggregateEPSFetchError,
    EPSWaybackSnapshot,
)


TARGET_URL = (
    "https://www.spglobal.com/spdji/en/documents/additional-material/"
    "sp-500-eps-est.xlsx"
)


def _snapshot(timestamp: str) -> EPSWaybackSnapshot:
    return EPSWaybackSnapshot(
        timestamp=timestamp,
        archive_url=f"https://web.archive.org/web/{timestamp}if_/{TARGET_URL}",
        snapshot_date=dt.datetime.strptime(timestamp[:8], "%Y%m%d").date(),
    )


def test_parse_wayback_cdx_json_filters_to_successful_workbook_snapshots() -> None:
    cdx_json = json.dumps(
        [
            ["timestamp", "original", "statuscode", "mimetype"],
            [
                "20200110123456",
                TARGET_URL,
                "200",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ],
            [
                "20200111123456",
                TARGET_URL,
                "404",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ],
            [
                "20200112123456",
                "https://example.test/other.xlsx",
                "200",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ],
            ["20200113123456", TARGET_URL, "200", "text/html"],
            ["20200114123456", TARGET_URL, "200", "application/vnd.ms-excel"],
        ]
    )

    assert parse_wayback_cdx_json(cdx_json, target_url=TARGET_URL) == [
        _snapshot("20200110123456"),
        _snapshot("20200114123456"),
    ]


@pytest.mark.parametrize(
    ("cdx_json", "match"),
    [
        ("not-json", "Wayback CDX response was not valid JSON"),
        (json.dumps([]), "Wayback CDX response contained no rows"),
        (
            json.dumps([["timestamp", "original"]]),
            "Unexpected Wayback CDX header",
        ),
        (
            json.dumps(
                [
                    ["timestamp", "original", "statuscode", "mimetype"],
                    ["20200110123456", TARGET_URL, "200"],
                ]
            ),
            "Wayback CDX row 2 had unexpected shape",
        ),
        (
            json.dumps(
                [
                    ["timestamp", "original", "statuscode", "mimetype"],
                    ["20200110123456", TARGET_URL, "200", "text/html"],
                ]
            ),
            "Wayback CDX response contained no usable workbook snapshots",
        ),
    ],
)
def test_parse_wayback_cdx_json_rejects_unusable_source_payloads(
    cdx_json: str,
    match: str,
) -> None:
    with pytest.raises(AggregateEPSFetchError, match=match):
        parse_wayback_cdx_json(cdx_json, target_url=TARGET_URL)


def test_filter_wayback_snapshots_sorts_bounds_and_caps_snapshots() -> None:
    snapshots = [
        _snapshot("20200320101010"),
        _snapshot("20200110123456"),
        _snapshot("20200214101010"),
        _snapshot("20200214090909"),
    ]

    assert filter_wayback_snapshots(
        snapshots,
        from_date=dt.date(2020, 2, 1),
        to_date=dt.date(2020, 3, 31),
        max_snapshots=2,
    ) == [
        _snapshot("20200214090909"),
        _snapshot("20200214101010"),
    ]


def test_append_wayback_status_writes_jsonl_records(tmp_path: Path) -> None:
    status_path = tmp_path / "aggregate_forward_eps_wayback" / "snapshot_status.jsonl"
    status_path.parent.mkdir(parents=True)
    snapshot = _snapshot("20200110123456")

    append_wayback_status(
        status_path,
        snapshot=snapshot,
        status="failed",
        detail="URLError: archive missing",
    )
    append_wayback_status(
        status_path,
        snapshot=_snapshot("20200214101010"),
        status="parsed_ok",
        detail="downloaded",
    )

    assert [json.loads(line) for line in status_path.read_text().splitlines()] == [
        {
            "snapshot_date": "2020-01-10",
            "timestamp": "20200110123456",
            "archive_url": snapshot.archive_url,
            "status": "failed",
            "detail": "URLError: archive missing",
        },
        {
            "snapshot_date": "2020-02-14",
            "timestamp": "20200214101010",
            "archive_url": _snapshot("20200214101010").archive_url,
            "status": "parsed_ok",
            "detail": "downloaded",
        },
    ]
