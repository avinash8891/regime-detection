#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
from xml.etree.ElementTree import Element

from defusedxml import ElementTree


def _nodeid_for(testcase: Element) -> str:
    classname = testcase.attrib.get("classname", "").strip()
    name = testcase.attrib.get("name", "").strip()
    if classname:
        return f"{classname}::{name}"
    return name


def _status_for(testcase: Element) -> str:
    if testcase.find("failure") is not None:
        return "failed"
    if testcase.find("error") is not None:
        return "error"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def collect_test_history(report_paths: list[Path]) -> dict[str, list[str]]:
    history: dict[str, list[str]] = defaultdict(list)
    for report_path in report_paths:
        root = cast(Element, ElementTree.parse(report_path).getroot())
        for testcase in root.iter("testcase"):
            history[_nodeid_for(testcase)].append(_status_for(testcase))
    return dict(history)


def build_flaky_report(history: dict[str, list[str]]) -> dict[str, Any]:
    flaky_tests: list[dict[str, Any]] = []
    stable_failures: list[dict[str, Any]] = []
    stable_tests = 0

    for nodeid in sorted(history):
        statuses = sorted(set(history[nodeid]))
        runs = len(history[nodeid])
        if len(statuses) > 1:
            flaky_tests.append(
                {
                    "nodeid": nodeid,
                    "statuses": statuses,
                    "runs": runs,
                }
            )
        if "passed" not in statuses:
            stable_failures.append(
                {
                    "nodeid": nodeid,
                    "statuses": statuses,
                    "runs": runs,
                }
            )
            continue
        if len(statuses) > 1:
            continue
        stable_tests += 1

    return {
        "summary": {
            "total_tests": len(history),
            "flaky_test_count": len(flaky_tests),
            "stable_failure_count": len(stable_failures),
            "stable_test_count": stable_tests,
        },
        "flaky_tests": flaky_tests,
        "stable_failures": stable_failures,
        "stable_tests": stable_tests,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect flaky tests by comparing multiple pytest JUnit XML reports."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="JUnit XML report paths from repeated pytest runs.",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Destination JSON report path.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    history = collect_test_history(args.reports)
    report = build_flaky_report(history)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["flaky_tests"] or report["stable_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
