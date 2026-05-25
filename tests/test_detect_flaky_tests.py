from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _write_junit_xml(path: Path, *, testcase_name: str, status: str) -> None:
    failure_block = ""
    if status == "failed":
        failure_block = '<failure message="boom">trace</failure>'
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuite name="pytest" tests="1" failures="{failures}" errors="0" skipped="0">'
            '<testcase classname="tests.test_sample" name="{name}" time="0.01">'
            "{failure_block}"
            "</testcase>"
            "</testsuite>"
        ).format(
            failures="1" if status == "failed" else "0",
            name=testcase_name,
            failure_block=failure_block,
        ),
        encoding="utf-8",
    )


def test_collect_test_history_flags_inconsistent_results(tmp_path: Path) -> None:
    detector = _load_module("detect_flaky_tests", "scripts/detect_flaky_tests.py")
    pass_xml = tmp_path / "run1.xml"
    fail_xml = tmp_path / "run2.xml"
    stable_xml = tmp_path / "run3.xml"
    _write_junit_xml(pass_xml, testcase_name="test_flaky", status="passed")
    _write_junit_xml(fail_xml, testcase_name="test_flaky", status="failed")
    _write_junit_xml(stable_xml, testcase_name="test_stable", status="passed")

    history = detector.collect_test_history([pass_xml, fail_xml, stable_xml])
    report = detector.build_flaky_report(history)

    assert report["flaky_tests"] == [
        {
            "nodeid": "tests.test_sample::test_flaky",
            "statuses": ["failed", "passed"],
            "runs": 2,
        }
    ]
    assert report["stable_failures"] == []
    assert report["stable_tests"] == 1


def test_cli_writes_report_and_fails_when_flakes_detected(tmp_path: Path) -> None:
    pass_xml = tmp_path / "run1.xml"
    fail_xml = tmp_path / "run2.xml"
    report_path = tmp_path / "flaky-report.json"
    _write_junit_xml(pass_xml, testcase_name="test_flaky", status="passed")
    _write_junit_xml(fail_xml, testcase_name="test_flaky", status="failed")

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/detect_flaky_tests.py",
            "--report",
            str(report_path),
            str(pass_xml),
            str(fail_xml),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, (result.stdout, result.stderr)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["flaky_test_count"] == 1
    assert payload["flaky_tests"][0]["nodeid"] == "tests.test_sample::test_flaky"
