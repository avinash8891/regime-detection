from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    error_block = ""
    skipped_block = ""
    if status == "failed":
        failure_block = '<failure message="boom">trace</failure>'
    if status == "error":
        error_block = '<error message="boom">trace</error>'
    if status == "skipped":
        skipped_block = '<skipped message="not today" />'
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuite name="pytest" tests="1" failures="{failures}" '
            'errors="{errors}" skipped="{skipped}">'
            '<testcase classname="tests.test_sample" name="{name}" time="0.01">'
            "{failure_block}"
            "{error_block}"
            "{skipped_block}"
            "</testcase>"
            "</testsuite>"
        ).format(
            failures="1" if status == "failed" else "0",
            errors="1" if status == "error" else "0",
            skipped="1" if status == "skipped" else "0",
            name=testcase_name,
            failure_block=failure_block,
            error_block=error_block,
            skipped_block=skipped_block,
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


def test_collect_test_history_rejects_xml_entities(tmp_path: Path) -> None:
    detector = _load_module("detect_flaky_tests", "scripts/detect_flaky_tests.py")
    report = tmp_path / "hostile.xml"
    report.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE testsuite [
  <!ENTITY injected "expanded-entity">
]>
<testsuite>
  <testcase classname="tests.test_sample" name="&injected;" />
</testsuite>
""",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="EntitiesForbidden|forbidden"):
        detector.collect_test_history([report])


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


def test_build_flaky_report_flags_mixed_non_pass_statuses() -> None:
    detector = _load_module("detect_flaky_tests", "scripts/detect_flaky_tests.py")

    report = detector.build_flaky_report(
        {
            "tests.test_sample::test_mixed_non_pass": ["failed", "skipped"],
            "tests.test_sample::test_stable": ["passed", "passed"],
        }
    )

    expected = {
        "nodeid": "tests.test_sample::test_mixed_non_pass",
        "statuses": ["failed", "skipped"],
        "runs": 2,
    }
    assert report["flaky_tests"] == [expected]
    assert report["stable_failures"] == [expected]
    assert report["stable_tests"] == 1


def test_build_flaky_report_counts_mixed_error_fail_as_stable_failure() -> None:
    detector = _load_module("detect_flaky_tests", "scripts/detect_flaky_tests.py")

    report = detector.build_flaky_report(
        {"tests.test_sample::test_never_passed": ["failed", "error"]}
    )

    assert report["stable_failures"] == [
        {
            "nodeid": "tests.test_sample::test_never_passed",
            "statuses": ["error", "failed"],
            "runs": 2,
        }
    ]
    assert report["summary"]["stable_failure_count"] == 1
