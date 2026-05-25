from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE_ROOTS = (Path("src/regime_detection"), Path("src/regime_data_fetch"))
SOURCE_SUFFIXES = {".py", ".yaml"}
FORBIDDEN_SOURCE_PATTERNS = re.compile(
    r"TODO\([^)]*ticket=|TD-|audit M[0-9]|spec_code_data_audit|"
    r"Ambiguity Log|Implementation Ambiguity Log|Slice[s]? [0-9]|slice-[0-9]|"
    r"documented implementation[^\n]*#|implementation phase\.[0-9]|"
    r"source-data audit[^\n]*M[0-9]|Log #[0-9]"
)


def test_source_comments_do_not_embed_task_or_audit_references() -> None:
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if FORBIDDEN_SOURCE_PATTERNS.search(line):
                    offenders.append(f"{path}:{line_number}: {line.strip()}")

    assert offenders == []


def test_axis_builders_do_not_use_common_typing_escape_hatch() -> None:
    assert not Path("src/regime_detection/axis_builders/common.py").exists()


def test_inflation_growth_axis_builder_does_not_reconstruct_rule_inputs() -> None:
    path = Path("src/regime_detection/axis_builders/inflation_growth.py")
    tree = ast.parse(path.read_text())

    constructors = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "InflationGrowthRuleInputs"
    ]

    assert constructors == []


def test_timeline_day_builder_stays_below_spaghetti_threshold() -> None:
    path = Path("src/regime_detection/timeline.py")
    tree = ast.parse(path.read_text())

    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    day_builder = functions["_build_timeline_output_for_day"]

    assert day_builder.end_lineno - day_builder.lineno + 1 <= 180


def test_market_context_has_single_sliced_context_constructor() -> None:
    path = Path("src/regime_detection/market_context.py")
    tree = ast.parse(path.read_text())

    constructors = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "MarketContext"
    ]

    assert len(constructors) <= 2
