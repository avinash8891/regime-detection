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
V1_CONTRACT_GUARD_PATHS = (
    Path("src/regime_detection/trend_direction.py"),
    Path("src/regime_detection/trend_character.py"),
    Path("src/regime_detection/volatility_state.py"),
    Path("src/regime_detection/breadth_state.py"),
    Path("src/regime_detection/event_calendar.py"),
    Path("src/regime_detection/strategy_response.py"),
    Path("src/regime_detection/legacy_v1_wire.py"),
)
FORBIDDEN_V1_CONTRACT_SCAFFOLDING_PATTERNS = re.compile(
    r"\b("
    r"GaussianHMM|hmm|GMM|gmm|ORCA|SRR|"
    r"hurst(?:_\w+)?|efficiency_ratio(?:_\w+)?|"
    r"eigenvalue|weighted_transition_score|"
    r"crash_condition"
    r")\b",
    re.IGNORECASE,
)
V1_CONTRACT_EXTENSION_ALLOWLIST: dict[Path, tuple[str, ...]] = {
    Path("src/regime_detection/trend_direction.py"): (
        '"efficiency_ratio_20d": _ev_float(features.efficiency_ratio_20d.loc[dt]),',
        '"hurst_250d": _ev_float(features.hurst_250d.loc[dt]),',
    ),
}


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


def test_v1_contract_paths_do_not_scaffold_unowned_v2_features() -> None:
    """V2 extends shared modules; V1 contract paths still reject stray V2-only hooks."""

    offenders: list[str] = []
    for path in V1_CONTRACT_GUARD_PATHS:
        allowed_snippets = V1_CONTRACT_EXTENSION_ALLOWLIST.get(path, ())
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if any(snippet in line for snippet in allowed_snippets):
                continue
            if FORBIDDEN_V1_CONTRACT_SCAFFOLDING_PATTERNS.search(line):
                offenders.append(f"{path}:{line_number}: {line.strip()}")

    assert offenders == []


def test_axis_builders_do_not_use_common_typing_escape_hatch() -> None:
    assert not Path("src/regime_detection/axis_builders/common.py").exists()


def test_data_fetch_user_agent_header_value_stays_in_shared_http_helper() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src/regime_data_fetch").rglob("*.py")):
        if path.name == "_http.py":
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if '"User-Agent"' in line or "'User-Agent'" in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")

    assert offenders == []


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


def test_bulk_rule_input_builders_do_not_scalar_loc_per_session() -> None:
    offenders: list[str] = []
    for path in (
        Path("src/regime_detection/credit_funding.py"),
        Path("src/regime_detection/inflation_growth.py"),
    ):
        tree = ast.parse(path.read_text())
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        bulk_builder = functions["build_rule_inputs_by_date"]
        for node in ast.walk(bulk_builder):
            if not isinstance(node, ast.Call):
                continue
            func_name = getattr(node.func, "id", "")
            if func_name in {"_scalar_at", "_scalar_at_lag"}:
                offenders.append(f"{path}:{node.lineno}: {func_name}")

    assert offenders == []


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
