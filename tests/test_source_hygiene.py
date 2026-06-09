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
    # trend_direction, trend_character, volatility_state, breadth_state were
    # previously guarded as V1-only contract paths. After the axis classifier
    # refactor merged V1+V2 into one module per axis (per CLAUDE.md framing
    # "V1 and V2 are phases of one engine, not two systems"), those modules
    # now formally own V2 features and were removed from the guard list. The
    # remaining paths are genuinely V1-only contract surfaces: the wire shim,
    # the V1-fixed event calendar feed, and the V1 strategy adapter.
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
V1_CONTRACT_EXTENSION_ALLOWLIST: dict[Path, tuple[str, ...]] = {}

# F-023: §12.4 requires a check that fails on V2 scaffolding/imports (HMM/clustering/
# change-point libraries) in V1 code. The pattern guard above covers only 7 enumerated
# V1-contract paths, so a stray HMM import in ANY other nominally-V1 module would slip
# through. Invert to a V2-EVIDENCE-IMPORT ALLOWLIST: name the V2-evidence modules that
# are permitted to import these libraries; every OTHER regime_detection module is V1-owned
# (§14.3: "V2 work occurs in V2 modules ... V1 modules either untouched or extended
# additively") and must NOT import them. A new V1 file importing hmmlearn fails CI until
# it is consciously added to this allowlist.
V2_EVIDENCE_IMPORT_ALLOWLIST: frozenset[Path] = frozenset(
    {
        Path("src/regime_detection/hmm_state.py"),
        Path("src/regime_detection/clustering.py"),
        Path("src/regime_detection/change_point.py"),
        Path("src/regime_detection/_config_evidence_strategy.py"),
    }
)
FORBIDDEN_V2_EVIDENCE_IMPORT_ROOTS = frozenset(
    {"hmmlearn", "sklearn", "bayesian_changepoint_detection"}
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


def test_v2_evidence_libraries_imported_only_by_allowlisted_modules() -> None:
    # F-023: every regime_detection module that imports an HMM/clustering/change-point
    # library must be in the V2-evidence allowlist — so a stray V2 import in any V1-owned
    # file is caught, not just within the 7 enumerated contract paths.
    offenders: list[str] = []
    for path in sorted(Path("src/regime_detection").rglob("*.py")):
        if path in V2_EVIDENCE_IMPORT_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_V2_EVIDENCE_IMPORT_ROOTS:
                    offenders.append(f"{path}: imports {module}")

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
    # After the Pattern A split (features vs classify), build_rule_inputs_by_date
    # lives in the *_rules.py file, not the features file.
    offenders: list[str] = []
    for path in (
        Path("src/regime_detection/credit_funding_rules.py"),
        Path("src/regime_detection/inflation_growth_rules.py"),
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


def test_spec_scope_decisions_are_documented() -> None:
    decision = Path("docs/decisions/0020-v2-prerequisite-and-shadow-scope.md")
    assert decision.exists()
    text = decision.read_text()

    required_fragments = (
        "F-019",
        "9-slice prerequisite",
        "process gate",
        "F-021",
        "ticker / start_date / end_date interval",
        "F-025",
        "HMM parameter-drift flags",
        "20% state-mean parameter-drift alert",
        "30% transition-probability review flag",
        "F-053",
        "Vol-crush exposure response",
        "downstream strategy-layer contract",
        "50% long-vol exposure reduction",
        "5-day cooldown",
        "F-045",
        "CPI-only dual-vintage store",
        "F-049",
        "local/Alpaca archived parquet is the shadow source of truth",
        "F-050",
        "daily fetch is upstream of the runner",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]

    assert missing == []


def test_shadow_runner_spec_pins_current_shadow_source_and_fetch_boundary() -> None:
    spec = Path("docs/shadow_runner_spec.md").read_text()

    assert "local/Alpaca archived parquet is the shadow source of truth" in spec
    assert "daily fetch is upstream of the runner" in spec


def test_bocpd_online_changepoint_dependency_api_is_stable() -> None:
    # F-046: the BOCPD (§4.6/§6.3) Adams-MacKay online implementation is pinned to the
    # bayesian-changepoint-detection 0.2.dev1 artifact (the only PyPI release carrying
    # the online API). Guard against a future version yanking or reshaping that API by
    # asserting the symbols import and online_changepoint_detection returns a run-length
    # posterior matrix of the expected (N+1, N+1) shape.
    from functools import partial

    import numpy as np
    from bayesian_changepoint_detection.online_changepoint_detection import (
        StudentT,
        constant_hazard,
        online_changepoint_detection,
    )

    data = np.concatenate([np.zeros(20, dtype=float), np.ones(20, dtype=float)])
    posterior, run_length_maxes = online_changepoint_detection(
        data,
        partial(constant_hazard, 250),
        StudentT(alpha=0.1, beta=0.01, kappa=1.0, mu=0.0),
    )

    assert posterior.shape == (len(data) + 1, len(data) + 1)
    assert run_length_maxes.shape[0] == len(data) + 1
