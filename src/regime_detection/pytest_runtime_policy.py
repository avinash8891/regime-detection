from __future__ import annotations

import os
import re

_MARK_TOKEN_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


def _last_markexpr(args: list[str]) -> str:
    markexpr = ""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m" and i + 1 < len(args):
            markexpr = args[i + 1]
            i += 2
            continue
        if arg.startswith("--markexpr="):
            markexpr = arg.split("=", 1)[1]
        i += 1
    return markexpr


def _integration_only_markexpr(markexpr: str) -> bool:
    normalized = " ".join(markexpr.lower().split())
    if not normalized or " or " in f" {normalized} ":
        return False
    positive_expr = re.sub(r"\bnot\s+[a-zA-Z_][a-zA-Z0-9_]*\b", "", normalized)
    marker_tokens = {
        token
        for token in _MARK_TOKEN_RE.findall(positive_expr)
        if token not in {"and", "not"}
    }
    return marker_tokens == {"integration"}


def pytest_load_initial_conftests(early_config, parser, args) -> None:  # type: ignore[no-untyped-def]
    if os.environ.get("SANTO_DOMINGO_FORCE_XDIST_AUTO") == "1":
        return
    if not _integration_only_markexpr(_last_markexpr(args)):
        return
    args.extend(["-n", "0"])
