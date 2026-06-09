from __future__ import annotations


def quote_sqlite_identifier(
    identifier: str, *, allowed_identifiers: frozenset[str] | set[str]
) -> str:
    """Return a safely quoted SQLite identifier from a closed allowlist."""
    if identifier not in allowed_identifiers:
        raise ValueError(f"Unexpected SQLite identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'
