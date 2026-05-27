# Pyright Pandas Stub Policy

Pyright strict coverage is ratcheted module by module. Known pandas-stub
surfaces may use narrow `# pyright: ignore[reportUnknownMemberType]` comments
when the value is immediately normalized into a domain type.

Current business-logic ratchet includes:

- `src/regime_detection/engine.py`
- `src/regime_detection/models.py`
- `src/regime_detection/axis_series.py`

New strict modules should be added only with a matching test or contract check
that prevents accidental removal from the include set.

Allowed ignores:

- pandas indexing/accessor calls where stubs cannot represent the runtime type
  and the next line casts or validates into a concrete domain type.
- compatibility shims whose only job is to isolate pandas typing noise.

Not allowed:

- blanket ignores across business logic modules.
- ignores that hide regime classification branches, thresholds, payload fields,
  date alignment, or dependency semantics.
- suppressing non-pandas errors under a pandas-stub rationale.

Each ignore should stay close to the pandas expression and include a short
reason. The strict include set should fail on new business logic type
regressions even while pandas `reportUnknownMemberType` debt is ratcheted down.

Pydantic model-compatibility suppressions are tracked separately from pandas.
They may be used only for framework protocol mismatches, such as dict-compatible
payload wrappers overriding `BaseModel.__iter__` or Pydantic subclass fields that
intentionally narrow a validated payload type. They must not hide classifier
branch logic or unvalidated input handling.
