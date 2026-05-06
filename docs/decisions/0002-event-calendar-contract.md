# Decision 0002: V1 Event Calendar Contract

**Status:** accepted

## Decision

V1 event calendar behavior is locked as follows:

- `RegimeEngine.classify(..., event_calendar=...)` accepts a normalized `pd.DataFrame` only. The engine does not load YAML/CSV paths.
- Scheduled event dates may be used even when `event_date > as_of_date`, but only when `publication_date <= as_of_date`.
- Event outcomes or realized values are never available before the event happens.
- FOMC, CPI, NFP, and ad-hoc events are sourced from YAML/CSV rows and normalized by a separate loader.
- `expiry_week` and `earnings_season` are sourced from config-defined calendar rules in `core3-v1.0.0.yaml`, not from event-calendar rows.
- `expiry_week` uses the configured monthly-options rule `third_friday_of_month` with trading-day window `[-2, 0]`.
- `earnings_season` uses configured quarter windows such as `second_monday_of_january` plus `end_offset_days: 35`.

## Why

- Scheduled macro dates are public knowledge in advance, so publication date is the real lookahead boundary, not event date.
- Keeping `classify()` DataFrame-only preserves separation of concerns between I/O and classification.
- Expiry and coarse earnings-season windows are deterministic calendar rules; encoding them as rows creates unnecessary maintenance drift.

## V1 Implementation Rule

Event-calendar normalization and file loading live outside the engine, for example in a helper such as:

```python
load_event_calendar(source: str | Path | pd.DataFrame, *, market: str = "US") -> pd.DataFrame
```

Normalized event-calendar DataFrame contract:

```text
date
market
type
importance
publication_date
```

Supported row types in V1:

```text
FOMC
CPI
NFP
ad_hoc
```

Loader defaults:

- For FOMC/CPI/NFP rows, if `publication_date` is absent, default it to `date - 90 calendar days`.
- For ad-hoc rows, if `publication_date` is absent, default it to `date`.

Optional operator guard:

- If a scheduled event row is dated more than 90 calendar days after the historical `as_of_date` being replayed, emit a warning for review. This does not fail classification.

## Precedence Reminder

If multiple windows match, V1 precedence remains:

```text
fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown
```

Importance does not override precedence in V1.

## Agent Instruction

Do not invent path-loading inside `RegimeEngine.classify(...)`. Do not convert config-defined expiry or earnings windows back into event rows. If a future change wants more convenience, add it as a helper or constructor outside the core classification path and update this decision record.
