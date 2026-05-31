# Golden Date Replacement Justification

The V1 spec section 12.2 remains the historical source set, but the active
regression fixture uses the 10 `tests/fixtures/derived/golden_dates.yaml` rows
because they are the dates with committed local OHLCV/VIX coverage and
hand-labeled V2 transition expectations.

Section 12.2 includes `2020-04-10`, which was Good Friday and not an NYSE
trading session. The engine contract rejects non-trading `as_of_date` values
instead of rolling them. The active replacement set therefore keeps explicit
trading-session dates around the same regimes and now verifies every row with
no silent pre-2019 or data-quality skips.

The replacement set is not engine-generated. When deterministic predicates
disagree with the old hand label, the row must be updated with the predicate
result or the implementation must be fixed before the test is allowed to pass.
