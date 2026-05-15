# Regime Engine V1 Data Requirements

**Scope:** US equities V1  
**Engine version:** `regime-engine-v1.0.0`  
**Source spec:** `docs/regime_engine_v1_final_spec.md`

This document lists all data needed to build V1 with real fixtures and real tests. V1 tests must use deterministic repo-local files and must not call the network.

---

## 0. Source, Cadence, and Availability Summary

| Data | Source | Cadence | Availability / Comment |
|---|---|---|---|
| `SPY` daily OHLCV | Alpaca REST | daily | fetch after NYSE close; use NYSE trading dates only |
| `RSP` daily OHLCV | Alpaca REST | daily | fetch after NYSE close; align exactly to `SPY` trading dates |
| `VIX` daily close or `VIXY` proxy close | Alpaca REST | daily | fetch after market close; use `VIXY` only when true `VIX` is unavailable |
| V1 scheduled event rows (`FOMC`, `CPI`, `NFP`) | generated repo-local YAML from Fed calendars + BLS release schedules | scheduled monthly / about 8 times per year | generate from official source schedules and release histories; store release timestamps with each row and consume through the event-label resolver |
| V1 rule-derived event windows (`expiry_week`, `earnings_season`) | deterministic config/runtime rules | monthly / quarterly windows | compute at runtime from NYSE calendar rules and fixed quarter-season rules; resolver applies the same precedence as the V1 spec and does not maintain hand-entered historical rows |
| NYSE trading calendar | `pandas_market_calendars` or equivalent | exchange calendar | session/holiday schedule must be available for the full fixture range |
| Golden fixture expectations | repo-local fixture files | static fixture set | updated only when fixture verification proves the labeled expectation is wrong |

This document defines the **required V1 data artifacts and semantics**. It does not imply that every source already has a production-ready live fetcher.

---

## 1. Raw Market Data

V1 requires daily price data for the market anchor, ETF breadth proxy, and volatility proxy.

### 1.1 SPY Daily OHLCV

Purpose:

- market anchor for all V1 labels;
- trend direction features;
- trend character features;
- realized volatility features;
- transition-risk recovery predicates;
- ETF breadth index-side features.

Required columns:

```text
date
symbol
open
high
low
close
volume
adjusted_close
```

Rules:

- `symbol` must be `SPY`.
- `date` must be NYSE trading dates only.
- `close` is the canonical price used for SMA, returns, drawdown, and realized volatility.
- `high` and `low` are required for ADX14.
- `adjusted_close` is retained for audit if the vendor provides it, but V1 formulas use the canonical `close` column unless the implementation spec is changed.

### 1.2 RSP Daily OHLCV

Purpose:

- ETF proxy breadth mode;
- `relative_breadth_ratio = RSP_close / SPY_close`;
- `relative_breadth_sma50`;
- `relative_breadth_return_20d`.

Required columns:

```text
date
symbol
open
high
low
close
volume
adjusted_close
```

Rules:

- `symbol` must be `RSP`.
- `date` must align to NYSE trading dates.
- `close` is required for all V1 breadth formulas.

### 1.3 VIX Daily Close

Purpose:

- `vix_percentile_252d`;
- volatility-state `high_vol` and `crisis_vol` predicates.

Required columns:

```text
date
symbol
close
```

Allowed symbols:

```text
VIX
VIXY (documented proxy when Alpaca does not provide true VIX)
```

OHLCV columns may be included if the vendor provides them, but V1 only requires `close`.

Availability note:

- fetch after market close;
- prefer true `VIX`;
- if using `VIXY`, treat it as an explicit operational proxy rather than silently labeling it as true `VIX`.

### 1.4 Required Date Range

Preferred raw range:

```text
2016-01-04 through 2024-12-31
```

Reason:

- covers every V1 golden date;
- provides enough lookback for SMA200, ADX14, 252-day volatility percentile, 320-day engine-wide history, and hysteresis replay;
- provides buffer for fixture verification and future V1 regression dates.

---

## 2. Event Calendar Data

V1 requires a generated US scheduled-event calendar plus deterministic runtime rules for rule-derived windows.

Availability note:

- scheduled rows should be generated from official source histories, not typed by hand;
- `FOMC` dates come from Federal Reserve meeting-calendar pages;
- `CPI` and `NFP` release dates come from BLS yearly release-schedule pages;
- `expiry_week` and `earnings_season` should be computed from deterministic rules rather than stored as historical rows;
- ad-hoc events are out of scope for V1.
- current FOMC coverage has been live-verified from `2007-10-31` through `2026-03-18`;
- current checked YAML/report contains `131` CPI rows (`2016-01-20` through `2026-12-10`) and `131` NFP rows (`2016-01-08` through `2026-12-04`), generated from the BLS yearly schedule structure via the repo-local archive-backed fetch path;
- current `expiry_week` and `earnings_season` runtime rules are implemented and wired through the repo event-label resolver.

Required event fields:

```text
date
market
type
importance
```

Required scheduled event types:

```text
FOMC
CPI
NFP
```

Required rule-derived event windows:

```text
monthly_options_expiry
earnings_season
```

Rules:

- use only event rows with date `<= as_of_date`;
- event windows use NYSE trading days, not calendar days;
- if multiple event windows match, active label follows V1 precedence:

```text
fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown
```

Evidence must preserve all matching event labels.

### 2.1 Scheduled Event Representation

Preferred generated row fields:

```text
date
release_timestamp_et
market
type
importance
source
```

Rules:

- generated rows must include only releases with date `<= as_of_date`;
- `release_timestamp_et` must preserve the real release time when available;
- `FOMC` rows should carry the Fed minutes release timestamp convention at `14:00 ET`;
- `CPI` and `NFP` release rows should carry the BLS scheduled release time at `08:30 ET`.
- implementation logic is:
  - fetch current Fed FOMC calendar page
  - fetch the Fed historical year index and yearly archive pages for pre-2021 meetings
  - dedupe FOMC rows by `meeting_end_date`
  - for `CPI` and `NFP`, fetch BLS yearly release-schedule pages under `/schedule/YYYY/` or `/schedule/YYYY/home.htm`
  - parse only `Consumer Price Index` / `Consumer Price Indexes` rows into `CPI`
  - parse only `The Employment Situation` / `Employment Situation` rows into `NFP`
  - stamp `CPI` and `NFP` releases at `08:30 ET`
  - sort all rows by `release_timestamp_et`
  - write generated YAML to `configs/events/us_events.yaml`
  - load scheduled YAML through `load_scheduled_events_yaml()`
  - resolve active event labels through `resolve_event_label()`
  - if BLS historical pages return access-control errors from the execution environment, treat that as a transport blocker, not as a reason to fall back silently to a different source

### 2.2 Earnings Season Representation

If represented as rows, use:

```text
start_date
end_date
market
type=earnings_season
importance
```

If represented as config, define the market-specific windows in `configs/core3-v1.0.0.yaml`.

Preferred V1 representation: config/runtime rule, not stored historical rows.

Current agreed rule:

- anchor months: `January`, `April`, `July`, `October`
- season start: **second Monday** of the anchor month
- season end: **35 calendar days after** the start date
- `as_of_date` is inside `earnings_season` when it falls within that inclusive window
- when multiple windows match, `earnings_season` loses to any higher-precedence scheduled event

### 2.3 Monthly Options Expiry Representation

Required fields:

```text
date
market
type=monthly_options_expiry
importance
```

Preferred V1 representation: config/runtime rule, not stored historical rows. If explicit rows are ever used in tests, they must match the deterministic config rule exactly.

If represented as config, the rule must be deterministic and documented in `configs/core3-v1.0.0.yaml`.

Current agreed rule:

- anchor date is the **third Friday** of the month
- if that Friday is not an NYSE trading day, roll back to the **previous NYSE trading day**
- the runtime `expiry_week` window is the inclusive NYSE trading-day range `[-2, 0]` around that anchor
- when multiple windows match, `expiry_week` loses to `FOMC`, `CPI`, and `NFP` according to the shared precedence rules

---

## 3. Trading Calendar Data

V1 uses the NYSE trading calendar through `pandas_market_calendars` or equivalent.

Availability note:

- this is derived from the installed exchange-calendar library, not from a repo-local raw download;
- the required availability is deterministic access to all NYSE sessions and holidays across the full test range.

No repo-local CSV is required if the calendar library is installed, but all tests and fixture verification depend on:

```text
NYSE sessions
NYSE holidays
NYSE trading-day offsets
```

Required coverage:

```text
2016-01-04 through 2024-12-31
```

Uses:

- reject non-trading `as_of_date`;
- rolling lookback counts;
- event windows;
- hysteresis day counts;
- contiguous `classify_window` output.

---

## 4. Golden Fixture Data

Golden fixtures pin expected V1 outputs against real market data.

Golden dates:

```text
2017-06-01
2018-02-05
2018-12-24
2019-09-13
2020-03-16
2020-04-10
2021-11-15
2022-06-13
2022-10-12
2024-01-16
```

Required fields per golden date:

```text
as_of_date
expected_trend_direction
expected_trend_character
expected_volatility_state
expected_breadth_state
expected_transition_risk
notes
```

Important rule:

- `2018-02-05` breadth must be pinned after fixture verification from real SPY/RSP data.
- If verified predicates contradict the hand-labeled table, the table is wrong. Replace the fixture date or expected label before classifier implementation continues.

---

## 5. Fixture Verification Data

Fixture verification must produce a report that traces each expected label to raw data, computed features, and rule predicates.

Required output file:

```text
tests/fixtures/verification/golden_dates_report.yaml
```

Required fields per golden date:

```text
as_of_date
raw_data_sources
raw_spy_open
raw_spy_high
raw_spy_low
raw_spy_close
raw_spy_volume
raw_rsp_open
raw_rsp_high
raw_rsp_low
raw_rsp_close
raw_rsp_volume
raw_vix_close
sma_50
sma_200
return_1d
return_5d
return_10d
return_21d
return_63d
prior_63d_drawdown
adx_14
realized_vol_21d
realized_vol_percentile_252d
vix_percentile_252d
relative_breadth_ratio
relative_breadth_sma50
relative_breadth_return_20d
index_distance_from_63d_high
predicate_evaluations
selected_labels
generated_by_commit
generated_at_utc
```

Rules:

- verification reads only repo-local raw files;
- verification performs no network calls;
- report values must be sufficient to explain each output in under 30 seconds.

---

## 6. Provenance Data

Every raw fixture file needs provenance.

Required file:

```text
tests/fixtures/raw/PROVENANCE.md
```

Required metadata per raw file:

```text
filename
vendor_or_source
download_url_or_source_identifier
download_date
symbol_mapping
date_range
timezone
calendar_assumption
adjustment_policy
license_or_usage_note
checksum
```

Rules:

- raw data is read-only after commit;
- vendor revisions require a new raw file or explicit provenance update;
- derived fixture changes must identify the raw files used.

---

## 7. Config Data

V1 requires:

```text
configs/core3-v1.0.0.yaml
```

Required contents:

```text
config_version
market
trading_calendar
hysteresis days
breadth_mode=etf_proxy
cap_weight_index=SPY
equal_weight_proxy=RSP
event calendar settings
scheduled event generator settings
earnings season windows if config-based
monthly expiry windows if config-based
data-quality thresholds
```

Rules:

- config keys must be validated with `extra="forbid"`;
- precedence orderings and risk-rank tables are hardcoded in code, not config;
- output `config_version` reflects the loaded config.

---

## 8. Recommended Repo Layout

Raw fixtures:

```text
tests/fixtures/raw/
  spy_2015_2024.csv
  rsp_2015_2024.csv
  vix_2015_2024.csv
  us_events_2015_2024.yaml
  PROVENANCE.md
```

Derived fixtures:

```text
tests/fixtures/derived/
  golden_dates.yaml
```

Verification artifacts:

```text
tests/fixtures/verification/
  golden_dates_report.yaml
```

Raw CSV diff policy:

```gitattributes
tests/fixtures/raw/*.csv linguist-generated=true
tests/fixtures/raw/*.csv -diff
```

---

## 9. Data Explicitly Not Needed For V1

Do not collect or wire these for V1:

```text
PIT S&P 500 constituents
individual stock OHLCV
2y yield
10y yield
DXY
credit spreads
macro series
options/implied volatility
HMM/GMM/change-point data
correlation/eigenvalue universe
ORCA/SRR data
Hurst inputs beyond SPY close
efficiency-ratio-specific data beyond SPY close
```

These are V2 or v1.1 concerns unless the V1 spec is changed.
