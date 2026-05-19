# Layer 2D Event-Calendar Pipeline — Spec 1: Group A (Scheduled-Calendar Coverage)

**Status:** draft — awaiting user review
**Date:** 2026-05-14
**Scope:** Spec 1 of 2. Group A only — `ECB_decision`, `BOE_decision`, `BOJ_decision`, `election`.
**Companion:** Spec 2 (Group B — `geopolitical_event`, `budget`) is a separate follow-up.

---

## 1. Problem & Context

`configs/events/us_events.yaml` feeds the V2 Layer 2D event-calendar classifier
(`regime_detection.event_calendar`). The runtime side is settled — labels,
windows, and precedence already support every V2 event type, and ADR 0002 locks
the consumed row schema. The gap is **data acquisition**: the V2 curated event
types are incomplete or unsourced.

Current state in `src/regime_data_fetch/event_calendar.py`
(`_build_v2_curated_candidate_events`, line 607):

- **ECB/BoE/BoJ** — now sourced from both current-calendar and historical archive pages. ECB: 88 decisions, BoE: 96 decisions, BoJ: 89 decisions, all covering 2016-2026. `HTTP_USER_AGENT` fixed from bot-like to browser-like string to avoid central-bank page blocks.
- **election** — pure date arithmetic (even-year first-Tuesday-after-first-Monday),
  tagged with a `fec.gov:election-dates` source id that is never actually consulted.
- **budget** — deterministic `Sep 30` per year (out of scope for Spec 1; see §3.3).
- **geopolitical_event** — no fetch path at all (Spec 2).

The function also **mixes fetch, parse, validation, and YAML rendering** in one
place and writes straight to YAML with no candidate layer, no source-evidence
record, no conflict detection. Spec 1 fixes Group A coverage *and* introduces the
structured candidate → validation → triangulation → YAML spine that Spec 2 extends.

### Acceptance criteria (from problem statement)

For every emitted event row we must be able to recover: exact date, event type,
source URL/source id, historical-actual vs. future-scheduled, confidence /
source count, and whether manual review was required. `us_events.yaml` must be
**regenerable from the candidate + validation artifacts** — never hand-merged.

---

## 2. Non-Goals (Spec 1)

- **Group B event types.** No `geopolitical_event` or `budget` candidate
  generation. The existing deterministic `budget` Sep-30 block stays untouched
  (§3.3).
- **LLM ambiguity resolver.** No LLM calls. Spec 1 adds only an unused
  resolver *hook point* on the orchestrator (§5.4).
- **TinyFish / GDELT / GPR / ACLED.** Spec 2 sources; not created here.
- **Changes to the consumed YAML row schema or `regime_detection`** classifier.
  ADR 0002 stays intact. We only change how the YAML is *produced*.
- **Incremental fetch.** Each run is a full re-fetch / full re-render (§7.3).

---

## 3. Architecture — Approach 3 (Adapters + Validators + Shared Triangulation)

### 3.1 Principle

- **Primary adapters** *originate* `EventCandidate` rows from official,
  source-grade feeds. One adapter per official source.
- **Secondary validators** *confirm or contradict* existing candidates. They
  never originate a final event on their own.
- **One shared triangulation pass** decides promote / quarantine / manual-review.

This adapter/validator split *is* the Q3 conflict rule expressed in types:
official-authoritative, secondary-advisory, conflict → quarantine.

### 3.2 Package layout (Spec 1 creates these files only)

```
src/regime_data_fetch/event_sources/
  __init__.py
  models.py                      # EventCandidate, ValidationResult, protocols, enums
  orchestrator.py                # registry, run(), triangulation/promotion, YAML render
  official_ecb.py                # primary adapter — ECB monetary-policy decisions
  official_boe.py                # primary adapter — BoE MPC announcements
  official_boj.py                # primary adapter — BoJ MPM decision days
  deterministic_election.py      # primary adapter — federal general elections
  validators_hf_central_bank.py  # secondary validator — HF central-bank dataset
```

The registry in `orchestrator.py` is **open for extension**: Spec 2 adds
`deterministic_budget.py`, `validators_tinyfish.py`, `validators_gpr_gdelt.py`
by registering them — no edits to `models.py` or the orchestrator core.

### 3.3 Wiring (AGENTS.md failure-mode A — wire-first)

`_build_v2_curated_candidate_events` in `event_calendar.py` is rewritten to:

1. Construct the orchestrator with the Group A adapters + validators registered.
2. Call `orchestrator.run(...)` for `ECB_decision`, `BOE_decision`,
   `BOJ_decision`, `election` → returns promoted `ScheduledEvent`s.
3. **Keep the existing inline `budget` Sep-30 block unchanged** — Spec 2 migrates
   it to `deterministic_budget.py`.

The orchestrator is invoked from the existing `run_us_event_calendar_fetch`
entrypoint behind the existing `include_v2_curated_candidates` flag. No new
top-level entrypoint. The integration test (§10) exercises the path end-to-end.

---

## 4. Core Models (`models.py`)

```python
EventConfidence = Literal["low", "medium", "high"]
Verdict = Literal["confirm", "contradict", "unknown"]
PromotionOutcome = Literal["promote", "quarantine"]

@dataclass(frozen=True)
class EventCandidate:
    date: date
    event_type: str                       # "ECB_decision" | "BOE_decision" | "BOJ_decision" | "election"
    market: str                           # "GLOBAL" for central banks, "US" for election
    importance: str                       # "high" for all Group A types
    source_id: str
    source_url: str | None
    raw_title: str | None
    raw_snippet: str | None
    is_future_scheduled: bool             # date > run as_of_date
    confidence: EventConfidence
    requires_manual_review: bool
    release_timestamp_et: datetime | None = None
    window_days: tuple[int, int] | None = None

@dataclass(frozen=True)
class ValidationResult:
    candidate_key: tuple[str, date]       # (event_type, date)
    validator_id: str
    verdict: Verdict
    evidence_url: str | None
    evidence_snippet: str | None

@dataclass(frozen=True)
class PromotionDecision:
    candidate_key: tuple[str, date]
    outcome: PromotionOutcome
    final_confidence: EventConfidence
    source_count: int
    requires_manual_review: bool
    reason: str                           # human-readable rule that fired
```

Protocols (structural, so Spec 2 plugs in without inheritance coupling):

```python
class PrimaryAdapter(Protocol):
    source_id: str
    def fetch(self, *, start_year: int, end_year: int,
              store: AcquisitionStore | None, run_id: int | None) -> list[EventCandidate]: ...

class SecondaryValidator(Protocol):
    validator_id: str
    def validate(self, candidates: list[EventCandidate], *,
                 store: AcquisitionStore | None, run_id: int | None) -> list[ValidationResult]: ...

class AmbiguityResolver(Protocol):   # hook only — NOT implemented in Spec 1
    def resolve(self, candidate_key: tuple[str, date],
                conflicting: list[EventCandidate]) -> EventCandidate | None: ...
```

`release_timestamp_et` renderer rule: if present on the candidate, emit it; if
`None`, fall back to `_midnight_et(date)` for Group A rows (matches today's
behavior for central-bank/election rows). Adapters may set a better timestamp if
the source page exposes an announcement time.

---

## 5. Orchestrator & Triangulation (`orchestrator.py`)

### 5.1 `run()` flow

1. **Originate** — invoke each registered `PrimaryAdapter.fetch(...)`; collect
   all `EventCandidate`s. Raw source pages are recorded as artifacts in
   `AcquisitionStore` by the adapters (provenance system-of-record).
2. **Validate** — invoke each registered `SecondaryValidator.validate(...)` over
   the candidate list; collect `ValidationResult`s.
3. **Triangulate** — group candidates by `(event_type, date)`; apply the
   promotion rules (§5.2) → one `PromotionDecision` per key.
4. **Persist** — write the three parquet tables (§7); register each in
   `AcquisitionStore.derived_outputs`.
5. **Render** — build `ScheduledEvent`s from promoted decisions, hand back to
   `_build_v2_curated_candidate_events` for merge + YAML render.

### 5.2 Promotion rules (Group A subset)

Evaluated per `(event_type, date)` group:

| Condition | Outcome | confidence | requires_review |
|---|---|---|---|
| Deterministic primary (election), no contradiction | promote | high | false |
| Official primary (central bank) + ≥1 validator `confirm` | promote | high | false |
| Official primary (central bank), validators all `unknown` | promote | medium | false |
| Any validator `contradict` | **quarantine** | low | **true** |
| Two primary candidates for the same key with different dates¹ | **quarantine** | low | **true** |

¹ Within Group A there is one primary adapter per event type, so this case is
not expected. It is kept in the rule set because the triangulation engine is
shared with Spec 2 and must handle it. If it fires in Group A it indicates an
adapter bug — quarantine is the safe default. No LLM resolution in Spec 1.

`source_count` = 1 (primary) + count of `confirm` verdicts.

### 5.3 Quarantine semantics

Quarantined rows are **not emitted to `us_events.yaml`**. They are written to
`quarantine.parquet` with `requires_manual_review = true` and the failing rule's
`reason`. The fetch report (§7.2) surfaces a quarantine count. Per AGENTS.md
rule I, a quarantine rate > 1% of candidates **stops the run** and reports to the
operator rather than silently shipping a thin calendar.

### 5.4 Resolver hook

`run()` accepts an optional `resolver: AmbiguityResolver | None = None`. In
Spec 1 it is always `None` and never invoked. The parameter exists so Spec 2 can
supply a bounded, optional resolver for genuinely ambiguous Group B dates without
changing the orchestrator signature.

---

## 6. Per-Adapter Source Specifications

URLs marked **(verify)** must be confirmed against the live site during
implementation (AGENTS.md rule F — no confident-wrong). Each adapter validates
its raw input row-by-row and quarantines malformed rows (AGENTS.md rule I).

### 6.1 `official_ecb.py` — `source_id: ecb.europa.eu:monetary-policy-decisions`

| Field | Value |
|---|---|
| Source URL / API | Historical: ECB monetary-policy decisions archive, by year — `https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html` **(verify)**. Forward: Governing Council calendar — `https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html` **(verify; current code uses `events/calendar/mgcgc`)**. Plain HTTPS, no auth. |
| Coverage start/end | Decisions archive ≥ 1999; we consume 2016→. Forward calendar publishes ≈ 12–18 months ahead. **Implemented:** 88 ECB decisions 2016-2026. |
| Future-date support | Yes (calendar page). |
| Self-updating | Yes — but pages **rotate**: prior years move to dated archive pages each January. Adapter must fetch *both* the current calendar page *and* the historical archive index (mirrors `_fetch_fomc_events`). |
| License / access risk | ECB public content; reproduction permitted with source acknowledgment. Low risk. |
| Parser fields extracted | Decision date = **Day 2 / press-conference date** (exclude non-monetary-policy Governing Council meetings, as the current ECB parser already does); `raw_title`, `source_url`. |
| Test fixture path | `tests/fixtures/event_sources/ecb_decisions_archive_2016.html`, `tests/fixtures/event_sources/ecb_calendar_current.html` |
| Failure behavior | Page unreachable → log + degrade (skip that page). Parse yields 0 rows from a page expected to have rows, or a completed year's meeting count is off the ≈8/year sanity check (§9) → **raise** (deterministic). Bank ends with 0 candidates → failed run (status integrity). |

### 6.2 `official_boe.py` — `source_id: bankofengland.co.uk:mpc-decisions`

| Field | Value |
|---|---|
| Source URL / API | Forward: upcoming MPC dates — `https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates`. Historical: Monetary Policy Summary & minutes listing / decisions archive — **(verify exact URL)**. Optional cross-check: Bank Rate history — `https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp` **(verify; rate-*change* meetings only, not a complete meeting list)**. Plain HTTPS, no auth. |
| Coverage start/end | Full history; MPC moved to 8 meetings/year in 2016 — clean fit for 2016→. **Implemented:** 96 BoE decisions 2016-2026. |
| Future-date support | Yes (upcoming dates page). |
| Self-updating | Yes — rotates yearly; fetch current + archive. |
| License / access risk | BoE public content. Low risk. |
| Parser fields extracted | **Announcement date** (the Thursday the MPC decision is published — "Super Thursday" since Aug 2018; the decision Thursday before that); `raw_title`, `source_url`. |
| Test fixture path | `tests/fixtures/event_sources/boe_upcoming_mpc.html`, `tests/fixtures/event_sources/boe_decisions_archive.html` |
| Failure behavior | Same policy as §6.1. |

### 6.3 `official_boj.py` — `source_id: boj.or.jp:monetary-policy-meetings`

| Field | Value |
|---|---|
| Source URL / API | MPM schedule index — `https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm`; historical per-year schedule pages linked from the index **(verify URL pattern)**. Plain HTTPS, no auth. |
| Coverage start/end | Full history; BoJ moved to 8 MPM/year in 2016 — clean fit for 2016→. **Implemented:** 89 BoJ decisions 2016-2026. |
| Future-date support | Yes — schedule published ≈ 1 year ahead. |
| Self-updating | Yes — rotates yearly; fetch current + archive. |
| License / access risk | BoJ public content. Low risk. |
| Parser fields extracted | **Final MPM day** = the decision day (current parser already takes the end day of the 2-day meeting); `raw_title`, `source_url`. |
| Test fixture path | `tests/fixtures/event_sources/boj_mpm_schedule_current.html`, `tests/fixtures/event_sources/boj_mpm_schedule_2016.html` |
| Failure behavior | Same policy as §6.1. |

### 6.4 `deterministic_election.py` — `source_id: fec.gov:election-dates`

| Field | Value |
|---|---|
| Source URL / API | Deterministic computation per **2 U.S.C. §7** (first Tuesday after the first Monday of November, even years). `source_url` records the FEC election-dates page for provenance. No network call at runtime. |
| Coverage start/end | 2016 → 2028 (presidential + midterm federal general elections). |
| Future-date support | Yes — formula. |
| Self-updating | N/A (formula). The FEC page is captured **as a test fixture** to cross-check that computed dates equal FEC's published dates. |
| License / access risk | None (statutory formula). |
| Parser fields extracted | Computed `date`; election kind (`presidential` / `midterm`) → `raw_title`; `window_days = (-5, 10)`; `confidence = high`. |
| Test fixture path | `tests/fixtures/event_sources/fec_election_dates.html` (cross-check only) |
| Failure behavior | Deterministic. If the FEC fixture cross-check disagrees with a computed date in tests, the test fails (formula or fixture is wrong) — not a runtime path. |

### 6.5 `validators_hf_central_bank.py` — `validator_id: hf:aufklarer-central-bank-communications`

| Field | Value |
|---|---|
| Source URL / API | Hugging Face dataset `aufklarer/central-bank-communications`. Accessed via the HF **datasets-server parquet HTTP endpoint** (public dataset, no auth) to **avoid adding `datasets` / `huggingface_hub` as a dependency** (AGENTS.md — stdlib/existing deps first). **(verify dataset schema, exact columns, coverage window, and license on HF.)** |
| Coverage start/end | Historical only — **static snapshot** (verify end date). No reliable forward coverage. |
| Future-date support | No (static snapshot). |
| Self-updating | No — updates only when the maintainer republishes. |
| License / access risk | **Medium** — third-party dataset, could be renamed/removed. Mitigation: cache the downloaded parquet as an artifact in `AcquisitionStore` so a past run stays reproducible even if the dataset disappears. |
| Parser fields extracted | `(central_bank, communication_date, doc_type, title/url)` per row. |
| Validator logic | For each central-bank candidate `(type, date)`: a **decision/statement-type** communication from the matching bank within ±N days (N to pin during implementation, default ±1) → `confirm`. `contradict` is **conservative** — it fires only when the dataset has a clearly decision-type communication for that bank on a *different* date inside the candidate's expected meeting window *and none* on the candidate date; ordinary speeches/non-decision docs never trigger `contradict`. If `doc_type` cannot be reliably classified, the verdict is `unknown`, never `contradict`. No coverage for the period → `unknown`. This conservatism is deliberate: `contradict` drives quarantine, and a noisy validator would trip the >1% quarantine stop (§5.3). |
| Test fixture path | `tests/fixtures/event_sources/hf_central_bank_sample.parquet` |
| Failure behavior | Dataset unreachable → **all verdicts `unknown`**, log + degrade. A validator failure must **not** fail the run — central-bank candidates still promote at `confidence = medium` (§5.2). |

---

## 7. Candidate Store & Outputs

### 7.1 Persistent artifacts

| Path | Content | Store role |
|---|---|---|
| `data/raw/acquisition.db` (`AcquisitionStore`) | Raw fetched HTML/parquet source pages + run metadata | **System of record** for provenance (unchanged role). |
| `data/raw/event_calendar/candidates/event_candidates.parquet` | All originated `EventCandidate`s, this run | Derived output. |
| `data/raw/event_calendar/candidates/event_validations.parquet` | All `ValidationResult`s, this run | Derived output. |
| `data/raw/event_calendar/candidates/quarantine.parquet` | Candidates whose `PromotionDecision.outcome == "quarantine"` | Derived output. |
| `configs/events/us_events.yaml` | Final promoted rows (ADR 0002 schema) | Final output. |
| `event_calendar_fetch_report.json` | Run summary (extended — §7.2) | Final output. |

Each parquet file is registered in `AcquisitionStore.derived_outputs` with path,
row count, and min/max date — so the SQLite DB stays the single index of all
persistent state (AGENTS.md rule 8). The parquet files are *derived and
reproducible*, not shadow state.

### 7.2 Fetch report extension

`event_calendar_fetch_report.json` gains a `group_a` block: per-type candidate
count, promoted count, quarantined count, and the source ids consulted. The
existing FOMC/CPI/NFP report fields are unchanged.

### 7.3 Refresh model

Full re-fetch, full re-render every run (Q4 decision A) — matches the existing
`run_us_event_calendar_fetch` pattern. "Self-updating" = a scheduled re-run, not
incremental-merge logic. The candidate parquet records `run_id` and the run's
`as_of` date, so history is preserved across runs without incremental state.

---

## 8. YAML Rendering & Reproducibility

The acceptance criterion "YAML regenerable from candidate + validation artifacts"
is met by a **pure render function**:

```python
render_events_from_candidates(
    candidates: list[EventCandidate],
    decisions: list[PromotionDecision],
) -> list[ScheduledEvent]
```

It takes only promoted candidates + decisions — no network, no clock. Given the
same parquet inputs it produces byte-identical YAML. This is enforced by a test
(§10) that loads the parquet fixtures and asserts the rendered YAML equals a
golden file. The existing `_render_events_yaml` is reused for the final string
form; rows are merged with FOMC/CPI/NFP and sorted by
`(release_timestamp_et, type)` exactly as today.

---

## 9. Error Handling & Validation

Per AGENTS.md error policy:

- **Deterministic errors** (parse mismatch, 0 rows from a page that must have
  rows, per-year count sanity failure) → **raise**, fail the run loud.
- **External flakiness** (5xx, timeout, dataset unreachable) → catch, log, degrade.
  A *validator* degrading is tolerable (verdicts become `unknown`). A *primary
  adapter* producing zero candidates for its bank is a **failed run** (status
  integrity — partial success with a silent skip = failed run).
- **Row-level bad external data** → quarantine file + log, continue; > 1%
  quarantine rate stops the run (§5.3).
- **Per-year count sanity check** — modeled on the existing
  `validate_fomc_listing_integrity` / `_validate_bls_events`: each completed year
  2016→ should have ≈ 8 decisions for ECB/BoE/BoJ. Exact expected counts and the
  handling of the current incomplete year are **pinned during implementation**
  against captured fixtures.

All logging uses the stdlib `logging` module, UTC timestamps, with actionable
ERROR lines.

---

## 10. Testing Strategy

Per AGENTS.md rule G (real tests, real fixtures, real names):

- **Per-adapter unit tests** — each adapter parsed against a **captured real**
  HTML/parquet fixture under `tests/fixtures/event_sources/`. Assertions check
  exact dates and per-year counts, not `is not None`. Fixtures include at least
  one historical-archive page and one current/forward page per central bank.
- **Validator unit test** — `validators_hf_central_bank` against
  `hf_central_bank_sample.parquet`, asserting `confirm` / `contradict` /
  `unknown` verdicts on known central-bank dates.
- **Triangulation unit tests** — each promotion rule in §5.2 exercised with
  realistic `EventCandidate` / `ValidationResult` inputs using real event types
  and real ECB/BoE/BoJ dates (no `step1`/`x` toy names).
- **Reproducibility test** — load candidate + validation parquet fixtures, run
  `render_events_from_candidates`, assert byte-identical to a golden YAML.
- **Integration test** — `run_us_event_calendar_fetch` with
  `include_v2_curated_candidates=True` and adapters pointed at fixture fetchers,
  asserting the end-to-end path writes the three parquet files, registers them in
  `derived_outputs`, and emits the expected Group A rows into `us_events.yaml`.
  Also asserts FOMC/CPI/NFP rows are unaffected and the `budget` Sep-30 block is
  unchanged.
- **Quarantine-path test** — a `contradict` verdict routes the row to
  `quarantine.parquet` and keeps it out of YAML.

---

## 11. Open Items to Verify During Implementation

1. Exact ECB decisions-archive and Governing Council calendar URLs, and which
   page gives the cleanest forward coverage.
2. Exact BoE historical MPC-decisions archive URL; whether the Bank Rate history
   CSV is worth wiring as a second validator-grade cross-check.
3. BoJ historical per-year MPM schedule URL pattern.
4. HF dataset `aufklarer/central-bank-communications` — schema, columns,
   coverage end date, license, and the parquet HTTP endpoint shape.
5. Per-year decision-count sanity thresholds for ECB/BoE/BoJ 2016→, and
   current-incomplete-year handling.
6. The ±N-day window for the HF validator's `confirm` verdict (default ±1).

---

## 12. Acceptance Criteria (Spec 1)

- [ ] `src/regime_data_fetch/event_sources/` package exists with the seven
      Group A/core files; registry is open for Spec 2 extension.
- [ ] ECB / BoE / BoJ decision dates cover 2016 → latest published forward date,
      sourced from official current **and** archive pages.
- [ ] `election` rows for 2016–2028 are produced by `deterministic_election.py`
      with FEC-page provenance and a fixture cross-check.
- [ ] Every promoted row traces to: exact date, event type, source id/URL,
      `is_future_scheduled`, and candidate `confidence` in
      `event_candidates.parquet`; promotion-level fields (`source_count`,
      `requires_manual_review`, outcome, reason) come from the matching
      `PromotionDecision` and may be denormalized into candidate artifacts
      for operator review.
- [ ] `event_candidates.parquet`, `event_validations.parquet`,
      `quarantine.parquet` are written and registered in
      `AcquisitionStore.derived_outputs`.
- [ ] `us_events.yaml` is regenerable from the parquet artifacts via
      `render_events_from_candidates` (reproducibility test passes).
- [ ] Quarantine path works; > 1% quarantine rate stops the run.
- [ ] All tests pass; FOMC/CPI/NFP rows and the `budget` Sep-30 block are
      byte-unchanged.
- [ ] No LLM calls; resolver exists only as an unused hook.
