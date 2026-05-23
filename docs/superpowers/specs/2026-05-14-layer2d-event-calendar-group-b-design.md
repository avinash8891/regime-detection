# Layer 2D Event-Calendar Pipeline — Spec 2: Group B (Curated + Approval-Gated Events)

**Status:** draft — awaiting user review
**Date:** 2026-05-14
**Scope:** Spec 2 of 2. Group B only — `geopolitical_event`, `budget`.
**Companion:** Spec 1 (Group A — `ECB/BoE/BoJ_decision`, `election`) —
`docs/superpowers/specs/2026-05-14-layer2d-event-calendar-group-a-design.md`.
Spec 2 **extends** Spec 1's `event_sources/` package; it does not re-design it.

---

## 1. Problem & Context

Group B covers the two V2 event types that are *not* schedulable from an
official calendar:

- **`geopolitical_event`** — war, invasion, sanctions, terrorism shocks. No
  authoritative registry; the inputs are *risk signals*, not event rows. Today:
  no fetch path at all. v2 spec §2D / Ambiguity Log #50 call it a "manual YAML
  flag," explicitly **excluded** from `macro_event_score` (it scores via the
  separate geopolitical path, not the routine scheduled-event score).
- **`budget`** — US fiscal-deadline risk. Today: a deterministic `Sep 30`
  fiscal-year-end row per year (the inline block Spec 1 §3.3 deliberately left
  untouched). Missing: the events that actually move markets — **debt-ceiling
  X-dates, government shutdowns, continuing-resolution (CR) expirations**.

### What carries over from Spec 1 (unchanged)

The `event_sources/` package, `models.py` core types (`EventCandidate`,
`ValidationResult`, `PromotionDecision`), `orchestrator.py`, the
parquet candidate/validation/quarantine store, `AcquisitionStore.derived_outputs`
registration, the full re-fetch/re-render refresh model, and the wiring through
`_build_v2_curated_candidate_events`. Spec 2 registers new source modules in the
orchestrator's open registry — **no edits to the Spec 1 core**.

### What is genuinely new in Spec 2

1. **Group B inverts the data shape.** Group A sources are clean date lists;
   Group B sources are *signals* (GPR index spikes, GDELT event volume). Spec 2
   adds a **candidate-generation** step — turning signals into dated candidate
   rows — that Group A never needed.
2. **A mandatory human-approval gate.** Group A promotes automatically because
   dates are facts. Group B candidates are *hypotheses* until a human signs off.
3. **The git-tracked approval overlay** — `configs/events/group_b_approvals.yaml`
   — is the persistence boundary for operator decisions (§5).

### Acceptance criteria (inherited + extended)

Every emitted Group B row must trace to: exact date, event type, source id(s)/URL,
`is_future_scheduled`, confidence/source count, `requires_manual_review`, **and
the approval record that promoted it**. `us_events.yaml` Group B rows must be
regenerable from *(regenerated candidates + validations + the approval overlay)*
— never hand-merged.

---

## 2. Non-Goals (Spec 2)

- **No re-design of the Spec 1 core.** New modules register into the existing
  registry; `models.py` only gains additive fields/protocols (§4).
- **No LLM ambiguity resolver.** The mandatory human approval gate *is* the
  resolver for geopolitical; budget auto-promotion uses an objective
  ≥2-official-source rule. The Spec 1 resolver hook stays unused. (Confirm item
  §14.1 — revisit only if review surfaces a concrete need.)
- **No auto-promotion of `geopolitical_event` — ever.** Even with three
  corroborating sources, geopolitical events reach `us_events.yaml` *only* via
  the approval overlay. This is the problem statement's hard rule: "do not
  directly promote noisy search results to final YAML."
- **No new YAML row fields.** ADR 0002's schema is frozen. Budget subtype
  (debt-ceiling / shutdown / CR) lives in the candidate parquet and overlay
  `notes`, not in `us_events.yaml` — the runtime classifier only knows `budget`.
- **No bulk historical scraping via TinyFish.** TinyFish is a *targeted*
  discovery/confirmation tool (§6.4), not a backfill crawler.

---

## 3. Architecture — Module Roles & the Generator/Validator Distinction

### 3.1 The taxonomy question (Confirm item #1)

Spec 1 defined a strict split: **primary adapters originate** candidates that can
auto-promote; **secondary validators only confirm/contradict**, never originate.
Group B does not fit that binary — `geopolitical_event` has no authoritative
primary source, yet *something* must originate its candidates.

Resolution, grounded in your own framing ("secondary validators should
confirm/contradict candidates, **not create final events** by themselves" — note:
final *events*, not *candidates*): Spec 2 introduces a third role.

| Role | Originates? | Can auto-promote? | Spec 1 / Spec 2 |
|---|---|---|---|
| `PrimaryAdapter` | Yes | Yes (via triangulation) | Spec 1 |
| `CandidateGenerator` | Yes — but every candidate carries `requires_manual_review=true` | **No** — overlay-gated only | **Spec 2 (new)** |
| `SecondaryValidator` | No | No | both |

Your original module list named the geopolitical/budget-discovery files
`validators_*`. That naming reflects **authority** (they cannot promote), but
their **role** is `CandidateGenerator` + `SecondaryValidator` combined. The spec
keeps your filenames; if you'd rather rename them `generators_*`, that is
Confirm item #1.

### 3.2 Module layout (Spec 2 adds these files)

```
src/regime_data_fetch/event_sources/
  ... (Spec 1 files unchanged) ...
  deterministic_budget.py        # PrimaryAdapter — formula FY-end deadline (auto-promote)
  budget_official_discovery.py   # CandidateGenerator — debt-ceiling/shutdown/CR via TinyFish+official sources
  validators_tinyfish.py         # TinyFish Search/Extract — confirms candidates, finds moved URLs
  validators_gpr_gdelt.py        # CandidateGenerator+Validator — GPR/GDELT geopolitical candidates
  approvals.py                   # load/validate group_b_approvals.yaml; CLI append helper
```

> Note: Spec 2 splits the budget work into a deterministic adapter
> (`deterministic_budget.py`) and an official-discovery generator
> (`budget_official_discovery.py`) rather than overloading one
> `deterministic_budget.py`. The original single-file name was a misnomer —
> debt-ceiling/CR dates are *not* deterministic. Confirm item #2.

### 3.3 Wiring

Spec 2 modifies `_build_v2_curated_candidate_events` to:

1. **Remove the inline `budget` Sep-30 block** (Spec 1 left it in place) and
   register `deterministic_budget.py` instead — it produces the identical Sep-30
   rows, now flowing through the candidate pipeline with provenance.
2. Register `budget_official_discovery.py`, `validators_tinyfish.py`,
   `validators_gpr_gdelt.py` as Group B sources.
3. Load the approval overlay and pass it to `orchestrator.run(...)`.

No new top-level entrypoint; still behind `include_v2_curated_candidates`.

---

## 4. Core Model Additions (`models.py`)

Additive only — Spec 1 fields/types unchanged.

```python
@dataclass(frozen=True)
class EventCandidate:
    # ... all Spec 1 fields, with their existing trailing defaults ...
    event_subtype: str | None = None  # NEW — "debt_ceiling" | "shutdown" | "cr_expiration"
                                      #       | "fy_deadline" | None (geopolitical: free-text shock label)
    candidate_id: str = ""            # NEW — group identity hash; stamped by the orchestrator
                                      #       post-triangulation, not by generators (see below)

@dataclass(frozen=True)
class ApprovalRecord:                 # NEW — mirrors one group_b_approvals.yaml entry
    event_type: str
    date: date
    approved_label: str
    approver: str
    approved_at: date
    evidence_candidate_id: str
    evidence_source_count: int
    importance: str | None = None     # optional override; default from generator heuristic
    window_days: tuple[int, int] | None = None  # optional override
    notes: str | None = None

class CandidateGenerator(Protocol):   # NEW — see §3.1
    source_id: str
    def generate(self, *, start_year: int, end_year: int,
                 store: AcquisitionStore | None, run_id: int | None) -> list[EventCandidate]: ...
```

**`candidate_id` definition.** Spec 1's model has no single merged candidate
object — a triangulated event is N per-source `EventCandidate` rows plus one
`PromotionDecision`. So `candidate_id` is a **group identity**: after
triangulation, the orchestrator computes
`sha256(f"{event_type}|{date.isoformat()}|{'|'.join(sorted(corroborating_source_ids))}")`,
hex-truncated, and stamps that same value onto **every** `EventCandidate` row for
that `(event_type, date)` key (via `dataclasses.replace`, since the dataclass is
frozen). It defaults to `""` on construction because generators cannot know the
corroborating set — only the orchestrator can, post-triangulation. Stable across
runs as long as the same set of sources corroborates the event; if the
corroborating source set changes, `candidate_id` changes — exactly the signal
used for **stale-evidence detection** (§5.3).

`is_future_scheduled` for `geopolitical_event` is **always `false`** — shocks are
historical actuals, never schedulable. For `budget`: `fy_deadline` and known
future CR expirations *can* be future-scheduled.

---

## 5. The Approval Overlay — `configs/events/group_b_approvals.yaml`

Git-tracked (the `configs/` tree is versioned; `data/raw/` is gitignored). This
is the operator-policy persistence boundary: candidate parquet is regenerated
every run; the overlay is stable, reviewable, portable, PR-auditable.

### 5.1 Schema

```yaml
approvals:
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "a1b2c3d4e5f6a7b8"
    evidence_source_count: 3
    importance: high            # optional; overrides generator heuristic
    window_days: [0, 0]         # optional; overrides SCHEDULED_EVENT_WINDOWS default
    notes: "Russia invasion of Ukraine; promoted after source triangulation."
```

`approvals.py` loads and **validates** this file on every run: well-formed dates,
known `event_type` values, no duplicate `(event_type, date)` keys, required
fields present. A malformed overlay is a **deterministic error → raise** (it is
operator config, not flaky external data).

### 5.2 Promotion rule (Group B)

A Group B row is rendered into `us_events.yaml` **iff** one of:

1. **Overlay-approved** — `(event_type, date)` exists in the overlay **AND** the
   regenerated candidate for that key still exists and still passes validation
   (no `contradict` verdict). Applies to **all `geopolitical_event` rows** and to
   non-deterministic `budget` rows.
2. **Deterministic budget** — `event_subtype == "fy_deadline"` from
   `deterministic_budget.py`. Auto-promotes (formula-derived, like Group A).
3. **Budget with ≥2 independent official confirmations** — a `budget` candidate
   (debt-ceiling/shutdown/CR) corroborated by ≥2 *independent official* sources
   (e.g. Treasury + Congress.gov + GovInfo, extracted via TinyFish). Auto-promotes
   *without* overlay approval — your stated rule. All other budget candidates
   need overlay approval.

Everything not promoted stays in `event_candidates.parquet` with
`requires_manual_review=true` and never reaches YAML.

### 5.3 Stale-approval handling

On a full re-run, an approved key may no longer be backed by the evidence that
justified it. Behavior:

| Situation | Action |
|---|---|
| Approved key, no regenerated candidate exists | Row **not rendered**; report lists it under `stale_approvals` for operator attention. |
| Approved key, candidate exists, `candidate_id` ≠ overlay `evidence_candidate_id` | Row **still rendered** (the `(event_type, date)` decision stands) but flagged `stale_evidence` in the report — the corroborating sources shifted; operator should re-confirm. |
| Approved key, candidate now has a `contradict` verdict | Row **not rendered**; report lists it under `contradicted_approvals` — loud, P1-style. |

This keeps `us_events.yaml` trustworthy: an approval is a decision about a
*(type, date)*, but it cannot resurrect an event the evidence no longer supports.

---

## 6. Per-Source Specifications

URLs/endpoints marked **(verify)** are confirmed during implementation
(AGENTS.md rule F). Every generator validates raw input row-by-row and
quarantines malformed rows (rule I).

### 6.1 `deterministic_budget.py` — `source_id: usa.gov:federal-budget-process`

| Field | Value |
|---|---|
| Source / API | Deterministic — US federal fiscal year ends **Sep 30**. No network call. Reuses the existing source id so historical YAML rows are byte-stable. |
| Coverage | 2016 → end_year, one `fy_deadline` row per year. |
| Future-date support | Yes (formula). |
| Self-updating | N/A. |
| Role | `PrimaryAdapter` — auto-promotes (§5.2 rule 2). |
| Parser fields | `date = Sep 30`, `event_subtype = "fy_deadline"`, `importance = "medium"`, `market = "US"`. |
| Test fixture | None needed (pure formula); covered by a unit test asserting exact dates 2016–2028. |
| Failure behavior | Deterministic; cannot fail externally. |

### 6.2 `budget_official_discovery.py` — `source_id: govinfo/congress/treasury (per row)`

| Field | Value |
|---|---|
| Source / API | Official US budget sources, reached via `validators_tinyfish.py` (search + extract) because the URLs for specific debt-ceiling/CR actions are not stable: **Congress.gov** (api.congress.gov, free key — appropriations bills, CR text) **(verify)**; **Treasury** (debt-limit press releases / `fiscaldata.treasury.gov`) **(verify)**; **GovInfo** (govinfo.gov API — enacted CRs, public laws) **(verify)**. |
| Coverage | 2016 → date. Debt-ceiling X-dates, shutdown start/end, CR expiration dates. Historical only for actuals; *known future* CR expirations may be future-scheduled. |
| Future-date support | Partial — future CR expiration dates yes; debt-ceiling X-dates only when Treasury has published one. |
| Self-updating | Yes (official sites update); fetched fresh each run. |
| License / access risk | Official US-government public domain — low risk. Congress.gov/GovInfo API keys are free; record them as env-var-backed (AGENTS.md — no secrets in code). |
| Role | `CandidateGenerator` — emits `requires_manual_review=true` budget candidates with `event_subtype` ∈ {`debt_ceiling`, `shutdown`, `cr_expiration`}. Auto-promotes only under §5.2 rule 3 (≥2 independent official sources). |
| Parser fields | `date`, `event_subtype`, `raw_title`, `raw_snippet` (the extracted official-text excerpt), `source_id`, `source_url`. |
| Test fixture | `tests/fixtures/event_sources/budget_congress_cr.json`, `budget_treasury_debt_limit.html` (captured real responses, redacted). |
| Failure behavior | A source unreachable → log + degrade (fewer confirmations → more rows fall to manual review, which is safe). All official sources down → budget discovery contributes nothing this run; `deterministic_budget.py` still produces `fy_deadline` rows, so the run is not failed by this alone. |

### 6.3 `validators_gpr_gdelt.py` — `source_id: gpr:caldara-iacoviello`, `gdelt:events-v2`

| Field | Value |
|---|---|
| Source / API | **GPR / AI-GPR** (Caldara-Iacoviello Geopolitical Risk index) — live fetch from `https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls`, monthly country context from `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls`, and AI-GPR context from `https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv`, `ai_gpr_eventtype_monthly.csv`, and `ai_gpr_country_monthly.csv`. **GDELT Event Database daily exports** — live fetch from `http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip`, parsed for CAMEO root event codes `14/18/19/20` and material-conflict `QuadClass=4`. **ACLED** — client implemented for `https://acleddata.com/api/acled/read`, but live raw-event pulls are TODO pending an entitled API key/account; a Gmail/Open myACLED token is not enough. **Uppsala/UCDP GED Candidate** — client implemented for `https://ucdpapi.pcr.uu.se/api/gedevents/26.0.3`, TODO pending `UCDP_ACCESS_TOKEN`. **HDX HAPI conflict-events** — `https://hapi.humdata.org/api/v2/coordination-context/conflict-events` monthly/admin evidence requiring `HDX_HAPI_APP_IDENTIFIER`, or both `HDX_HAPI_APP_NAME` and `HDX_HAPI_APP_EMAIL`; missing app identity is logged and skipped. |
| Coverage | GPR: daily 1985→ (monthly republish). GDELT: 2015→ present. ACLED / Uppsala-UCDP coverage is pending API-key entitlement; HDX is monthly/admin aggregate evidence, not a daily shock row. |
| Future-date support | **None** — geopolitical events are unscheduled by nature. |
| Self-updating | GPR: monthly maintainer republish (static snapshot between). GDELT: continuous. |
| License / access risk | GPR: free for research, cite the paper. GDELT: open event exports. ACLED and Uppsala/UCDP require entitled credentials/tokens and remain TODO for live raw-event pulls; HDX HAPI requires an app identifier and is transformed from ACLED into monthly/admin aggregates. Cache available pulls as `AcquisitionStore` artifacts for reproducibility. |
| Role | `CandidateGenerator` (spike/event/aggregate → `requires_manual_review=true`) **+** `SecondaryValidator` (nearby independent source dates corroborate one another → `confirm`). GPR evidence never auto-promotes a `geopolitical_event`; rendering still requires the approval overlay. |
| Parser fields | GPR daily parser retains `GPRD`, `GPRD_ACT`, `GPRD_THREAT`, `GPRD_MA7`, `GPRD_MA30`, `N10D`, and optional `event` text. The detector still requires a headline `GPRD` spike before emitting a candidate, then uses acts/threats/persistence/article-count evidence to set candidate subtype, confidence, importance, suggested `window_days`, and review snippets. Monthly GPR adds top `GPRC_*` country context for the candidate month. AI-GPR adds same-day `GPR_AI`, top monthly event type, and top monthly country/role context. GDELT daily export: `SQLDATE`, `EventRootCode`, `QuadClass`, `NumMentions`, `SOURCEURL` → per-day `event_count`, `raw_title`, `raw_snippet`, `source_url`. ACLED: `event_date`, `event_type`, `country`, `fatalities`. UCDP: `date_start`, `country`, `best` / death fields, `source_article`. HDX HAPI: `reference_period_start`, `event_type`, `events`, `fatalities`, `location_name`. |
| Test fixture | Inline fixture rows in `tests/test_event_source_group_b.py` cover GPR spike detection, injected GDELT volume CSV, real-shaped GDELT daily export ZIP/TSV parsing, and ACLED/UCDP/HDX JSON parsing/generator wiring. |
| Failure behavior | Source unreachable → log + degrade; that source contributes no candidates/verdicts. A geopolitical run with *zero* generated candidates is **not** a failed run (there may simply be no shocks in range) — but it is reported explicitly. |

### 6.4 `validators_tinyfish.py` — `validator_id: tinyfish:search-extract`

| Field | Value |
|---|---|
| Source / API | TinyFish Search/Extract MCP (`mcp__tinyfish__*`). Requires authentication (`mcp__tinyfish__authenticate`) — Confirm item #4: is TinyFish auth provisioned for the fetch environment? |
| Role | (a) **Confirmation** — for each generated `geopolitical_event` candidate, TinyFish Search retrieves news/official coverage for that date; corroborating coverage → `confirm` `ValidationResult` **and** fills `raw_title`/`raw_snippet` with a human-readable description (this is what makes a candidate *reviewable* — the operator sees "2022-02-24: Russia invades Ukraine," not "GPR percentile 99.4"). (b) **URL discovery** — when an official budget-source URL has moved, TinyFish Search locates the current page so `budget_official_discovery.py` can extract it. |
| Coverage | On-demand per candidate; not a bulk source. |
| Future-date support | N/A. |
| Self-updating | N/A. |
| License / access risk | Per-call external dependency. **Hard rule:** TinyFish output **never auto-promotes** — it only enriches/confirms candidates that still pass through the approval overlay. |
| Test fixture | `tests/fixtures/event_sources/tinyfish_extract_sample.json` (captured real response, redacted). |
| Failure behavior | Unauthenticated or unreachable → all verdicts `unknown`, candidates keep their generator-supplied `raw_*` fields, log + degrade. A TinyFish failure must **not** fail the run and must **not** block manual review — it only means less enrichment. |

---

## 7. Candidate Generation — Signal → Candidate

This is the step Group A never needed. Pinned design, with thresholds as confirm items.

### 7.1 Geopolitical

1. **Seed dates.** `validators_gpr_gdelt.py` scans the GPR daily index over
   2016→ and flags **spike days** — days where headline `GPRD` exceeds a
   trailing-window threshold (Confirm item #5 — default: value above
   trailing-252-day mean + 3·std). The daily parser retains `GPRD`, `GPRD_ACT`,
   `GPRD_THREAT`, `GPRD_MA7`, `GPRD_MA30`, `N10D`, and optional `event` text.
   A headline `GPRD` spike is still required before emitting a candidate;
   acts, threats, moving-average persistence, article count, optional event
   text, monthly country GPR, and AI-GPR context only enrich the emitted candidate.
   For each requested-year GPR spike
   date, the generator fetches GDELT daily Event export ZIPs for the spike
   window and flags material conflict/protest volume rows from the raw export.
   HDX HAPI adds extra aggregate candidate rows for the requested years when
   `HDX_HAPI_APP_IDENTIFIER` is configured, or when both `HDX_HAPI_APP_NAME`
   and `HDX_HAPI_APP_EMAIL` are configured. TODO: ACLED and Uppsala/UCDP raw-event
   rows remain pending entitled API keys/account access; missing credentials or
   denied access degrade to skipped sources.
2. **Merge / dedup.** Spike days from GPR and GDELT within a small window of
   each other (Confirm item #6 — default ±2 calendar days) collapse to a single
   candidate keyed `(geopolitical_event, anchor_date)`. The anchor date is the
   GPR spike day if present, else the GDELT peak day. `source_count` reflects how
   many independent sources flagged it.
3. **Enrich.** `validators_tinyfish.py` retrieves coverage for the anchor date →
   fills `raw_title`/`raw_snippet` and adds a `confirm` verdict.
4. **GPR evidence enrichment.** GPR acts/threats/persistence/article-count
   evidence sets candidate subtype (`gpr_acts_spike`, `gpr_threats_spike`, or
   `gpr_headline_spike`), confidence, importance, suggested event `window_days`
   (`(0, 0)`, `(-1, 3)`, or `(-2, 5)`), and review snippets. This is
   quantitative review evidence, not a promotion rule; the operator can still
   override `importance` via the overlay's optional `importance`.
5. **Output.** A `geopolitical_event` candidate, `requires_manual_review=true`,
   `is_future_scheduled=false`, into `event_candidates.parquet`. **Never
   auto-promoted** — overlay only.

### 7.2 Budget

1. `deterministic_budget.py` emits `fy_deadline` rows (auto-promote).
2. `budget_official_discovery.py` searches official sources (via TinyFish for
   moved URLs) for debt-ceiling X-date announcements, shutdown start/end dates,
   and CR expiration dates → emits `requires_manual_review=true` candidates with
   the appropriate `event_subtype`.
3. Triangulation counts *independent official* corroborations. ≥2 → auto-promote
   (§5.2 rule 3); else → overlay-gated.

---

## 8. Orchestrator Extensions (`orchestrator.py`)

Additive to Spec 1's `run()`:

- Accepts `approval_overlay: list[ApprovalRecord]` (loaded by `approvals.py`).
- After triangulation, the **Group B promotion path** applies §5.2: deterministic
  budget auto-promotes; ≥2-official budget auto-promotes; everything else is
  rendered only if overlay-approved and not stale (§5.3).
- Group A promotion (Spec 1 §5.2) is untouched and runs alongside.
- The fetch report (§9) gains a `group_b` block: per-type candidate count,
  promoted count, manual-review-pending count, `stale_approvals`,
  `stale_evidence`, `contradicted_approvals`.
- The resolver hook stays `None` (§2).

---

## 9. Candidate Store & Outputs

Extends Spec 1 §7 — same three parquet files, now also carrying Group B rows,
same `AcquisitionStore.derived_outputs` registration.

| Path | Group B addition |
|---|---|
| `event_candidates.parquet` | Group B candidates incl. `candidate_id`, `event_subtype`, `requires_manual_review`. |
| `event_validations.parquet` | TinyFish + cross-source verdicts. |
| `quarantine.parquet` | Malformed Group B rows + `contradict`-quarantined candidates. |
| `configs/events/group_b_approvals.yaml` | **New, git-tracked** — the approval overlay. |
| `configs/events/us_events.yaml` | Now also carries overlay-promoted + auto-promoted Group B rows. |
| `event_calendar_fetch_report.json` | New `group_b` block (§8). |

---

## 10. CLI Approval Helper

`approvals.py` exposes the load/validate API plus an append helper, wired to a
thin script `scripts/approve_group_b_candidate.py` (matches the flat
`scripts/` layout). Workflow:

1. Operator runs the fetch → unapproved candidates land in
   `event_candidates.parquet` with `requires_manual_review=true`.
2. Operator inspects candidates (a `--list-pending` mode prints them with
   `candidate_id`, sources, `raw_title`/`raw_snippet`).
3. Operator approves: `python scripts/approve_group_b_candidate.py --candidate-id <id> --approver <name> --notes "..."`.
   The helper looks up the candidate, appends a fully-formed `ApprovalRecord` to
   `group_b_approvals.yaml` (no raw hand-editing), and re-validates the overlay.
4. Next fetch run renders the now-approved row into `us_events.yaml`.

The helper never edits `us_events.yaml` directly — promotion always flows through
the orchestrator's render path, preserving "regenerable from artifacts."

---

## 11. YAML Rendering & Reproducibility

Spec 1's `render_events_from_candidates` gains a third input:

```python
render_events_from_candidates(
    candidates: list[EventCandidate],
    decisions: list[PromotionDecision],
    approval_overlay: list[ApprovalRecord],
) -> list[ScheduledEvent]
```

Still pure — no network, no clock. Given the same parquet inputs **and** the same
overlay, output YAML is byte-identical. Reproducibility test (extends Spec 1 §10):
load candidate/validation parquet fixtures + a fixture overlay, render, assert
byte-identical golden YAML — including the auto-promote, overlay-promote, and
stale-approval paths.

---

## 12. Error Handling & Validation

Per AGENTS.md error policy:

- **Overlay malformed** → deterministic error, **raise** (operator config).
- **A `CandidateGenerator` source unreachable** → log + degrade; fewer candidates
  / fewer confirmations. Safe — it pushes rows toward manual review, not toward
  silent promotion.
- **TinyFish unavailable** → verdicts `unknown`, candidates keep generator `raw_*`;
  run continues.
- **Zero geopolitical candidates generated** → not a failure (range may be quiet),
  but reported explicitly.
- **`contradict` on an approved key** → row withheld, surfaced loud in the report
  (§5.3) — treated as a P1-style data-integrity signal.
- **Quarantine rate > 1%** → stop the run (inherited from Spec 1 §5.3).
- All logging via stdlib `logging`, UTC, actionable ERROR lines.

---

## 13. Testing Strategy

Per AGENTS.md rule G — real fixtures, real names (real shock dates like
`2022-02-24`, real CR dates; no `step1`/`x`):

- **`deterministic_budget`** — unit test asserting exact `Sep 30` dates 2016–2028.
- **`budget_official_discovery`** — parse captured Congress.gov/Treasury/GovInfo
  fixtures; assert exact debt-ceiling/CR dates and `event_subtype`.
- **`validators_gpr_gdelt`** — spike detection over `gpr_index_sample.csv` asserts
  known shock dates surface (e.g. 2022-02-24) and quiet periods do not; GDELT
  corroboration produces `confirm` verdicts.
- **`validators_tinyfish`** — against `tinyfish_extract_sample.json`: enrichment
  fills `raw_title`/`raw_snippet`; unauthenticated path yields `unknown` and does
  not raise.
- **`approvals`** — overlay load/validate: rejects duplicate keys, bad dates,
  unknown `event_type`; round-trips the append helper.
- **Group B promotion tests** — each §5.2 path: deterministic auto-promote,
  ≥2-official budget auto-promote, overlay-approved geopolitical promote,
  unapproved geopolitical withheld, all three §5.3 stale paths.
- **Reproducibility test** — §11.
- **Integration test** — `run_us_event_calendar_fetch` with
  `include_v2_curated_candidates=True`, Group B sources pointed at fixture
  fetchers + a fixture overlay: asserts parquet written + registered, Group B
  rows in YAML match the overlay, Group A + FOMC/CPI/NFP rows unaffected, and the
  old inline Sep-30 budget rows are reproduced byte-identically by
  `deterministic_budget.py`.

---

## 14. Open Items to Confirm During Review/Implementation

1. **Module naming** — keep `validators_gpr_gdelt.py` / `validators_tinyfish.py`,
   or rename to `generators_*` to match the `CandidateGenerator` role (§3.1).
2. **Budget file split** — `deterministic_budget.py` +
   `budget_official_discovery.py` (recommended) vs. one file (§3.2).
3. **ACLED / Uppsala-UCDP API keys** — client code is implemented in
   `validators_gpr_gdelt.py`, but live raw-event fetches remain TODO pending
   entitled API keys/account access. A Gmail/Open myACLED token currently returns
   an API denial for raw ACLED data. HDX HAPI remains monthly/admin aggregate
   corroboration, not daily truth.
4. **TinyFish auth** — is `mcp__tinyfish__*` authentication provisioned for the
   fetch environment? (§6.4)
5. **GPR spike threshold** — default trailing-252-day mean + 3·std or top-0.5
   percentile (§7.1).
6. **Cross-source merge window** — default ±2 calendar days (§7.1).
7. **GPR/GDELT exact source URLs/endpoints** (§6.3).
8. **Congress.gov / GovInfo / Treasury endpoints + free API key handling** (§6.2).

---

## 15. Acceptance Criteria (Spec 2)

- [ ] `event_sources/` gains `deterministic_budget.py`,
      `budget_official_discovery.py`, `validators_tinyfish.py`,
      `validators_gpr_gdelt.py`, `approvals.py` — registered in the existing
      orchestrator registry with **no edits to the Spec 1 core**.
- [ ] `models.py` gains `candidate_id`, `event_subtype`, `ApprovalRecord`,
      `CandidateGenerator` — additive only; Spec 1 types unchanged.
- [ ] `configs/events/group_b_approvals.yaml` exists, git-tracked, schema-validated
      on load.
- [ ] `geopolitical_event` rows reach `us_events.yaml` **only** via the overlay —
      never auto-promoted, even with multiple corroborations.
- [ ] `budget` `fy_deadline` rows auto-promote and reproduce the prior inline
      Sep-30 rows byte-identically; debt-ceiling/shutdown/CR rows auto-promote
      only with ≥2 independent official confirmations, else overlay-gated.
- [ ] Every promoted Group B row traces to candidate evidence + (where applicable)
      an `ApprovalRecord`; `us_events.yaml` is regenerable from
      *(candidates + validations + overlay)*.
- [ ] Stale-approval, stale-evidence, and contradicted-approval paths behave per
      §5.3 and are surfaced in the fetch report.
- [ ] `scripts/approve_group_b_candidate.py` lists pending candidates and appends
      validated `ApprovalRecord`s without hand-editing YAML.
- [ ] No LLM calls; resolver hook stays unused.
- [ ] All tests pass; Group A + FOMC/CPI/NFP rows byte-unchanged.
