# GPR Context And AI-GPR Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the GPR geopolitical evidence slice by adding persistence-derived review windows, monthly country context, AI-GPR context, and source-provided qualitative snippets while preserving approval-gated `geopolitical_event` promotion.

**Architecture:** Keep `GPRGDELTSignalGenerator` as the single geopolitical candidate generator. Add optional fetchers/parsers for monthly GPR export and AI-GPR CSVs, derive context keyed by month, and enrich only the existing GPR candidate fields (`raw_title`, `raw_snippet`, `event_subtype`, `confidence`, `importance`, `window_days`). Do not change the Group B promotion rules or event-calendar classifier.

**Tech Stack:** Python, pandas, stdlib URL fetchers, existing `EventCandidate` and `SourceFetchStatus`, RTK/pytest.

---

## Task List

- [ ] Add tests for persistence-derived `window_days` on GPR candidates:
  - one-day spike remains `(0, 0)`
  - `persistent_7d` suggests `(-1, 3)`
  - `persistent_30d` suggests `(-2, 5)`
- [ ] Add tests for monthly country GPR parsing from `data_gpr_export.xls`/CSV-shaped fixtures:
  - parse month/date column
  - detect top country columns for the candidate month
  - include country context in GPR candidate evidence
- [ ] Add tests for AI-GPR parsing from official CSV-shaped fixtures:
  - daily AI-GPR context confirms the candidate date
  - event-type monthly context adds top event type
  - country monthly context adds top country/role
- [ ] Add fetchers and constants for:
  - `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls`
  - `https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv`
  - `https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_eventtype_monthly.csv`
  - `https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_country_monthly.csv`
- [ ] Wire fetch/parse failures as degraded source statuses, not fatal run failures.
- [ ] Update candidate evidence:
  - keep headline `GPRD` spike required
  - add source-provided `event` text to title when present
  - add `suggested_window_days` through `EventCandidate.window_days`
  - add monthly country and AI-GPR context to `raw_snippet`
- [ ] Update docs to state what is implemented and what remains out of scope.
- [ ] Run focused Group B tests, event-calendar approval tests, ruff, and diff check.

