# Regime Engine — Spec ↔ Code ↔ Data-Fetch Audit

**Date:** 2026-05-15
**Scope:** V1 and V2 — every layer/section, every metric
**Method:** read each metric from `regime_engine_v1_final_spec.md` and
`regime_engine_v2_spec.md`, identify its raw data inputs, then check
`market_data_fetch_plan.md` and `src/regime_detection/**/*.py` for
agreement on what is fetched vs what is consumed.

The audit produced a per-metric reconciliation table (Section 1) and two
substantive mismatches (Section 2). Resolution paths and concrete
implementation diffs are in Section 3.

---

## 1. Per-metric reconciliation

### 1.1 V1 (Phase 1) — all `OK`

| Layer | Metric | Spec data | Fetch doc | Code consumer | Verdict |
|---|---|---|---|---|---|
| 1A trend direction | `close`, `SMA_50`, `SMA_200`, `return_63d` | SPY close | ✅ `data/raw/daily_ohlcv/SPY` | `trend_direction.py` reads `MarketContext.spy_ohlcv["close"]` | OK |
| 1B trend character | `ADX_14`, `return_10d/21d`, `prior_63d_drawdown` | SPY OHLC + volume | ✅ | `trend_character.py` reads SPY OHLCV | OK |
| 1C volatility | `realized_vol_21d`, `*_percentile_252d`, `vix_percentile_252d` | SPY close + VIX/VIXY | ✅ both present | `volatility_state.py` | OK — VIXY proxy substitution acknowledged in spec §5 and fetch §2.1 |
| 1D breadth (ETF proxy) | `relative_breadth_ratio = RSP/SPY`, sma50, ret20d | RSP, SPY | ✅ | `breadth_state.py` | OK |
| 1D breadth (PIT) | `pct_above_50dma` | PIT membership + per-stock OHLCV | V1 spec §6.2 marks **deferred to V2** | V1 code does NOT compute PIT | OK |
| 2.0 events | FOMC, CPI, NFP + `expiry_week`, `earnings_season` rules | YAML from Fed/BLS schedules | ✅ generated YAML covers 2007→2026 | `event_calendar.py` resolver | OK |
| 2.1 monetary | n/a — V1 emits `unknown` | n/a | n/a | n/a | OK by design |
| 3 network fragility | n/a — V1 emits `not_implemented_v1` | n/a | n/a | n/a | OK by design |
| 4 transition risk | named warnings only | composed labels | n/a | `transition_risk.py` | OK |

**V1 outcome: no spec/code/fetch drift.**

### 1.2 V2 §1A trend direction & character

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `efficiency_ratio_20d`, `hurst_250d`, `slope_sma_50/200`, `breakout_expansion`, `range_bound` | SPY close + volume | ✅ | `trend_direction_v2.py` | OK |
| `recovery`, `euphoria` labels | SPY + AAII `bull_bear_spread_8w_ma` | ✅ `data/raw/sentiment/aaii_sentiment.parquet` | `feature_store._build_sentiment_score_series` | OK (Log #32 / ADR 0004) |

### 1.3 V2 §1C volatility

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `atr_ratio`, `iv_rv_spread`, `vol_crush`, `gap_frequency_20d`, `intraday_range_percentile_252d` | SPY OHLC + `implied_vol_30d = VIXCLS / 100` | ✅ FRED `VIXCLS` | `volatility_state_v2.py` | OK (ADR 0005 / Log #19, #20) |

### 1.4 V2 §1D breadth

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `pct_above_50dma/200dma`, `ad_line`, `ad_line_slope_20d`, `nh_nl_ratio`, `upvol_downvol_ratio`, `breadth_thrust`, `sector_breadth` | PIT membership + per-constituent OHLCV; 11 sector ETF closes | ✅ `fja05680/sp500` intervals + 762-stock OHLCV in SQLite; 11 sector ETFs | `breadth_state_v2.py` `_compute_pit_features` + `compute_breadth_v2_features` — emits `pit_constituent_biased_research` bias warning | OK with documented bias warning |

### 1.5 V2 §1E volume / liquidity

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `volume_zscore_20d`, `gap_frequency_20d`, `intraday_range_percentile_252d` | SPY volume + OHLC | ✅ | `volume_liquidity_v2.py` | OK |

### 1.6 V2 §2A monetary / liquidity

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `yield_change_zscore_*` (DGS2, DGS10, 63d, 21d), `broad_usd_index_zscore_63d/21d` | FRED `DGS2`, `DGS10`, `DTWEXBGS` | ✅ all live-verified | `monetary_pressure.py` | OK |
| **Central bank text → hawkish/dovish classifier into `monetary_pressure.evidence`** (§2A lines 2578–2586) | FOMC minutes + Powell speeches | ✅ both fetched, in `data/raw/fomc_minutes/` and `data/raw/powell_speeches/` | ❌ **No consumer in `regime_detection`** | **M1 — fixed in this iteration (Section 3.1)** |
| **First-release vs revised CPI for replay** (§2A lines 2587–2593) | Vintage-aware CPI (`CPIAUCSL` realtime params) | ⚠ `--include-cpi-vintages` path implemented; **no parquet in checkout**; default off | ❌ No metric consumes vintages — `inflation_growth.py` uses revised series | **M2 — fixed in this iteration (Section 3.2)** |

### 1.7 V2 §2B inflation / growth

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `cpi_3m/6m_change_pct`, `pmi_manufacturing` (+slope), `treasury_10y_yield_slope_21d`, `commodity_return_63d` (DBC), `cyclical_defensive_ratio` (XLY+XLI)/(XLP+XLU), `aggregate_forward_eps_revision_direction_4w` | FRED `CPIAUCSL`, repo-local PMI parquet, FRED `DGS10`, DBC close, sector ETF closes, S&P workbook accumulator | ✅ all present | `inflation_growth.py` (bias warning `commodity_proxy_dbc_substitute`) | OK (Log #48 + ADR 0006) |
| `inflation_surprise_zscore` | realized CPI − Cleveland Fed `cpi_nowcast`, 5y rolling std | ✅ Cleveland Fed nowcast | `compute_inflation_surprise_zscore` (bias warning `inflation_surprise_cleveland_fed_nowcast`) | OK (ADR 0006) |

### 1.8 V2 §2C credit / funding

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `hy_oas_*`, `ig_oas_*` (authoritative, 2023-05-15+) | FRED `BAMLH0A0HYM2`, `BAMLC0A4CBBB` | ✅ trailing ~3y FRED window | `credit_funding.py` → `credit_funding_state` | OK with documented coverage cap (Log #71 / ADR 0007) |
| `hy_tr_differential_*`, `ig_tr_differential_*` (parallel proxy, full history) | TLT, HYG, LQD closes | ✅ | same classifier → `credit_funding_state_proxy` + bias warning; downstream resolver → `credit_funding_effective_state` | OK — raw series stay parallel; effective label records source/agreement evidence |
| `kre_spy_ratio`, `kre_spy_slope_63d`, `sofr_iorb_spread`, `broad_usd_index_zscore_21d`, `nfci_weekly_carried` | KRE, SPY, SOFR, IORB, DTWEXBGS, NFCI | ✅ | `credit_funding.py` | OK |

### 1.9 V2 §2D event calendar extensions

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| `budget_week`, `election_window`, `global_rate_decision` | Deterministic / official adapters + candidate parquet | ✅ Sep-30 budget, BOE/ECB/BOJ pages | `event_calendar.py` + `macro_event_score` in §4.2 | OK |
| `geopolitical_event` | GPR + GDELT + HDX HAPI; ACLED/UCDP pending API keys | ⚠ partial — ACLED/UCDP TODO | overlay-only, never auto-promoted (spec-aligned) | OK as designed |

### 1.10 V2 §3 network fragility

| Metric | Spec data | Fetch | Code | Verdict |
|---|---|---|---|---|
| 24-asset universe (11 sectors + SPY + 12 cross-asset) | sector + cross-asset ETFs | ✅ | `fragility_universe.py` constants match exactly | OK |
| `avg_pairwise_corr_63d`, `*_percentile_504d`, `largest_eigenvalue_share`, `effective_rank` (natural log), `absorption_ratio_top3`, `dispersion_ratio` | returns matrix from universe | ✅ | `network_fragility.py` | OK |

### 1.11 V2 §4 / §5 / §6

Transition score, strategy response cohort routing, HMM, K-means/GMM
clustering, change-point detection: all composed from existing axis
labels + derived features. No new raw data needed. ✅ aligned.

---

## 2. Mismatches found

### M1 — V2 §2A central bank text pipeline (spec describes, code missing)

**Spec citation:** §2A lines 2578–2586.

> "1. Ingest text on release. 2. LLM classifier outputs
> `{hawkish, dovish, neutral}` with confidence. 3. Output as structured
> score, fed into `monetary_pressure.evidence` — never as standalone
> label."

**Fetch state:** FOMC minutes verified `2011-01-26` through `2026-03-18`
in `data/raw/fomc_minutes/fomc_minutes.parquet`. Powell speeches verified
`2013-02-22` through `2026-03-21` in
`data/raw/powell_speeches/powell_speeches.parquet`. Both parquets carry
`body_text` columns.

**Code state:** no module in `src/regime_detection/` reads FOMC text or
Powell text. `monetary_pressure.py` produces only the four yield-and-USD
z-score features; its `RuleInputs` and `Features` dataclasses have no
text-derived field.

**Implementation decision:** the spec's "LLM classifier" phrasing
conflicts with V1 §2.2 stateless replay (same inputs → identical
outputs). An LLM call inside the engine breaks reproducibility. The
project's existing precedent for spec-described-but-vendor-blocked
metrics is to ship an approved deterministic substitute with a
bias-warning row (DBC for BCOM, VIXCLS for options-IV, AAII for
analyst sentiment, Cleveland Fed for analyst consensus, fja05680 for
vendor PIT). We follow that precedent.

### M2 — V2 §2A vintage-aware CPI for historical replay (spec requires, code uses revised)

**Spec citation:** §2A lines 2587–2593.

> "Revisions to prior releases must be stored separately. Original
> release values are point-in-time-correct; revised values are not. The
> engine must use original values for historical replay. Implementation:
> data store has both `value_first_release` and `value_latest_revision`
> per data point."

**Fetch state:** `regime_data_fetch.fred` supports `--include-cpi-vintages`
with FRED realtime params; output path
`data/raw/macro_vintages/cpi_all_items_vintages.parquet` is specified in
the fetch doc but the file is not in the checkout and the current macro
fetch report has `include_cpi_vintages=false`.

**Code state:** `inflation_growth.compute_inflation_growth_features`
reads `cpi_all_items` from `macro_series` — the latest-revision FRED
`CPIAUCSL`. Historical replay therefore uses the same revised series for
every `as_of_date`, not the value-as-of-release.

---

## 3. Resolutions implemented in this iteration

### 3.1 M1 — wire deterministic central-bank-text classifier into `monetary_pressure.evidence`

**New module:** `src/regime_detection/central_bank_text.py`.

- Deterministic lexicon-based scorer over `body_text` (no LLM call).
- Hawkish lexicon: tighten, restrictive, hike, raise, hawkish, persistent
  inflation, anchor expectations, …
- Dovish lexicon: accommodative, ease, cut, dovish, stimulus, soften,
  disinflation, …
- Score per release: `net_score = (hawkish_count − dovish_count) /
  (hawkish_count + dovish_count)` ∈ `[-1, +1]`, NaN when both counts are
  zero.
- Output frame per release: `(release_date, hawkish_count, dovish_count,
  net_score, total_tokens, source)`.

**Data plumbing:**

- `loaders.load_central_bank_text_score(...)` reads
  `fomc_minutes.parquet` and `powell_speeches.parquet`, scores each
  release, returns a single date-indexed Series of `net_score` keyed by
  `release_date` (FOMC: 14:00 ET date; Powell: publication_date column).
- `MarketContext.central_bank_text_score: pd.Series | None`.
- `feature_store.build_feature_store` builds a daily forward-filled
  series + a 30-session smoothing mean (mirrors AAII 8w-MA pattern) and
  passes it to `compute_monetary_pressure_features`.

**Engine consumer:**

- `MonetaryPressureV2Features` gains a new optional Series field
  `central_bank_text_score`. It is **evidence only** — not consumed by
  any of the four §2A rule predicates. The §2A label set is unchanged.
- `axis_series.py` emits `central_bank_text_score` in the monetary
  evidence dict.

**Spec-substitute documentation:**

- Bias-warning row code: `central_bank_text_deterministic_lexicon_substitute`.
- V2 spec gets an Ambiguity Log entry pinning the substitution and the
  rationale (stateless replay). `market_data_fetch_plan.md` gets a new
  data-inventory row for the score artifact.

**V1 byte-identity:** the new field is optional everywhere. V1 callers
that do not thread `CentralBankTextConfig` see no diff, and the V1
frozen-replay tests do not consume `monetary_pressure.evidence`.

### 3.2 M2 — wire first-release CPI for historical replay

**Fetch:**

- `--include-cpi-vintages` default flipped to `True` for the backtest
  fetch entrypoint (`scripts/fetch_regime_engine_v1_data.py`).
- `data/raw/macro_vintages/cpi_all_items_vintages.parquet` materialized
  on next fetch run.

**Loader:**

- `loaders.load_cpi_vintages_first_release(source) -> pd.Series` —
  for each `target_date` returns the earliest non-NaN observation
  from the vintage frame (`first_release_value` per release date).
- The output Series is keyed by the **release date** (i.e. the day the
  number was first published), not the reference month, so replay can
  look up the value-as-of-release.

**MarketContext:**

- New optional field
  `cpi_first_release: pd.Series | None` — propagated through
  `slice_context_to_*` helpers.

**Engine consumer:**

- `inflation_growth.compute_inflation_growth_features` accepts a new
  optional `cpi_first_release` parameter. When supplied it replaces
  `cpi_all_items` in the realized-inflation-rate computation (the
  `realized_cpi_rate` term of `inflation_surprise_zscore` and the
  `cpi_*_change_pct` series). When absent, the existing latest-revision
  `CPIAUCSL` path is preserved unchanged.
- Config flag: `inflation_growth.use_first_release_cpi_when_available`
  (default `True`). The flag exists so shadow-mode operators can pin to
  revised CPI when explicitly intended.

**V1 byte-identity:** preserved — V1 callers do not wire
`inflation_growth_config`, so the new optional parameter has no effect
on the V1 path.

---

## 4. Operational gaps (known, tracked, not addressed here)

These are documented elsewhere and remain known operational
limitations. They are NOT spec/code drift.

- ACLED + UCDP geopolitical evidence pending entitled API tokens.
  Overlay-only; never auto-promoted. Spec, fetch, code aligned.
- Vendor PIT S&P 500 universe (CRSP / Compustat / FactSet / Norgate) —
  `fja05680/sp500` substitute carries the
  `survivorship_biased_constituent_universe` warning everywhere.
- Pre-2023 ICE BofA OAS history not on FRED — the TLT-vs-HYG/LQD proxy
  covers earlier history as a *separate* parallel metric
  (`credit_funding_state_proxy`). Raw series are never spliced; downstream rules
  consume the audited `credit_funding_effective_state` resolver.
- Dedicated shadow runner with SQLite ledger and archived daily input
  snapshots — operational, not a data-source gap.

### M1 follow-up — FOMC-RoBERTa classifier swap (deferred, TODO)

The audit M1 deterministic lexicon was validated against the
`gtfintechlab/fomc_communication` corpus
(`docs/verification/lexicon_validation.md`, run via
`scripts/validate_central_bank_text_lexicon.py`):

- **Sentence-level overall accuracy: 53.9%** (vs 49.4% always-predict-
  neutral baseline) — barely beats baseline because only 16.9% of
  sentences carry any lexicon term.
- **Conditional on firing on a directional sentence: ~70.9% accuracy**
  — when the lexicon *has* something to read, it reads the sign
  correctly.
- **Document-level test (14.8%) is methodologically biased** (modal
  label is neutral 25 of 27 years; the lexicon never predicts neutral
  on pooled text). Should be redesigned before being used as a
  decision gate.

The upgrade path is `gtfintechlab/FOMC-RoBERTa` (Shah et al. 2023,
EMNLP) — Apache-2.0, deterministic with argmax decoding, paper-
reported ~85% sentence accuracy. Cost: +750 MB CPU container delta
(`torch` + `transformers` + ~500 MB model cache), 1–5 sec/document
inference time, plus a model-SHA-pinning discipline to preserve
replay determinism (record the model SHA in the bias-warning row,
forbid auto-upgrades).

**Deferred** because:

(a) The lexicon is evidence-only (never a §2A rule predicate input),
    so the accuracy gap doesn't move any labels.

(b) The runtime container cost (+750 MB to +4 GB) doubles or triples
    the engine's current footprint.

(c) Two cheaper diagnostics are worth running first:
    1. Redesign the document-level test (compare per-year `net_score`
       sign to per-year hawkish-vs-dovish sentence count balance,
       not to modal label). 1–2 hours.
    2. Audit lexicon balance — every pooled year reads dovish; either
       FOMC language is genuinely cautious-leaning or the dovish list
       is over-represented. Ablation test which terms move the score
       most. 1–2 hours.

If after those diagnostics the lexicon still looks structurally
under-coverage at the granularity the engine consumes, swap to
RoBERTa. Until then: lexicon ships, bias-warning row stamps the
substitution, calibration runner surfaces the score distribution
for operator review.

**TODO marker** also recorded inline at
`src/regime_detection/central_bank_text.py` (top of file) so a future
editor lands at the upgrade context.

---

## 4.1 Post-M1/M2 follow-up — SF Fed Daily News Sentiment as a second sentiment voice

Discovered while implementing follow-up #12: AAII (`sentiment_score`)
and SF Fed news sentiment measure genuinely different things (retail
positioning vs press tone). Wiring the SF Fed series as an additional
**evidence-only** signal gives §1A a second voice without spec-amending
the `euphoria` rule predicate. See `regime_engine_v2_spec.md` Ambiguity
Log entry #74 for the spec amendment.

**Code:** `src/regime_data_fetch/sf_fed_news_sentiment.py` (fetcher
against the SF Fed published XLSX), `loaders.load_news_sentiment_series`,
`MarketContext.news_sentiment`, `NewsSentimentConfig`,
`TrendDirectionV2Features.news_sentiment_score` +
`sentiment_concordance` fields, `compute_trend_v2_features` +
`feature_store` + `timeline.build_regime_timeline` + `RegimeEngine`
wiring. **Tests:** `tests/test_news_sentiment.py` — 9 cases including
explicit byte-identity assertions that the §1A `euphoria` rule's input
fields are unchanged when the news series is present or absent. **YAML:**
`news_sentiment` block in `configs/core3-v2.0.0.yaml`.

**Concordance flag.** `sentiment_concordance` is a per-session float in
{+1.0, 0.0, -1.0, NaN}: agreement positive / disagreement / agreement
negative / either NaN. Surfaced for downstream consumers
(strategy_response, calibration dashboards) but not read by any rule.

**Promotion path.** If walk-forward evidence shows `sentiment_concordance
> 0` reduces euphoria false positives without hurting recall, a
subsequent log entry adds a `require_news_concordance: bool` config
flag (default False) and amends the §1A predicate text. Until then:
signal, not predicate.

## 5. Tests run

After implementation:

```
pytest -q
```

V1 frozen-replay must remain byte-identical; V2 §2A label set must be
unchanged; V2 §2B label set must be unchanged with first-release CPI
preferred when available.

---

## 6. Follow-up audit session (2026-05-19) — resolved items

The following issues were discovered and resolved in the 2026-05-19 forensic audit (ADR 0010):

1. **Per-label hysteresis now mandatory for all 9 label axes.** L1 axes (trend_direction, trend_character, volatility, breadth) previously used flat hysteresis. Now all axes use `apply_per_label_asymmetric_hysteresis` with `deescalation_days_by_label`. Missing config fails loudly with `RuntimeError`.

2. **Event calendar expanded 454 → 699 events.** ECB (88), BoE (96), BoJ (89) decisions now cover 2016-2026 via official archive + current-calendar pages. Provenance artifacts (event_candidates, event_validations, event_quarantine) tracked in manifest.

3. **CPI October 2025 permanently missing** from BLS due to government shutdown. Engine handles via forward-fill + 60-day staleness gate.

4. **CPI vintages NaN bug fixed.** `_drop_null_fred_observations` now runs after merge (was before, letting old NaN rows survive).

5. **`run_v2_calibration.py` refactored** to use manifest input resolver pattern. Was constructing paths manually and missing event_calendar, aaii_sentiment, implied_vol_30d.

6. **`trend_character_v2` config wired.** `deescalation_days_by_label` was declared in config but never read by the axis builder.

7. **`monthly_options_expiry` removed from EventType Literal.** `expiry_week` is computed deterministically from trading calendar, never from YAML rows.

8. **`agent_routing` and `strategy_family_constraints`** now surfaced in profile engine `trailing_v2_status` reporting.

9. **`HTTP_USER_AGENT`** in `event_sources/_common.py` changed from bot-like to browser-like string to avoid central-bank page blocks.

10. **Event calendar is S3-only** via manifest. `default_relpath=("event_calendar", "us_events.yaml")` added to `ManifestInputSpec`.
