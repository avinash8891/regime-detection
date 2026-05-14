# Decision 0006: V2 §2B `inflation_surprise_zscore` — Cleveland Fed nowcast substitutes for analyst `consensus_estimate`

**Status:** accepted — `consensus_estimate` is substituted with the Cleveland Fed inflation nowcast (model-derived current-period CPI inflation rate).

Spec owner directed this path explicitly ("fednowcast spec amend. do all"). The recommendation below became the chosen pin; §2B spec text amended and Ambiguity Log #48's `inflation_surprise_zscore` blocker closed in the same commit as this status flip.

**Context:**
The V2 §2B `inflation_shock` label has a two-limb OR rule (spec lines 2550-2555):

```text
inflation_shock:
  (inflation_surprise_zscore > +1.5)            # single-signal limb
  OR (commodity_return_63d > 0.15 AND ...)       # composite limb — already ships
```

The composite limb ships today. The single-signal limb consumes `inflation_surprise_zscore`, defined (spec line 2505) as:

```text
inflation_surprise_zscore = (actual_release - consensus_estimate) / std_of_surprise_history_5y
```

The original deferral (Log #48) classified this as "needs the BLS consensus-vs-actual feed (not yet ingested)" and the spec amendment work surfaced that `consensus_estimate` — an *analyst-survey* aggregate — is the genuinely-paid half (Bloomberg / Reuters / Action Economics; or Investing.com scrape with Akamai risk). The `actual_release` half is free (FRED `CPIAUCSL`).

**The pick:** substitute the **Cleveland Fed inflation nowcast** — a free, publicly-published, model-derived estimate of the current-period CPI inflation rate — for the analyst `consensus_estimate`. The Cleveland Fed's "Inflation Nowcasting" updates as new daily data arrives and publishes a current-month CPI inflation-rate estimate ahead of the BLS release.

---

## Why the Cleveland Fed nowcast (not the alternatives)

| Option | Cadence | Cost | Verdict |
|---|---|---|---|
| **Cleveland Fed inflation nowcast** | Updates intra-month; current-period CPI rate | **Free** (Cleveland Fed publication) | **Picked** — it IS a "what will the release print" estimate, the exact role `consensus_estimate` fills |
| Analyst-consensus survey (Bloomberg / Reuters) | Monthly, pre-release | **Paid** | The spec's literal source; deferred indefinitely on budget grounds |
| Investing.com scrape | Monthly | Free-but-fragile | Akamai bot-mitigation risk (same blocker class as the spdji EPS workbook); not a reliable long-term feed |
| UMich / NY Fed consumer expectations (`MICH`) | Monthly | Free on FRED | Wrong concept — *consumer* 1-year-ahead expectations, not a next-release nowcast |
| Market breakeven (`T5YIEM`) | Daily | Free on FRED | Wrong horizon — 5-year market-implied, not a next-release nowcast |

The Cleveland Fed nowcast is the only free option that occupies the same conceptual slot as the spec's `consensus_estimate`: a published, point-in-time estimate of what the upcoming CPI release will show.

---

## The spec deviation this introduces (and why it's acceptable)

The spec's `consensus_estimate` denotes an **analyst-survey aggregate** — a poll of human forecasters. The Cleveland Fed nowcast is a **statistical model output**. These are not the same epistemic object:

- A survey aggregates judgment; a nowcast extrapolates from high-frequency data (daily gas prices, etc.).
- Survey-vs-actual "surprise" measures *forecaster* error; nowcast-vs-actual "surprise" measures *model* error.

For the `inflation_shock` rule's purpose — detecting a *large positive inflation surprise* — both forms answer the same question ("did inflation come in materially hotter than the best available pre-release estimate?"). The model-nowcast substitution is therefore **semantically faithful to the rule's intent** even though it deviates from the literal source named in the spec text. Per V2 §10 ("do not invent — when the spec is ambiguous or silent, stop and ask"), this substitution is NOT an invention: it is a spec-owner-directed amendment, recorded here and in the §2B spec text, replacing one named source with another.

The `inflation_surprise_zscore` feature carries a **bias-warning row** with provenance code `inflation_surprise_cleveland_fed_nowcast` so downstream consumers can see at a glance that the surprise is model-relative, not survey-relative — the same precedent as the §2C credit-spread proxy and §2B DBC commodity proxy bias warnings.

---

## Operational pins

### `cpi_nowcast` input

Supplied via `MarketContext.macro_series["cpi_nowcast"]` — the Cleveland Fed nowcast of the current-period CPI inflation **rate** (not a price-index level), as a monthly time series. Source-agnostic at the engine boundary: the engine consumes whatever is in that key; `regime_data_fetch.cleveland_fed_nowcast` produces it (see "Fetch path" below). When `cpi_nowcast` is absent from `macro_series`, `inflation_surprise_zscore` stays all-NaN and the single-signal limb falsifies — V1 byte-identity preserved, identical to the pre-substitution behavior.

### `inflation_surprise_zscore` computation

```text
realized_cpi_rate[t]   = 1-month % change of CPIAUCSL (matches the nowcast's
                         monthly cadence; CPIAUCSL is already in macro_series
                         as `cpi_all_items`)
inflation_surprise[t]  = realized_cpi_rate[t] - cpi_nowcast[t]
inflation_surprise_zscore[t] =
    inflation_surprise[t] / rolling_std_5y(inflation_surprise)
```

- The 5-year normalizer window is **1260 trading days** — the same convention as the §2A yield z-scores and the §1D `nh_nl_ratio` percentile windows. `min_periods=1260` (the z-score is NaN until a full 5y of surprise history exists — V1 §2.7 cold-start).
- Both `realized_cpi_rate` and `cpi_nowcast` are forward-filled onto the SPY session index (CPI / nowcast are monthly; the daily classifier reads the most-recent-release value carried forward — same pattern as `cpi_all_items` already uses).
- NaN at either operand falsifies the surprise (hence the z-score) at that session.

### Rule threshold

`inflation_surprise_zscore > +1.5` — the spec's verbatim threshold (line 2551). Exposed as `InflationGrowthRulesConfig.inflation_surprise_zscore_threshold` (default `1.5`) for V2 §9.1 walk-forward calibration.

---

## Cross-cutting impact

### Spec amendments (§2B, doc-only)

- §2B line 2504-2507 `inflation_surprise_zscore` comment: rewrite the formula to name the Cleveland Fed nowcast as the `consensus_estimate` substitute; drop "DEFERRED: requires consensus-vs-actual feed".
- §2B cross-axis / short-circuit note (line ~2573 area): the single-signal limb is no longer permanently short-circuited — it consumes `inflation_surprise_zscore` and is silent only during cold-start / when `cpi_nowcast` is unwired.
- Log #48: status update — `inflation_surprise_zscore` blocker closed via the Cleveland Fed nowcast substitution.

### Code wiring

- `compute_inflation_growth_features` gains optional `cpi_nowcast: pd.Series | None = None`; when supplied, computes the real `inflation_surprise_zscore` (+ a provenance bias-warning row); when None, the all-NaN placeholder stays.
- `InflationGrowthRuleInputs` += `inflation_surprise_zscore: float`.
- `evaluate_inflation_shock` single-signal limb flips from hardcoded-skip to `inflation_surprise_zscore > threshold` (NaN falsifies).
- `InflationGrowthRulesConfig` += `inflation_surprise_zscore_threshold: float = 1.5`.
- `feature_store.build_feature_store` passes `context.macro_series.get("cpi_nowcast")` through.
- `cpi_nowcast` is NOT added to `V2_FRED_SERIES` — the Cleveland Fed nowcast is published on the Cleveland Fed site, not on FRED (FRED carries the Cleveland Fed *median* / *trimmed-mean* CPI, which are core-inflation measures, not the nowcast). It is sourced by its own fetcher (below) and wired into `macro_series`; the engine code path is complete and the limb unlocks the moment `cpi_nowcast` data is present.

### Fetch path

`regime_data_fetch/cleveland_fed_nowcast.py` is the dedicated fetch path, and the data source is **verified**: the Cleveland Fed "Inflation Nowcasting" page backs its month-over-month chart with a single JSON archive at `https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json` — reachable directly over `urllib` (only the human-facing HTML page 403s programmatic clients). The feed is the **full history**: one FusionCharts-style chart object per monthly vintage, ~2013-08 to present (154 usable CPI vintages as of the 2026-05 data vintage — far past the 1260-session normalizer, so the limb is not cold-start-blocked).

- `download_cleveland_fed_nowcast_json` fetches the feed; `parse_cleveland_fed_nowcast_json` extracts, per vintage, the **last non-empty `CPI Inflation` value** (the settled nowcast right before the BLS release), keyed to the **1st of the target month** (`chart.subcaption` `"YYYY-M"`). The 1st-of-month anchor matches FRED `CPIAUCSL`'s reference-date convention, so `realized − nowcast` forward-fills like-for-like.
- `value_scale = 0.01` converts the feed's percent-m/m publication to the fractional monthly rate `compute_inflation_surprise_zscore` expects.
- `series_name` / `value_scale` are parameterized (an operator could switch to Core CPI / PCE) but the headline-CPI defaults are verified, not guessed. A structurally-wrong feed raises `ClevelandFedNowcastError` loudly rather than producing a silently-wrong series.
- Manual-drop is a *fallback only* — if the download fails, `run_cleveland_fed_nowcast_fetch` parses an already-present `nowcast_month.json`. No dual-source path.

### Tests

`_compute_inflation_surprise_zscore` unit tests (hand-computed surprise + z-score; cold-start all-NaN below 5y; NaN-operand falsification); single-signal limb fires on `zscore > +1.5` and falsifies on NaN / below threshold; `inflation_shock` OR-rule fires via EITHER limb; integration through `compute_inflation_growth_features` with and without `cpi_nowcast`.

---

## Decision

Accepted as above. Spec amendment + code wiring land as the follow-on commit(s).
