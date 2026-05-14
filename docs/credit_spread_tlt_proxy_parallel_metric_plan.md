# §2C TLT-Proxy Parallel Credit Metric — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TLT-vs-HYG/LQD total-return-differential credit-spread proxy as a *separate, parallel* §2C metric that produces its own `RegimeOutput.credit_funding_state_proxy` label — covering history the real ICE BofA OAS series (FRED, 2023-05-15+ only) cannot.

**Architecture:** The §2C rule predicates are scale-invariant (percentile + slope only), so the *same* `CreditFundingSeriesClassifier` rule logic runs twice — once on the real-OAS feature series, once on the proxy series — producing two never-blended `CreditFundingOutput`s. Legacy `hy_spread_proxy_*` feature fields (which hold *real* OAS since Ambiguity Log #49) are renamed `hy_oas_*`; the new proxy fields are `hy_tr_differential_*`. `CreditFundingRuleInputs` spread fields become source-neutral so one builder serves both runs.

**Tech Stack:** Python 3.14, pandas, pydantic v2, pytest (`rtk proxy python3 -m pytest`, xdist `--dist loadfile`). Spec authority: `docs/regime_engine_v2_spec.md` §2C + Ambiguity Log #71.

---

## File Structure

| File | Responsibility / change |
|---|---|
| `src/regime_detection/credit_funding.py` | Rename `hy_spread_proxy_*`→`hy_oas_*` (features) + `CreditFundingRuleInputs` spread fields→source-neutral; restore `hyg_close`/`lqd_close` params + proxy bias-warning constants; add 5 `hy_tr_differential_*`/`ig_tr_differential_*` fields + compute; parameterize rule-input builders |
| `src/regime_detection/feature_store.py` | Pass `hyg_close`/`lqd_close` into `compute_credit_funding_features` |
| `src/regime_detection/axis_series.py` | `CreditFundingSeriesClassifier`: factor the per-session loop into a source-parameterized helper, run twice; `AxisSeriesBundle.credit_funding_proxy` field; bundle assembly runs the proxy |
| `src/regime_detection/models.py` | `RegimeOutput.credit_funding_state_proxy: CreditFundingOutput \| None = None` (+ docstring rename) |
| `src/regime_detection/timeline.py` | Wire `axis_bundle.credit_funding_proxy` → `RegimeOutput.credit_funding_state_proxy` |
| `tests/test_credit_funding.py` | Rename refs; new proxy-compute + parameterized-builder tests |
| `tests/test_v2_data_ingestion.py` | Integration test: `engine.classify` emits both `credit_funding_state` and `credit_funding_state_proxy` |
| `docs/decisions/0007-credit-spread-tlt-proxy-parallel-metric.md` | New ADR (decision record) |
| `src/regime_detection/configs/core3-v2.0.0.yaml` | §2C comment touch-up only (no config-value change) |

**Pre-flight (run once before Task 1):**
```bash
rtk proxy python3 -m pytest tests/test_credit_funding.py tests/test_v1_frozen_replay.py tests/test_v2_data_ingestion.py -q
```
Expected: all green. This is the baseline the rename (Task 1) must preserve.

---

### Task 1: Rename — `hy_spread_proxy_*` → `hy_oas_*` (features) and `CreditFundingRuleInputs` spread fields → source-neutral

Pure mechanical refactor. **No behavior change.** Two distinct rename targets — do not blind-sed, the two dataclasses currently share field names:

**Rename map A — `CreditFundingFeatures` fields + everything that reads them as a *feature series*** (real OAS):
| old | new |
|---|---|
| `hy_spread_proxy_63d` | `hy_oas_63d` |
| `ig_spread_proxy_63d` | `ig_oas_63d` |
| `hy_spread_proxy_percentile_504d` | `hy_oas_percentile_504d` |
| `hy_spread_proxy_slope_21d` | `hy_oas_slope_21d` |
| `ig_spread_proxy_slope_21d` | `ig_oas_slope_21d` |

**Rename map B — `CreditFundingRuleInputs` fields + everything that reads them as a *rule-input scalar*** (source-neutral — the same rule logic runs on either metric):
| old | new |
|---|---|
| `hy_spread_proxy_percentile_504d` | `hy_spread_percentile_504d` |
| `hy_spread_proxy_slope_21d` | `hy_spread_slope_21d` |
| `ig_spread_proxy_slope_21d` | `ig_spread_slope_21d` |

**Files:**
- Modify: `src/regime_detection/credit_funding.py` (62 occurrences — `CreditFundingFeatures` dataclass + `feature_names` + `_BIAS_FEATURE_NAMES` + `compute_credit_funding_features` locals/renames/return → map A; `CreditFundingRuleInputs` dataclass + `build_rule_inputs_for_date` + `build_rule_inputs_by_date` + `evaluate_credit_calm`/`evaluate_spread_widening`/`evaluate_credit_stress` → map B)
- Modify: `src/regime_detection/axis_series.py` (6 occurrences — `required_inputs` list reads `features.hy_spread_proxy_63d`/`ig_spread_proxy_63d` → map A; the `rule_evidence` dict reads `rule_inputs.hy_spread_proxy_*` → map B)
- Modify: `src/regime_detection/models.py` (1 occurrence — `CreditFundingOutput` docstring mention)
- Modify: `tests/test_credit_funding.py` (32 occurrences — `_rule_inputs` helper + assertions: feature-series refs → map A, rule-input refs → map B)
- Check: `src/regime_data_fetch/fetch_workflow.py` — its `spread_proxy` hit is a comment; update the comment text if it names the field, otherwise leave.

- [ ] **Step 1: Apply Rename map A in `credit_funding.py`** — the `CreditFundingFeatures` dataclass fields, the `feature_names` property tuple, the `_BIAS_FEATURE_NAMES` tuple, and inside `compute_credit_funding_features` the local variables / `.rename(...)` strings / the `return CreditFundingFeatures(...)` kwargs. Do NOT touch `CreditFundingRuleInputs` yet.

- [ ] **Step 2: Apply Rename map B in `credit_funding.py`** — the `CreditFundingRuleInputs` dataclass fields, the `build_rule_inputs_for_date` + `build_rule_inputs_by_date` kwargs (the *left* side stays a rule-input name → map B; the *right* side `_scalar_at(features.hy_oas_*, dt)` is a feature read → already map A from Step 1), and the three predicates `evaluate_credit_calm` / `evaluate_spread_widening` / `evaluate_credit_stress` (`inputs.hy_spread_proxy_*` → map B).

- [ ] **Step 3: Apply renames in `axis_series.py`** — `required_inputs` list: `features.hy_spread_proxy_63d`/`ig_spread_proxy_63d` → `features.hy_oas_63d`/`ig_oas_63d` (map A). The `rule_evidence` dict in the per-session evidence: `rule_inputs.hy_spread_proxy_percentile_504d` etc. → `rule_inputs.hy_spread_percentile_504d` etc. (map B); also rename the dict *keys* to match.

- [ ] **Step 4: Apply renames in `models.py` + `tests/test_credit_funding.py`** — `models.py`: the one docstring mention. `tests/test_credit_funding.py`: feature-series assertions (`features.hy_spread_proxy_63d` etc.) → map A; the `_rule_inputs` helper defaults + rule-input assertions → map B.

- [ ] **Step 5: Run the full §2C + frozen-replay suite — must stay green**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py tests/test_v1_frozen_replay.py tests/test_v2_data_ingestion.py -q`
Expected: PASS, identical pass count to the pre-flight baseline (rename only — no behavior change).

- [ ] **Step 6: Lint + commit**

```bash
python3 -m ruff check src/regime_detection/credit_funding.py src/regime_detection/axis_series.py tests/test_credit_funding.py
git add src/regime_detection/credit_funding.py src/regime_detection/axis_series.py src/regime_detection/models.py tests/test_credit_funding.py
git commit -m "refactor(credit_funding): rename hy_spread_proxy_* -> hy_oas_* (features) + source-neutral rule inputs (Log #71)"
```

---

### Task 2: TLT-proxy feature compute on `CreditFundingFeatures`

Add the proxy series. The `hyg_close`/`lqd_close` params were removed by commit `9cad7e7`; restore them (they are already `REQUIRED_CROSS_ASSET_KEYS`, so `feature_store` already gates on their presence).

**Files:**
- Modify: `src/regime_detection/credit_funding.py`
- Modify: `src/regime_detection/feature_store.py`
- Test: `tests/test_credit_funding.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_credit_funding.py` (use the existing `_bdate_index`, `_make_constant_series`, `_make_random_walk`, `_default_rules` helpers; supply `hyg_close`/`lqd_close` to the existing `compute_credit_funding_features` call shape):

```python
def test_tlt_proxy_differential_hand_computed() -> None:
    """hy_tr_differential_63d = tlt_total_return_63d - hyg_total_return_63d,
    with total_return_lookback_days = 63 (config default)."""
    idx = _bdate_index(periods=300)
    hyg = _make_random_walk(idx, seed=11, start=80.0, sigma=0.4)
    lqd = _make_random_walk(idx, seed=12, start=110.0, sigma=0.3)
    tlt = _make_random_walk(idx, seed=13, start=95.0, sigma=0.5)
    kre = _make_random_walk(idx, seed=14, start=55.0, sigma=0.6)
    spy = _make_random_walk(idx, seed=15, start=420.0, sigma=1.0)
    hy_oas = _make_constant_series(idx, 3.5, "hy_oas")
    ig_oas = _make_constant_series(idx, 1.2, "ig_oas")
    sofr = _make_constant_series(idx, 5.3, "sofr")
    iorb = _make_constant_series(idx, 5.4, "iorb")
    nfci = _make_constant_series(idx, -0.2, "nfci")
    usd = _make_random_walk(idx, seed=16, start=120.0, sigma=0.2)

    features = compute_credit_funding_features(
        hyg_close=hyg, lqd_close=lqd, tlt_close=tlt, kre_close=kre,
        spy_close=spy, sofr=sofr, iorb=iorb, nfci_weekly=nfci,
        broad_usd_index=usd, hy_oas=hy_oas, ig_oas=ig_oas,
        config=_default_rules(),
    )

    w = _default_rules().total_return_lookback_days  # 63
    t = 200
    tlt_tr = tlt.iloc[t] / tlt.iloc[t - w] - 1.0
    hyg_tr = hyg.iloc[t] / hyg.iloc[t - w] - 1.0
    assert features.hy_tr_differential_63d.iloc[t] == pytest.approx(tlt_tr - hyg_tr)
    # the percentile / slope derivations exist and are scale-invariant transforms
    assert features.hy_tr_differential_percentile_504d.name == "hy_tr_differential_percentile_504d"
    assert features.hy_tr_differential_slope_21d.name == "hy_tr_differential_slope_21d"
    assert features.ig_tr_differential_slope_21d.name == "ig_tr_differential_slope_21d"


def test_tlt_proxy_features_carry_proxy_bias_warning() -> None:
    """The five hy_tr_differential_* / ig_tr_differential_* features each
    carry a credit_spread_proxy_total_return_differential bias-warning row;
    the hy_oas_* features keep the credit_spread_ice_bofa_oas_fred row."""
    idx = _bdate_index(periods=300)
    common = dict(
        hyg_close=_make_constant_series(idx, 80.0, "HYG"),
        lqd_close=_make_constant_series(idx, 110.0, "LQD"),
        tlt_close=_make_constant_series(idx, 95.0, "TLT"),
        kre_close=_make_constant_series(idx, 55.0, "KRE"),
        spy_close=_make_constant_series(idx, 420.0, "SPY"),
        sofr=_make_constant_series(idx, 5.3, "sofr"),
        iorb=_make_constant_series(idx, 5.4, "iorb"),
        nfci_weekly=_make_constant_series(idx, -0.2, "nfci"),
        broad_usd_index=_make_constant_series(idx, 120.0, "usd"),
        hy_oas=_make_constant_series(idx, 3.5, "hy_oas"),
        ig_oas=_make_constant_series(idx, 1.2, "ig_oas"),
        config=_default_rules(),
    )
    features = compute_credit_funding_features(**common)
    codes = set(
        zip(
            features.bias_warnings["feature_name"],
            features.bias_warnings["warning_code"],
        )
    )
    assert ("hy_tr_differential_63d", "credit_spread_proxy_total_return_differential") in codes
    assert ("hy_oas_63d", "credit_spread_ice_bofa_oas_fred") in codes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py::test_tlt_proxy_differential_hand_computed -q`
Expected: FAIL — `compute_credit_funding_features() got an unexpected keyword argument 'hyg_close'` (and `AttributeError: ... 'hy_tr_differential_63d'`).

- [ ] **Step 3: Implement in `credit_funding.py`**

(a) Restore the proxy bias-warning constants near `CREDIT_SPREAD_SOURCE_CODE` (verbatim from commit `9cad7e7~1`):
```python
# Proxy provenance — the TLT-vs-HYG/LQD total-return-differential metric
# (Ambiguity Log #71). Distinct from the real-OAS source code above; the
# proxy is a similar measure that exists because FRED's ICE BofA OAS
# series lack pre-2023 history.
CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE = "credit_spread_proxy_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE = "tlt_minus_hyg_lqd_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL = "internal:tlt_minus_hyg_lqd_total_return_differential"

_PROXY_BIAS_FEATURE_NAMES: tuple[str, ...] = (
    "hy_tr_differential_63d",
    "ig_tr_differential_63d",
    "hy_tr_differential_percentile_504d",
    "hy_tr_differential_slope_21d",
    "ig_tr_differential_slope_21d",
)
```

(b) Add the 5 fields to `CreditFundingFeatures` (after `ig_oas_slope_21d`) and to the `feature_names` tuple:
```python
    hy_tr_differential_63d: pd.Series
    ig_tr_differential_63d: pd.Series
    hy_tr_differential_percentile_504d: pd.Series
    hy_tr_differential_slope_21d: pd.Series
    ig_tr_differential_slope_21d: pd.Series
```

(c) Add `hyg_close: pd.Series` and `lqd_close: pd.Series` to the `compute_credit_funding_features` signature (place them first, before `tlt_close`, matching the `9cad7e7~1` order).

(d) Inside `compute_credit_funding_features`, after the existing `tlt = tlt_close.reindex(...)` line, add:
```python
    hyg = hyg_close.reindex(spy_index).astype(float)
    lqd = lqd_close.reindex(spy_index).astype(float)
```
and after the `ig_oas_slope_21d` block, add the proxy compute:
```python
    # §2C proxy metric (Ambiguity Log #71) — TLT-vs-HYG/LQD total-return
    # differential. Rising = Treasury outperforming credit = widening
    # spreads (matches the §2C line 2033 sign convention). A SEPARATE
    # parallel metric — never blended with the real-OAS series above.
    total_return_window = config.total_return_lookback_days
    hyg_tr = (hyg / hyg.shift(total_return_window)) - 1.0
    lqd_tr = (lqd / lqd.shift(total_return_window)) - 1.0
    tlt_tr = (tlt / tlt.shift(total_return_window)) - 1.0
    hy_tr_differential_63d = (tlt_tr - hyg_tr).rename("hy_tr_differential_63d")
    ig_tr_differential_63d = (tlt_tr - lqd_tr).rename("ig_tr_differential_63d")
    hy_tr_differential_percentile_504d = (
        hy_tr_differential_63d.rolling(pct_window).rank(pct=True)
        .rename("hy_tr_differential_percentile_504d")
    )
    hy_tr_differential_slope_21d = _rolling_ols_slope(
        hy_tr_differential_63d, window=slope_21d
    ).rename("hy_tr_differential_slope_21d")
    ig_tr_differential_slope_21d = _rolling_ols_slope(
        ig_tr_differential_63d, window=slope_21d
    ).rename("ig_tr_differential_slope_21d")
```

(e) Extend the `bias_warnings` frame so it carries BOTH the OAS provenance rows (existing `_BIAS_FEATURE_NAMES` with `CREDIT_SPREAD_SOURCE_CODE`) AND the proxy rows. Build the proxy rows alongside and concatenate into the single `make_bias_warnings_frame([...])` call:
```python
    bias_warnings = make_bias_warnings_frame(
        [
            {
                "warning_code": CREDIT_SPREAD_SOURCE_CODE,
                "feature_name": feat,
                "source": CREDIT_SPREAD_SOURCE,
                "source_url": CREDIT_SPREAD_SOURCE_URL,
            }
            for feat in _BIAS_FEATURE_NAMES
        ]
        + [
            {
                "warning_code": CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
                "feature_name": feat,
                "source": CREDIT_SPREAD_PROXY_BIAS_SOURCE,
                "source_url": CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL,
            }
            for feat in _PROXY_BIAS_FEATURE_NAMES
        ]
    )
```

(f) Add the 5 new fields to the `return CreditFundingFeatures(...)` kwargs.

- [ ] **Step 4: Thread `hyg_close`/`lqd_close` through `feature_store.py`**

In `feature_store.py`, find the `compute_credit_funding_features(...)` call site (the §2C seam block). The `cross_asset_closes` dict already carries `HYG`/`LQD` (they are `REQUIRED_CROSS_ASSET_KEYS` — the gate above the call already requires them). Add `hyg_close=cross_asset_closes["HYG"]` and `lqd_close=cross_asset_closes["LQD"]` to the call (match the existing kwarg style for `tlt_close`/`kre_close`).

Run: `grep -n "compute_credit_funding_features" src/regime_detection/feature_store.py` to locate the exact call site.

- [ ] **Step 5: Run tests to verify they pass**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py -q`
Expected: PASS (the two new tests + all pre-existing).

- [ ] **Step 6: Lint + commit**

```bash
python3 -m ruff check src/regime_detection/credit_funding.py src/regime_detection/feature_store.py tests/test_credit_funding.py
git add src/regime_detection/credit_funding.py src/regime_detection/feature_store.py tests/test_credit_funding.py
git commit -m "feat(credit_funding): add TLT-vs-HYG/LQD proxy feature compute (Log #71)"
```

---

### Task 3: Parameterize the rule-input builders for either spread source

`build_rule_inputs_for_date` / `build_rule_inputs_by_date` currently read the spread triple off `features.hy_oas_*` directly. Make them take the three spread series explicitly so the SAME builder serves both the real-OAS run and the proxy run.

**Files:**
- Modify: `src/regime_detection/credit_funding.py`
- Test: `tests/test_credit_funding.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_rule_inputs_accepts_either_spread_source() -> None:
    """The same builder, pointed at the OAS triple vs the proxy triple,
    yields rule inputs whose source-neutral spread fields differ."""
    idx = _bdate_index(periods=600)
    common = dict(
        hyg_close=_make_random_walk(idx, seed=21, start=80.0, sigma=0.4),
        lqd_close=_make_random_walk(idx, seed=22, start=110.0, sigma=0.3),
        tlt_close=_make_random_walk(idx, seed=23, start=95.0, sigma=0.5),
        kre_close=_make_random_walk(idx, seed=24, start=55.0, sigma=0.6),
        spy_close=_make_random_walk(idx, seed=25, start=420.0, sigma=1.0),
        sofr=_make_constant_series(idx, 5.3, "sofr"),
        iorb=_make_constant_series(idx, 5.4, "iorb"),
        nfci_weekly=_make_constant_series(idx, -0.2, "nfci"),
        broad_usd_index=_make_random_walk(idx, seed=26, start=120.0, sigma=0.2),
        hy_oas=_make_random_walk(idx, seed=27, start=3.5, sigma=0.05),
        ig_oas=_make_random_walk(idx, seed=28, start=1.2, sigma=0.03),
        config=_default_rules(),
    )
    f = compute_credit_funding_features(**common)
    rvp = _make_constant_series(idx, 0.5, "rvp")
    acp = _make_constant_series(idx, 0.5, "acp")
    dt = idx[550]

    oas_inputs = build_rule_inputs_for_date(
        features=f, dt=dt,
        hy_spread_percentile_504d=f.hy_oas_percentile_504d,
        hy_spread_slope_21d=f.hy_oas_slope_21d,
        ig_spread_slope_21d=f.ig_oas_slope_21d,
        realized_vol_21d_percentile_252d=rvp,
        avg_pairwise_corr_percentile_504d=acp,
    )
    proxy_inputs = build_rule_inputs_for_date(
        features=f, dt=dt,
        hy_spread_percentile_504d=f.hy_tr_differential_percentile_504d,
        hy_spread_slope_21d=f.hy_tr_differential_slope_21d,
        ig_spread_slope_21d=f.ig_tr_differential_slope_21d,
        realized_vol_21d_percentile_252d=rvp,
        avg_pairwise_corr_percentile_504d=acp,
    )
    # Source-neutral spread fields differ; the shared macro/vol fields match.
    assert oas_inputs.hy_spread_percentile_504d != proxy_inputs.hy_spread_percentile_504d
    assert oas_inputs.spy_21d_return == proxy_inputs.spy_21d_return
    assert oas_inputs.sofr_iorb_slope_21d == proxy_inputs.sofr_iorb_slope_21d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py::test_build_rule_inputs_accepts_either_spread_source -q`
Expected: FAIL — `build_rule_inputs_for_date() got an unexpected keyword argument 'hy_spread_percentile_504d'`.

- [ ] **Step 3: Implement — parameterize both builders**

In `credit_funding.py`, change `build_rule_inputs_for_date` to take the three spread series as required keyword args and read the scalars from THEM instead of from `features.hy_oas_*`:
```python
def build_rule_inputs_for_date(
    *,
    features: CreditFundingFeatures,
    dt: pd.Timestamp,
    hy_spread_percentile_504d: pd.Series,
    hy_spread_slope_21d: pd.Series,
    ig_spread_slope_21d: pd.Series,
    realized_vol_21d_percentile_252d: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
) -> CreditFundingRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``.

    The spread triple is passed explicitly (source-neutral) so the same
    builder serves both the real-OAS run (pass ``features.hy_oas_*``) and
    the proxy run (pass ``features.hy_tr_differential_*``) — Log #71.
    """
    return CreditFundingRuleInputs(
        hy_spread_percentile_504d=_scalar_at(hy_spread_percentile_504d, dt),
        hy_spread_slope_21d=_scalar_at(hy_spread_slope_21d, dt),
        ig_spread_slope_21d=_scalar_at(ig_spread_slope_21d, dt),
        broad_usd_index_zscore_21d=_scalar_at(features.broad_usd_index_zscore_21d, dt),
        sofr_iorb_slope_21d=_scalar_at(features.sofr_iorb_slope_21d, dt),
        spy_21d_return=_scalar_at(features.spy_21d_return, dt),
        tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
        realized_vol_21d_percentile_252d=_scalar_at(realized_vol_21d_percentile_252d, dt),
        avg_pairwise_corr_percentile_504d=_scalar_at(avg_pairwise_corr_percentile_504d, dt),
    )
```
Apply the same parameter change to `build_rule_inputs_by_date` (it takes the same three new series args; iterate `hy_spread_percentile_504d.index`; build each `CreditFundingRuleInputs` reading the spread scalars from the three passed series and the rest from `features`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py -q`
Expected: FAIL — `tests/test_credit_funding.py::test_build_rule_inputs_by_date_matches_single_day_builder` and any caller now passes the old signature.

- [ ] **Step 5: Update existing callers/tests to the new signature**

In `tests/test_credit_funding.py`, update `test_build_rule_inputs_by_date_matches_single_day_builder` (and any other builder caller) to pass `hy_spread_percentile_504d=feats.hy_oas_percentile_504d`, `hy_spread_slope_21d=feats.hy_oas_slope_21d`, `ig_spread_slope_21d=feats.ig_oas_slope_21d`. (`axis_series.py`'s `build_credit_funding_rule_inputs_by_date` call is updated in Task 4 — leave it for now; this step's test run is `tests/test_credit_funding.py` only.)

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py -q`
Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
python3 -m ruff check src/regime_detection/credit_funding.py tests/test_credit_funding.py
git add src/regime_detection/credit_funding.py tests/test_credit_funding.py
git commit -m "refactor(credit_funding): source-neutral rule-input builders (Log #71)"
```

---

### Task 4: `CreditFundingSeriesClassifier` — parallel real + proxy run; `AxisSeriesBundle.credit_funding_proxy`

Factor the per-session loop in `CreditFundingSeriesClassifier.build` into a private helper parameterized by the spread-source triple + bias-warning code, then run it twice.

**Files:**
- Modify: `src/regime_detection/axis_series.py`
- Test: `tests/test_credit_funding.py` (classifier-level) — reuse existing classifier-test scaffolding if present, else add to `tests/test_v2_data_ingestion.py`

- [ ] **Step 1: Write the failing test**

Add a classifier-level test (in `tests/test_credit_funding.py` if it already imports `CreditFundingSeriesClassifier`, else `tests/test_v2_data_ingestion.py`). It builds a `MarketContext` + `FeatureStore` with the §2C seam lit, then asserts both outputs exist and the proxy evidence carries the proxy bias code:
```python
def test_credit_funding_classifier_emits_real_and_proxy_outputs() -> None:
    context, feature_store = _build_v2_context_with_credit_funding()  # existing helper / fixture
    classifier = CreditFundingSeriesClassifier()
    real = classifier.build(context, feature_store)
    proxy = classifier.build_proxy(context, feature_store)
    assert real is not None and proxy is not None
    assert set(real.keys()) == set(proxy.keys())  # one output per session, both runs
    a_day = next(iter(proxy))
    assert proxy[a_day].evidence["bias_warning_code"] == "credit_spread_proxy_total_return_differential"
```
If no `_build_v2_context_with_credit_funding` helper exists, build the context inline from the synthetic series used in Task 2's tests via `build_market_context` + `build_feature_store` (mirror the pattern in `tests/test_v2_data_ingestion.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py::test_credit_funding_classifier_emits_real_and_proxy_outputs -q` (adjust path if placed elsewhere)
Expected: FAIL — `'CreditFundingSeriesClassifier' object has no attribute 'build_proxy'`.

- [ ] **Step 3: Implement — factor the loop, run twice**

In `axis_series.py`, refactor `CreditFundingSeriesClassifier`:
- Extract the body of `build` (from `cf_config = ...` through the `outputs` construction) into a private method `_build_for_spread_source(self, context, feature_store, *, hy_spread_percentile_504d, hy_spread_slope_21d, ig_spread_slope_21d, bias_warning_code) -> dict[date, CreditFundingOutput] | None`. Inside it:
  - call `build_credit_funding_rule_inputs_by_date(features=features, hy_spread_percentile_504d=..., hy_spread_slope_21d=..., ig_spread_slope_21d=..., realized_vol_21d_percentile_252d=realized_vol_pct, avg_pairwise_corr_percentile_504d=avg_corr_pct_series)` (the Task-3 signature)
  - in the per-session `rule_evidence` dict, set `"bias_warning_code": bias_warning_code` (passed in — `"credit_spread_ice_bofa_oas_fred"` for the real run, `"credit_spread_proxy_total_return_differential"` for the proxy run; this also fixes the existing hardcoded-string staleness)
  - `required_inputs` for the real run uses `features.hy_oas_63d`/`features.ig_oas_63d`; for the proxy run uses `features.hy_tr_differential_63d`/`features.ig_tr_differential_63d` — pass the two `_63d` series in too, or derive them inside from the same source kwargs.
- `build(...)` becomes: `return self._build_for_spread_source(context, feature_store, hy_spread_percentile_504d=features.hy_oas_percentile_504d, hy_spread_slope_21d=features.hy_oas_slope_21d, ig_spread_slope_21d=features.ig_oas_slope_21d, bias_warning_code="credit_spread_ice_bofa_oas_fred")` — but `features` is read inside the helper, so pass the source SELECTOR instead: add a `spread_source: Literal["oas", "proxy"]` param to the helper and resolve the triple from `feature_store.credit_funding` inside it. Keep `build` returning the real run.
- Add `build_proxy(self, context, feature_store) -> dict[date, CreditFundingOutput] | None` returning `self._build_for_spread_source(..., spread_source="proxy", ...)`.

- [ ] **Step 4: Add the `AxisSeriesBundle.credit_funding_proxy` field**

In `axis_series.py`, add to `AxisSeriesBundle` (after `credit_funding`):
```python
    # V2 §2C credit/funding PROXY label — None in pure-v1 mode; populated by
    # CreditFundingSeriesClassifier.build_proxy on the TLT-vs-HYG/LQD
    # differential. Parallel to `credit_funding`, never blended (Log #71).
    credit_funding_proxy: dict[date, CreditFundingOutput] | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `rtk proxy python3 -m pytest tests/test_credit_funding.py tests/test_v2_data_ingestion.py -q`
Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
python3 -m ruff check src/regime_detection/axis_series.py tests/test_credit_funding.py
git add src/regime_detection/axis_series.py tests/test_credit_funding.py
git commit -m "feat(axis_series): parallel CreditFundingSeriesClassifier proxy run (Log #71)"
```

---

### Task 5: `RegimeOutput.credit_funding_state_proxy` + bundle assembly + timeline wiring

**Files:**
- Modify: `src/regime_detection/models.py`
- Modify: `src/regime_detection/axis_series.py` (bundle-assembly function)
- Modify: `src/regime_detection/timeline.py`
- Test: `tests/test_v2_data_ingestion.py`

- [ ] **Step 1: Write the failing integration test**

In `tests/test_v2_data_ingestion.py`, add a test that runs `engine.classify` end-to-end with the §2C seam lit and asserts both wire fields are present and independent:
```python
def test_engine_emits_credit_funding_real_and_proxy(market_df_for_asof) -> None:
    # build a v2 context with HYG/LQD/TLT/KRE + SOFR/IORB/NFCI/broad_usd_index
    # + hy_oas/ig_bbb_oas on macro_series (mirror existing §2C integration setup)
    ...
    timeline = RegimeEngine().classify(...)
    out = timeline.outputs[-1]
    assert out.credit_funding_state is not None
    assert out.credit_funding_state_proxy is not None
    # they are independent CreditFundingOutput objects
    assert out.credit_funding_state_proxy.evidence["bias_warning_code"] == \
        "credit_spread_proxy_total_return_differential"
```
Use the existing §2C integration fixture/setup in `tests/test_v2_data_ingestion.py` as the template.

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_v2_data_ingestion.py::test_engine_emits_credit_funding_real_and_proxy -q`
Expected: FAIL — `RegimeOutput` has no field `credit_funding_state_proxy`.

- [ ] **Step 3: Add the `RegimeOutput` field**

In `models.py`, in `RegimeOutput`, after `credit_funding_state`:
```python
    credit_funding_state_proxy: CreditFundingOutput | None = None  # v2 §2C proxy (Log #71)
```

- [ ] **Step 4: Wire the bundle assembly**

In `axis_series.py`, find the function that constructs `AxisSeriesBundle` (it calls `CreditFundingSeriesClassifier().build(...)`). Run `grep -n "CreditFundingSeriesClassifier()" src/regime_detection/axis_series.py` to locate it. Add a `.build_proxy(...)` call alongside and pass `credit_funding_proxy=<result>` to the `AxisSeriesBundle(...)` constructor.

- [ ] **Step 5: Wire the timeline**

In `timeline.py`, near the existing `credit_funding_by_date = axis_bundle.credit_funding` (line ~214) add `credit_funding_proxy_by_date = axis_bundle.credit_funding_proxy`; near the per-day `credit_funding_output = credit_funding_by_date.get(day) if ...` (line ~244) add the parallel `credit_funding_proxy_output = credit_funding_proxy_by_date.get(day) if credit_funding_proxy_by_date is not None else None`; and in the `RegimeOutput(...)` construction (line ~341) add `credit_funding_state_proxy=credit_funding_proxy_output`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `rtk proxy python3 -m pytest tests/test_v2_data_ingestion.py tests/test_v1_frozen_replay.py -q`
Expected: PASS — including frozen-replay (the new field defaults `None` → `exclude_none` → V1 byte-identity preserved).

- [ ] **Step 7: Lint + commit**

```bash
python3 -m ruff check src/regime_detection/models.py src/regime_detection/axis_series.py src/regime_detection/timeline.py tests/test_v2_data_ingestion.py
git add src/regime_detection/models.py src/regime_detection/axis_series.py src/regime_detection/timeline.py tests/test_v2_data_ingestion.py
git commit -m "feat(models): RegimeOutput.credit_funding_state_proxy + timeline wiring (Log #71)"
```

---

### Task 6: ADR 0007 + docstring / yaml provenance touch-ups

**Files:**
- Create: `docs/decisions/0007-credit-spread-tlt-proxy-parallel-metric.md`
- Modify: `src/regime_detection/credit_funding.py` (module docstring)
- Modify: `src/regime_detection/configs/core3-v2.0.0.yaml` (§2C comment block)

- [ ] **Step 1: Write ADR 0007**

Create `docs/decisions/0007-credit-spread-tlt-proxy-parallel-metric.md` following the structure of `docs/decisions/0006-*.md`. Content (the decision is already pinned in spec Ambiguity Log #71 — the ADR is the concise standalone record):
- **Status:** accepted.
- **Context:** FRED truncated the ICE BofA OAS series to a trailing ~3-year window (`BAMLH0A0HYM2` / `BAMLC0A4CBBB` start 2023-05-15, confirmed against FRED `/series` metadata); commit `9cad7e7`'s "fallback unreachable" reasoning is now invalid.
- **Decision:** (a) accept 2023+ depth for the real-OAS metric; (b) reintroduce the TLT-vs-HYG/LQD proxy as a SEPARATE parallel metric producing its own `credit_funding_state_proxy` label via the same scale-invariant classifier — never blended with the real-OAS series; (c) rename `hy_spread_proxy_*` → `hy_oas_*`.
- **Why this is not the dual-sourcing `9cad7e7` removed:** that was one column fed by either source (mixing); this is two distinct metrics, two distinct label outputs.
- **Consequences:** §2C real-OAS backtest depth capped at ~2023; the proxy covers ~2018→current; cross-reference spec §2C + Ambiguity Log #71.

- [ ] **Step 2: Update the `credit_funding.py` module docstring**

The module docstring (lines 11-32) currently says "single source ... There is NO proxy fallback." Rewrite that block to describe the two parallel metrics (real `hy_oas_*` + the `hy_tr_differential_*` proxy), the rename, and the "similar measure; proxy exists because FRED OAS lacks pre-2023 history; never spliced" note — consistent with spec Ambiguity Log #71.

- [ ] **Step 3: Touch up the §2C yaml comment**

In `core3-v2.0.0.yaml`, the §2C `credit_funding` comment block (around line 288-293) describes only the real OAS. Add one line noting the parallel `hy_tr_differential_*` proxy metric + `credit_funding_state_proxy` label (Log #71). No config-value changes — `total_return_lookback_days: 63` already present and is the proxy's lookback.

- [ ] **Step 4: Commit**

```bash
git add docs/decisions/0007-credit-spread-tlt-proxy-parallel-metric.md src/regime_detection/credit_funding.py src/regime_detection/configs/core3-v2.0.0.yaml
git commit -m "docs(adr): ADR 0007 credit-spread TLT-proxy parallel metric + provenance notes"
```

---

### Task 7: Full-suite verification + V1 byte-identity

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `rtk proxy python3 -m pytest`
Expected: PASS — all tests, no regressions. New tests from Tasks 2-5 included.

- [ ] **Step 2: Confirm V1 byte-identity explicitly**

Run: `rtk proxy python3 -m pytest tests/test_v1_frozen_replay.py -q`
Expected: PASS — `credit_funding_state_proxy` defaults `None`, `exclude_none=True` omits it from the wire, frozen fixtures unchanged.

- [ ] **Step 3: Lint the full changeset**

Run: `python3 -m ruff check src/regime_detection/ tests/test_credit_funding.py tests/test_v2_data_ingestion.py`
Expected: `All checks passed!`

- [ ] **Step 4: Final commit (if any lint fixups were needed; otherwise skip)**

```bash
git add -A && git commit -m "chore(credit_funding): lint fixups for §2C proxy slice"
```

---

## Self-Review

**Spec coverage** (against `docs/regime_engine_v2_spec.md` Ambiguity Log #71 + §2C amendment):
- #71(a) accept 2023+ real-OAS depth → no code needed; existing NaN-falsify behavior. Verified in Task 7.
- #71(b) parallel proxy metric + own label, never blended → Tasks 2, 4, 5.
- #71(c) rename `hy_spread_proxy_*` → `hy_oas_*` → Task 1.
- §2C Features (two metrics) → Tasks 1, 2. §2C "Two label outputs" → Tasks 4, 5. Proxy bias warning → Task 2 (compute), Task 4 (evidence code).
- ADR 0007 + provenance note → Task 6.

**Placeholder scan:** No "TBD"/"add appropriate ..."; the two grep-to-locate steps (feature_store call site, bundle-assembly call site) give the exact command, not a vague instruction.

**Type consistency:** `CreditFundingFeatures` fields `hy_oas_*` + `hy_tr_differential_*` (Tasks 1, 2) ↔ builder kwargs `hy_spread_percentile_504d` / `hy_spread_slope_21d` / `ig_spread_slope_21d` (Task 3) ↔ `CreditFundingRuleInputs` source-neutral fields (Task 1) ↔ `_build_for_spread_source` helper params (Task 4) — all consistent. `build` keeps its signature; `build_proxy` added (Task 4). `RegimeOutput.credit_funding_state_proxy` (Task 5) matches `AxisSeriesBundle.credit_funding_proxy` (Task 4).
