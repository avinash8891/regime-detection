from __future__ import annotations

from typing import Literal

from pydantic import Field

from regime_detection._config_core import StrictBaseModel


class MonetaryPressureV2FeaturesConfig(StrictBaseModel):
    """v2 §2A — Layer 2A Monetary/Liquidity V2 feature config.

    Configures the yield_change_zscore template (§2A line 2855) applied to
    DGS2 (2y) and DGS10 (10y), plus the broad_usd_index_zscore_63d and
    21d-variant features. All feed the §2A axis classifier
    (tightening_pressure, easing_pressure, rate_shock, neutral_monetary,
    unknown) with precedence ordering, risk-rank table, and per-label
    hysteresis days defined in ``MonetaryPressureV2Config``.
    """

    # v2 §2A line 2854 — `yield_change_63d[t] = yield[t] - yield[t-63]`.
    # Must be > 0 because the change is computed by `yield - yield.shift(N)`
    # with N >= 1; N == 0 would produce an identically-zero change series.
    yield_change_lookback_days: int = Field(gt=0, default=63)

    # v2 §2A line 2850 — mean/std normalizer window ("5y"). 5y ≈ 1260
    # trading days under NYSE calendar conventions used throughout V2.
    # Must be > 0 (rolling mean/std requires at least one observation).
    zscore_normalizer_window_days: int = Field(gt=0, default=1260)

    # v2 §2A lines 2864-2865 — 21d-variant rate_shock predicate lookback. Mechanical
    # generalization of the line-2854 template using a 21d change window.
    rate_shock_lookback_days: int = Field(gt=0, default=21)

    # v2 §2A line 2861 — broad_usd_index z-score lookback. Mechanical generalization
    # of the line-2854 template applied to a USD-index level series.
    broad_usd_lookback_days: int = Field(gt=0, default=63)


class MonetaryPressureV2RulesConfig(StrictBaseModel):
    """v2 §2A monetary-pressure rule thresholds.

    Each value pins the verbatim §2A rule predicate threshold. Precedence
    is enforced in ``monetary_pressure.evaluate_rules``.
    """

    # §2A lines 2881-2884 tightening_pressure: yield_change_zscore_*_63d > +1.5 OR broad_usd > +1.5.
    tightening_pressure_zscore_threshold: float = Field(default=1.5, gt=0.0)
    # §2A lines 2886-2888 easing_pressure: yield_change_zscore_*_63d < -1.5 on either tenor.
    easing_pressure_zscore_threshold: float = Field(default=-1.5, lt=0.0)
    # §2A lines 2890-2892 rate_shock: abs(yield_change_zscore_21d_*) > 2.0.
    rate_shock_zscore_threshold: float = Field(default=2.0, gt=0.0)


class MonetaryPressureV2Config(StrictBaseModel):
    """v2 §2A monetary-pressure axis classifier config.

    Separate from ``MonetaryPressureV2FeaturesConfig`` (features vs
    classifier), mirroring the ``volume_liquidity_v2`` vs
    ``volume_liquidity_state`` split.
    """

    rules: MonetaryPressureV2RulesConfig = Field(
        default_factory=MonetaryPressureV2RulesConfig
    )
    # §2A per-label hysteresis days.
    deescalation_days_by_label: dict[str, int]
    # Default for labels NOT listed.
    default_deescalation_days: int = Field(default=0, ge=0)


class NewsSentimentConfig(StrictBaseModel):
    """v2 §1A SF Fed Daily News Sentiment evidence config.

    EVIDENCE-only second sentiment voice alongside the AAII bull-bear 8w-MA
    `sentiment_score`. The §1A `euphoria` rule predicate consumes only the
    AAII series per spec line 205; this config does NOT modify that rule.
    The news sentiment score and the derived `sentiment_concordance` flag
    surface in evidence dicts so downstream consumers can treat divergent
    euphoria firings as lower-conviction.

    Bias-warning code emitted in feature output:
    ``news_sentiment_sf_fed_daily_news_index``.
    """

    # Smoothing window over the daily SF Fed news sentiment. Default 21
    # NYSE sessions ≈ 1 month — short enough to react to material
    # narrative shifts, long enough to dampen single-day noise.
    smoothing_window_sessions: int = Field(default=21, gt=0)


class CentralBankTextConfig(StrictBaseModel):
    """v2 §2A central-bank-text classifier config (spec lines 2943-2950).

    Pinned as a deterministic-lexicon substitute for the spec's "LLM
    classifier" phrasing. The substitution preserves V1 §2.2 stateless
    replay (LLM calls are non-deterministic; the lexicon is pure-function).
    The resulting score is fed into ``monetary_pressure.evidence`` — never
    a standalone label per spec.

    Bias-warning code emitted in the feature output:
    ``central_bank_text_deterministic_lexicon_substitute``.
    """

    # Smoothing window in NYSE sessions over the forward-filled per-release
    # net_score series. Default 30 sessions ≈ 6 weeks ≈ four FOMC-cycle
    # releases, mirrors the AAII 8w-MA smoothing pattern §1A uses for
    # ``sentiment_score``.
    smoothing_window_sessions: int = Field(default=30, gt=0)

    # Optional safety cap: drop releases older than this many calendar
    # days at score time. Default 365 keeps an entire policy cycle of
    # history while excluding stale rows that pre-date the OHLCV window.
    max_release_age_days: int = Field(default=365, gt=0)

    # Same-date collision strategy. When FOMC minutes and a Powell speech
    # share a release date, this picks which voice wins. `pick_longer`
    # uses token-count as a proxy for material content;
    # `token_weighted_average` averages all same-date rows by token weight;
    # `fomc_priority` favours FOMC minutes unconditionally.
    same_date_aggregation: Literal[
        "pick_longer", "token_weighted_average", "fomc_priority"
    ] = Field(default="pick_longer")


class InflationGrowthRulesConfig(StrictBaseModel):
    """v2 §2B inflation/growth rule thresholds.

    Defaults match the spec verbatim (§2B lines 3046-3107).
    """

    # §2B line 3048 — goldilocks "abs drift over 21d <= 0.005" (50bps).
    cpi_drift_threshold: float = Field(default=0.005, gt=0.0)
    # §2B lines 3051 / 3080 — pmi > 50.
    pmi_goldilocks_threshold: float = Field(default=50.0, gt=0.0)
    pmi_recovery_threshold: float = Field(default=50.0, gt=0.0)
    # §2B line 3067 — disinflation "pmi > 45".
    pmi_disinflation_threshold: float = Field(default=45.0, gt=0.0)
    # ADR 0012 R2 (ratified) — 3m CPI acceleration threshold for the third
    # inflation_shock OR-limb (rapid-onset). Limb fires when
    # `cpi_3m_change_pct > threshold AND treasury_10y_yield_slope_21d > 0`.
    # Rationale: catches sharp 3-month CPI acceleration that the commodity
    # composite (lagging) and surprise-z-score (history-gap) limbs miss.
    # Tested at tests/test_inflation_growth.py::test_inflation_shock_rapid_onset_limb_*.
    cpi_3m_acceleration_threshold: float = Field(default=0.02, gt=0.0)
    # §2B line 3057 — inflation_shock "commodity_return_63d > 0.15".
    commodity_return_threshold: float = Field(default=0.15, gt=0.0)
    # §2B line 3073 — recession_scare "spy_21d_return < -0.05".
    spy_recession_threshold: float = Field(default=-0.05, lt=0.0)
    # SPY threshold for recession_scare when credit IS confirmed stressed.
    # Default -0.05 matches spec §2B line 3073 (ADR 0012 R2 ratified 2026-05-22:
    # the mild-decline scenario stays a known coverage gap until a future ADR
    # introduces a `credit_watch` label; ADR 0011 "Remaining Gaps" line 70-72).
    # The knob is preserved so a future credit_watch ADR has an existing seam
    # to differentiate confirmed vs unconfirmed credit thresholds.
    spy_recession_credit_confirmed_threshold: float = Field(default=-0.05, lt=0.0)
    # §2B lines 2987-2988 — CPI 3m / 6m percent-change lookbacks.
    cpi_lookback_3m_sessions: int = Field(default=63, ge=20)
    cpi_lookback_6m_sessions: int = Field(default=126, ge=20)
    # §2B line 3049 — 21d slope window on cpi_6m_change_pct.
    cpi_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3017 — 21d OLS slope on pmi_manufacturing.
    pmi_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3034 — DBC 63d return.
    commodity_return_lookback_sessions: int = Field(default=63, ge=5)
    # §2B line 3037 — DGS10 21d slope.
    treasury_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3041 — cyclical/defensive 21d slope.
    cyclical_defensive_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3052 — SPY 21d return.
    spy_return_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3060 — TLT 21d return.
    tlt_return_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3056 — inflation_shock single-signal limb threshold
    # (`inflation_surprise_zscore > +1.5`). Must be > 0: the limb gates on
    # a strictly-positive (hotter-than-nowcast) surprise.
    inflation_surprise_zscore_threshold: float = Field(default=1.5, gt=0.0)
    # 5y rolling-std normalizer window for the inflation surprise
    # (1260 trading days, same convention as §2A yield z-scores).
    inflation_surprise_normalizer_window_sessions: int = Field(default=1260, ge=20)
    # Lookback for the realized 1-month CPI inflation rate (~21 trading days
    # = 1 month, matches the Cleveland Fed nowcast cadence).
    inflation_surprise_realized_rate_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 3101 — earnings_expansion "aggregate_forward_eps_revision_
    # direction_4w > +0.02". Must be > 0 (a strictly-positive 4-week
    # forward-EPS revision).
    eps_revision_expansion_threshold: float = Field(default=0.02, gt=0.0)
    # §2B line 3105 — earnings_contraction "... < -0.02". Must be < 0
    # (a strictly-negative 4-week revision).
    eps_revision_contraction_threshold: float = Field(default=-0.02, lt=0.0)
    # §2B Release Timestamp Handling — first-release vs latest-revision CPI for replay.
    # When True AND ``MarketContext.cpi_first_release`` is supplied, the
    # realized inflation rate (and the cpi 3m/6m change series) read from
    # the first-release vintage so historical replay is PIT-accurate. When
    # False or when the vintage seam is absent, the existing revised
    # CPIAUCSL path is preserved unchanged. Default True per spec contract.
    use_first_release_cpi_when_available: bool = Field(default=True)
    # ADR 0011 Fix 2 — Goldilocks benign-CPI ceiling: when cpi_6m_change_pct
    # is below this level, CPI is treated as benign regardless of
    # drift/slope direction. 4% annualized is the upper bound of
    # "manageable" inflation. See docs/decisions/0011-inflation-growth-rule-coverage-fix.md.
    cpi_goldilocks_benign_ceiling: float = Field(default=0.04, gt=0.0)
    # ADR 0011 Fix 1 — When True, disinflation fires on cpi_slope < 0 alone
    # (PMI > 45 still required) without requiring yield_slope < 0 confirmation.
    # See docs/decisions/0011-inflation-growth-rule-coverage-fix.md.
    disinflation_yield_independent: bool = Field(default=True)
    # ADR 0011 Fix 3 — When True, goldilocks / recession_scare /
    # recovery_growth can fire with strengthened non-credit conditions when
    # credit_funding is unavailable (None). Prevents the cross-axis
    # coverage cliff. See docs/decisions/0011-inflation-growth-rule-coverage-fix.md.
    allow_credit_independent_fallback: bool = Field(default=True)
    # Strengthened SPY threshold for recession_scare when credit is unavailable.
    spy_recession_credit_independent_threshold: float = Field(default=-0.07, lt=0.0)


class InflationGrowthConfig(StrictBaseModel):
    """v2 §2B Inflation/Growth axis configuration.

    Wires the rule thresholds, per-label hysteresis days, and the
    unknown-gate staleness thresholds (§2B lines 3149-3155).
    """

    series_ids: dict[str, str] = Field(default_factory=dict)
    rules: InflationGrowthRulesConfig = Field(default_factory=InflationGrowthRulesConfig)
    # §2B lines 3132-3146 — per-label asymmetric hysteresis days.
    deescalation_days_by_label: dict[str, int]
    default_deescalation_days: int = Field(default=0, ge=0)
    # §2B line 3152 — "CPI stale > 60 calendar days" (2× monthly cycle).
    cpi_stale_calendar_days: int = Field(default=60, ge=1)
    # §2B line 3153 — "PMI stale > 45 calendar days" (1.5× monthly cycle).
    pmi_stale_calendar_days: int = Field(default=45, ge=1)
    # §2B line 3154 — "DGS10 stale > 5 sessions".
    dgs10_stale_sessions: int = Field(default=5, ge=1)
    # EPS revision staleness threshold — matches EPS_REVISION_STALE_DAYS = 35
    # in profile_engine_reporting.py. When the aggregate_forward_eps_revision
    # series has no non-NaN value within this many calendar days of the session,
    # the EPS signal is treated as NaN and earnings_expansion /
    # earnings_contraction labels do not fire.
    eps_revision_stale_calendar_days: int = Field(default=35, ge=1)
    # Cleveland Fed nowcast cadence is monthly/current-period; stale nowcast
    # suppresses only the optional inflation-surprise limb.
    nowcast_stale_calendar_days: int = Field(default=60, ge=1)


class CreditFundingRulesConfig(StrictBaseModel):
    """v2 §2C rule thresholds.

    Defaults match the spec verbatim (§2C lines 3249-3272).
    """

    # §2C line 3250 — credit_calm "<0.50"
    hy_percentile_calm_max: float = Field(default=0.50, ge=0.0, le=1.0)
    # §2C line 3258 — credit_stress ">0.80"
    hy_percentile_stress_min: float = Field(default=0.80, ge=0.0, le=1.0)
    # §2C lines 3259/3267 — equities-falling threshold "< -0.05".
    spy_drop_threshold: float = Field(default=-0.05, le=0.0)
    # §2C line 3262 — funding_squeeze "> +1.5" (USD z-score 21d-change variant).
    broad_usd_zscore_funding_threshold: float = Field(default=1.5, gt=0.0)
    # §2C line 3269 — deleveraging "> 0" (USD z-score 21d-change variant).
    broad_usd_zscore_deleveraging_threshold: float = Field(default=0.0)
    # §2C line 3270 — deleveraging "realized_vol_21d_percentile_252d > 0.75".
    realized_vol_percentile_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # §2C line 3271 — deleveraging "avg_pairwise_corr_percentile_504d > 0.75".
    correlation_percentile_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # Opt-in calibration lever (default False matches spec §2C lines 3254-3255
    # which requires hy_oas_slope_21d > 0 AND ig_oas_slope_21d > 0). When True,
    # the rule also fires on HY slope > 0 alone — useful for catching
    # early-stage widening where HY leads IG. Not a governance-ratified
    # default; production yaml (core3-v2.0.0.yaml:376) keeps False.
    spread_widening_hy_only: bool = Field(default=False)
    # §2C line 3221 — 504d percentile lookback ("scale-invariant predicate").
    hy_percentile_504d_lookback: int = Field(default=504, ge=20)
    # §2C lines 3222-3223 — 21d OLS slope window.
    slope_21d_lookback: int = Field(default=21, ge=5)
    # §2C line 3230 — KRE/SPY ratio 63d OLS slope window.
    slope_63d_lookback: int = Field(default=63, ge=5)
    # §2C lines 3216-3217 — 63d total-return lookback.
    total_return_lookback_days: int = Field(default=63, ge=5)
    # §2C line 3259 — spy_21d_return lookback.
    spy_return_lookback_days: int = Field(default=21, ge=5)
    # §2C line 3268 — tlt_21d_return lookback.
    tlt_return_lookback_days: int = Field(default=21, ge=5)
    # §2C lines 3235-3239 — 21d-change variant (vs §2A 63d-change variant).
    broad_usd_change_window_days: int = Field(default=21, ge=5)
    # §2C lines 3237-3238 normalizer window (~5y trading days, same as §2A).
    broad_usd_normalizer_window_days: int = Field(default=1260, ge=100)


class CreditFundingConfig(StrictBaseModel):
    """v2 §2C Credit/Funding axis configuration.

    Wires the rule thresholds, per-label hysteresis days, and the
    unknown-gate staleness thresholds. The 8-symbol universe
    (HYG/LQD/TLT/KRE/SOFR/IORB/NFCI/broad_usd_index) is hard-pinned in
    code per spec §2C feature block (lines 3186-3244) — no yaml override.
    """

    rules: CreditFundingRulesConfig = Field(default_factory=CreditFundingRulesConfig)
    # §2C lines 3292-3302 — per-label asymmetric hysteresis days.
    deescalation_days_by_label: dict[str, int]
    # Labels not listed take this default.
    default_deescalation_days: int = Field(default=0, ge=0)
    # §2C line 3308 — NFCI weekly: "stale > 14 days (2× weekly release cycle)".
    nfci_stale_days: int = Field(default=14, ge=1)
    # §2C line 3307 — HYG/LQD/TLT stale > 5 sessions.
    etf_stale_sessions: int = Field(default=5, ge=1)

