from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from regime_detection._config_core import AxisName, StrictBaseModel
from regime_detection.event_calendar_labels import EventCalendarLabel


class TransitionComponentScales(StrictBaseModel):
    """Normalization scales for the §4.2 component-score formulas.

    Each component is a clipped, normalized 0..1 stress signal. The scales
    here used to live as inline literals in
    ``transition_score.compose_transition_score_for_session``; promoting them
    to config lets a deployment recalibrate stress sensitivity without code
    changes. Defaults match the historical literals exactly so existing
    runs are byte-identical.
    """

    # vol_acc = clip((rv_short / rv_long - 1) / vol_acc_full_stress_ratio,
    #                0, 1). A full-stress ratio of 0.5 means short-window
    # realized vol must run 50% above long-window for vol_acc to saturate.
    vol_acc_full_stress_ratio: float = Field(default=0.5, gt=0.0, le=1.0)

    # breadth_det = clip((breadth_zero_stress_pct - pct_above_50dma) /
    #                    breadth_full_stress_range, 0, 1).
    # Defaults: zero stress at 50% above 50dma; full stress at 20% (drop of 30 pts).
    breadth_zero_stress_pct: float = Field(default=0.50, gt=0.0, lt=1.0)
    breadth_full_stress_range: float = Field(default=0.30, gt=0.0, le=1.0)

    # trend_drawdown = clip(-drawdown_252d / drawdown_full_stress, 0, 1).
    # Default 0.15 → a 15% 252d drawdown saturates the component.
    drawdown_full_stress: float = Field(default=0.15, gt=0.0, le=1.0)

    # ma_break = clip((sma_50 - close) / sma_50 / ma_break_full_stress, 0, 1).
    # Default 0.05 → SPY 5% below its 50-DMA saturates ma_break.
    ma_break_full_stress: float = Field(default=0.05, gt=0.0, le=1.0)

    # absorption_stress = clip((absorption_ratio_top3 - absorption_floor) /
    #                          absorption_range, 0, 1).
    # Defaults: floor 0.70, range 0.25 → linear ramp from 0.70 to 0.95.
    absorption_floor: float = Field(default=0.70, ge=0.0, lt=1.0)
    absorption_range: float = Field(default=0.25, gt=0.0, le=1.0)

    # volume_stress = clip((volume_zscore_20d - volume_zscore_floor) /
    #                      volume_zscore_range, 0, 1).
    # Defaults: floor z=1.0, range 2.0 → linear ramp from z=1 to z=3.
    volume_zscore_floor: float = Field(default=1.0)
    volume_zscore_range: float = Field(default=2.0, gt=0.0)


class TransitionOverrideThresholds(StrictBaseModel):
    """Numeric thresholds gating the named hard-override rules.

    These values used to live as inline literals inside
    ``transition_risk_series.build_transition_risk_outputs_by_date``. Promoting
    them to config makes the override ladder fully tunable and auditable; see
    ``docs/transition_risk.md`` §4 for what each one gates.
    """

    # Component-score thresholds that promote a stress signal into a hard
    # override (e.g. fragile_bull fires for a bull regime when credit OR
    # correlation are sufficiently elevated, independent of the weighted score).
    credit_stress: float = Field(default=0.70, ge=0.0, le=1.0)
    correlation_fragility: float = Field(default=0.70, ge=0.0, le=1.0)

    # Combined gate for event_transition_watch: macro_event must clear
    # macro_event_min AND the weighted score must be ≥ score_elevated_min AND
    # macro_event must be the dominant component.
    macro_event_min: float = Field(default=1.0, ge=0.0, le=1.0)
    score_elevated_min: float = Field(default=0.35, ge=0.0, le=1.0)

    # Component-value floor for inclusion in transition_risk.primary_drivers.
    primary_driver_min: float = Field(default=0.35, ge=0.0, le=1.0)


class TransitionScoreConfig(StrictBaseModel):
    """Transition-risk score and final-state band config (v2 spec §4.3 / §4.4)."""

    # Component weights for the full transition-pressure score. At runtime,
    # unavailable optional components are omitted and the remaining weights are
    # normalized, provided minimum_component_weight_coverage is still met.
    weights: dict[str, float]

    minimum_component_weight_coverage: float = Field(default=0.75, gt=0.0, le=1.0)

    # V2 §4.4 score bands. These are interpreted first, then transition_risk
    # combines the band with named rule overrides to select the final state.
    bands: dict[str, tuple[float, float]]

    # Post-axis-switch cooldown window in NYSE sessions. Default 5 means switch
    # day through five sessions later; the transition-risk composer turns this
    # into a watch state when the pressure score is otherwise stable.
    cooldown_window_days: int = Field(default=5, ge=0)

    # Final-state debounce. Value is the number of consecutive raw prints
    # required before the public transition_risk.state changes to that state.
    state_confirmation_days: dict[str, int]

    # Override thresholds. Default values preserve the historical inline
    # literals so existing configs / golden fixtures remain byte-identical.
    overrides: TransitionOverrideThresholds = Field(
        default_factory=TransitionOverrideThresholds
    )

    # Component-score normalization scales. Default values match the
    # historical inline literals in compose_transition_score_for_session so
    # adding this field does not perturb existing scores or fixtures.
    scales: TransitionComponentScales = Field(
        default_factory=TransitionComponentScales
    )

    # Optional seed for the public-state debounce. When None (default), the
    # first session's raw state is accepted immediately — matching the
    # historical behavior and existing golden fixtures. When set, the
    # debounce starts with this state, so any first-session promotion to a
    # non-matching state must clear its configured confirmation window
    # before becoming public. Useful for live streaming, where there is no
    # warm-up history to bootstrap from.
    initial_active_state: str | None = None

    @model_validator(mode="after")
    def _validate_bands_monotonic(self) -> "TransitionScoreConfig":
        # Required band labels in ascending order; interpret_transition_score
        # in transition_score.py depends on this exact set.
        required = ("stable", "weakening", "transition_warning", "high")
        missing = [name for name in required if name not in self.bands]
        if missing:
            raise ValueError(
                f"transition_score.bands missing required entries: {missing}"
            )
        for name in required:
            lo, hi = self.bands[name]
            if lo < 0.0 or hi > 1.0 or lo >= hi:
                raise ValueError(
                    f"transition_score.bands[{name!r}] must satisfy 0.0 <= lo < hi <= 1.0, "
                    f"got [{lo}, {hi}]"
                )
        for prev, nxt in zip(required, required[1:], strict=False):
            prev_lo = self.bands[prev][0]
            nxt_lo = self.bands[nxt][0]
            if nxt_lo <= prev_lo:
                raise ValueError(
                    "transition_score.bands lower bounds must be strictly increasing in order "
                    f"{required}; got {prev}={prev_lo}, {nxt}={nxt_lo}"
                )
        if self.initial_active_state is not None:
            if self.initial_active_state not in self.state_confirmation_days:
                raise ValueError(
                    "transition_score.initial_active_state must appear in "
                    f"state_confirmation_days; got {self.initial_active_state!r}, "
                    f"known states: {sorted(self.state_confirmation_days)}"
                )
        return self


class HMMConfig(StrictBaseModel):
    """Hidden Markov Model regime probability configuration (v2 spec §6.1)."""

    n_states: int = Field(ge=2)
    training_window_days: int = Field(ge=100)
    retrain_cadence_days: int = Field(ge=1)
    # Deterministic seed: reproducibility gate — same inputs + same seed → byte-identical posterior.
    random_state: int = Field(default=42, ge=0)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    min_covar: float = Field(default=0.001, ge=0.0)
    standardize_inputs: bool = True
    random_seeds: tuple[int, ...] = Field(
        default=(42, 101, 202, 303, 404, 505, 606, 707, 808, 909),
        min_length=1,
    )
    # EM convergence tolerance for hmmlearn.GaussianHMM. Pinned to the
    # hmmlearn default so EM converges to the same fixed point as the
    # legacy sequential implementation — wall-clock is reduced via the
    # per-checkpoint seed-sweep parallelization in compute_hmm_features
    # rather than by relaxing convergence.
    tol: float = Field(default=0.01, gt=0.0)
    # ADR 0013 R2 (ratified) — V2 ship default is 4-state HMM
    # (calm_bull / trending_bull / choppy_normal / stress_crash). Spec §6.1
    # line 4074 explicitly authorizes "Optionally 4 states (split bull into
    # trending vs euphoric) once 3-state version validates". The validation
    # work is recorded in tests/test_hmm_state.py + the committed label maps
    # at docs/verification/hmm_state_label_map.yaml.
    model_version: str = "hmm_4state_v1.0"
    label_map_required_for_output: bool = Field(default=False)
    state_label_map: dict[int, str] | None = None


class ClusteringConfig(StrictBaseModel):
    """v2 §6.2 K-Means/GMM clustering configuration.

    GMM is the V2 ship default; K-Means is documented as an acceptable
    fallback only (spec §6.2 line 4191). Mapping cluster_id →
    economic_label is operator-side via cluster_label_map.yaml
    (spec §6.2 line 4233); not part of this slice.
    """

    n_clusters: int = Field(default=8, ge=2)
    training_window_days: int = Field(default=1260, ge=100)
    # Retrain cadence in sessions: GMM is refit at every cadence step over
    # the training window, then the latest fit's predictions are written
    # forward to the next checkpoint. Mirrors the HMM design (§6.1). The
    # legacy per-session refit (cadence=1) is supported but ~20x slower
    # without improving label stability — adjacent k-means initializations
    # routinely permute labels, so the checkpoint cadence is also the
    # more stable assignment regime.
    retrain_cadence_days: int = Field(default=21, ge=1)
    random_state: int = Field(default=42, ge=0)
    # `full` is the V2 ship default — required for the Mahalanobis
    # ``distance_to_centroid`` formula in ClusteringFeatures (uses
    # sklearn's per-cluster ``precisions_`` shape). Setting `tied` /
    # `diag` / `spherical` silently swaps the formula to Euclidean
    # (see ClusteringFeatures.distance_to_centroid docstring).
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    # sklearn GaussianMixture EM iteration cap. sklearn's default is 100;
    # V2 ships 200 to give the V2 §6.2 covariance-`full` fit more
    # iterations before triggering the fail-open `precisions_` singularity
    # path (`reg_covar` below is the other half of that contract).
    max_iter: int = Field(default=200, ge=1)
    # Diagonal load applied to each covariance matrix before inversion.
    # Equal to sklearn's documented default (1e-6) — pinned explicitly so
    # behavior survives sklearn minor-version drift in the default value.
    reg_covar: float = Field(default=1e-6, gt=0.0)
    model_version: str = Field(default="gmm_8cluster_v1.0")
    label_map_required_for_output: bool = Field(default=False)
    cluster_label_map: dict[int, str] | None = None


class ChangePointConfig(StrictBaseModel):
    """v2 §6.3 BOCPD change-point detection (evidence-only).

    Library: bayesian-changepoint-detection.
    Observation series: realized_vol_21d.
    Score = 5-session rolling max of recent short-run posterior mass.
    Break = posterior >= 0.5 threshold.
    """

    hazard_lambda: float = Field(default=250.0, gt=0.0)  # spec §6.3 line 4263: 1/250 → lambda=250
    # Score = 5-session rolling max of recent short-run posterior mass.
    score_window_days: int = Field(default=5, ge=1)
    # realized_vol_21d is already a 21-session rolling statistic; abrupt
    # market breaks appear as BOCPD probability mass over short run lengths
    # rather than only R[1]. Sum rows 1..21 as the data-conditioned
    # "new regime started recently" posterior.
    recent_run_length_window_days: int = Field(default=21, ge=1)
    # A break occurs when posterior >= break_threshold (default 0.5).
    break_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    # 2705 non-null realized-vol observations matches the runtime YAML and
    # gives 2016+ historical replay a strict pre-window warmup from SPY history.
    training_window_days: int = Field(default=2705, ge=100)
    # ADR 0013 R3 (ratified) — StudentT conjugate-prior hyperparameters for
    # the BOCPD Gaussian-with-unknown-mean-and-variance observation likelihood
    # (Adams & MacKay 2007 §3.2). Passed unchanged to
    # `bayesian_changepoint_detection`. Spec §6.3 cites the algorithm but
    # does not pin numeric priors; ADR 0013 ratifies these as the library
    # defaults so calibration §9.1 has an explicit baseline.
    student_t_alpha: float = Field(default=0.1, gt=0.0)
    student_t_beta: float = Field(default=0.01, gt=0.0)
    student_t_kappa: float = Field(default=1.0, gt=0.0)
    student_t_mu: float = Field(default=0.0)
    method: str = Field(default="BOCPD")


class NoFlipFlopConfig(StrictBaseModel):
    """No-flip-flop timing-control knobs exposed for downstream consumers."""

    window_trading_days: int = Field(ge=0)


class CohortRoutingRulePredicate(StrictBaseModel):
    """v2 §5.1 single-axis predicate (member-match against active label)."""

    axis: AxisName
    values: list[str]


class CohortRoutingRule(StrictBaseModel):
    """v2 §5.1 cohort routing rule.

    `any_of` predicates form an OR-match group; `all_of` predicates form an
    AND-match group. A rule fires when each non-empty group matches per its
    own quantifier. An empty rule (both lists empty) never fires —
    default_neutral is handled by the walker as the universal fallback.
    """

    any_of: list[CohortRoutingRulePredicate] = Field(default_factory=list)
    all_of: list[CohortRoutingRulePredicate] = Field(default_factory=list)


class CohortRoutingConfig(StrictBaseModel):
    """v2 §5.1 Agent Cohort Routing configuration."""

    routing_rules: dict[str, CohortRoutingRule]
    # Values are strategy modes/families suppressed by the active cohort, not
    # alternate agent cohorts.
    blocked_strategy_modes: dict[str, list[str]]


class FamilyOverride(StrictBaseModel):
    """v2 §5.2 — one family's constraint values under one specialist cohort.

    All fields Optional so a specialist cohort can override just one
    dimension (e.g. just ``allowed``) and inherit the rest from
    ``default_neutral``. ``allowed`` is REQUIRED on the ``default_neutral``
    entry per the spec baseline contract (enforced in
    ``resolve_strategy_family_constraints``); on cohort overrides it stays
    Optional so a cohort can re-tune just a non-``allowed`` knob.
    """

    allowed: bool | None = None
    max_lookback_days: int | None = None
    max_holding_days: int | None = None
    max_position_pct: float | None = None
    min_adx: int | None = None
    require_breadth_confirmation: bool | None = None
    require_volume_confirmation: bool | None = None
    event_window_only: bool | None = None
    reason: str | None = None


class StrategyFamilyConstraintsConfig(StrictBaseModel):
    """v2 §5.2 family constraints — override-on-default inheritance.

    ``default_neutral`` carries the baseline for every strategy family the
    engine constrains. Specialist cohorts declare ONLY the field-level
    overrides that diverge from the baseline; unspecified families inherit
    the ``default_neutral`` values verbatim.
    """

    # Keyed by family name (e.g. ``trend_following``). ``allowed`` is
    # REQUIRED on every default_neutral entry; the resolver enforces this.
    default_neutral: dict[str, FamilyOverride]
    # First key = cohort name, second key = family name.
    overrides: dict[str, dict[str, FamilyOverride]]


class StrategyEventModifierRule(StrictBaseModel):
    """v2 strategy modifiers keyed by active event-calendar labels."""

    labels: tuple[EventCalendarLabel, ...] = Field(min_length=1)
    position_size_cap: float | None = Field(default=None, gt=0.0, le=1.0)
    leverage_allowed: bool | None = None
    allow_leverage_expansion: bool | None = None
    require_confirmation_for_new_longs: bool | None = None
    prefer_cash_or_hedges: bool | None = None

    @model_validator(mode="after")
    def _validate_action_fields(self) -> "StrategyEventModifierRule":
        if not any(
            action is not None
            for action in (
                self.position_size_cap,
                self.leverage_allowed,
                self.allow_leverage_expansion,
                self.require_confirmation_for_new_longs,
                self.prefer_cash_or_hedges,
            )
        ):
            raise ValueError(
                "strategy event modifier rule must set at least one action field"
            )
        loosening_actions = []
        if self.leverage_allowed is True:
            loosening_actions.append("leverage_allowed=True")
        if self.allow_leverage_expansion is True:
            loosening_actions.append("allow_leverage_expansion=True")
        if self.require_confirmation_for_new_longs is False:
            loosening_actions.append("require_confirmation_for_new_longs=False")
        if self.prefer_cash_or_hedges is False:
            loosening_actions.append("prefer_cash_or_hedges=False")
        if loosening_actions:
            raise ValueError(
                "strategy event modifier rule cannot loosen risk controls: "
                f"{loosening_actions}"
            )
        return self


class StrategyEventModifiersConfig(StrictBaseModel):
    """v2 strategy event-calendar modifier rule set."""

    rules: dict[str, StrategyEventModifierRule] = Field(min_length=1)
