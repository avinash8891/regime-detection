from __future__ import annotations

from typing import Literal

from pydantic import Field

from regime_detection._config_core import AxisName, StrictBaseModel


class TransitionScoreConfig(StrictBaseModel):
    """Composite transition risk score configuration (v2 spec §4.3 / §4.4)."""

    # V2 §4.3 weights when HMM regime-probability shift is available.
    weights_with_hmm: dict[str, float]

    # V2 §4.3 weights when HMM is unavailable (5-component renormalization).
    weights_without_hmm: dict[str, float]

    # V2 §4.3 — weights when change_point evidence is available but HMM is not (6 components).
    weights_with_change_point: dict[str, float]

    # V2 §4.3 — weights when both HMM and change_point evidence are available (7 components).
    weights_with_hmm_with_change_point: dict[str, float]

    # V2 §4.4 interpretation bands: stable / weakening / transition_warning / high.
    bands: dict[str, tuple[float, float]]

    cooldown_window_days: int = Field(default=3, ge=0)


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
    model_version: str = "hmm_4state_v1.0"
    state_label_map: dict[int, str] | None = None


class ClusteringConfig(StrictBaseModel):
    """v2 §6.2 K-Means/GMM clustering configuration.

    GMM is the V2 ship default; K-Means support deferred per spec line
    2835. Mapping cluster_id → economic_label is operator-side
    (cluster_label_map.yaml per spec line 2842); not part of this slice.
    """

    n_clusters: int = Field(default=8, ge=2)
    training_window_days: int = Field(default=1260, ge=100)
    random_state: int = Field(default=42, ge=0)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    model_version: str = Field(default="gmm_8cluster_v1.0")
    cluster_label_map: dict[int, str] | None = None


class ChangePointConfig(StrictBaseModel):
    """v2 §6.3 BOCPD change-point detection (evidence-only).

    Library: bayesian-changepoint-detection.
    Observation series: realized_vol_21d.
    Score = 5-session rolling max of posterior P(run_length=0).
    Break = posterior >= 0.5 threshold.
    """

    hazard_lambda: float = Field(default=250.0, gt=0.0)  # spec line 2872: 1/250 → lambda=250
    score_window_days: int = Field(default=5, ge=1)
    break_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    training_window_days: int = Field(default=1260, ge=100)  # 5y, matches HMM/GMM
    # Cap BOCPD input length. The library is O(n²); 3000 sessions covers
    # 2014-2026 with full warmup while keeping runtime reasonable.
    max_bocpd_window: int = Field(default=3000, ge=1260)
    # StudentT prior hyperparameters (Adams-MacKay defaults — conservative).
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

