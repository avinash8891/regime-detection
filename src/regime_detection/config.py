from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator

from regime_detection import __version__


from regime_detection._config_core import (
    DataQualityConfig,
    EarningsSeasonConfig,
    ETFProxyConfig,
    EventCalendarConfig,
    ExpiryRulesConfig,
    MonthlyOptionsExpiryRuleConfig,
    NetworkFragilityConfig,
    NetworkFragilityRulesConfig,
    StrictBaseModel,
)
from regime_detection._config_layer1 import (
    AxisHysteresisConfig,
    BreadthV2Config,
    TrendCharacterV2Config,
    TrendDirectionV2Config,
    TrendDirectionV2RulesConfig,
    VolatilityV2Config,
    VolatilityV2RulesConfig,
    VolumeLiquidityConfig,
    VolumeLiquidityRulesConfig,
    VolumeLiquidityV2Config,
)
from regime_detection._config_layer2 import (
    CentralBankTextConfig,
    CreditFundingConfig,
    CreditFundingRulesConfig,
    InflationGrowthConfig,
    InflationGrowthRulesConfig,
    MonetaryPressureV2Config,
    MonetaryPressureV2FeaturesConfig,
    MonetaryPressureV2RulesConfig,
    NewsSentimentConfig,
)
from regime_detection._config_evidence_strategy import (
    ChangePointConfig,
    ClusteringConfig,
    CohortRoutingConfig,
    CohortRoutingRule,
    CohortRoutingRulePredicate,
    FamilyOverride,
    HMMConfig,
    NoFlipFlopConfig,
    StrategyFamilyConstraintsConfig,
    StrategyEventModifierRule,
    StrategyEventModifiersConfig,
    TransitionScoreConfig,
)

__all__ = [
    "BreadthV2Config",
    "AxisHysteresisConfig",
    "CentralBankTextConfig",
    "ChangePointConfig",
    "ClusteringConfig",
    "CohortRoutingConfig",
    "CohortRoutingRule",
    "CohortRoutingRulePredicate",
    "CreditFundingConfig",
    "CreditFundingRulesConfig",
    "DataQualityConfig",
    "ETFProxyConfig",
    "EarningsSeasonConfig",
    "EventCalendarConfig",
    "ExpiryRulesConfig",
    "FamilyOverride",
    "HMMConfig",
    "InflationGrowthConfig",
    "InflationGrowthRulesConfig",
    "MonetaryPressureV2Config",
    "MonetaryPressureV2FeaturesConfig",
    "MonetaryPressureV2RulesConfig",
    "MonthlyOptionsExpiryRuleConfig",
    "NetworkFragilityConfig",
    "NetworkFragilityRulesConfig",
    "NewsSentimentConfig",
    "NoFlipFlopConfig",
    "RegimeConfig",
    "StrategyFamilyConstraintsConfig",
    "StrategyEventModifierRule",
    "StrategyEventModifiersConfig",
    "TransitionScoreConfig",
    "TrendCharacterV2Config",
    "TrendDirectionV2Config",
    "TrendDirectionV2RulesConfig",
    "VolatilityV2Config",
    "VolatilityV2RulesConfig",
    "VolumeLiquidityConfig",
    "VolumeLiquidityRulesConfig",
    "VolumeLiquidityV2Config",
    "load_default_regime_config",
    "load_regime_config",
]


class RegimeConfig(StrictBaseModel):
    config_version: Literal["core3-v1.0.0", "core3-v2.0.0"]
    market: Literal["US"]
    trading_calendar: Literal["NYSE"]
    breadth_mode: Literal["etf_proxy"]
    etf_proxy: ETFProxyConfig
    event_calendar: EventCalendarConfig
    expiry_rules: ExpiryRulesConfig
    earnings_seasons: list[EarningsSeasonConfig]
    data_quality: DataQualityConfig

    # Axis-level hysteresis. These neutral sections are used regardless of
    # whether the raw labels came from V1 rules or V2 feature overrides.
    trend_direction: AxisHysteresisConfig
    trend_character: AxisHysteresisConfig
    volatility_state: AxisHysteresisConfig
    breadth_state: AxisHysteresisConfig

    # V2 optional sub-configs (default None so V2 slices can land independently).
    network_fragility: NetworkFragilityConfig | None = None
    trend_direction_v2: TrendDirectionV2Config | None = None
    volatility_state_v2: VolatilityV2Config | None = None
    breadth_state_v2: BreadthV2Config | None = None
    volume_liquidity_v2: VolumeLiquidityV2Config | None = None
    # v2 §1E axis classifier configuration.
    volume_liquidity_state: VolumeLiquidityConfig | None = None
    transition_score: TransitionScoreConfig | None = None
    # v2 §1A trend-character V2 axis configuration.
    trend_character_v2: TrendCharacterV2Config | None = None
    monetary_pressure_v2: MonetaryPressureV2FeaturesConfig | None = None
    # v2 §2A axis classifier configuration.
    monetary_pressure_state: MonetaryPressureV2Config | None = None
    # v2 §2A central-bank-text evidence config (deterministic-lexicon
    # substitute for the spec's "LLM classifier" phrasing; spec line 2569
    # ratifies the substitution).
    central_bank_text: CentralBankTextConfig | None = None
    # v2 §1A SF Fed news sentiment evidence config. Evidence only —
    # never read by the `euphoria` rule.
    news_sentiment: NewsSentimentConfig | None = None
    inflation_growth: InflationGrowthConfig | None = None
    credit_funding: CreditFundingConfig | None = None
    hmm: HMMConfig | None = None
    # v2 §6.2 GMM clustering evidence layer.
    clustering: ClusteringConfig | None = None
    # v2 §6.3 BOCPD change-point evidence layer.
    change_point: ChangePointConfig | None = None
    no_flip_flop: NoFlipFlopConfig | None = None  # v2 §5.4
    cohort_routing: CohortRoutingConfig | None = None  # v2 §5.1
    strategy_family_constraints: StrategyFamilyConstraintsConfig | None = (
        None  # v2 §5.2
    )
    strategy_event_modifiers: StrategyEventModifiersConfig | None = None

    @model_validator(mode="after")
    def _validate_v2_cross_section_dependencies(self) -> "RegimeConfig":
        if (
            self.config_version == "core3-v2.0.0"
            and self.volume_liquidity_state is not None
            and self.volatility_state_v2 is None
        ):
            raise ValueError(
                "volume_liquidity_state requires volatility_state_v2 because "
                "liquidity_gap_behavior consumes volatility-v2 gap/range percentiles"
            )
        return self


def load_regime_config(path: str | Path) -> RegimeConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)


def _default_config_resource_name_for_version(version: str) -> str:
    major_text = version.split(".", 1)[0]
    if not major_text.isdigit():
        raise ValueError(
            f"Unsupported package __version__ for default config dispatch: {version!r}"
        )
    major = int(major_text)
    if major == 2:
        return "configs/core3-v2.0.0.yaml"
    if major == 1:
        return "configs/core3-v1.0.0.yaml"
    raise ValueError(
        f"Unsupported package __version__ for default config dispatch: {version!r}"
    )


def default_config_text() -> str:
    """The packaged default-config YAML text, dispatched on package ``__version__`` major
    (2 -> core3-v2.0.0.yaml, 1 -> core3-v1.0.0.yaml). Loaded as resource content so it
    works even when the package is distributed as a zip/egg. Used by the walk-forward
    runner to archive the frozen config when no explicit ``--config-path`` is given."""
    resource_name = _default_config_resource_name_for_version(__version__)
    pkg_file = importlib.resources.files("regime_detection").joinpath(resource_name)
    return pkg_file.read_text(encoding="utf-8")


def load_default_regime_config() -> RegimeConfig:
    """
    Load the packaged default config shipped with the library.

    Dispatches on package ``__version__`` major:
        - 2  -> configs/core3-v2.0.0.yaml
        - 1  -> configs/core3-v1.0.0.yaml

    NOTE: We load the resource content directly (instead of returning a filesystem
    Path) so this works even when the package is distributed as a zip/egg.
    """
    data = yaml.safe_load(default_config_text())
    if not isinstance(data, dict):
        raise ValueError("Default config must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)
