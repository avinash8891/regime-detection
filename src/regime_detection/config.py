from __future__ import annotations

import copy
import importlib.resources
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

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
    SentimentScoreConfig,
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
    "SentimentScoreConfig",
    "StrategyFamilyConstraintsConfig",
    "StrategyEventModifierRule",
    "StrategyEventModifiersConfig",
    "TransitionScoreConfig",
    "TrendCharacterV2Config",
    "TrendDirectionV2Config",
    "TrendDirectionV2RulesConfig",
    "V2FeatureBuildConfigs",
    "VolatilityV2Config",
    "VolatilityV2RulesConfig",
    "VolumeLiquidityConfig",
    "VolumeLiquidityRulesConfig",
    "VolumeLiquidityV2Config",
    "load_default_regime_config",
    "load_regime_config",
]


class V2FeatureBuildConfigs(TypedDict):
    network_fragility_config: NetworkFragilityConfig | None
    trend_direction_v2_config: TrendDirectionV2Config | None
    volatility_state_v2_config: VolatilityV2Config | None
    breadth_state_v2_config: BreadthV2Config | None
    volume_liquidity_v2_config: VolumeLiquidityV2Config | None
    monetary_pressure_v2_config: MonetaryPressureV2FeaturesConfig | None
    credit_funding_config: CreditFundingConfig | None
    inflation_growth_config: InflationGrowthConfig | None
    central_bank_text_config: CentralBankTextConfig | None
    news_sentiment_config: NewsSentimentConfig | None
    sentiment_score_config: SentimentScoreConfig | None


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
    # AAII sentiment staleness guard — prevents unbounded ffill from
    # keeping stale euphoria readings alive indefinitely.
    sentiment_score: SentimentScoreConfig | None = None
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
        if self.config_version == "core3-v2.0.0":
            required_v2_sections = (
                "network_fragility",
                "trend_direction_v2",
                "volatility_state_v2",
                "breadth_state_v2",
                "volume_liquidity_v2",
                "volume_liquidity_state",
                "transition_score",
                "trend_character_v2",
                "monetary_pressure_v2",
                "monetary_pressure_state",
                "central_bank_text",
                "news_sentiment",
                "sentiment_score",
                "inflation_growth",
                "credit_funding",
                "hmm",
                "clustering",
                "change_point",
                "no_flip_flop",
                "cohort_routing",
                "strategy_family_constraints",
                "strategy_event_modifiers",
            )
            missing = [
                section
                for section in required_v2_sections
                if getattr(self, section) is None
            ]
            if missing:
                raise ValueError(
                    "core3-v2.0.0 config missing required V2 sections: "
                    + ", ".join(missing)
                )
        if (
            self.config_version == "core3-v2.0.0"
            and self.volume_liquidity_state is not None
        ):
            if self.volume_liquidity_v2 is None:
                raise ValueError(
                    "volume_liquidity_state requires volume_liquidity_v2 because "
                    "the volume/liquidity axis consumes volume z-score features"
                )
            if self.volatility_state_v2 is None:
                raise ValueError(
                    "volume_liquidity_state requires volatility_state_v2 because "
                    "liquidity_gap_behavior consumes volatility-v2 gap/range percentiles"
                )
        return self

    def v2_feature_build_configs(self) -> V2FeatureBuildConfigs:
        """Return V2 feature config kwargs for ``build_feature_store``.

        All 10 V2 optional seam configs are returned as a single dict
        suitable for ``**``-unpacking into ``build_feature_store``.  When
        ``config_version`` is ``core3-v1.0.0`` every value is ``None`` so
        V1 byte-identity is preserved without the caller having to re-derive
        the version gate.
        """
        if self.config_version == "core3-v1.0.0":
            return {
                "network_fragility_config": None,
                "trend_direction_v2_config": None,
                "volatility_state_v2_config": None,
                "breadth_state_v2_config": None,
                "volume_liquidity_v2_config": None,
                "monetary_pressure_v2_config": None,
                "credit_funding_config": None,
                "inflation_growth_config": None,
                "central_bank_text_config": None,
                "news_sentiment_config": None,
                "sentiment_score_config": None,
            }
        return {
            "network_fragility_config": self.network_fragility,
            "trend_direction_v2_config": self.trend_direction_v2,
            "volatility_state_v2_config": self.volatility_state_v2,
            "breadth_state_v2_config": self.breadth_state_v2,
            "volume_liquidity_v2_config": self.volume_liquidity_v2,
            "monetary_pressure_v2_config": self.monetary_pressure_v2,
            "credit_funding_config": self.credit_funding,
            "inflation_growth_config": self.inflation_growth,
            "central_bank_text_config": self.central_bank_text,
            "news_sentiment_config": self.news_sentiment,
            "sentiment_score_config": self.sentiment_score,
        }


@lru_cache(maxsize=32)
def _config_data_from_file(
    resolved_path: str, mtime_ns: int, size: int
) -> dict[str, object]:
    del mtime_ns, size
    data = yaml.safe_load(Path(resolved_path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return data


def load_regime_config(path: str | Path) -> RegimeConfig:
    config_path = Path(path)
    stat = config_path.stat()
    data = _config_data_from_file(
        str(config_path.resolve()), stat.st_mtime_ns, stat.st_size
    )
    return RegimeConfig.model_validate(copy.deepcopy(data))


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


@lru_cache(maxsize=1)
def _default_config_data() -> dict[str, object]:
    data = yaml.safe_load(default_config_text())
    if not isinstance(data, dict):
        raise ValueError("Default config must contain a YAML mapping at the top level")
    return data


def load_default_regime_config() -> RegimeConfig:
    """
    Load the packaged default config shipped with the library.

    Dispatches on package ``__version__`` major:
        - 2  -> configs/core3-v2.0.0.yaml
        - 1  -> configs/core3-v1.0.0.yaml

    NOTE: We load the resource content directly (instead of returning a filesystem
    Path) so this works even when the package is distributed as a zip/egg.
    """
    return RegimeConfig.model_validate(copy.deepcopy(_default_config_data()))
