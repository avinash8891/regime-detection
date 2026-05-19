from regime_detection.models import AxisOutput, CreditFundingOutput, DataQuality, VolumeLiquidityOutput


def test_unknown_with_ok_data_quality_is_no_rule_fired() -> None:
    out = CreditFundingOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"rule_evidence": {"hy_spread_percentile_504d": 0.60}},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "no_rule_fired"
    assert out.classification_reason == "no_rule_fired"
    assert out.reporting_label == "no_rule_fired"


def test_unknown_held_by_hysteresis_reports_no_rule_fired_hysteresis() -> None:
    out = AxisOutput(
        raw_label="sideways",
        stable_label="unknown",
        active_label="unknown",
        evidence={"rule_evidence": {"within_5pct_sma200": True}},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "no_rule_fired_hysteresis"
    assert out.classification_reason == "hysteresis_held_unknown"
    assert out.reporting_label == "no_rule_fired_hysteresis"


def test_unknown_with_missing_rule_feature_reports_no_rule_fired_warmup() -> None:
    out = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"rule_evidence": {"hy_spread_percentile_504d": None}},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "no_rule_fired_warmup"
    assert out.classification_reason == "required_rule_feature_is_nan"
    assert out.reporting_label == "no_rule_fired_warmup"


def test_wrapped_evidence_with_missing_rule_feature_reports_no_rule_fired_warmup() -> None:
    out = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"root": {"rule_evidence": {"hy_spread_percentile_504d": None}}},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )

    assert out.classification_status == "no_rule_fired_warmup"
    assert out.classification_reason == "required_rule_feature_is_nan"
    assert out.reporting_label == "no_rule_fired_warmup"


def test_unknown_with_stale_data_quality_is_stale_data() -> None:
    out = CreditFundingOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "nfci_stale_21d"},
        data_quality=DataQuality(
            status="stale_data",
            freshness_days=None,
            completeness=None,
            reason="nfci_stale_21d",
        ),
    )

    assert out.classification_status == "stale_data"
    assert out.classification_reason == "nfci_stale_21d"


def test_non_unknown_label_is_classified_even_when_data_quality_is_degraded() -> None:
    out = AxisOutput(
        raw_label="bull",
        stable_label="bull",
        active_label="bull",
        evidence={},
        data_quality=DataQuality(
            status="degraded",
            freshness_days=1,
            completeness=0.85,
            reason="incomplete_data",
        ),
    )

    assert out.classification_status == "classified"
    assert out.classification_reason is None
    assert out.reporting_label == "bull"


def test_unknown_with_insufficient_data_reports_data_unavailable() -> None:
    out = AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={},
        data_quality=DataQuality(
            status="insufficient_data",
            freshness_days=None,
            completeness=0.0,
            reason="source_missing",
        ),
    )

    assert out.classification_status == "data_unavailable"
    assert out.reporting_label == "data_unavailable"


def test_volume_liquidity_reporting_label_uses_same_contract() -> None:
    unknown = VolumeLiquidityOutput(
        label="unknown",
        evidence={},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )
    classified = VolumeLiquidityOutput(
        label="normal_volume",
        evidence={},
        data_quality=DataQuality(status="ok", freshness_days=0, completeness=1.0),
    )
    missing = VolumeLiquidityOutput(
        label="unknown",
        evidence={},
        data_quality=DataQuality(
            status="insufficient_data",
            freshness_days=None,
            completeness=0.0,
        ),
    )

    assert unknown.classification_status == "no_rule_fired"
    assert unknown.reporting_label == "no_rule_fired"
    assert classified.classification_status == "classified"
    assert classified.reporting_label == "normal_volume"
    assert missing.classification_status == "data_unavailable"
    assert missing.reporting_label == "data_unavailable"
