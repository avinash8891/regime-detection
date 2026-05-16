from regime_detection.models import AxisOutput, CreditFundingOutput, DataQuality


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
