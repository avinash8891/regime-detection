from __future__ import annotations


def dependency_payload_contracts_report() -> dict[str, dict[str, str]]:
    """Operator-facing summary of V2 cross-axis payload contracts."""

    return {
        "network_fragility": {
            "breadth_state": "label_only",
            "volatility_state": "label_only",
            "credit_funding_effective": "label_only",
        },
        "inflation_growth_state": {
            "credit_funding_effective": "label_only",
        },
        "transition_score": {
            "event_calendar": "matching_labels",
            "credit_funding_effective": "label_and_status",
            "volume_liquidity_state": "label_and_status",
        },
    }
