from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass


class AggregateEPSFetchError(RuntimeError):
    pass


@dataclass(frozen=True, init=False)
class AggregateEPSSnapshot:
    observation_date: dt.date
    observation_label: str
    forward_estimate_label: str | None
    forward_estimate_value: float | None
    estimates_by_label: Mapping[str, float | None]
    price: float | None
    pe_by_label: Mapping[str, float | None]
    change_vs_prior_observation_by_label: Mapping[str, float | None]
    change_vs_prior_observation_price: float | None
    change_vs_prior_observation_pe_by_label: Mapping[str, float | None]

    def __init__(
        self,
        *,
        observation_date: dt.date,
        observation_label: str,
        forward_estimate_label: str | None,
        forward_estimate_value: float | None,
        estimates_by_label: Mapping[str, float | None] | None = None,
        estimate_2025e: float | None = None,
        estimate_q4_2025e: float | None = None,
        estimate_2026e: float | None = None,
        price: float | None = None,
        pe_by_label: Mapping[str, float | None] | None = None,
        pe_2025e: float | None = None,
        pe_2026e: float | None = None,
        change_vs_prior_observation_by_label: Mapping[str, float | None] | None = None,
        change_vs_prior_observation_2025e: float | None = None,
        change_vs_prior_observation_q4_2025e: float | None = None,
        change_vs_prior_observation_2026e: float | None = None,
        change_vs_prior_observation_price: float | None = None,
        change_vs_prior_observation_pe_by_label: Mapping[str, float | None] | None = None,
        change_vs_prior_observation_pe_2025e: float | None = None,
        change_vs_prior_observation_pe_2026e: float | None = None,
    ) -> None:
        object.__setattr__(self, "observation_date", observation_date)
        object.__setattr__(self, "observation_label", observation_label)
        object.__setattr__(self, "forward_estimate_label", forward_estimate_label)
        object.__setattr__(self, "forward_estimate_value", forward_estimate_value)
        object.__setattr__(
            self,
            "estimates_by_label",
            dict(
                estimates_by_label
                or {
                    "2025E": estimate_2025e,
                    "Q4 2025E": estimate_q4_2025e,
                    "2026E": estimate_2026e,
                }
            ),
        )
        object.__setattr__(self, "price", price)
        object.__setattr__(
            self,
            "pe_by_label",
            dict(pe_by_label or {"2025E": pe_2025e, "2026E": pe_2026e}),
        )
        object.__setattr__(
            self,
            "change_vs_prior_observation_by_label",
            dict(
                change_vs_prior_observation_by_label
                or {
                    "2025E": change_vs_prior_observation_2025e,
                    "Q4 2025E": change_vs_prior_observation_q4_2025e,
                    "2026E": change_vs_prior_observation_2026e,
                }
            ),
        )
        object.__setattr__(
            self,
            "change_vs_prior_observation_price",
            change_vs_prior_observation_price,
        )
        object.__setattr__(
            self,
            "change_vs_prior_observation_pe_by_label",
            dict(
                change_vs_prior_observation_pe_by_label
                or {
                    "2025E": change_vs_prior_observation_pe_2025e,
                    "2026E": change_vs_prior_observation_pe_2026e,
                }
            ),
        )

    @property
    def estimate_2025e(self) -> float | None:
        return self.estimates_by_label.get("2025E")

    @property
    def estimate_q4_2025e(self) -> float | None:
        return self.estimates_by_label.get("Q4 2025E")

    @property
    def estimate_2026e(self) -> float | None:
        return self.estimates_by_label.get("2026E")

    @property
    def pe_2025e(self) -> float | None:
        return self.pe_by_label.get("2025E")

    @property
    def pe_2026e(self) -> float | None:
        return self.pe_by_label.get("2026E")

    @property
    def change_vs_prior_observation_2025e(self) -> float | None:
        return self.change_vs_prior_observation_by_label.get("2025E")

    @property
    def change_vs_prior_observation_q4_2025e(self) -> float | None:
        return self.change_vs_prior_observation_by_label.get("Q4 2025E")

    @property
    def change_vs_prior_observation_2026e(self) -> float | None:
        return self.change_vs_prior_observation_by_label.get("2026E")

    @property
    def change_vs_prior_observation_pe_2025e(self) -> float | None:
        return self.change_vs_prior_observation_pe_by_label.get("2025E")

    @property
    def change_vs_prior_observation_pe_2026e(self) -> float | None:
        return self.change_vs_prior_observation_pe_by_label.get("2026E")

    def to_legacy_row(self) -> dict[str, object]:
        """Return the persisted flat schema while internal values stay label-keyed."""
        return {
            "observation_date": self.observation_date,
            "observation_label": self.observation_label,
            "forward_estimate_label": self.forward_estimate_label,
            "forward_estimate_value": self.forward_estimate_value,
            "estimate_2025e": self.estimate_2025e,
            "estimate_q4_2025e": self.estimate_q4_2025e,
            "estimate_2026e": self.estimate_2026e,
            "price": self.price,
            "pe_2025e": self.pe_2025e,
            "pe_2026e": self.pe_2026e,
            "change_vs_prior_observation_2025e": self.change_vs_prior_observation_2025e,
            "change_vs_prior_observation_q4_2025e": self.change_vs_prior_observation_q4_2025e,
            "change_vs_prior_observation_2026e": self.change_vs_prior_observation_2026e,
            "change_vs_prior_observation_price": self.change_vs_prior_observation_price,
            "change_vs_prior_observation_pe_2025e": self.change_vs_prior_observation_pe_2025e,
            "change_vs_prior_observation_pe_2026e": self.change_vs_prior_observation_pe_2026e,
        }


@dataclass(frozen=True)
class ParsedAggregateEPSWorkbook:
    workbook_as_of_date: dt.date
    public_files_discontinued: bool
    historical_snapshots: list[AggregateEPSSnapshot]
    current_snapshot: AggregateEPSSnapshot


@dataclass(frozen=True)
class EPSWaybackSnapshot:
    timestamp: str
    archive_url: str
    snapshot_date: dt.date
