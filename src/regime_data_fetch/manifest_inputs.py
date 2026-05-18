from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from regime_data_fetch.artifact_manifest import ManifestArtifact, load_manifest
from regime_data_fetch.materialization import destination_for


class ManifestInputResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class RequiredRunnerInputPaths:
    daily_dir: Path
    constituent_tree: Path
    macro_parquet: Path
    pit_parquet: Path
    event_calendar: Path


@dataclass(frozen=True)
class OptionalRunnerInputPaths:
    pmi_path: Path | None
    aaii_sentiment_parquet: Path | None
    news_sentiment_parquet: Path | None
    fomc_minutes_parquet: Path | None
    powell_speeches_parquet: Path | None
    cpi_vintages_parquet: Path | None


@dataclass(frozen=True)
class RunnerInputPaths:
    required: RequiredRunnerInputPaths
    optional: OptionalRunnerInputPaths
    resolved_from_manifest: tuple[str, ...]
    cli_overrides: tuple[str, ...]

    @property
    def daily_dir(self) -> Path:
        return self.required.daily_dir

    @property
    def constituent_tree(self) -> Path:
        return self.required.constituent_tree

    @property
    def macro_parquet(self) -> Path:
        return self.required.macro_parquet

    @property
    def pit_parquet(self) -> Path:
        return self.required.pit_parquet

    @property
    def event_calendar(self) -> Path:
        return self.required.event_calendar

    @property
    def pmi_path(self) -> Path | None:
        return self.optional.pmi_path

    @property
    def aaii_sentiment_parquet(self) -> Path | None:
        return self.optional.aaii_sentiment_parquet

    @property
    def news_sentiment_parquet(self) -> Path | None:
        return self.optional.news_sentiment_parquet

    @property
    def fomc_minutes_parquet(self) -> Path | None:
        return self.optional.fomc_minutes_parquet

    @property
    def powell_speeches_parquet(self) -> Path | None:
        return self.optional.powell_speeches_parquet

    @property
    def cpi_vintages_parquet(self) -> Path | None:
        return self.optional.cpi_vintages_parquet


ARTIFACT_BY_FIELD: dict[str, str] = {
    "macro_parquet": "fred_macro_series",
    "pit_parquet": "sp500_pit_constituents",
    "event_calendar": "event_calendar_us",
    "pmi_path": "ism_pmi_history",
    "aaii_sentiment_parquet": "aaii_sentiment",
    "news_sentiment_parquet": "sf_fed_news_sentiment",
    "fomc_minutes_parquet": "fomc_minutes",
    "powell_speeches_parquet": "powell_speeches",
    "cpi_vintages_parquet": "cpi_all_items_vintages",
}

REQUIRED_INPUT_FIELDS: frozenset[str] = frozenset(
    field.name for field in fields(RequiredRunnerInputPaths)
)


def resolve_runner_input_paths(
    *,
    manifest_path: Path,
    data_root: Path,
    runner_name: str,
    cli_values: dict[str, Path | None],
    cli_overrides: set[str] | frozenset[str],
    repo_root: Path | None = None,
    required_fields: frozenset[str] = REQUIRED_INPUT_FIELDS,
) -> RunnerInputPaths:
    manifest = load_manifest(manifest_path)
    artifacts = manifest.required_for(runner_name)
    if not artifacts:
        raise ManifestInputResolutionError(
            f"manifest has no artifacts required for {runner_name}"
        )

    by_name = {artifact.name: artifact for artifact in artifacts}
    resolved: dict[str, Path | None] = {}
    resolved_from_manifest: list[str] = []

    tree_root = _constituent_tree_root(
        artifacts=artifacts,
        data_root=data_root,
        repo_root=repo_root,
    )
    for field in ("daily_dir", "constituent_tree"):
        if field in cli_overrides:
            resolved[field] = cli_values.get(field)
        else:
            resolved[field] = tree_root
            resolved_from_manifest.append(field)

    for field, artifact_name in ARTIFACT_BY_FIELD.items():
        if field in cli_overrides:
            resolved[field] = cli_values.get(field)
            continue
        artifact = by_name.get(artifact_name)
        if artifact is None:
            if field in required_fields:
                raise ManifestInputResolutionError(
                    f"manifest missing required artifact {artifact_name} for {field}"
                )
            resolved[field] = None
            continue
        resolved[field] = destination_for(artifact, data_root, repo_root=repo_root)
        resolved_from_manifest.append(field)

    for field in required_fields:
        if resolved.get(field) is None:
            raise ManifestInputResolutionError(
                f"required runner input {field} was not resolved"
            )

    return RunnerInputPaths(
        required=RequiredRunnerInputPaths(
            daily_dir=_require_resolved_path(resolved, "daily_dir"),
            constituent_tree=_require_resolved_path(resolved, "constituent_tree"),
            macro_parquet=_require_resolved_path(resolved, "macro_parquet"),
            pit_parquet=_require_resolved_path(resolved, "pit_parquet"),
            event_calendar=_require_resolved_path(resolved, "event_calendar"),
        ),
        optional=OptionalRunnerInputPaths(
            pmi_path=resolved["pmi_path"],
            aaii_sentiment_parquet=resolved["aaii_sentiment_parquet"],
            news_sentiment_parquet=resolved["news_sentiment_parquet"],
            fomc_minutes_parquet=resolved["fomc_minutes_parquet"],
            powell_speeches_parquet=resolved["powell_speeches_parquet"],
            cpi_vintages_parquet=resolved["cpi_vintages_parquet"],
        ),
        resolved_from_manifest=tuple(sorted(resolved_from_manifest)),
        cli_overrides=tuple(sorted(cli_overrides)),
    )


def _constituent_tree_root(
    *,
    artifacts: list[ManifestArtifact],
    data_root: Path,
    repo_root: Path | None,
) -> Path:
    constituent_artifacts = [
        artifact
        for artifact in artifacts
        if _is_constituent_ohlcv_artifact(artifact)
    ]
    if not constituent_artifacts:
        raise ManifestInputResolutionError(
            "manifest missing required daily OHLCV artifacts"
        )
    # Emitted partition artifacts point at symbol=XYZ/ohlcv.parquet leaves.
    # Runners need the shared tree root, so this contract deliberately walks
    # two parents up from the first verified partition file.
    first_destination = destination_for(
        constituent_artifacts[0],
        data_root,
        repo_root=repo_root,
    )
    if len(first_destination.parents) < 2:
        raise ManifestInputResolutionError(
            f"constituent OHLCV artifact path is too shallow: {first_destination}"
        )
    return first_destination.parents[1]


def _is_constituent_ohlcv_artifact(artifact: ManifestArtifact) -> bool:
    if artifact.name.startswith(("constituent_ohlcv_", "daily_ohlcv_parquet_")):
        return True
    parts = Path(artifact.local_path).parts
    return (
        len(parts) >= 5
        and parts[0:2] == ("data", "raw")
        and parts[2].startswith("daily_ohlcv")
        and parts[3].startswith("symbol=")
    )


def _require_resolved_path(resolved: dict[str, Path | None], field: str) -> Path:
    path = resolved.get(field)
    if path is None:
        raise ManifestInputResolutionError(f"required runner input {field} is missing")
    return path
