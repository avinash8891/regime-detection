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
    cpi_nowcast_parquet: Path | None = None
    aggregate_forward_eps_weekly_history_parquet: Path | None = None


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

    @property
    def cpi_nowcast_parquet(self) -> Path | None:
        return self.optional.cpi_nowcast_parquet

    @property
    def aggregate_forward_eps_weekly_history_parquet(self) -> Path | None:
        return self.optional.aggregate_forward_eps_weekly_history_parquet


@dataclass(frozen=True)
class ManifestInputSpec:
    """Single source of truth for a manifest-routed runner input.

    Declaring a spec here automatically registers it in:
      * ``ARTIFACT_BY_FIELD``     — used by ``resolve_runner_input_paths``
      * ``MANIFEST_INPUT_FLAGS``  — used by the CLI plumbing
      * ``apply_manifest_input_defaults`` — fills ``args.<field>`` from ``data_root``

    A registry-driven coverage test enforces that no spec drifts away from any
    of these surfaces. To wire a new manifest input, add exactly one
    ``ManifestInputSpec`` entry to ``MANIFEST_INPUT_SPECS`` — no other dict,
    dataclass, or call-site edit is required for the routing to take effect.
    """

    field: str
    cli_flag: str
    artifact_name: str | None
    is_required: bool
    default_relpath: tuple[str, ...] | None


MANIFEST_INPUT_SPECS: tuple[ManifestInputSpec, ...] = (
    # Required: derived structurally from the constituent OHLCV tree, not by
    # artifact-name lookup. They participate in the CLI-override rail so an
    # operator can point a runner at an alternate tree.
    ManifestInputSpec(
        field="daily_dir",
        cli_flag="--daily-dir",
        artifact_name=None,
        is_required=True,
        default_relpath=None,
    ),
    ManifestInputSpec(
        field="constituent_tree",
        cli_flag="--constituent-tree",
        artifact_name=None,
        is_required=True,
        default_relpath=None,
    ),
    # Required: resolved by direct artifact-name lookup.
    ManifestInputSpec(
        field="macro_parquet",
        cli_flag="--macro-parquet",
        artifact_name="fred_macro_series",
        is_required=True,
        default_relpath=("macro", "fred_macro_series.parquet"),
    ),
    ManifestInputSpec(
        field="pit_parquet",
        cli_flag="--pit-parquet",
        artifact_name="sp500_pit_constituents",
        is_required=True,
        default_relpath=("pit_constituents", "sp500_ticker_intervals.parquet"),
    ),
    ManifestInputSpec(
        field="event_calendar",
        cli_flag="--event-calendar",
        artifact_name="event_calendar_us",
        is_required=True,
        default_relpath=("event_calendar", "us_events.yaml"),
    ),
    # Optional inputs. Each carries a canonical default relpath under
    # ``data_root`` so runners that bypass the manifest still locate the file.
    ManifestInputSpec(
        field="pmi_path",
        cli_flag="--pmi-path",
        artifact_name="ism_pmi_history",
        is_required=False,
        default_relpath=("pmi", "us_ism_pmi_history.parquet"),
    ),
    ManifestInputSpec(
        field="aaii_sentiment_parquet",
        cli_flag="--aaii-sentiment-parquet",
        artifact_name="aaii_sentiment",
        is_required=False,
        default_relpath=("sentiment", "aaii_sentiment.parquet"),
    ),
    ManifestInputSpec(
        field="news_sentiment_parquet",
        cli_flag="--news-sentiment-parquet",
        artifact_name="sf_fed_news_sentiment",
        is_required=False,
        default_relpath=("news_sentiment", "sf_fed_news_sentiment.parquet"),
    ),
    ManifestInputSpec(
        field="fomc_minutes_parquet",
        cli_flag="--fomc-minutes-parquet",
        artifact_name="fomc_minutes",
        is_required=False,
        default_relpath=("fomc_minutes", "fomc_minutes.parquet"),
    ),
    ManifestInputSpec(
        field="powell_speeches_parquet",
        cli_flag="--powell-speeches-parquet",
        artifact_name="powell_speeches",
        is_required=False,
        default_relpath=("powell_speeches", "powell_speeches.parquet"),
    ),
    ManifestInputSpec(
        field="cpi_vintages_parquet",
        cli_flag="--cpi-vintages-parquet",
        artifact_name="cpi_all_items_vintages",
        is_required=False,
        default_relpath=("macro_vintages", "cpi_all_items_vintages.parquet"),
    ),
    ManifestInputSpec(
        field="cpi_nowcast_parquet",
        cli_flag="--cpi-nowcast-parquet",
        artifact_name="cleveland_fed_cpi_nowcast",
        is_required=False,
        default_relpath=("cleveland_fed_nowcast", "cpi_nowcast.parquet"),
    ),
    ManifestInputSpec(
        field="aggregate_forward_eps_weekly_history_parquet",
        cli_flag="--aggregate-forward-eps-weekly-history-parquet",
        artifact_name="sp500_eps_weekly_history",
        is_required=False,
        default_relpath=("aggregate_forward_eps", "sp500_eps_weekly_history.parquet"),
    ),
)

_SPEC_BY_FIELD: dict[str, ManifestInputSpec] = {}
# Derived views. These are public for back-compat with callers that import
# the older flat dicts, but the registry is the source of truth.
ARTIFACT_BY_FIELD: dict[str, str] = {}
MANIFEST_INPUT_FLAGS: dict[str, str] = {}
for _spec in MANIFEST_INPUT_SPECS:
    _SPEC_BY_FIELD[_spec.field] = _spec
    MANIFEST_INPUT_FLAGS[_spec.field] = _spec.cli_flag
    if _spec.artifact_name is not None:
        ARTIFACT_BY_FIELD[_spec.field] = _spec.artifact_name

REQUIRED_INPUT_FIELDS: frozenset[str] = frozenset(
    field.name for field in fields(RequiredRunnerInputPaths)
)


def get_manifest_input_spec(field: str) -> ManifestInputSpec:
    try:
        return _SPEC_BY_FIELD[field]
    except KeyError as exc:
        raise KeyError(
            f"{field!r} is not a registered manifest input "
            f"(see MANIFEST_INPUT_SPECS in regime_data_fetch.manifest_inputs)"
        ) from exc


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
            cpi_nowcast_parquet=resolved["cpi_nowcast_parquet"],
            aggregate_forward_eps_weekly_history_parquet=resolved[
                "aggregate_forward_eps_weekly_history_parquet"
            ],
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
        artifact for artifact in artifacts if _is_constituent_ohlcv_artifact(artifact)
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
