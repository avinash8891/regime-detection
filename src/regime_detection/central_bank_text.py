"""v2 §2A central-bank-text classifier (FOMC minutes + Powell speeches).

Implements the §2A lines 2578-2586 pipeline as a deterministic
lexicon-based scorer. The spec phrasing names an "LLM classifier" but
the V1 §2.2 stateless-replay rule (inherited by V2) forbids any
non-deterministic step: same inputs must produce identical outputs.
A free, deterministic lexicon scorer is the spec-amendment substitute,
following the same precedent as:

- DBC for the Bloomberg Commodity Index (documented implementation decision)
- AAII bull-bear for survey sentiment (documented implementation decision)
- Cleveland Fed nowcast for analyst CPI consensus (ADR 0006)
- VIXCLS/100 for options-implied 30d vol (ADR 0005)
- fja05680 for vendor PIT membership

Bias-warning row code: ``central_bank_text_deterministic_lexicon_substitute``.

TODO(future): replace this lexicon with `gtfintechlab/FOMC-RoBERTa`
(Shah et al. 2023, EMNLP — Apache-2.0, deterministic via argmax
decoding). Lexicon validated against the labeled
`gtfintechlab/fomc_communication` corpus in
`docs/verification/lexicon_validation.md`: 53.9% sentence-level
accuracy (vs 49.4% baseline), but ~70.9% accuracy CONDITIONAL on the
lexicon firing on a directional sentence. The upgrade is gated on
the +750MB CPU container delta + model-SHA pinning discipline +
running two cheaper diagnostics first (test redesign + lexicon
ablation). See the source-data audit
follow-up — FOMC-RoBERTa classifier swap (deferred, TODO)" for the
full deferral rationale and the trigger criteria.

Per V2 §2A line 2585 the score feeds ``monetary_pressure.evidence`` and
is **never** consumed by the §2A rule predicates as a standalone label —
that contract is preserved here by surfacing the score only on the
``MonetaryPressureV2Features`` dataclass, not on ``RuleInputs``.

Score schema per release row:

    hawkish_count   int   number of distinct hawkish lexicon hits in body_text
    dovish_count    int   number of distinct dovish lexicon hits in body_text
    total_tokens    int   simple whitespace token count of body_text
    net_score       float (hawkish - dovish) / (hawkish + dovish) in [-1, +1];
                          NaN when both counts are zero

The classifier is intentionally lightweight: it counts whole-word
occurrences against curated hawkish/dovish vocabularies derived from
the central-banking literature (Romer & Romer, Apel & Blix-Grimaldi,
Bennani & Neuenkirch lexicons). The output is an evidence-grade signal
to surface in the engine's monetary axis, not a primary rule input.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexicons.
#
# Curated whole-word vocabularies. Entries are lowercased and matched as
# word-boundary regex tokens, so "hike" matches "hike" / "hikes" / "hiked"
# (via the trailing optional-suffix group) but not the unrelated substring
# inside another word. Multi-word entries (e.g. "rate increase") are
# matched literally.
#
# Calibration sources:
# - Apel & Blix-Grimaldi (2014) "How informative are central bank minutes?"
# - Bennani & Neuenkirch (2017) "The Federal Reserve's communication"
# - Romer & Romer (2004) narrative monetary policy shocks
# - Bank of England MPC sentiment dictionary (public domain)
# ---------------------------------------------------------------------------


HAWKISH_TERMS: tuple[str, ...] = (
    "hawkish",
    "hike",
    "tighten",
    "tightening",
    "restrictive",
    "raise",
    "raising",
    "rate increase",
    "rate increases",
    "higher rates",
    "above target",
    "persistent inflation",
    "elevated inflation",
    "inflationary pressure",
    "inflationary pressures",
    "overheating",
    "anchor expectations",
    "anchored expectations",
    "withdraw accommodation",
    "remove accommodation",
    "policy firming",
    "firmer policy",
    "upside risk",
    "upside risks",
    "stronger than expected",
    "robust growth",
    "tight labor market",
    "wage pressure",
)

DOVISH_TERMS: tuple[str, ...] = (
    "dovish",
    "accommodative",
    "accommodation",
    "ease",
    "easing",
    "cut",
    "cuts",
    "lower rates",
    "lowering rates",
    "rate cut",
    "rate cuts",
    "soften",
    "softening",
    "moderate",
    "moderating",
    "disinflation",
    "disinflationary",
    "below target",
    "patient",
    "patience",
    "stimulus",
    "stimulative",
    "support the economy",
    "supportive policy",
    "downside risk",
    "downside risks",
    "weaker than expected",
    "slack",
    "labor market slack",
    "unemployment elevated",
)


def _compile_lexicon(terms: tuple[str, ...]) -> re.Pattern[str]:
    # Word-boundary anchored, case-insensitive. Multi-word entries match
    # the exact sequence with single-space separators. Single-word
    # entries match common -s / -d / -ed / -ing inflections via the
    # optional suffix group.
    parts: list[str] = []
    for term in terms:
        escaped = re.escape(term.lower())
        if " " in term:
            parts.append(rf"\b{escaped}\b")
        else:
            parts.append(rf"\b{escaped}(?:s|d|ed|ing)?\b")
    pattern = "|".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE)


_HAWKISH_PATTERN = _compile_lexicon(HAWKISH_TERMS)
_DOVISH_PATTERN = _compile_lexicon(DOVISH_TERMS)
_TOKEN_PATTERN = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Per-release score dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentralBankTextScore:
    """One scored release (one FOMC minutes or one Powell speech).

    ``net_score`` is in ``[-1, +1]``; positive = hawkish, negative =
    dovish, NaN when ``hawkish_count + dovish_count == 0`` (the document
    contains no lexicon hits — common for very short releases).
    """

    hawkish_count: int
    dovish_count: int
    total_tokens: int
    net_score: float


def score_text(body_text: str) -> CentralBankTextScore:
    """Score a single body of text with the hawkish/dovish lexicons.

    Pure function — same input always yields the same output, satisfying
    V1 §2.2 stateless replay.
    """
    if not isinstance(body_text, str) or not body_text:
        return CentralBankTextScore(
            hawkish_count=0, dovish_count=0, total_tokens=0, net_score=float("nan")
        )
    hawkish_count = len(_HAWKISH_PATTERN.findall(body_text))
    dovish_count = len(_DOVISH_PATTERN.findall(body_text))
    total_tokens = len(_TOKEN_PATTERN.findall(body_text))
    denom = hawkish_count + dovish_count
    if denom == 0:
        net = float("nan")
    else:
        net = (hawkish_count - dovish_count) / denom
    return CentralBankTextScore(
        hawkish_count=hawkish_count,
        dovish_count=dovish_count,
        total_tokens=total_tokens,
        net_score=net,
    )


# ---------------------------------------------------------------------------
# Frame scoring.
# ---------------------------------------------------------------------------




def score_release_frame(
    df: pd.DataFrame,
    *,
    date_column: str,
    source_label: str,
) -> pd.DataFrame:
    """Score every row's ``body_text`` and return a per-release frame.

    Output columns: ``release_date`` (date), ``hawkish_count``,
    ``dovish_count``, ``total_tokens``, ``net_score``, ``source``.
    The output is sorted by ``release_date``; multiple releases on the
    same date are kept as separate rows (the engine's session-aligned
    forward-fill consumes the chronologically latest one).
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "release_date",
                "hawkish_count",
                "dovish_count",
                "total_tokens",
                "net_score",
                "source",
            ]
        )
    if date_column not in df.columns:
        raise ValueError(
            f"central_bank_text source missing required date column: {date_column}"
        )
    if "body_text" not in df.columns:
        raise ValueError(
            "central_bank_text source missing required column: body_text"
        )
    out_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        score = score_text(row["body_text"])
        release_date = pd.to_datetime(row[date_column]).date()
        out_rows.append(
            {
                "release_date": release_date,
                "hawkish_count": score.hawkish_count,
                "dovish_count": score.dovish_count,
                "total_tokens": score.total_tokens,
                "net_score": score.net_score,
                "source": source_label,
            }
        )
    out = pd.DataFrame(out_rows).sort_values("release_date").reset_index(drop=True)
    return out


def combine_release_frames(
    *frames: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate scored release frames and return a sorted union."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "release_date",
                "hawkish_count",
                "dovish_count",
                "total_tokens",
                "net_score",
                "source",
            ]
        )
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["release_date", "source"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Daily session-aligned series.
# ---------------------------------------------------------------------------


SameDateAggregation = Literal[
    "pick_longer",
    "token_weighted_average",
    "fomc_priority",
]


def _aggregate_same_date_rows(
    scored_releases: pd.DataFrame,
    *,
    same_date_aggregation: SameDateAggregation,
) -> pd.DataFrame:
    """Collapse same-date rows according to the configured strategy.

    Returns one row per ``release_date`` with a ``net_score`` column.
    Strategies:

    - ``pick_longer`` (default): the row with the larger ``total_tokens``
      wins. Mirrors the original behavior — longer documents tend to
      carry more substantive content (FOMC minutes vs a short Powell
      speech on the same date).
    - ``token_weighted_average``: net_score is averaged across all
      same-date rows weighted by ``total_tokens``. Handles the case
      where two documents on the same date carry equally substantive
      but slightly different signals.
    - ``fomc_priority``: FOMC minutes win unconditionally over Powell
      speeches when both land on the same date. Useful when callers
      want a single canonical voice per release date.
    """
    if same_date_aggregation == "pick_longer":
        deduped = (
            scored_releases.sort_values(
                ["release_date", "total_tokens"], ascending=[True, False]
            )
            .drop_duplicates(subset="release_date", keep="first")
            .reset_index(drop=True)
        )
        return deduped[["release_date", "net_score"]]
    if same_date_aggregation == "fomc_priority":
        # Stable sort: fomc_minutes first, then powell_speech, then any
        # other source alphabetically. drop_duplicates keeps the first.
        source_rank = scored_releases["source"].map(
            lambda s: 0 if s == "fomc_minutes" else (1 if s == "powell_speech" else 2)
        )
        ordered = scored_releases.assign(_rank=source_rank).sort_values(
            ["release_date", "_rank"], ascending=[True, True]
        )
        deduped = ordered.drop_duplicates(subset="release_date", keep="first").drop(
            columns="_rank"
        )
        return deduped[["release_date", "net_score"]].reset_index(drop=True)
    if same_date_aggregation == "token_weighted_average":
        out: list[dict[str, object]] = []
        for release_date, group in scored_releases.groupby("release_date", sort=True):
            # Drop NaN net_score rows from the average — they have no
            # signal. If every row is NaN, the date is NaN.
            valid = group.dropna(subset=["net_score"])
            if valid.empty:
                out.append({"release_date": release_date, "net_score": float("nan")})
                continue
            weights = valid["total_tokens"].astype(float)
            if (weights <= 0).all():
                out.append({"release_date": release_date, "net_score": float(valid["net_score"].mean())})
                continue
            weighted = (valid["net_score"].astype(float) * weights).sum() / weights.sum()
            out.append({"release_date": release_date, "net_score": float(weighted)})
        return pd.DataFrame(out)
    raise ValueError(
        f"unknown same_date_aggregation strategy: {same_date_aggregation!r} — "
        f"expected one of pick_longer / token_weighted_average / fomc_priority"
    )


def to_daily_score_series(
    scored_releases: pd.DataFrame,
    *,
    session_index: pd.DatetimeIndex,
    smoothing_window_sessions: int = 30,
    same_date_aggregation: SameDateAggregation = "pick_longer",
) -> pd.Series:
    """Build a daily forward-filled, smoothed central-bank-text score.

    Per V1 §2.2 stateless replay: every session ``t`` reads the latest
    ``net_score`` whose ``release_date <= t`` (no future-dated reading).

    Same-date collisions (FOMC minutes and a Powell speech on the same
    date, for example) are resolved by the ``same_date_aggregation``
    strategy — see ``_aggregate_same_date_rows`` for the options. The
    default ``pick_longer`` matches the source-data audit initial wiring; the
    other two options exist as v2 §9.1 walk-forward calibration knobs.

    A trailing ``smoothing_window_sessions`` rolling mean (default 30
    NYSE sessions ≈ 6 weeks ≈ four FOMC-cycle releases) dampens
    single-document outliers; mirrors the AAII 8-week MA pattern §1A
    uses for ``sentiment_score``.

    Returns an all-NaN series on the session index when no releases are
    available (lets the monetary evidence emit NaN per the spec
    "evidence only — never a standalone label" contract).
    """
    if scored_releases is None or scored_releases.empty:
        return pd.Series(
            float("nan"),
            index=session_index,
            name="central_bank_text_score",
            dtype=float,
        )
    deduped = _aggregate_same_date_rows(
        scored_releases, same_date_aggregation=same_date_aggregation
    )
    deduped_index = pd.DatetimeIndex(pd.to_datetime(deduped["release_date"]))
    aligned = pd.Series(
        deduped["net_score"].astype(float).to_numpy(),
        index=deduped_index,
        name="central_bank_text_score",
    )
    # Forward-fill onto the session calendar.
    daily = aligned.reindex(session_index, method="ffill")
    # Smooth with a trailing rolling mean. ``min_periods=1`` so the
    # series is populated from the first available release rather than
    # waiting the full ``smoothing_window_sessions`` (matches AAII's
    # min_periods=1 ffill semantic).
    if smoothing_window_sessions > 1:
        daily = daily.rolling(
            smoothing_window_sessions, min_periods=1
        ).mean()
    daily.name = "central_bank_text_score"
    return daily
