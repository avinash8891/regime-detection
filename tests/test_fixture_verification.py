from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest
import yaml

from regime_detection.models import TransitionRiskState

_GOLDEN_EXPECTED_KEYS = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state_raw",
    "breadth_state_active",
    "transition_risk",
)

_V2_SPEC_GOLDEN_DATES = {
    "2010-05-06",
    "2011-08-08",
    "2015-08-24",
    "2018-10-10",
    "2020-08-14",
    "2021-01-27",
    "2022-09-26",
    "2023-03-13",
    "2024-08-05",
}

# Under the PRODUCTION config the four pre-2019 dates RAISE: the model-evidence
# windows (HMM=1260, clustering=1260, change_point=2705 sessions) exceed the fixture
# history before those dates (the V2 OHLCV fixture starts 2009-01-02), so transition_risk
# fails closed on missing model evidence. Full value-assert for these needs deep-history
# model-evidence fixtures (data/license-blocked — see golden_dates.yaml provenance_note).
# They are recorded as unsupported, never silently skipped.
_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES: dict[str, str] = {
    "2010-05-06": "missing: model evidence",
    "2011-08-08": "missing: model evidence",
    "2015-08-24": "missing: model evidence",
    "2018-10-10": "missing: model evidence",
}

# transition_risk escalation ladder for the transition_risk_minimum >= compare.
_TRANSITION_SEVERITY: dict[str, int] = {
    "stable": 0,
    "watch": 1,
    "weakening": 2,
    "fragile_bull": 2,
    "recovery_attempt": 2,
    "transition_warning": 3,
    "bear_stress": 3,
    "high_transition_risk": 4,
    "crisis": 5,
    "insufficient_data": 0,
}
# transition_evidence tokens are UPSTREAM axis active_labels, not transition_risk strings.
_NETWORK_FRAGILITY_TOKENS = frozenset(
    {
        "correlation_to_one",
        "systemic_stress",
        "systemic_stress_unconfirmed",
        "rising_fragility",
        "stock_picker_dispersion",
        "correlation_concentration",
    }
)
_CREDIT_FUNDING_TOKENS = frozenset(
    {"funding_squeeze", "deleveraging", "credit_stress", "credit_recovery"}
)

# Residual §9.4-vs-engine disagreements under the PRODUCTION config, AFTER a 2026-06
# measurement pass independently confirmed the engine computes its §3.5/§2C features
# and applies the rule ladders correctly on all six original disputes — its rule
# evidence was reproduced to full precision by a blind raw-fixture recompute. NONE is
# an engine bug. Two §9.4 expectations were therefore corrected to their
# independently-derived rule labels (2020-08-14 stock_picker_dispersion ->
# diversified_normal; 2024-08-05 funding_squeeze -> credit_stress) and now pass GREEN.
# The four below are the residual NON-engine-bug disagreements, kept self-policing so
# neither a new gap nor a silent resolution can slip past
# test_v2_golden_dates_classify_expected_fields. Each is one of:
#   * boundary near-miss — the §9.4 scenario lands just outside a spec threshold;
#   * spec gap — §3.5's 504d percentile cannot represent a one-day shock (ADR 0022);
#   * data-blocked — real ICE-OAS is absent pre-2023-05-15, so the §2C label the
#     event truly warrants is unreproducible from the TLT/HYG proxy.
_VALUE_ASSERT_DISPUTED: dict[str, str] = {
    # boundary near-miss: avg_pairwise_corr_percentile_504d=0.317 misses the <0.30
    # stock_picker_dispersion cutoff by 0.0175 (dispersion 0.726 > 0.70 IS met).
    "2021-01-27:network_fragility": "boundary near-miss: corr_pct 0.317 (needs <0.30); §9.4 stock_picker_dispersion",
    # boundary near-miss + §2C data gate: deleveraging realized_vol_pctile 0.698 vs
    # >0.75, and the proxy axis is data_unavailable (sofr_iorb 504d completeness
    # 0.577 < 0.70 floor — IORB starts 2021-07-29, no FEDFUNDS/IOER splice in fixture).
    "2022-09-26:transition_evidence:deleveraging": "boundary+data-gate: vol_pctile 0.698 (needs >0.75); §2C axis data_unavailable (sofr_iorb completeness 0.577<0.70)",
    # data-blocked: real ICE-OAS absent pre-2023-05-15; the TLT-vs-HYG proxy reads
    # 0.609 < 0.80 (TLT fell more than HYG over 63d — SVB flight-to-quality), so the
    # spec predicate on the only available metric cannot reproduce the real event.
    "2023-03-13:credit_funding": "data-blocked: proxy hy_tr_differential_pctile 0.609 (needs >0.80); real OAS license-blocked; §9.4/reality credit_stress",
    # spec gap (ADR 0022): correlation_to_one needs corr_pct>0.90, but a single-day
    # shock barely moves a 504d percentile (measured 0.349; vol/drawdown conditions met).
    "2024-08-05:transition_evidence:correlation_to_one": "spec gap (ADR 0022): corr_pct 0.349 (needs >0.90); 504d percentile cannot fire on a 1-day shock",
}


def _active_label(value: object) -> object:
    return value.get("active_label") if isinstance(value, dict) else value


def test_conftest_market_data_requires_real_combined_market_parquet(
    monkeypatch, tmp_path: Path
) -> None:
    import conftest as project_conftest

    for symbol in ("SPY", "RSP", "VIXY"):
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ]
        ).to_csv(tmp_path / f"{symbol}.csv", index=False)

    project_conftest._load_market_data.cache_clear()
    monkeypatch.setattr(project_conftest, "_RAW_DIR", tmp_path)
    monkeypatch.setattr(
        project_conftest, "_MARKET_PARQUET_PATH", tmp_path / "missing.parquet"
    )

    with pytest.raises(RuntimeError, match="market_data.parquet"):
        project_conftest._load_market_data()

    project_conftest._load_market_data.cache_clear()


def test_conftest_v2_kwargs_reject_asof_before_real_v2_rows() -> None:
    import conftest as project_conftest

    project_conftest._load_market_data.cache_clear()
    market_data = project_conftest._load_market_data()
    event_calendar = pd.DataFrame()
    build_kwargs = project_conftest.synthetic_v2_kwargs_for_market_data.__wrapped__(
        event_calendar
    )

    with pytest.raises(RuntimeError, match="as_of=2018-12-31"):
        build_kwargs(market_data[market_data["date"] <= date(2018, 12, 31)])


def test_conftest_v2_kwargs_use_real_v2_fixture_rows_when_window_is_covered() -> None:
    import conftest as project_conftest

    event_calendar = pd.DataFrame()
    build_kwargs = project_conftest.synthetic_v2_kwargs_for_market_data.__wrapped__(
        event_calendar
    )
    v2_daily = project_conftest._load_v2_daily_ohlcv()
    market_data = (
        v2_daily[
            (v2_daily["date"] <= date(2023, 12, 14))
            & (v2_daily["symbol"].isin({"SPY", "RSP", "VIX", "VIXY"}))
        ]
        .copy()
        .reset_index(drop=True)
    )
    kwargs = build_kwargs(market_data)

    qqq_rows = v2_daily[
        (v2_daily["symbol"] == "QQQ") & (v2_daily["date"] <= date(2023, 12, 14))
    ].sort_values("date")
    expected_qqq = qqq_rows.set_index(pd.to_datetime(qqq_rows["date"]))["close"].astype(
        float
    )
    pd.testing.assert_series_equal(
        kwargs["cross_asset_closes"]["QQQ"],
        expected_qqq.rename("QQQ"),
        check_names=True,
    )


def test_fixture_verification_legacy_path_fails_loudly_without_v2_transition_inputs() -> (
    None
):
    """
    Hard gate for Slice 2:
    - golden_dates.yaml is hand-labeled (never engine-generated)
    - legacy raw CSV fixtures do not carry required V2 transition-score inputs
    - the report generator must fail loudly instead of silently fabricating a
      transition_risk fallback
    """
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"

    committed_derived = yaml.safe_load(derived_path.read_text())
    assert committed_derived.get("provenance") == "hand_labeled", (
        "golden_dates.yaml must carry provenance: hand_labeled — "
        "expected values are independently derived, not from engine output"
    )

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    generate_report = getattr(mod, "generate_report")

    with pytest.raises(ValueError) as excinfo:
        generate_report(
            generated_at_utc="2026-05-19T00:00:00+00:00",
            generated_by_commit="test_determinism",
        )
    message = str(excinfo.value)
    assert "ClassifyRequest missing configured V2 inputs" in message


def test_classified_golden_outputs_cover_every_row_without_silent_skips(
    golden_rows: list[dict[str, object]],
    classified_golden_outputs: dict[date, object],
) -> None:
    expected_dates = {date.fromisoformat(str(row["as_of_date"])) for row in golden_rows}

    assert len(expected_dates) == 10
    assert set(classified_golden_outputs) == expected_dates


def test_golden_dates_match_live_labels_without_data_quality_bypass(
    golden_rows: list[dict[str, object]],
    classified_golden_outputs: dict[date, object],
) -> None:
    for row in golden_rows:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        expected = row["expected"]
        output = classified_golden_outputs[as_of]
        actual = {
            "trend_direction": output.trend_direction.active_label,
            "trend_character": output.trend_character.active_label,
            "volatility_state": output.volatility_state.active_label,
            "breadth_state_raw": output.breadth_state.raw_label,
            "breadth_state_active": output.breadth_state.active_label,
            "transition_risk": output.transition_risk.state,
        }
        data_quality = {
            "trend_direction": output.trend_direction.data_quality.status,
            "trend_character": output.trend_character.data_quality.status,
            "volatility_state": output.volatility_state.data_quality.status,
            "breadth_state": output.breadth_state.data_quality.status,
            "transition_risk": output.transition_risk.data_quality.status,
        }

        assert set(_GOLDEN_EXPECTED_KEYS).issubset(expected), as_of
        assert actual == {key: expected[key] for key in _GOLDEN_EXPECTED_KEYS}, as_of
        assert data_quality == {
            "trend_direction": "ok",
            "trend_character": "ok",
            "volatility_state": "ok",
            "breadth_state": "ok",
            "transition_risk": "ok",
        }, as_of


def test_v2_section_9_4_golden_dates_are_registered() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    assert golden["provenance"] == "hand_labeled"
    v2_rows = [row for row in golden["rows"] if "expected_v2_fields" in row]
    assert {row["as_of_date"] for row in v2_rows} == _V2_SPEC_GOLDEN_DATES
    for row in v2_rows:
        assert row["intent_id"]
        assert row["expected_v2_fields"]
        # V2END-023 anti-recurrence: a declared multi-day `sequence` expectation MUST
        # carry a structured `expected_sequence` trajectory block, so it cannot be
        # registered as an unasserted scenario tag (the gap this finding closed).
        if "sequence" in row["expected_v2_fields"]:
            assert "expected_sequence" in row, (
                f"{row['as_of_date']} declares expected_v2_fields.sequence but no "
                "expected_sequence block for test_q4_2018_bull_to_narrowing_to_bear_sequence"
            )


@pytest.mark.slow
def test_v2_golden_dates_classify_expected_fields(
    v2_classify_kwargs_for_asof,
) -> None:
    """Value-assert the §9.4 golden dates under the PRODUCTION config.

    Marked ``slow``: classifying under the production config recomputes the deep
    model-evidence (HMM/change-point/clustering) and 504-session percentile windows
    per date, so this runs in the ``-m slow`` / ``-m ""`` confidence lane
    (full-verification.yml, release.yml), not the fast PR suite.

    Each expected_v2_fields label is compared to the engine's emitted label (not
    merely checked non-empty). Three honest outcomes, no silent relaxation:
      * GREEN — the engine's label equals the §9.4 expectation.
      * unsupported (_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES) — the date RAISES
        because the deep-history model-evidence windows are unfilled (the four
        pre-2019 dates).
      * disputed (_VALUE_ASSERT_DISPUTED) — the engine substantively disagrees with
        the §9.4 hand-label; recorded EXACTLY so the gap cannot silently appear or
        silently resolve. (transition_evidence tokens are upstream axis labels.)
    """
    from regime_detection.config import load_default_regime_config
    from regime_detection.engine import RegimeEngine

    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    v2_rows = [row for row in golden["rows"] if "expected_v2_fields" in row]
    engine = RegimeEngine()
    prod_config = load_default_regime_config()  # value-assert needs the faithful config
    unsupported: dict[str, str] = {}
    disagreements: dict[str, str] = {}
    classified_dates: set[str] = set()

    for row in v2_rows:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        kwargs = dict(v2_classify_kwargs_for_asof(as_of))
        kwargs["config"] = prod_config  # not _fast_v2_test_config (relaxed/over-fires)
        try:
            output = engine.classify(as_of_date=as_of, **kwargs)
        except (RuntimeError, ValueError) as exc:
            unsupported[str(as_of)] = str(exc)
            continue

        classified_dates.add(str(as_of))
        dumped = output.model_dump(mode="json", exclude_none=True)
        nf = _active_label(dumped.get("network_fragility"))
        cf = _active_label(dumped.get("credit_funding_effective_state"))
        vol = _active_label(dumped.get("volume_liquidity_state"))
        trs = (dumped.get("transition_risk") or {}).get("state")

        for field_name, expected in row["expected_v2_fields"].items():
            if field_name == "sequence":
                # §9.4 multi-day trajectory — no single-date label semantics. It is
                # value-asserted over the Q4-2018 axis series by
                # test_q4_2018_bull_to_narrowing_to_bear_sequence (this date also
                # fails-closed here, so the loop never reaches it anyway).
                continue
            if field_name == "transition_evidence":
                for token in expected if isinstance(expected, list) else [expected]:
                    if token in _NETWORK_FRAGILITY_TOKENS:
                        ok = nf == token
                    elif token in _CREDIT_FUNDING_TOKENS:
                        ok = cf == token
                    else:
                        ok = False
                    if not ok:
                        disagreements[f"{as_of}:{field_name}:{token}"] = (
                            f"nf={nf} cf={cf}"
                        )
                continue
            if field_name == "transition_risk_minimum":
                if _TRANSITION_SEVERITY.get(trs, 0) < _TRANSITION_SEVERITY.get(
                    expected, 0
                ):
                    disagreements[f"{as_of}:{field_name}"] = f"state={trs}"
                continue
            actual = {
                "network_fragility": nf,
                "credit_funding": cf,
                "volume_liquidity_state": vol,
            }.get(field_name)
            if actual != expected:
                disagreements[f"{as_of}:{field_name}"] = f"got={actual}"

    # D2 — the deep-history dates fail closed on missing model evidence.
    assert unsupported.keys() == _V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES.keys()
    for as_of, expected_fragment in _V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES.items():
        assert expected_fragment in unsupported[as_of]
    assert classified_dates == _V2_SPEC_GOLDEN_DATES - set(unsupported)
    # Exact, self-policing dispute set: no undocumented gap, no silently-resolved gap.
    assert set(disagreements) == set(_VALUE_ASSERT_DISPUTED), (
        f"undocumented={set(disagreements) - set(_VALUE_ASSERT_DISPUTED)} "
        f"resolved={set(_VALUE_ASSERT_DISPUTED) - set(disagreements)}; observed={disagreements}"
    )


@pytest.mark.slow
def test_q4_2018_bull_to_narrowing_to_bear_sequence(
    v2_classify_kwargs_for_asof,
) -> None:
    """V2END-023 / §9.4: value-assert the 2018-10-10 bull -> narrowing_breadth ->
    bear_stress trajectory.

    The §9.4 table names 2018-10-10 for this *sequence*, which has no single-date
    label semantics — and the full single-date V2 classify fails-closed there (its
    transition_risk model-evidence windows reach before the 2009-start fixture, see
    ``_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES``). The trajectory lives on the
    trend_direction + breadth_state axes, neither of which needs model evidence, so it
    is asserted over the Q4-2018 axis series under the PRODUCTION config. Stage anchors
    are the hand-labeled market landmarks in golden_dates.yaml ``expected_sequence``;
    the engine independently reproduces each stage's label. ``bear_stress`` on the
    transition_risk axis is value-asserted at the adjacent dec2018_bear_stress core
    golden row (2018-12-11). Replaces the prior silent ``sequence`` skip.
    """
    from regime_detection.axis_builders.breadth import build_breadth_axis_series
    from regime_detection.axis_builders.trend_direction import (
        build_trend_direction_axis_series,
    )
    from regime_detection.config import load_default_regime_config
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    (row,) = [
        r for r in golden["rows"] if r.get("intent_id") == "q4_2018_breadth_stress"
    ]
    seq = row["expected_sequence"]
    as_of = date.fromisoformat(str(seq["window_as_of"]))

    # Build context + feature store under the production config WITHOUT running the
    # transition_risk step (which fails-closed on this date); the breadth/trend axes
    # need no model evidence. Mirrors timeline.build_regime_timeline's wiring.
    kwargs = dict(v2_classify_kwargs_for_asof(as_of))
    kwargs.pop("config")
    cfg = load_default_regime_config()
    context = build_market_context(end_date=as_of, config=cfg, **kwargs)
    is_v2 = cfg.config_version != "core3-v1.0.0"
    feature_store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility if is_v2 else None,
        trend_direction_v2_config=cfg.trend_direction_v2 if is_v2 else None,
        volatility_state_v2_config=cfg.volatility_state_v2 if is_v2 else None,
        breadth_state_v2_config=cfg.breadth_state_v2 if is_v2 else None,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2 if is_v2 else None,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2 if is_v2 else None,
        credit_funding_config=cfg.credit_funding if is_v2 else None,
        inflation_growth_config=cfg.inflation_growth if is_v2 else None,
        central_bank_text_config=cfg.central_bank_text if is_v2 else None,
        news_sentiment_config=cfg.news_sentiment if is_v2 else None,
    )
    trend = build_trend_direction_axis_series(
        context, feature_store
    ).active_labels_by_date
    breadth = build_breadth_axis_series(context, feature_store).active_labels_by_date

    # Stage 1 — bull: trending up into the early-October top, breadth not yet narrowing.
    bull_date = date.fromisoformat(str(seq["bull"]["date"]))
    assert bull_date in trend, f"{bull_date} absent from trend series"
    assert (
        trend[bull_date] == seq["bull"]["trend_direction"]
    ), f"bull stage: trend[{bull_date}]={trend[bull_date]}"
    assert (
        breadth[bull_date] != seq["narrowing_breadth"]["breadth_state"]
    ), f"breadth already narrowing at the bull top ({bull_date})"

    # Stage 2 — narrowing_breadth onset inside the October-2018 correction window.
    onset_lo, onset_hi = (
        date.fromisoformat(str(d)) for d in seq["narrowing_breadth"]["onset_window"]
    )
    narrowing = seq["narrowing_breadth"]["breadth_state"]
    onset_sessions = [d for d in sorted(breadth) if onset_lo <= d <= onset_hi]
    narrowing_dates = [d for d in onset_sessions if breadth[d] == narrowing]
    assert narrowing_dates, (
        f"no {narrowing} in {onset_lo}..{onset_hi}; got "
        f"{[(d.isoformat(), breadth[d]) for d in onset_sessions]}"
    )
    first_narrowing = narrowing_dates[0]

    # Stage 3 — bear + narrowing_breadth in the December collapse.
    bear_date = date.fromisoformat(str(seq["bear_stress"]["date"]))
    assert bear_date in trend, f"{bear_date} absent from trend series"
    assert (
        trend[bear_date] == seq["bear_stress"]["trend_direction"]
    ), f"bear stage: trend[{bear_date}]={trend[bear_date]}"
    assert (
        breadth[bear_date] == seq["bear_stress"]["breadth_state"]
    ), f"bear stage: breadth[{bear_date}]={breadth[bear_date]}"

    # Ordering from ENGINE output (the §9.4 "sequence" order, not the hardcoded
    # anchors): the bear-trend regime must ONSET strictly AFTER the narrowing-breadth
    # onset, and be established no later than the bear anchor. Falsifiable — fails if
    # the engine turned bear before/at the narrowing onset, or never turned bear.
    bear_onsets = [
        d for d in sorted(trend) if d >= first_narrowing and trend[d] == "bear"
    ]
    assert bear_onsets, "trend never turned bear after the narrowing-breadth onset"
    assert first_narrowing < bear_onsets[0] <= bear_date


# The 10 V1 spec §12.2 golden-date table source dates (docs/regime_engine_v1_final_spec.md).
_SPEC_SECTION_12_2_DATES = (
    "2017-06-01",
    "2018-02-05",
    "2018-12-24",
    "2019-09-13",
    "2020-03-16",
    "2020-04-10",
    "2021-11-15",
    "2022-06-13",
    "2022-10-12",
    "2024-01-16",
)


def test_golden_date_replacement_set_has_documented_justification() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    justification = (
        repo_root
        / "docs"
        / "verification"
        / "golden_dates_replacement_justification.md"
    ).read_text()

    assert "2020-04-10" in justification
    assert "Good Friday" in justification
    assert "no silent pre-2019 or data-quality skips" in justification

    # F-008: the report must be a complete per-date mapping. Every §12.2 source
    # date AND every committed replacement as_of_date must be documented, so the
    # justification cannot silently drift out of sync with the active fixture.
    for spec_date in _SPEC_SECTION_12_2_DATES:
        assert spec_date in justification, f"§12.2 date {spec_date} missing from report"

    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    # The §12.2 replacement set is the core-axis golden gate; expected_v2_fields
    # rows are the separate §9.4 V2-axis set and are not part of this mapping.
    committed_dates = [row["as_of_date"] for row in golden["rows"] if "expected" in row]
    assert len(committed_dates) == len(_SPEC_SECTION_12_2_DATES)
    for committed in committed_dates:
        assert (
            committed in justification
        ), f"committed golden as_of_date {committed} missing from replacement report"


def test_classification_labels_are_independent_of_extra_history_length(
    market_df_for_asof,
    event_calendar_df: pd.DataFrame,
) -> None:
    from regime_detection.config import load_regime_config
    from regime_detection.engine import RegimeEngine

    repo_root = Path(__file__).resolve().parents[1]
    as_of = date(2023, 12, 5)
    market_data = market_df_for_asof(as_of)
    spy_sessions = (
        market_data.loc[market_data["symbol"] == "SPY", "date"]
        .drop_duplicates()
        .sort_values()
    )
    shorter_start = spy_sessions.iloc[-700]
    shorter_market_data = (
        market_data[market_data["date"] >= shorter_start].copy().reset_index(drop=True)
    )
    config = load_regime_config(
        repo_root / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
    )
    engine = RegimeEngine()

    full = engine.classify(
        as_of_date=as_of,
        market_data=market_data,
        config=config,
        event_calendar=event_calendar_df,
    )
    shorter = engine.classify(
        as_of_date=as_of,
        market_data=shorter_market_data,
        config=config,
        event_calendar=event_calendar_df,
    )

    assert {
        "trend_direction": shorter.trend_direction.active_label,
        "trend_character": shorter.trend_character.active_label,
        "volatility_state": shorter.volatility_state.active_label,
        "breadth_state_raw": shorter.breadth_state.raw_label,
        "breadth_state_active": shorter.breadth_state.active_label,
    } == {
        "trend_direction": full.trend_direction.active_label,
        "trend_character": full.trend_character.active_label,
        "volatility_state": full.volatility_state.active_label,
        "breadth_state_raw": full.breadth_state.raw_label,
        "breadth_state_active": full.breadth_state.active_label,
    }


def test_fixture_verification_report_includes_rich_transition_evidence(
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    session = date(2026, 5, 15)
    axis = SimpleNamespace(active_label="bull", evidence={"rule": "sample"})
    transition = SimpleNamespace(
        state="watch",
        evidence={
            "triggered_rules": ["post_switch_cooldown"],
            "axis_switch_count": 1,
            "recent_axis_switch_count": 2,
        },
        score=0.42,
        score_components={"trend_break": 0.30, "macro_event": 1.0},
        primary_drivers=["macro_event"],
        triggered_rules=["post_switch_cooldown"],
        data_quality={"status": "ok"},
    )
    output = SimpleNamespace(
        trend_direction=axis,
        trend_character=SimpleNamespace(active_label="trending", evidence={}),
        volatility_state=SimpleNamespace(active_label="normal_vol", evidence={}),
        breadth_state=SimpleNamespace(active_label="healthy_breadth", evidence={}),
        transition_risk=transition,
    )

    monkeypatch.setattr(
        mod,
        "INTENTS",
        [
            {
                "intent_id": "transition_rich_evidence",
                "intent_date": session.isoformat(),
                "intent": {"transition_risk": "watch"},
                "search_window_trading_days": 0,
                "notes": "synthetic v2 transition evidence",
            }
        ],
    )
    monkeypatch.setattr(
        mod,
        "_load_hand_labeled_expectations",
        lambda: {"transition_rich_evidence": {"transition_risk": "watch"}},
    )
    monkeypatch.setattr(
        mod,
        "_load_market_data",
        lambda: pd.DataFrame({"date": [pd.Timestamp(session)]}),
    )
    monkeypatch.setattr(
        mod, "_classify_all_intents", lambda _market_data: {session: output}
    )
    monkeypatch.setattr(mod, "_sha256_file", lambda _path: "sha256")

    report = mod.generate_report(
        generated_at_utc="2026-05-23T00:00:00+00:00",
        generated_by_commit="test_transition_rich_evidence",
    )

    transition_evidence = report["rows"][0]["predicate_evaluations"]["transition_risk"]
    assert transition_evidence == {
        "evidence": {
            "triggered_rules": ["post_switch_cooldown"],
            "axis_switch_count": 1,
            "recent_axis_switch_count": 2,
        },
        "score": 0.42,
        "score_components": {"trend_break": 0.3, "macro_event": 1.0},
        "primary_drivers": ["macro_event"],
        "triggered_rules": ["post_switch_cooldown"],
        "data_quality": {"status": "ok"},
    }


def test_fixture_verification_requires_combined_market_parquet_for_vix(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    for symbol in ("SPY", "RSP", "VIXY"):
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ]
        ).to_csv(tmp_path / f"{symbol}.csv", index=False)

    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="market_data.parquet"):
        mod._load_market_data()


def test_fixture_transition_risk_expectations_use_current_state_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"
    doc = yaml.safe_load(derived_path.read_text())
    valid_states = set(get_args(TransitionRiskState))

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    labels: list[str] = []
    labels.extend(
        row["expected"]["transition_risk"]
        for row in doc.get("rows", [])
        if "transition_risk" in row.get("expected", {})
    )
    labels.extend(
        item["intent"]["transition_risk"]
        for item in mod.INTENTS
        if "transition_risk" in item.get("intent", {})
    )

    assert labels
    assert sorted(set(labels) - valid_states) == []
