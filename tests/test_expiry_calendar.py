from __future__ import annotations

import datetime as dt

from regime_data_fetch.expiry_calendar import (
    build_monthly_options_expiry_anchors,
    compute_monthly_options_expiry_anchor,
    expand_trading_day_window,
)


def test_compute_monthly_options_expiry_anchor_uses_regular_third_friday_when_open() -> None:
    assert compute_monthly_options_expiry_anchor(year=2025, month=11) == dt.date(2025, 11, 21)


def test_compute_monthly_options_expiry_anchor_rolls_back_for_good_friday_2019() -> None:
    # Official ICE/NYSE 2018-2019 holiday release includes Good Friday closure on 2019-04-19.
    assert compute_monthly_options_expiry_anchor(year=2019, month=4) == dt.date(2019, 4, 18)


def test_compute_monthly_options_expiry_anchor_rolls_back_for_good_friday_2022() -> None:
    # Official NYSE 2022 calendar marks Good Friday closure on 2022-04-15.
    assert compute_monthly_options_expiry_anchor(year=2022, month=4) == dt.date(2022, 4, 14)


def test_compute_monthly_options_expiry_anchor_rolls_back_for_juneteenth_2026() -> None:
    # Official NYSE/ICE 2025-2027 holiday release marks Juneteenth closure on 2026-06-19.
    assert compute_monthly_options_expiry_anchor(year=2026, month=6) == dt.date(2026, 6, 18)


def test_expand_trading_day_window_uses_nyse_sessions() -> None:
    window = expand_trading_day_window(anchor_date=dt.date(2026, 6, 18), lookback_trading_days=2, lookahead_trading_days=0)
    assert window == [
        dt.date(2026, 6, 16),
        dt.date(2026, 6, 17),
        dt.date(2026, 6, 18),
    ]


def test_build_monthly_options_expiry_anchors_covers_2015_through_current_month() -> None:
    anchors = build_monthly_options_expiry_anchors(
        start_date=dt.date(2015, 1, 1),
        end_date=dt.date(2026, 5, 7),
    )

    assert len(anchors) == 137
    assert anchors[0] == dt.date(2015, 1, 16)
    assert anchors[-1] == dt.date(2026, 5, 15)
