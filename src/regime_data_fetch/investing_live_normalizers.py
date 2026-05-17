from __future__ import annotations

import datetime as dt

from regime_data_fetch.investing_live_constants import SOURCE_EARNINGS_URL


def normalize_event_rows(
    occurrences: list[dict[str, object]],
    events: list[dict[str, object]],
    start: dt.date,
    end: dt.date,
    countries: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    mapped = {
        str(event.get("id") or event.get("event_id") or ""): event for event in events
    }
    rows = []
    for occurrence in occurrences:
        event_id = str(occurrence.get("event_id") or "")
        event = mapped.get(event_id, {})
        country_id = str(event.get("country_id") or "")
        country = countries.get(country_id, {})
        rows.append(
            {
                "kind": "event",
                "requested_date_from": start.isoformat(),
                "requested_date_to": end.isoformat(),
                "occurrence_id": occurrence.get("occurrence_id", ""),
                "event_id": event_id,
                "occurrence_time_utc": occurrence.get("occurrence_time", ""),
                "actual_time_utc": occurrence.get("actual_time", ""),
                "country_id": country_id,
                "country_code": country.get("country_code", ""),
                "country": country.get("name", ""),
                "currency": event.get("currency", ""),
                "category": event.get("category", ""),
                "importance": event.get("importance", ""),
                "event_type": event.get("event_type", ""),
                "is_speech": event.get("event_type", "") == "speech",
                "is_report": event.get("event_type", "") == "report",
                "event": event.get("event_translated", ""),
                "event_short_name": event.get("short_name", ""),
                "event_long_name": event.get("long_name", ""),
                "event_description": event.get("description", ""),
                "period": occurrence.get("reference_period", ""),
                "unit": event.get("unit", ""),
                "precision": occurrence.get("precision", event.get("precision", "")),
                "actual": occurrence.get("actual", ""),
                "forecast": occurrence.get("forecast", ""),
                "previous": occurrence.get("previous", ""),
                "revised_from": occurrence.get("revised_from", ""),
                "preliminary": occurrence.get("preliminary", ""),
                "event_source": event.get("source", ""),
                "event_source_url": event.get("source_url", ""),
                "event_path": event.get("page_link", ""),
            }
        )
    return rows


def normalize_holiday_rows(
    holidays: list[dict[str, object]], start: dt.date, end: dt.date
) -> list[dict[str, object]]:
    rows = []
    for holiday in holidays:
        exchange = (
            holiday.get("exchange") if isinstance(holiday.get("exchange"), dict) else {}
        )
        rows.append(
            {
                "kind": "holiday",
                "requested_date_from": start.isoformat(),
                "requested_date_to": end.isoformat(),
                "holiday_id": holiday.get("holiday_id", ""),
                "holiday_start_utc": holiday.get("holiday_start", ""),
                "holiday_end_utc": holiday.get("holiday_end", ""),
                "country_id": exchange.get("country_id", ""),
                "country": exchange.get("country", ""),
                "exchange_id": holiday.get("exchange_id", ""),
                "exchange_short_name": exchange.get("short_name", ""),
                "exchange_long_name": exchange.get("long_name", ""),
                "exchange_time_zone": exchange.get("time_zone", ""),
                "name": holiday.get("holiday_name", ""),
                "exchange_closed": holiday.get("exchange_closed", ""),
            }
        )
    return rows


def normalize_earnings_rows(
    earnings: list[dict[str, object]],
    instruments: dict[str, dict[str, object]],
    key_metrics: dict[str, dict[str, object]],
    countries: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for earning in earnings:
        instrument_id = str(earning.get("instrument_id") or "")
        instrument = instruments.get(instrument_id, {})
        metrics = key_metrics.get(instrument_id, {}).get("key_metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        attributes = instrument.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        price = instrument.get("price") or {}
        if not isinstance(price, dict):
            price = {}
        country_id = str(
            instrument.get("country_id") or earning.get("country_id") or ""
        )
        country = countries.get(country_id, {})
        rows.append(
            {
                "kind": "earnings",
                "source_url": SOURCE_EARNINGS_URL,
                "date": earning.get("date", ""),
                "instrument_id": instrument_id,
                "company": instrument.get("long_name", earning.get("company", "")),
                "short_name": instrument.get("short_name", ""),
                "symbol": instrument.get("symbol", earning.get("symbol", "")),
                "display_symbol": instrument.get("display_symbol", ""),
                "country_id": country_id,
                "country": instrument.get("country", country.get("name", "")),
                "country_code": country.get(
                    "country_code", earning.get("country_code", "")
                ),
                "exchange_id": instrument.get("exchange_id", ""),
                "exchange_short_name": instrument.get("exchange_short_name", ""),
                "currency_id": earning.get(
                    "currency_id", instrument.get("currency_id", "")
                ),
                "currency_code": instrument.get("currency_code", ""),
                "sector_id": attributes.get("sector_id", ""),
                "importance": attributes.get("importance", ""),
                "instrument_type": instrument.get(
                    "type", metrics.get("instrument_type", "")
                ),
                "market_phase": earning.get("market_phase", ""),
                "earning_date_type": earning.get("earning_date_type", ""),
                "report_month": earning.get("report_month", ""),
                "report_year": earning.get("report_year", ""),
                "eps_actual": earning.get("eps_actual", ""),
                "eps_forecast": earning.get("eps_forecast", ""),
                "revenue_actual": earning.get("revenue_actual", ""),
                "revenue_forecast": earning.get("revenue_forecast", ""),
                "market_cap": metrics.get("market_cap", ""),
                "price_last": price.get("last", ""),
                "price_change": price.get("change", ""),
                "price_change_percent": price.get("change_percent", ""),
                "last_price_timestamp_utc": price.get("last_price_timestamp", ""),
                "instrument_link": instrument.get("link", ""),
                "market_link": instrument.get("market_link", ""),
                "active": instrument.get("active", ""),
                "realtime": instrument.get("realtime", ""),
            }
        )
    return rows
