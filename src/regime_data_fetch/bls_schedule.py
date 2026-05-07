from __future__ import annotations

import datetime as dt
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


US_EASTERN = ZoneInfo("America/New_York")
MONTH_NAME_RE = r"(?:Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|Jun\.?|June|Jul\.?|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)"
_MODERN_ROW_RE = re.compile(
    r"(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday),\s+"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})\s+"
    r"(?P<time>\d{2}:\d{2}\s+[AP]M)\s+"
    r"(?P<title>Employment Situation|Consumer Price Index(?:es)?)\s+for\s+(?P<period>[A-Za-z]+\s+\d{4})",
    flags=re.IGNORECASE,
)
_LEGACY_CPI_RE = re.compile(
    rf"Consumer Price Index(?:es)?(?:,\s*(?P<period>[A-Za-z]+\s+\d{{4}}))?\s+"
    rf"(?P<month>{MONTH_NAME_RE})\s+(?P<day>\d{{1,2}})(?:,\s*(?P<year>\d{{4}}))?\s+"
    r"(?P<time>\d{1,2}:\d{2}\s*[ap]m)",
    flags=re.IGNORECASE,
)
_LEGACY_NFP_RE = re.compile(
    rf"The Employment Situation(?:,\s*(?P<period>[A-Za-z]+\s+\d{{4}}))?\s+"
    rf"(?P<month>{MONTH_NAME_RE})\s+(?P<day>\d{{1,2}})(?:,\s*(?P<year>\d{{4}}))?\s+"
    r"(?P<time>\d{1,2}:\d{2}\s*[ap]m)",
    flags=re.IGNORECASE,
)


class BLSScheduleFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class BLSReleaseDate:
    date: dt.date
    release_timestamp_et: dt.datetime
    type: str
    source_url: str
    reference_period: str


def build_bls_schedule_year_urls(*, start_year: int, end_year: int) -> list[str]:
    urls: list[str] = []
    for year in range(start_year, end_year + 1):
        urls.append(f"https://www.bls.gov/schedule/{year}/")
        urls.append(f"https://www.bls.gov/schedule/{year}/home.htm")
    return urls


def parse_bls_schedule_page(html: str, *, source_url: str, default_year: int) -> list[BLSReleaseDate]:
    normalized = _normalize_html_text(html)
    releases: list[BLSReleaseDate] = []
    releases.extend(_parse_modern_rows(normalized, source_url=source_url))
    releases.extend(_parse_legacy_rows(normalized, source_url=source_url, default_year=default_year))
    deduped: dict[tuple[dt.date, str], BLSReleaseDate] = {}
    for release in releases:
        deduped[(release.date, release.type)] = release
    return sorted(deduped.values(), key=lambda item: (item.date, item.type))


def fetch_bls_year_releases(
    *,
    start_year: int,
    end_year: int,
    page_fetcher=None,
) -> list[BLSReleaseDate]:
    page_fetcher = page_fetcher or fetch_bls_schedule_page_text
    deduped: dict[tuple[dt.date, str], BLSReleaseDate] = {}
    failures: list[str] = []
    for year in range(start_year, end_year + 1):
        year_releases: list[BLSReleaseDate] | None = None
        for url in build_bls_schedule_year_urls(start_year=year, end_year=year):
            try:
                html = page_fetcher(url)
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                continue
            parsed = parse_bls_schedule_page(html, source_url=url, default_year=year)
            if parsed:
                year_releases = parsed
                break
        if year_releases is None:
            raise BLSScheduleFetchError(
                f"BLS schedule fetch failed for {year}; tried yearly schedule URLs with no parseable CPI/NFP rows. Failures: {failures[-4:]}"
            )
        for release in year_releases:
            deduped[(release.date, release.type)] = release
    return sorted(deduped.values(), key=lambda item: (item.date, item.type))


def build_bls_local_archive_page_fetcher(
    *,
    schedule_dir: Path,
    fallback_page_fetcher=None,
):
    fallback_page_fetcher = fallback_page_fetcher or fetch_bls_schedule_page_text

    def _page_fetcher(url: str) -> str:
        year = _extract_year_from_schedule_url(url)
        local_path = schedule_dir / f"bls_schedule_{year}.html"
        if local_path.exists():
            return local_path.read_text()
        return fallback_page_fetcher(url)

    return _page_fetcher


def fetch_bls_schedule_page_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _extract_year_from_schedule_url(url: str) -> int:
    match = re.search(r"/schedule/(?P<year>\d{4})/", url)
    if not match:
        raise BLSScheduleFetchError(f"Could not determine BLS schedule year from URL: {url}")
    return int(match.group("year"))


def _parse_modern_rows(normalized: str, *, source_url: str) -> list[BLSReleaseDate]:
    releases: list[BLSReleaseDate] = []
    for match in _MODERN_ROW_RE.finditer(normalized):
        title = match.group("title").lower()
        release_type = "NFP" if "employment situation" in title else "CPI"
        release_date = _parse_month_day_year(
            month_text=match.group("month"),
            day_text=match.group("day"),
            year_text=match.group("year"),
        )
        release_timestamp = _parse_release_timestamp(
            release_date=release_date,
            time_text=match.group("time"),
        )
        releases.append(
            BLSReleaseDate(
                date=release_date,
                release_timestamp_et=release_timestamp,
                type=release_type,
                source_url=source_url,
                reference_period=match.group("period").strip(),
            )
        )
    return releases


def _parse_legacy_rows(normalized: str, *, source_url: str, default_year: int) -> list[BLSReleaseDate]:
    releases: list[BLSReleaseDate] = []
    for pattern, release_type in ((_LEGACY_CPI_RE, "CPI"), (_LEGACY_NFP_RE, "NFP")):
        for match in pattern.finditer(normalized):
            reference_period = (match.group("period") or "").strip()
            if not reference_period:
                continue
            inferred_year = _infer_legacy_release_year(
                release_month_text=match.group("month"),
                default_year=default_year,
                reference_period=reference_period,
                explicit_year_text=match.group("year"),
            )
            release_date = _parse_month_day_year(
                month_text=match.group("month"),
                day_text=match.group("day"),
                year_text=str(inferred_year),
            )
            release_timestamp = _parse_release_timestamp(
                release_date=release_date,
                time_text=match.group("time"),
            )
            releases.append(
                BLSReleaseDate(
                    date=release_date,
                    release_timestamp_et=release_timestamp,
                    type=release_type,
                    source_url=source_url,
                    reference_period=reference_period,
                )
            )
    return releases


def _normalize_html_text(html: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?i)</(p|div|li|tr|td|h1|h2|h3|h4|h5|h6)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def _parse_month_day_year(*, month_text: str, day_text: str, year_text: str) -> dt.date:
    month_clean = month_text.strip().rstrip(".")
    if month_clean == "Sept":
        month_clean = "Sep"
    normalized = f"{month_clean} {int(day_text)} {int(year_text)}"
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return dt.datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    raise BLSScheduleFetchError(f"Unsupported BLS month/day/year token: {month_text!r} {day_text!r} {year_text!r}")


def _parse_release_timestamp(*, release_date: dt.date, time_text: str) -> dt.datetime:
    time_clean = re.sub(r"\s+", " ", time_text.strip()).upper()
    release_time = dt.datetime.strptime(time_clean, "%I:%M %p").time()
    return dt.datetime.combine(release_date, release_time, tzinfo=US_EASTERN)


def _infer_legacy_release_year(
    *,
    release_month_text: str,
    default_year: int,
    reference_period: str,
    explicit_year_text: str | None,
) -> int:
    if explicit_year_text:
        return int(explicit_year_text)

    release_month = _parse_month_number(release_month_text)
    reference_month, reference_year = _parse_reference_period(reference_period)
    if release_month < reference_month:
        return reference_year + 1
    return reference_year


def _parse_reference_period(reference_period: str) -> tuple[int, int]:
    parsed = dt.datetime.strptime(reference_period, "%B %Y")
    return parsed.month, parsed.year


def _parse_month_number(month_text: str) -> int:
    month_clean = month_text.strip().rstrip(".")
    if month_clean == "Sept":
        month_clean = "Sep"
    for fmt in ("%b", "%B"):
        try:
            return dt.datetime.strptime(month_clean, fmt).month
        except ValueError:
            continue
    raise BLSScheduleFetchError(f"Unsupported BLS month token: {month_text!r}")
