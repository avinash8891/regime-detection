from __future__ import annotations

from html import unescape
import logging
import re
from urllib.parse import urljoin
from urllib.error import URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

ECB_BASE_URL = "https://www.ecb.europa.eu"
BOE_BASE_URL = "https://www.bankofengland.co.uk"
BOJ_BASE_URL = "https://www.boj.or.jp"

MONTHS = {
    "january": 1,
    "jan": 1,
    "jan.": 1,
    "february": 2,
    "feb": 2,
    "feb.": 2,
    "march": 3,
    "mar": 3,
    "mar.": 3,
    "april": 4,
    "apr": 4,
    "apr.": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "jun.": 6,
    "july": 7,
    "jul": 7,
    "jul.": 7,
    "august": 8,
    "aug": 8,
    "aug.": 8,
    "september": 9,
    "sept": 9,
    "sept.": 9,
    "sep": 9,
    "sep.": 9,
    "october": 10,
    "oct": 10,
    "oct.": 10,
    "november": 11,
    "nov": 11,
    "nov.": 11,
    "december": 12,
    "dec": 12,
    "dec.": 12,
}


def fetch_text_url(url: str, *, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "regime-detection-event-fetch/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        LOGGER.error("event source fetch failed for %s; skipping source for this run: %s", url, exc)
        return ""


def strip_tags(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def absolute_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    return urljoin(base_url, unescape(href))
