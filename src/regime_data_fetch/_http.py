from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

UrlOpen = Callable[..., Any]


def fetch_bytes(
    url: str,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    method: str | None = None,
    retries: int = 1,
    backoff_seconds: float = 0.0,
    retry_status_codes: frozenset[int] = TRANSIENT_HTTP_STATUS_CODES,
    urlopen: UrlOpen | None = None,
) -> bytes:
    _require_fetchable_url(url)
    opener = urllib.request.urlopen if urlopen is None else urlopen
    request = urllib.request.Request(  # noqa: S310 - _require_fetchable_url allows only http(s).
        url,
        data=data,
        headers=headers_with_user_agent(headers),
        method=method,
    )
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in retry_status_codes or attempt >= retries:
                raise
            last_exc = exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise
            last_exc = exc
        _close_url_exception(last_exc)
        if backoff_seconds > 0:
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError("HTTP retry loop exited without returning or raising")


def _require_fetchable_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"Unsupported URL scheme for HTTP fetch: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"HTTP fetch URL must include a network location: {url!r}")


def fetch_text(
    url: str,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    method: str | None = None,
    retries: int = 1,
    backoff_seconds: float = 0.0,
    retry_status_codes: frozenset[int] = TRANSIENT_HTTP_STATUS_CODES,
    errors: str = "replace",
    urlopen: UrlOpen | None = None,
) -> str:
    return fetch_bytes(
        url,
        timeout=timeout,
        headers=headers,
        data=data,
        method=method,
        retries=retries,
        backoff_seconds=backoff_seconds,
        retry_status_codes=retry_status_codes,
        urlopen=urlopen,
    ).decode("utf-8", errors=errors)


def headers_with_user_agent(headers: dict[str, str] | None) -> dict[str, str]:
    merged = dict(headers or {})
    if not any(key.lower() == "user-agent" for key in merged):
        merged["User-Agent"] = DEFAULT_USER_AGENT
    return merged


def _close_url_exception(exc: BaseException | None) -> None:
    if exc is None:
        return
    close = getattr(exc, "close", None)
    if callable(close):
        close()
