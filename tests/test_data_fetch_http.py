from __future__ import annotations

import urllib.error
from urllib.request import Request

import pytest

from regime_data_fetch import _http


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_fetch_bytes_sends_default_user_agent_and_extra_headers() -> None:
    seen: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, *, timeout: float) -> _Response:
        seen.append((request, timeout))
        return _Response(b"payload")

    payload = _http.fetch_bytes(
        "https://example.test/data",
        timeout=12.5,
        headers={"Accept": "application/json"},
        urlopen=fake_urlopen,
    )

    assert payload == b"payload"
    assert seen[0][1] == 12.5
    assert seen[0][0].headers["User-agent"] == _http.DEFAULT_USER_AGENT
    assert seen[0][0].headers["Accept"] == "application/json"


def test_fetch_text_decodes_with_replacement() -> None:
    def fake_urlopen(_request: Request, *, timeout: float) -> _Response:
        assert timeout == 30.0
        return _Response(b"ok \xff")

    assert (
        _http.fetch_text("https://example.test/page", urlopen=fake_urlopen)
        == "ok \ufffd"
    )


def test_fetch_bytes_retries_transient_http_errors() -> None:
    attempts = 0
    first_error = urllib.error.HTTPError(
        "https://example.test/data", 503, "unavailable", {}, None
    )

    def fake_urlopen(_request: Request, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise first_error
        return _Response(b"after-retry")

    payload = _http.fetch_bytes(
        "https://example.test/data",
        retries=2,
        backoff_seconds=0,
        urlopen=fake_urlopen,
    )

    assert payload == b"after-retry"
    assert attempts == 2


def test_fetch_bytes_does_not_retry_non_transient_http_errors() -> None:
    attempts = 0

    def fake_urlopen(_request: Request, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://example.test/data", 404, "missing", {}, None
        )

    with pytest.raises(urllib.error.HTTPError):
        _http.fetch_bytes(
            "https://example.test/data",
            retries=3,
            backoff_seconds=0,
            urlopen=fake_urlopen,
        )

    assert attempts == 1


def test_fetch_bytes_rejects_non_http_urls_before_opening() -> None:
    called = False

    def fake_urlopen(_request: Request, *, timeout: float) -> _Response:
        nonlocal called
        called = True
        return _Response(b"local file contents")

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        _http.fetch_bytes("file:///tmp/secret.txt", urlopen=fake_urlopen)

    assert called is False
