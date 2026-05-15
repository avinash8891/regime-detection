from __future__ import annotations

from urllib.error import URLError

import pytest

from regime_data_fetch.event_sources import _common


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_fetch_text_url_returns_empty_text_on_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        raise URLError("timeout")

    monkeypatch.setattr(_common, "urlopen", fake_urlopen)

    assert _common.fetch_text_url("https://example.test/source") == ""


def test_fetch_text_url_raises_on_invalid_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "urlopen", lambda *_args, **_kwargs: _Response(b"\xff"))

    with pytest.raises(UnicodeDecodeError):
        _common.fetch_text_url("https://example.test/source")
