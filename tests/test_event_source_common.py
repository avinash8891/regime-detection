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


def test_fetch_text_result_distinguishes_network_failure_from_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        raise URLError("timeout")

    monkeypatch.setattr(_common, "urlopen", fake_urlopen)

    result = _common.fetch_text_result("https://example.test/source")

    assert result.ok is False
    assert result.text is None
    assert result.error == "timeout"


def test_fetch_text_result_accepts_valid_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "urlopen", lambda *_args, **_kwargs: _Response(b""))

    result = _common.fetch_text_result("https://example.test/source")

    assert result.ok is True
    assert result.text == ""
    assert result.error is None


def test_fetch_text_result_raises_on_invalid_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "urlopen", lambda *_args, **_kwargs: _Response(b"\xff"))

    with pytest.raises(UnicodeDecodeError):
        _common.fetch_text_result("https://example.test/source")
