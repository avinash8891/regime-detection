from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

import base64
import binascii
import datetime as dt
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch.investing_live_constants import SOURCE_EARNINGS_URL

LOGGER = logging.getLogger(__name__)


class InvestingEarningsBrowserCaptureError(RuntimeError):
    """Browser capture reached a modeled failure state."""


@dataclass(frozen=True)
class CapturedEarningsPage:
    path: Path
    access_token: str


def capture_investing_earnings_loaded_page(
    *,
    output_path: Path,
    user_data_dir: Path | None = None,
    executable_path: Path | None = None,
    headless: bool | None = None,
    timeout_ms: int | None = None,
) -> Path:
    """Capture a browser-loaded Investing.com earnings page with a fresh token."""
    return capture_investing_earnings_page_with_token(
        output_path=output_path,
        user_data_dir=user_data_dir,
        executable_path=executable_path,
        headless=headless,
        timeout_ms=timeout_ms,
    ).path


def capture_investing_earnings_page_with_token(
    *,
    output_path: Path,
    user_data_dir: Path | None = None,
    executable_path: Path | None = None,
    headless: bool | None = None,
    timeout_ms: int | None = None,
) -> CapturedEarningsPage:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for automatic Investing.com earnings browser capture; "
            "install the browser extra or pass --investing-earnings-loaded-page"
        ) from exc

    resolved_user_data_dir = user_data_dir or Path(
        os.environ.get(
            "INVESTING_BROWSER_USER_DATA_DIR", output_path.parent / "browser_profile"
        )
    )
    env_executable = os.environ.get("INVESTING_BROWSER_EXECUTABLE", "").strip()
    resolved_executable_path = executable_path or (
        Path(env_executable) if env_executable else None
    )
    resolved_headless = (
        os.environ.get("INVESTING_BROWSER_HEADLESS", "0").strip().lower()
        in {"1", "true", "yes"}
        if headless is None
        else headless
    )
    resolved_timeout_ms = timeout_ms or int(
        os.environ.get("INVESTING_BROWSER_TIMEOUT_MS", "120000")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_user_data_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch_kwargs: dict[str, object] = {
            "headless": resolved_headless,
            "user_data_dir": str(resolved_user_data_dir),
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if resolved_executable_path:
            launch_kwargs["executable_path"] = str(resolved_executable_path)
        context = playwright.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(
                    SOURCE_EARNINGS_URL,
                    wait_until="domcontentloaded",
                    timeout=resolved_timeout_ms,
                )
            except PlaywrightTimeoutError as exc:
                raise InvestingEarningsBrowserCaptureError(
                    "Investing.com earnings browser capture navigation timed out; "
                    "complete the browser challenge and retry"
                ) from exc
            try:
                page.wait_for_function(
                    "() => document.documentElement.innerHTML.includes('accessToken')",
                    timeout=resolved_timeout_ms,
                )
            except PlaywrightTimeoutError as exc:
                partial_path = output_path.with_suffix(output_path.suffix + ".partial")
                partial_path.write_text(page.content())
                if output_path.exists():
                    output_path.unlink()
                raise InvestingEarningsBrowserCaptureError(
                    "Investing.com earnings browser capture did not expose accessToken; "
                    f"saved current page to {partial_path}. Complete the browser challenge and retry."
                ) from exc
            html = page.content()
            token = access_token_from_page(html)
            validate_token_not_expired(token)
            output_path.write_text(redact_access_token(html, token))
        finally:
            context.close()
    return CapturedEarningsPage(path=output_path, access_token=token)


def investing_earnings_access_token() -> str:
    return os.environ.get("INVESTING_EARNINGS_ACCESS_TOKEN", "").strip()


def loaded_earnings_page_html(path: Path | None) -> str:
    configured = path or investing_earnings_loaded_page_path()
    if configured is None:
        return ""
    return configured.read_text(errors="replace")


def investing_earnings_loaded_page_path() -> Path | None:
    configured = os.environ.get("INVESTING_EARNINGS_LOADED_PAGE", "").strip()
    return Path(configured) if configured else None


def validate_token_not_expired(token: str) -> None:
    parts = token.split(".")
    if len(parts) != 3:
        return
    try:
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        payload = json.loads(payload_bytes)
    except (binascii.Error, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error(
            "Investing.com earnings accessToken is malformed; reload the earnings calendar page"
        )
        raise RuntimeError(
            "Investing.com earnings accessToken is malformed; reload the earnings calendar page"
        ) from exc
    exp = payload.get("exp")
    if not isinstance(exp, int | float):
        return
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    if exp <= now:
        raise RuntimeError(
            "Investing.com earnings accessToken is expired; reload the earnings calendar page and retry"
        )


def redact_access_token(html: str, token: str) -> str:
    redacted = html.replace(token, "[redacted]")
    return re.sub(
        r'("accessToken"\s*:\s*")[^"]+(")',
        r"\1[redacted]\2",
        redacted,
    )


def page_data(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html
    )
    if not match:
        return {}
    return json.loads(match.group(1))


def access_token_from_page(html: str) -> str:
    token = str(
        page_data(html).get("props", {}).get("pageProps", {}).get("accessToken") or ""
    )
    if not token:
        raise RuntimeError("Investing.com earnings page did not expose accessToken")
    return token


def country_map_from_page(html: str, *, key: str) -> dict[str, dict[str, object]]:
    data = page_data(html)
    groups = (
        data.get("props", {})
        .get("pageProps", {})
        .get("state", {})
        .get("countryStore", {})
        .get(key, [])
    )
    mapped: dict[str, dict[str, object]] = {}
    for group in groups:
        for country in group.get("countries", []):
            if isinstance(country, dict):
                mapped[str(country.get("id"))] = country
    return mapped
