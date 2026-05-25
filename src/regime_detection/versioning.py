from __future__ import annotations

from regime_detection import __version__

ENGINE_VERSION_PREFIX = "regime-engine-v"


def engine_version() -> str:
    return f"{ENGINE_VERSION_PREFIX}{__version__}"
