"""Shared CORS origin lists for FastAPI and Socket.IO."""

from __future__ import annotations

from src.config.settings import Settings


def build_allowed_origins(settings: Settings) -> list[str]:
    """Origins allowed for REST and Socket.IO (explicit list)."""
    origins = [
        settings.WEB_ORIGIN,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    extra = (settings.CORS_EXTRA_ORIGINS or "").strip()
    if extra:
        origins.extend(o.strip() for o in extra.split(",") if o.strip())
    # Preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for o in origins:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def lan_origin_regex(settings: Settings) -> str | None:
    """Regex for dev frontends on RFC1918 addresses (port 3000)."""
    if not settings.CORS_ALLOW_LAN_REGEX:
        return None
    return r"http://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}):3000"
