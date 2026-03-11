"""Factory for creating an HTTP-based progress callback for Temporal workers.

The callback POSTs progress events to the API server's internal endpoint,
which then broadcasts them to the Socket.IO room for the mission.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
INTERNAL_SECRET = os.getenv("INTERNAL_API_SECRET", "")
PROGRESS_ENDPOINT = "/api/v1/internal/research-progress"


def create_progress_callback(
    mission_id: str,
    *,
    api_base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    """Create an async progress callback that POSTs to the internal endpoint.

    Returns a callable with signature:
        async def callback(event_type: str, payload: dict) -> None
    """
    base_url = api_base_url or API_BASE_URL
    url = base_url.rstrip("/") + PROGRESS_ENDPOINT
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if INTERNAL_SECRET:
        headers["X-Internal-Secret"] = INTERNAL_SECRET

    _client = client

    async def _get_client() -> httpx.AsyncClient:
        nonlocal _client
        if _client is None:
            _client = httpx.AsyncClient(timeout=5.0)
        return _client

    async def callback(event_type: str, payload: dict[str, Any]) -> None:
        body = {
            "mission_id": mission_id,
            "event_type": event_type,
            "payload": payload,
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        try:
            http = await _get_client()
            resp = await http.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "Progress POST returned %d for mission %s event %s",
                    resp.status_code, mission_id, event_type,
                )
        except Exception:
            logger.debug(
                "Progress POST failed for mission %s event %s (non-blocking)",
                mission_id, event_type, exc_info=True,
            )

    return callback
