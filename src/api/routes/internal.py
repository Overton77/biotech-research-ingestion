"""Internal routes — progress relay from Temporal workers to Socket.IO rooms."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.socketio.server import get_sio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

INTERNAL_SECRET = os.getenv("INTERNAL_API_SECRET", "")


class ResearchProgressRequest(BaseModel):
    """Payload from Temporal worker → API server for Socket.IO broadcast."""

    mission_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


@router.post("/research-progress", status_code=status.HTTP_204_NO_CONTENT)
async def research_progress(
    req: ResearchProgressRequest,
    x_internal_secret: str | None = Header(default=None),
) -> None:
    """Receive progress from a Temporal worker and broadcast to the Socket.IO room."""
    if INTERNAL_SECRET and x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    sio = get_sio()
    room = f"mission:{req.mission_id}"
    await sio.emit(
        "research_progress",
        req.model_dump(),
        room=room,
        namespace="/research",
    )
