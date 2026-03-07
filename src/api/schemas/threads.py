"""Thread and message API schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ThreadCreate(BaseModel):
    """Request body for POST /threads."""

    title: str = Field(default="New research", max_length=500)


class ThreadResponse(BaseModel):
    """Thread as returned by API."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    status: Literal["active", "archived"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Message as returned by API."""

    id: str
    thread_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]]
    created_at: datetime
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
