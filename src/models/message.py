"""Message Beanie document — chat message within a thread."""

from datetime import datetime
from typing import Any, Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import Field


class Message(Document):
    """A single message in a thread (user, assistant, system, or tool)."""

    thread_id: PydanticObjectId
    role: Literal["user", "assistant", "system", "tool"] = Field(...)
    content: str | list[dict[str, Any]] = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    run_id: str | None = Field(default=None, description="LangSmith run ID")
    metadata: dict = Field(default_factory=dict)

    class Settings:
        name = "messages"
