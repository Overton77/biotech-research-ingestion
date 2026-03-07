"""Thread Beanie document — research conversation container."""

from datetime import datetime
from typing import Literal

from beanie import Document
from pydantic import Field


class Thread(Document):
    """A research conversation thread."""

    title: str = Field(default="New research", description="Thread title")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["active", "archived"] = Field(default="active")
    metadata: dict = Field(default_factory=dict)

    class Settings:
        name = "threads"
