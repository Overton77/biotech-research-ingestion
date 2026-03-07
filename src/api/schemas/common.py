"""Common API envelope and pagination."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiEnvelope(BaseModel, Generic[T]):
    """Standard response envelope: { data, error }."""

    data: T | None = Field(default=None, description="Response payload")
    error: dict[str, str] | None = Field(default=None, description="Error code and message")


class CursorPage(BaseModel, Generic[T]):
    """Cursor-based page of items."""

    items: list[T] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, description="Cursor for next page, null if last")
    has_more: bool = Field(default=False)


def envelope(data: Any) -> dict[str, Any]:
    """Build envelope with data."""
    return {"data": data, "error": None}
