# src/api/schemas/openai_research.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OpenAISeededSourceIn(BaseModel):
    type: Literal[
        "url",
        "domain",
        "query",
        "file_id",
        "vector_store_id",
        "note",
        "internal_reference",
    ]
    value: str
    label: str | None = None
    description: str | None = None
    required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateOpenAIResearchRunRequest(BaseModel):
    thread_id: str | None = None 
    title: str = "OpenAI Research Plan"
    objective: str
    user_prompt: str
    model: Literal["o3-deep-research", "o4-mini-deep-research"] = "o3-deep-research"
    coordinator_notes: str | None = None
    system_instructions: str | None = None
    expected_output_format: str | None = None
    seeded_sources: list[OpenAISeededSourceIn] = Field(default_factory=list)
    tools: list[Literal["web_search_preview", "file_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["web_search_preview"]
    )
    approver_notes: str | None = None


class CreateOpenAIResearchRunResponse(BaseModel):
    plan_id: str
    run_id: str
    workflow_id: str
    status: str