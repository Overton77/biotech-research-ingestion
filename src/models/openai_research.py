from datetime import datetime
from typing import Any, Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field

from src.utils.now import utc_now


class OpenAISeededSource(BaseModel):
    """Optional source/context hints provided by a coordinator before kickoff."""

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


class OpenAIResearchPlan(Document):
    """Approved plan for handing off a research mission to OpenAI Deep Research."""

    thread_id: PydanticObjectId
    title: str = "OpenAI Research Plan"
    objective: str
    user_prompt: str
    model: Literal["o3-deep-research", "o4-mini-deep-research"] = "o3-deep-research"

    # Prompt-building / orchestration fields
    coordinator_notes: str | None = None
    system_instructions: str | None = None
    expected_output_format: str | None = None

    # Optional seeded sources from coordinator
    seeded_sources: list[OpenAISeededSource] = Field(default_factory=list)

    # Tooling the eventual OpenAI run should use
    tools: list[Literal["web_search_preview", "file_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["web_search_preview"]
    )

    status: Literal[
        "draft",
        "pending_approval",
        "approved",
        "rejected",
        "submitted",
        "complete",
        "failed",
    ] = "draft"

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    approved_at: datetime | None = None
    approver_notes: str | None = None
    version: int = 1

    class Settings:
        name = "openai_research_plans"
        indexes = [
            "thread_id",
            "status",
            "created_at",
        ]





class OpenAIResearchRun(Document):
    """Execution record for a single OpenAI Deep Research background run."""

    thread_id: PydanticObjectId
    openai_research_plan_id: PydanticObjectId

    # OpenAI API identifiers
    openai_response_id: str | None = None
    openai_status: str | None = None
    output_dir: str | None = None

    # Execution config captured at launch time
    model: Literal["o3-deep-research", "o4-mini-deep-research"]
    request_input: str
    request_tools: list[dict[str, Any]] = Field(default_factory=list)
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    openai_request_payload: dict[str, Any] | None = None
    webhook_url: str | None = None

    status: Literal[
        "queued",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
        "incomplete",
    ] = "queued"

    # Persisted outputs
    final_report_text: str | None = None
    output_items: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None

    # Raw API / webhook payloads
    openai_initial_response: dict[str, Any] | None = None
    openai_final_response: dict[str, Any] | None = None
    openai_usage: dict[str, Any] | None = None
    openai_incomplete_details: dict[str, Any] | None = None
    filesystem_artifacts: dict[str, str] = Field(default_factory=dict)
    status_history: list[dict[str, Any]] = Field(default_factory=list)


    # Failure / retry handling
    error_message: str | None = None
    error_payload: dict[str, Any] | None = None
    retry_count: int = 0

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    last_polled_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None

    class Settings:
        name = "openai_research_runs"
        indexes = [
            "thread_id",
            "openai_research_plan_id",
            "openai_response_id",
            "status",
            "created_at",
        ]