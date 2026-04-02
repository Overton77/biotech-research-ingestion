"""Research plan and task models — Pydantic + Beanie."""

from datetime import datetime
from typing import Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field


class StarterSource(BaseModel):
    """A starter reference (URL + description) for the Mission Compiler to use."""

    url: str = ""
    description: str = ""


class TaskInputRef(BaseModel):
    """Declaration of a task input and its source."""

    name: str
    source: Literal["task_output", "user_provided", "external"]
    source_task_id: str | None = None
    output_name: str | None = None
    description: str = ""


class TaskOutputSpec(BaseModel):
    """Declaration of a task output."""

    name: str
    type: Literal["text", "markdown", "json", "file", "s3_ref"]
    description: str = ""
    required: bool = True


class ResearchTask(BaseModel):
    """A single task in a research plan. Agent configuration is refined by the LangChain mission compiler."""

    id: str
    title: str
    description: str
    stage: str
    sub_stage: str | None = None
    inputs: list[TaskInputRef] = Field(default_factory=list)
    outputs: list[TaskOutputSpec] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    estimated_duration_minutes: int | None = None
    # Optional — set by the coordinator for LangChain ``MissionSliceInput`` compilation
    selected_tool_names: list[str] | None = None
    selected_subagent_names: list[str] | None = None
    stage_type: (
        Literal[
            "discovery",
            "entity_validation",
            "official_site_mapping",
            "targeted_extraction",
            "report_synthesis",
        ]
        | None
    ) = None


class ResearchPlan(Document):
    """Research plan document — stages and tasks, status, approval."""

    thread_id: PydanticObjectId
    title: str = "Research Plan"
    objective: str = ""
    stages: list[str] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
    starter_sources: list[StarterSource] = Field(default_factory=list)
    status: Literal[
        "draft",
        "pending_approval",
        "approved",
        "rejected",
        "executing",
        "complete",
        "failed",
    ] = "draft"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: datetime | None = None
    approver_notes: str | None = None
    version: int = 1

    class Settings:
        name = "research_plans"
