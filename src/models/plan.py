"""Research plan and task models — Pydantic + Beanie."""

from datetime import datetime
from typing import Any, Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Per-task agent configuration."""

    model: str = "openai:gpt-5"
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    backend_type: Literal["state", "filesystem", "composite"] = "state"
    backend_root_dir: str | None = None
    interrupt_on: dict[str, Any] | None = None
    max_retries: int = 6
    timeout: int = 120


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
    """A single task in a research plan."""

    id: str
    title: str
    description: str
    stage: str
    sub_stage: str | None = None
    agent_config: AgentConfig
    inputs: list[TaskInputRef] = Field(default_factory=list)
    outputs: list[TaskOutputSpec] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    estimated_duration_minutes: int | None = None


class ResearchPlan(Document):
    """Research plan document — stages and tasks, status, approval."""

    thread_id: PydanticObjectId
    title: str = "Research Plan"
    objective: str = ""
    stages: list[str] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
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
