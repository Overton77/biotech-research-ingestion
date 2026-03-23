"""Domain models for the Deep Agent Research Mission system.

All mission-related models in one file for v1. Covers:
- Mission compilation (InputBinding, TaskDef, ResearchMissionDraft)
- Mission execution (ResearchMission Beanie document, TaskResult, ResearchRun)
- Artifacts and events
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 4.1 InputBinding
# ---------------------------------------------------------------------------

class InputBinding(BaseModel):
    """Declares how a task input is resolved from prior task outputs."""

    source_task_id: str
    source_key: str
    required: bool = True
    transform: str | None = None      # "summarize" triggers LLM summarization
    max_tokens: int | None = None     # if set, truncate/summarize value to this size


# ---------------------------------------------------------------------------
# 4.2 MainDeepAgentConfig
# ---------------------------------------------------------------------------

class MainDeepAgentConfig(BaseModel):
    """Configuration for the primary create_deep_agent for a TaskDef."""

    model_name: str = "openai:gpt-5"
    model_tier: Literal["fast", "standard", "powerful"] = "standard"
    system_prompt: str
    tool_profile_name: str = "default_research"
    filesystem_profile: str = "task_local"
    memory_profile: str = "in_memory"
    allow_general_purpose_subagent: bool = True
    max_iterations: int | None = None
    skills: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.3 CompiledSubAgentConfig
# ---------------------------------------------------------------------------

class CompiledSubAgentConfig(BaseModel):
    """Configuration for a compiled subagent worker attached to a task agent."""

    name: str
    description: str
    system_prompt: str
    model_name: str | None = None
    tool_profile_name: str = "default_research"
    filesystem_profile: str = "subagent_local"
    use_todo_middleware: bool = False
    memory_profile: str = "in_memory"
    workspace_suffix: str
    max_invocations: int = 1
    expected_output_format: str | None = None
    expected_output_path: str | None = None
    skills: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4.4 TaskExecutionPolicy
# ---------------------------------------------------------------------------

class TaskExecutionPolicy(BaseModel):
    timeout_seconds: int = 300
    max_retries: int = 1
    persist_run_after_completion: bool = True
    fallback_models: list[str] = Field(default_factory=list)
    require_human_review: bool = False


# ---------------------------------------------------------------------------
# 4.5 TaskDef
# ---------------------------------------------------------------------------

class TaskDef(BaseModel):
    """The schedulable runtime unit. One TaskDef = one Deep Agent invocation."""

    task_id: str
    name: str
    stage_label: str | None = None
    description: str
    depends_on: list[str] = Field(default_factory=list)
    input_bindings: dict[str, InputBinding] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    main_agent: MainDeepAgentConfig
    compiled_subagents: list[CompiledSubAgentConfig] = Field(default_factory=list)
    execution: TaskExecutionPolicy = Field(default_factory=TaskExecutionPolicy)


# ---------------------------------------------------------------------------
# 4.6 ResearchMissionDraft (structured output from the compiler agent)
# ---------------------------------------------------------------------------

class ResearchMissionDraft(BaseModel):
    """
    Structured output from the Mission Compiler Agent.
    Validated by Pydantic before post-processing. Not saved to Mongo directly.
    """

    title: str
    goal: str
    global_context: dict[str, Any] = Field(default_factory=dict)
    global_constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    task_defs: list[TaskDef] = Field(default_factory=list)

    # model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 4.6b SourceReference
# ---------------------------------------------------------------------------

class SourceReference(BaseModel):
    """A web or literature source discovered during research."""

    url: str
    title: str | None = None
    snippet: str | None = None
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    relevance_score: float | None = None
    source_type: str = "web"         # "web" | "paper" | "patent" | "clinical_trial" | "database"
    tool_name: str | None = None
    query: str | None = None


# ---------------------------------------------------------------------------
# 4.6c QualityAssessment
# ---------------------------------------------------------------------------

class QualityAssessment(BaseModel):
    """Result of an acceptance criteria check against a task output."""

    task_id: str
    criteria_results: list[dict[str, Any]] = Field(default_factory=list)
    overall_pass: bool = True
    suggestions: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 4.6d MissionSummary
# ---------------------------------------------------------------------------

class MissionSummary(BaseModel):
    """High-level summary populated after mission finalization."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_sources: int = 0
    total_artifacts: int = 0
    has_final_report: bool = False
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# 4.7 ArtifactRef
# ---------------------------------------------------------------------------

class ArtifactRef(BaseModel):
    """Reference to an artifact produced by a task."""

    task_id: str
    name: str
    artifact_type: str  # "report", "document", "json", "log"
    storage: Literal["filesystem", "s3", "mongo_inline"] = "filesystem"
    path: str | None = None
    content_inline: str | None = None
    content_type: str = "text/plain"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 4.7b FileReference
# ---------------------------------------------------------------------------

class FileReference(BaseModel):
    """Metadata for a file produced by the deep agent or a subagent (path, optional name/description)."""

    path: str
    name: str | None = None
    description: str | None = None


# ---------------------------------------------------------------------------
# 4.7c TaskExecutionStructuredOutput
# ---------------------------------------------------------------------------

class TaskExecutionStructuredOutput(BaseModel):
    """Structured summary of a task run: subagent output locations and main agent synthesis reports."""

    subagent_final_output_locations: list[FileReference] = Field(default_factory=list)
    final_synthesis_reports: list[FileReference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4.8 ResearchEvent
# ---------------------------------------------------------------------------

class ResearchEvent(BaseModel):
    """An event emitted during task or mission execution."""

    event_type: str  # "task_started", "task_completed", "agent_token", etc.
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 4.9 TaskResult
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """Normalized output of a single TaskDef execution."""

    task_id: str
    status: Literal["completed", "failed"]
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    events: list[ResearchEvent] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_number: int = 1
    structured_execution_summary: TaskExecutionStructuredOutput | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    quality_assessment: QualityAssessment | None = None


# ---------------------------------------------------------------------------
# 4.10 ResearchMission (Beanie Document)
# ---------------------------------------------------------------------------

class ResearchMission(Document):
    """Fully compiled, executable mission. Stored in MongoDB."""

    research_plan_id: PydanticObjectId
    thread_id: PydanticObjectId
    title: str
    goal: str
    global_context: dict[str, Any] = Field(default_factory=dict)
    global_constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    task_defs: list[TaskDef] = Field(default_factory=list)
    dependency_map: dict[str, list[str]] = Field(default_factory=dict)
    reverse_dependency_map: dict[str, list[str]] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    summary: MissionSummary | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "research_missions"
        indexes = [
            [("research_plan_id", 1)],
            [("thread_id", 1)],
            [("status", 1)],
            [("created_at", -1)],
        ]


# ---------------------------------------------------------------------------
# 4.11 ResearchRun (Beanie Document)
# ---------------------------------------------------------------------------

class ResearchRun(Document):
    """One task execution record. Written after every task completion."""

    mission_id: PydanticObjectId
    task_id: str
    attempt_number: int = 1
    status: Literal["completed", "failed"]
    resolved_inputs_snapshot: dict[str, Any] = Field(default_factory=dict)
    outputs_snapshot: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "research_runs"
        indexes = [
            [("mission_id", 1), ("task_id", 1), ("attempt_number", 1)],
            [("mission_id", 1)],
            [("status", 1)],
        ]
