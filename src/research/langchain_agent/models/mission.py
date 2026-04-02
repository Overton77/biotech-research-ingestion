"""
Research mission models: configuration and run persistence.

Configuration hierarchy (loaded from JSON mission files):
  ResearchMission               ← top-level mission config
    └─ List[MissionStage]       ← each stage: slice input + prompt spec + dependencies
         └─ IterativeStageConfig (optional) ← bounded iteration loop settings

Persistence hierarchy (stored in MongoDB via Beanie):
  MissionRunDocument            ← one per run_mission() call (root document)
    └─ List[StageRunRecord]     ← one per run_single_mission_slice() (embedded)
         └─ StageArtifacts      ← S3 refs + local paths for all written artifacts
              └─ List[ArtifactRef] ← lightweight pointer (URI + metadata, no payload)
         └─ MemoryReportRecord  ← inline LangMem memory report summary

All S3 uploads are fire-and-forget (failures are logged, not raised).
Filesystem outputs are NEVER replaced — S3 refs are additive.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from beanie import Document, init_beanie
from pydantic import BaseModel, Field

from src.research.langchain_agent.agent.config import MissionSliceInput, ResearchPromptSpec
from src.research.langchain_agent.agent.prompts.prompt_builders import PROMPT_SPEC
from src.infrastructure.mongo.async_mongo_client import mongo_client
from src.research.langchain_agent.models.plan import ResearchPlan as LangchainResearchPlan
from src.research.langchain_agent.unstructured.models import UnstructuredIngestionConfig

logger = logging.getLogger(__name__)

_beanie_initialized = False


# ---------------------------------------------------------------------------
# Mission configuration models
# ---------------------------------------------------------------------------
# Research plans persisted for this pipeline live in
# ``src.research.langchain_agent.models.plan.ResearchPlan`` (collection ``langchain_research_plans``).


class IterativeStageConfig(BaseModel):
    """
    Configuration for stage-level iteration.

    When attached to a MissionStage, the DAG runner will execute that stage
    in a loop rather than as a single pass.  Each iteration produces a report
    and a NextStepsArtifact; the runner evaluates stop conditions between
    iterations and carries forward context.
    """

    max_iterations: int = 3
    completion_criteria: str = ""
    carry_forward_reports: int = 2
    stop_on_no_next_steps: bool = True


class MissionStage(BaseModel):
    """One stage of a research mission: slice input + prompt spec + optional dependencies.

    When ``iterative_config`` is set the DAG runner will execute this stage
    in a bounded iteration loop instead of a single pass.
    """

    slice_input: MissionSliceInput
    prompt_spec: ResearchPromptSpec
    execution_reminders: list[str] = Field(
        default_factory=lambda: [
            "Use runs/, reports/, and scratch/ as main folders.",
            "Save intermediate data; write the final report to reports/.",
            "Use recalled memories as hints, not unquestioned truth.",
        ]
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="task_slug of stages that must complete before this one; this stage receives their final_report.",
    )
    iterative_config: IterativeStageConfig | None = Field(
        default=None,
        description="When set, the stage runs iteratively (multiple bounded passes) instead of a single pass.",
    )


class ResearchMission(BaseModel):
    """Multi-stage research mission, typically loaded from a JSON config file."""

    mission_id: str
    mission_name: str = ""
    base_domain: str = ""
    stages: list[MissionStage] = Field(default_factory=list)
    run_kg: bool = Field(
        default=False,
        description="Run KG ingestion on completed stage reports after the mission finishes.",
    )
    unstructured_ingestion: UnstructuredIngestionConfig = Field(
        default_factory=UnstructuredIngestionConfig,
        description="Staged unstructured document ingestion after mission candidate manifests are gathered.",
    )


# ---------------------------------------------------------------------------
# Mission compiler — LLM draft (structured output) → ResearchMission
# ---------------------------------------------------------------------------


class ResearchPromptSpecModel(BaseModel):
    """Pydantic twin of ``ResearchPromptSpec`` for LLM structured output."""

    agent_identity: str = Field(default=PROMPT_SPEC.agent_identity, min_length=4)
    domain_scope: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.domain_scope))
    workflow: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.workflow))
    tool_guidance: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.tool_guidance))
    subagent_guidance: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.subagent_guidance))
    practical_limits: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.practical_limits))
    filesystem_rules: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.filesystem_rules))
    intermediate_files: list[str] = Field(default_factory=lambda: list(PROMPT_SPEC.intermediate_files))


def research_prompt_spec_model_to_dataclass(m: ResearchPromptSpecModel) -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=m.agent_identity,
        domain_scope=m.domain_scope,
        workflow=m.workflow,
        tool_guidance=m.tool_guidance,
        subagent_guidance=m.subagent_guidance,
        practical_limits=m.practical_limits,
        filesystem_rules=m.filesystem_rules,
        intermediate_files=m.intermediate_files,
    )


class TemporalScopeDraft(BaseModel):
    mode: Literal["current", "as_of_date", "date_range", "unknown"] = "current"
    as_of_date: str | None = None
    range_start: str | None = None
    range_end: str | None = None
    description: str = "Current state as of research date."


class MissionSliceInputDraft(BaseModel):
    """Compiler-safe subset of MissionSliceInput without runtime-only fields."""

    task_id: str
    mission_id: str
    task_slug: str
    user_objective: str
    targets: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(
        default_factory=lambda: ["search_web", "extract_from_urls", "map_website"]
    )
    selected_subagent_names: list[str] = Field(default_factory=list)
    report_required_sections: list[str] = Field(
        default_factory=lambda: [
            "Executive Summary",
            "Key Findings",
            "Sources",
            "Open Questions and Next Steps",
        ]
    )
    guidance_notes: list[str] = Field(default_factory=list)
    stage_type: Literal[
        "discovery",
        "entity_validation",
        "official_site_mapping",
        "targeted_extraction",
        "report_synthesis",
    ] = "discovery"
    max_step_budget: int = 12
    temporal_scope: TemporalScopeDraft | None = None
    research_date: str | None = None


class MissionStageDraft(BaseModel):
    """One mission stage as produced by the compiler LLM (before ``mission_id`` is fixed)."""

    slice_input: MissionSliceInputDraft
    prompt_spec: ResearchPromptSpecModel = Field(default_factory=ResearchPromptSpecModel)
    execution_reminders: list[str] = Field(
        default_factory=lambda: [
            "Use runs/, reports/, and scratch/ as main folders.",
            "Save intermediate data; write the final report to reports/.",
            "Use recalled memories as hints, not unquestioned truth.",
        ]
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="task_slug values that must complete before this stage.",
    )
    iterative_config: IterativeStageConfig | None = None


class ResearchMissionDraft(BaseModel):
    """
    Structured LLM output for an executable LangChain research mission.

    After validation, call :func:`draft_to_research_mission` with a new ``mission_id``.
    Each ``slice_input`` should use ``mission_id=\"pending\"``; it is rewritten on finalize.
    """

    mission_name: str = ""
    base_domain: str = ""
    stages: list[MissionStageDraft] = Field(min_length=1)
    run_kg: bool = False
    unstructured_ingestion: UnstructuredIngestionConfig = Field(
        default_factory=UnstructuredIngestionConfig,
    )


def draft_to_research_mission(draft: ResearchMissionDraft, mission_id: str) -> ResearchMission:
    """Attach a real ``mission_id`` to the mission and every stage ``slice_input``."""
    out_stages: list[MissionStage] = []
    for sd in draft.stages:
        si = sd.slice_input.model_copy(
            update={
                "mission_id": mission_id,
                "task_id": sd.slice_input.task_id or sd.slice_input.task_slug,
            }
        )
        temporal_scope = (
            si.temporal_scope.model_dump()
            if isinstance(si.temporal_scope, TemporalScopeDraft)
            else {}
        )
        ps = research_prompt_spec_model_to_dataclass(sd.prompt_spec)
        out_stages.append(
            MissionStage(
                slice_input=MissionSliceInput(
                    task_id=si.task_id or si.task_slug,
                    mission_id=mission_id,
                    task_slug=si.task_slug,
                    user_objective=si.user_objective,
                    targets=list(si.targets),
                    selected_tool_names=list(si.selected_tool_names),
                    selected_subagent_names=list(si.selected_subagent_names),
                    report_required_sections=list(si.report_required_sections),
                    guidance_notes=list(si.guidance_notes),
                    stage_type=si.stage_type,
                    max_step_budget=si.max_step_budget,
                    temporal_scope=temporal_scope,
                    research_date=si.research_date,
                    dependency_reports={},
                ),
                prompt_spec=ps,
                execution_reminders=sd.execution_reminders,
                dependencies=sd.dependencies,
                iterative_config=sd.iterative_config,
            )
        )
    return ResearchMission(
        mission_id=mission_id,
        mission_name=draft.mission_name,
        base_domain=draft.base_domain,
        stages=out_stages,
        run_kg=draft.run_kg,
        unstructured_ingestion=draft.unstructured_ingestion,
    )


# ---------------------------------------------------------------------------
# Persistence models (Beanie / MongoDB)
# ---------------------------------------------------------------------------


class ArtifactRef(BaseModel):
    """
    Lightweight pointer to a persisted artifact.
    Stores the S3 URI + key and the original local path.
    Never duplicates the payload — the content lives in S3 or on the filesystem.
    """

    s3_uri: str
    s3_key: str
    local_path: str = ""           # relative to agent_outputs/ root (empty for non-file artifacts)
    artifact_type: Literal[
        "final_report",
        "intermediate_file",
        "memory_report",
        "agent_state",
    ]
    filename: str
    size_bytes: int | None = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageArtifacts(BaseModel):
    """
    All S3-persisted artifacts produced by a single stage run.
    final_report and memory_report_json are always attempted.
    intermediate_files covers every path written to runs/<task_slug>/.
    """

    final_report: ArtifactRef | None = None
    intermediate_files: list[ArtifactRef] = Field(default_factory=list)
    memory_report_json: ArtifactRef | None = None
    agent_state_json: ArtifactRef | None = None


class MemoryReportRecord(BaseModel):
    """
    Inline copy of the LangMem memory report produced after each slice.
    The full JSON is also uploaded to S3 (referenced in StageArtifacts.memory_report_json).
    """

    mission_id: str
    summary: str
    file_paths: list[str] = Field(default_factory=list)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageRunRecord(BaseModel):
    """
    Persisted record of one run_single_mission_slice() execution.
    Embedded in MissionRunDocument.stages — not a separate collection.

    iteration is set for iterative missions (cycle number, 1-based).
    It is None for stage-based missions.
    """

    task_id: str
    task_slug: str
    stage_type: str
    targets: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)

    iteration: int | None = None

    status: Literal["running", "completed", "failed"] = "completed"
    error: str | None = None

    final_report_text: str = ""

    artifacts: StageArtifacts = Field(default_factory=StageArtifacts)
    memory_report: MemoryReportRecord | None = None

    # LangSmith trace ID for cross-referencing with the LangSmith UI
    langsmith_run_id: str | None = None

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class IterativeStageRecord(BaseModel):
    """
    Tracks one iterative stage across all its iterations.

    Embedded inside MissionRunDocument.iterative_stages.
    Each element in ``iterations`` is a StageRunRecord produced by one pass
    of run_single_mission_slice().
    """

    task_slug: str
    max_iterations: int
    iterations_completed: int = 0
    stop_reason: str = ""
    iterations: list[StageRunRecord] = Field(default_factory=list)
    combined_report_text: str = ""
    next_steps_history: list[dict] = Field(default_factory=list)


class MissionRunDocument(Document):
    """
    Root persistence document for one research mission execution.

    Stage-based missions:  stages contains one StageRunRecord per MissionStage.
    Iterative missions:    stages contains one StageRunRecord per (cycle, stage) pair.
                          max_cycles and cycles_completed track progress.

    Collection: biotech_research.research_mission_runs
    """

    mission_id: str
    mission_name: str
    research_plan_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    mission_type: Literal["stage_based", "iterative"] = "stage_based"
    base_domain: str = ""
    targets: list[str] = Field(default_factory=list)

    status: Literal["running", "completed", "partial", "failed"] = "running"
    error: str | None = None

    stages: list[StageRunRecord] = Field(default_factory=list)
    iterative_stages: list[IterativeStageRecord] = Field(default_factory=list)

    max_cycles: int | None = None
    cycles_completed: int = 0

    # LangSmith trace ID for the entire mission run
    langsmith_run_id: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    class Settings:
        name = "research_mission_runs"

    def append_stage(self, record: StageRunRecord) -> None:
        self.stages.append(record)
        self.updated_at = datetime.now(timezone.utc)

    def append_iterative_stage(self, record: IterativeStageRecord) -> None:
        self.iterative_stages.append(record)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        self.status = "completed"
        now = datetime.now(timezone.utc)
        self.updated_at = now
        self.completed_at = now

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Beanie initialization
# ---------------------------------------------------------------------------


async def init_research_agent_beanie() -> None:
    """
    Initialize Beanie for the langchain_agent storage models.
    Idempotent — safe to call multiple times.
    """
    global _beanie_initialized
    if _beanie_initialized:
        return
    db = mongo_client["biotech_research"]
    await init_beanie(
        database=db,
        document_models=[MissionRunDocument, LangchainResearchPlan],
    )
    _beanie_initialized = True
    logger.info("Beanie initialized for langchain_agent (biotech_research)")
