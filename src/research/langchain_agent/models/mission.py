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
from src.research.langchain_agent.storage.async_mongo_client import mongo_client

logger = logging.getLogger(__name__)

_beanie_initialized = False


# ---------------------------------------------------------------------------
# Mission configuration models
# ---------------------------------------------------------------------------


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
    mission_type: Literal["stage_based", "iterative"] = "stage_based"
    base_domain: str = ""
    targets: list[str] = Field(default_factory=list)

    status: Literal["running", "completed", "failed"] = "running"
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
        document_models=[MissionRunDocument],
    )
    _beanie_initialized = True
    logger.info("Beanie initialized for langchain_agent (biotech_research)")
