"""Serializable models for Temporal workflow/activity payloads.

All models here must be JSON-serializable (Pydantic BaseModel) so they can
cross the Temporal serialization boundary.  Heavy objects like LangGraph
stores, checkpointers, and Neo4j clients are never passed through Temporal —
they are created inside the activity worker process.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage activity I/O
# ---------------------------------------------------------------------------


class StageActivityInput(BaseModel):
    """Serializable input for the execute_research_stage activity."""

    mission_id: str
    stage_json: dict = Field(
        description="MissionStage.model_dump() — reconstructed inside the activity.",
    )
    dependency_reports: dict[str, str] = Field(
        default_factory=dict,
        description="task_slug -> final_report_text from completed dependency stages.",
    )
    root_filesystem: str | None = None
    snapshot_output_dir: str | None = None


class StageActivityOutput(BaseModel):
    """Serializable output from the execute_research_stage activity."""

    task_slug: str
    final_report_text: str = ""
    status: str = "completed"
    error: str | None = None
    stage_candidate_manifest_path: str = ""


# ---------------------------------------------------------------------------
# KG ingestion activity I/O
# ---------------------------------------------------------------------------


class KGIngestionInput(BaseModel):
    """Serializable input for the ingest_kg_from_report activity."""

    report_text: str
    source_report: str
    targets: list[str]
    stage_type: str = "targeted_extraction"
    research_date: str | None = None
    temporal_scope: str | None = None
    context: str = ""


class KGIngestionOutput(BaseModel):
    """Serializable output from the ingest_kg_from_report activity."""

    source_report: str
    total_nodes: int = 0
    total_rels_written: int = 0
    states_created: int = 0
    status: str = "completed"
    error: str | None = None


# ---------------------------------------------------------------------------
# Unstructured ingestion activity I/O
# ---------------------------------------------------------------------------


class UnstructuredIngestionInput(BaseModel):
    """Serializable input for the ingest_unstructured_documents activity."""

    mission_json: dict = Field(
        description="ResearchMission.model_dump() — needed for config + mission_id.",
    )
    stage_manifest_paths: list[str] = Field(
        default_factory=list,
        description="Relative paths to stage_candidate_manifest.json files from completed stages.",
    )


class UnstructuredIngestionOutput(BaseModel):
    """Serializable output from the ingest_unstructured_documents activity."""

    status: str = "completed"
    candidates_found: int = 0
    candidates_ingested: int = 0
    candidates_failed: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Workflow-level I/O
# ---------------------------------------------------------------------------


class MissionWorkflowInput(BaseModel):
    """Top-level input to ResearchMissionWorkflow."""

    mission_json: dict = Field(
        description="ResearchMission.model_dump() — reconstructed inside activities.",
    )
    run_kg: bool = False
    output_dir: str | None = None


class MissionWorkflowOutput(BaseModel):
    """Top-level output from ResearchMissionWorkflow."""

    mission_id: str
    status: str = "completed"
    stages_completed: int = 0
    stages_failed: int = 0
    kg_ingestions_completed: int = 0
    unstructured_ingestion: UnstructuredIngestionOutput | None = None
    stage_results: list[StageActivityOutput] = Field(default_factory=list)
    kg_results: list[KGIngestionOutput] = Field(default_factory=list)
