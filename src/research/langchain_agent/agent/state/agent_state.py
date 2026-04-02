from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date
from src.research.langchain_agent.kg.extraction_models import TemporalScope
from src.research.langchain_agent.agent.constants import TOOLS_MAP
from src.research.langchain_agent.agent.subagent_types import (
    ALL_SUBAGENT_NAMES,
    DEFAULT_STAGE_SUBAGENT_NAMES,
)


class MissionSliceInput(BaseModel):
    """
    One bounded agent run representing a major stage or sub-stage
    within a larger mission.
    """

    task_id: str
    mission_id: str
    task_slug: str
    user_objective: str

    targets: List[str] = Field(default_factory=list)
    dependency_reports: Dict[str, str] = Field(
        default_factory=dict,
        description="task_slug -> final report markdown from stages this one depends on (set by runner)",
    )

    selected_tool_names: List[str] = Field(
        default_factory=lambda: ["search_web", "extract_from_urls", "map_website"]
    )
    selected_subagent_names: List[str] = Field(
        default_factory=lambda: list(DEFAULT_STAGE_SUBAGENT_NAMES)
    )

    report_required_sections: List[str] = Field(
        default_factory=lambda: [
            "Executive Summary",
            "Key Findings",
            "Sources",
            "Open Questions and Next Steps",
        ]
    )

    guidance_notes: List[str] = Field(default_factory=list)

    stage_type: Literal[
        "discovery",
        "entity_validation",
        "official_site_mapping",
        "targeted_extraction",
        "report_synthesis",
    ] = "discovery"

    max_step_budget: int = 12

    # --- Temporal configuration ---
    temporal_scope: TemporalScope = Field(
        default_factory=TemporalScope,
        description="Temporal scope for this research stage. Defaults to 'current'.",
    )
    research_date: Optional[str] = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD) of when the research is considered current. "
            "Defaults to today if not set. Used as validFrom for ingested facts."
        ),
    )

    @field_validator("selected_tool_names")
    @classmethod
    def validate_tool_names(cls, value: List[str]) -> List[str]:
        unknown = [name for name in value if name not in TOOLS_MAP]
        if unknown:
            raise ValueError(f"Unknown tool names: {unknown}")
        return value

    @field_validator("selected_subagent_names")
    @classmethod
    def validate_subagent_names(cls, value: List[str]) -> List[str]:
        unknown = [name for name in value if name not in ALL_SUBAGENT_NAMES]
        if unknown:
            raise ValueError(f"Unknown subagent names: {unknown}")
        return value

    @property
    def effective_research_date(self) -> str:
        """Return the research_date or today's date as ISO string."""
        return self.research_date or date.today().isoformat()

    @property
    def effective_current_date(self) -> str:
        """Return today's date as ISO string (always real wall-clock date)."""
        return date.today().isoformat()


class RunPathsInput(BaseModel):
    run_dir: str
    report_path: str
    scratch_dir: str


class ResearchTaskMemoryReport(BaseModel):
    """Structured artifact used to feed a clean summary of the run into LangMem."""

    mission_id: str
    summary: str
    file_paths: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Iterative stage: next-steps artifact
# -----------------------------------------------------------------------------


class NextStepItem(BaseModel):
    """One open question or action item identified at the end of an iteration."""

    question: str
    priority: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


class NextStepsArtifact(BaseModel):
    """Structured next-steps produced by the evaluator after each iteration.

    The iterative runner uses this to decide whether to continue and what
    the next iteration should focus on.
    """

    stage_complete: bool = False
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed completeness (0.0 = nothing done, 1.0 = fully complete)",
    )
    open_questions: List[NextStepItem] = Field(default_factory=list)
    suggested_focus: str = ""
    key_findings_this_iteration: List[str] = Field(default_factory=list)