"""
LangChain-aligned research plan models.

``ResearchPlanOutput`` validates coordinator tool output before HITL / persistence.
``ResearchPlan`` is the Beanie document for plans that compile into
``ResearchMission`` (mission stages use the same tool and subagent registries
as ``MissionSliceInput``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field, field_validator, model_validator

from src.research.langchain_agent.agent.constants import TOOLS_MAP
from src.research.langchain_agent.agent.subagent_types import (
    ALL_SUBAGENT_NAMES,
    DEFAULT_STAGE_SUBAGENT_NAMES,
)

# ---------------------------------------------------------------------------
# Pydantic — shared pieces
# ---------------------------------------------------------------------------


class StarterSource(BaseModel):
    """Starter URL + description for later mission compilation."""

    url: str = ""
    description: str = ""


class ResearchPlanTask(BaseModel):
    """
    One schedulable task in a research plan.

    ``selected_tool_names`` and ``selected_subagent_names`` use the same allowlists
    as ``MissionSliceInput``. The coordinator should prefer values the user
    specifies; otherwise choose sensible defaults from the pre-selected sets.
    """

    id: str
    title: str
    description: str
    stage: str
    dependencies: list[str] = Field(default_factory=list)
    estimated_duration_minutes: int | None = None

    selected_tool_names: list[str] = Field(
        default_factory=lambda: ["search_web", "extract_from_urls", "map_website"],
    )
    selected_subagent_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_STAGE_SUBAGENT_NAMES),
    )

    stage_type: Literal[
        "discovery",
        "entity_validation",
        "official_site_mapping",
        "targeted_extraction",
        "report_synthesis",
    ] | None = Field(
        default=None,
        description="Optional hint for mission compilation; defaults can be inferred later.",
    )

    @field_validator("selected_tool_names")
    @classmethod
    def validate_tool_names(cls, value: list[str]) -> list[str]:
        if not value:
            return ["search_web", "extract_from_urls", "map_website"]
        unknown = [name for name in value if name not in TOOLS_MAP]
        if unknown:
            raise ValueError(f"Unknown tool names: {unknown}. Allowed: {sorted(TOOLS_MAP.keys())}")
        return value

    @field_validator("selected_subagent_names")
    @classmethod
    def validate_subagent_names(cls, value: list[str]) -> list[str]:
        if not value:
            return list(DEFAULT_STAGE_SUBAGENT_NAMES)
        unknown = [name for name in value if name not in ALL_SUBAGENT_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown subagent names: {unknown}. Allowed: {list(ALL_SUBAGENT_NAMES)}"
            )
        return value


class ResearchPlanOutput(BaseModel):
    """
    Validated payload returned by ``create_research_plan`` (no Mongo ids).

    This is the shape embedded under the ``plan`` key in the tool result.
    """

    title: str
    objective: str
    stages: list[str]
    tasks: list[ResearchPlanTask]
    context: str = ""
    starter_sources: list[StarterSource] = Field(default_factory=list)
    status: Literal["draft", "pending_approval", "approved"] = "approved"
    version: int = 1

    @model_validator(mode="after")
    def stages_cover_tasks(self) -> ResearchPlanOutput:
        stage_set = set(self.stages)
        for t in self.tasks:
            if t.stage not in stage_set:
                raise ValueError(
                    f"Task {t.id!r} references stage {t.stage!r} which is not in stages: {self.stages}"
                )
        return self


# ---------------------------------------------------------------------------
# Beanie — persisted plan (langchain pipeline)
# ---------------------------------------------------------------------------


class ResearchPlan(Document):
    """
    Research plan stored for the LangChain research pipeline.

    Collection is separate from ``src.models.plan.ResearchPlan`` (legacy) so
    migrations can be done explicitly.
    """

    thread_id: PydanticObjectId
    title: str = "Research Plan"
    objective: str = ""
    stages: list[str] = Field(default_factory=list)
    tasks: list[ResearchPlanTask] = Field(default_factory=list)
    starter_sources: list[StarterSource] = Field(default_factory=list)
    context: str = ""

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
        name = "langchain_research_plans"

    def to_output(self) -> ResearchPlanOutput:
        """Projection suitable for API / coordinator responses."""
        out_status: Literal["draft", "pending_approval", "approved"]
        if self.status in ("draft", "pending_approval", "approved"):
            out_status = self.status
        else:
            out_status = "approved"
        return ResearchPlanOutput(
            title=self.title,
            objective=self.objective,
            stages=list(self.stages),
            tasks=[ResearchPlanTask.model_validate(t.model_dump()) for t in self.tasks],
            context=self.context,
            starter_sources=list(self.starter_sources),
            status=out_status,
            version=self.version,
        )
