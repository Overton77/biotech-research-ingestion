"""create_research_plan tool — validates and returns a LangChain-aligned plan.

HITL is handled by HumanInTheLoopMiddleware which intercepts BEFORE this
tool executes. If approved, the tool runs normally. If rejected, the
middleware returns an error ToolMessage and the agent can revise.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.research.langchain_agent.models.plan import (
    ResearchPlanOutput,
    ResearchPlanTask,
    StarterSource,
)

logger = logging.getLogger(__name__)


def _normalize_tasks(raw: list[dict[str, Any]]) -> list[ResearchPlanTask]:
    return [ResearchPlanTask.model_validate(t) for t in raw]


def _normalize_starter_sources(raw: list[dict[str, Any]] | None) -> list[StarterSource]:
    if not raw:
        return []
    return [StarterSource.model_validate(s) for s in raw]


@tool
def create_research_plan(
    objective: str,
    title: str,
    stages: list[str],
    tasks: list[dict[str, Any]],
    context: str = "",
    starter_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a structured research plan for human review and approval.

    Provide the complete plan upfront — the middleware will pause for human review
    before this tool executes. If approved, the plan is returned to the agent.

    Each task dict must include:
        id, title, description, stage, dependencies (list of task ids),
        estimated_duration_minutes (optional),
        selected_tool_names (optional; must be from the allowed tool set),
        selected_subagent_names (optional; must be from the allowed subagent set),
        stage_type (optional): discovery | entity_validation | official_site_mapping |
            targeted_extraction | report_synthesis.

    If the user does not specify tools or subagents for a task, use sensible defaults
    (see coordinator system prompt) and still list them explicitly in each task.

    Args:
        objective: The main research objective.
        title: A concise descriptive title for the plan.
        stages: Ordered list of high-level stage names.
        tasks: List of task objects (validated against ResearchPlanTask).
        context: Optional summary of web research that informed the plan.
        starter_sources: Optional list of {url, description} for key sources.
    """
    output = ResearchPlanOutput(
        title=title,
        objective=objective,
        stages=stages,
        tasks=_normalize_tasks(tasks),
        context=context,
        starter_sources=_normalize_starter_sources(starter_sources),
        status="approved",
        version=1,
    )
    logger.info(
        "Research plan created: %s (%d tasks)",
        title,
        len(output.tasks),
    )
    return {"plan": output.model_dump(mode="json"), "status": "approved"}
