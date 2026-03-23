"""create_research_plan tool — generates and returns a plan.

HITL is handled by HumanInTheLoopMiddleware which intercepts BEFORE this
tool executes. If approved, the tool runs normally. If rejected, the
middleware returns an error ToolMessage and the agent can revise.
"""

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def create_research_plan(
    objective: str,
    title: str,
    stages: list[str],
    tasks: list[dict],
    context: str = "",
    starter_sources: list[dict] | None = None,
) -> dict[str, Any]:
    """Create a structured research plan for human review and approval.

    Provide the complete plan upfront — the middleware will pause for human review
    before this tool executes. If approved, the plan is returned to the agent.

    Args:
        objective: The main research objective.
        title: A concise descriptive title for the plan.
        stages: Ordered list of high-level stage names,
                e.g. ["Literature Review", "Analysis", "Synthesis"].
        tasks: List of task objects. Each task should have:
               - id (str): unique task identifier, e.g. "task-1"
               - title (str): short task title
               - description (str): what this task does
               - stage (str): which stage it belongs to
               - dependencies (list[str]): ids of tasks this depends on
               - estimated_duration_minutes (int): rough estimate
        context: Optional summary of web research that informed the plan.
        starter_sources: Optional list of {url, description} for key sources to use
                         when the mission is compiled (e.g. papers, docs, pages).
    """
    plan: dict[str, Any] = {
        "title": title,
        "objective": objective,
        "stages": stages,
        "tasks": tasks,
        "context": context,
        "starter_sources": starter_sources or [],
        "status": "approved",
        "version": 1,
    }
    logger.info("Research plan created: %s (%d tasks)", title, len(tasks))
    return {"plan": plan, "status": "approved"}
