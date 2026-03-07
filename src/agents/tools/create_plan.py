"""create_research_plan tool — builds plan and calls interrupt() for HITL."""

import logging
import uuid
from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt

from src.models.plan import ResearchPlan, ResearchTask, AgentConfig, TaskInputRef, TaskOutputSpec

logger = logging.getLogger(__name__)


def _default_plan_draft(objective: str, context: str) -> dict[str, Any]:
    """Build a minimal draft plan from objective and context (LLM can replace with richer output)."""
    return {
        "title": "Research Plan",
        "objective": objective,
        "stages": ["Literature Review", "Analysis", "Synthesis"],
        "tasks": [
            {
                "id": "task-1",
                "title": "Web and literature search",
                "description": f"Search for relevant literature and sources for: {objective[:200]}.",
                "stage": "Literature Review",
                "sub_stage": None,
                "agent_config": {
                    "model": "anthropic:claude-sonnet-4-20250514",
                    "system_prompt": "You are a research assistant.",
                    "tools": ["web_search"],
                    "backend_type": "state",
                    "backend_root_dir": None,
                    "interrupt_on": None,
                    "max_retries": 6,
                    "timeout": 120,
                },
                "inputs": [
                    {
                        "name": "objective",
                        "source": "user_provided",
                        "source_task_id": None,
                        "output_name": None,
                        "description": "Research objective",
                    }
                ],
                "outputs": [
                    {
                        "name": "literature_summary",
                        "type": "markdown",
                        "description": "Summary of findings",
                        "required": True,
                    }
                ],
                "dependencies": [],
                "estimated_duration_minutes": 15,
            },
        ],
        "status": "pending_approval",
        "version": 1,
    }


@tool
def create_research_plan(objective: str, context: str = "") -> dict[str, Any]:
    """Create a structured research plan for the given objective and optional context. Pauses for human approval before proceeding.
    Use this when the user asks for a plan or is ready to formalize their research into stages and tasks."""
    draft = _default_plan_draft(objective, context)
    interrupt_id = str(uuid.uuid4())
    # interrupt() raises; value is sent to client. On resume, the same tool returns the resume value.
    approved = interrupt({
        "type": "plan_review",
        "plan": draft,
        "message": "Please review this research plan before execution begins.",
        "interrupt_id": interrupt_id,
    })
    # When resumed, approved is the value passed to Command(resume=...)
    if isinstance(approved, dict) and approved.get("approved"):
        return {"plan": approved.get("plan", draft), "approved": True}
    return {"plan": draft, "approved": False}
