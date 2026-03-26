"""Temporal activity — run a full LangGraph deep-research mission."""

from __future__ import annotations

import logging

from temporalio import activity 


logger = logging.getLogger(__name__)


@activity.defn
async def run_deep_research_mission(mission_id: str) -> dict:
    """Execute the entire LangGraph MissionRunner graph for *mission_id*.

    This is intentionally a single, long-running activity that wraps the
    LangGraph StateGraph.  Temporal provides durable execution guarantees
    around it (retry, timeout, visibility).  We can decompose into finer
    activities later.
    """
    from src.research.deepagent.middleware.progress_callback import create_progress_callback
    from src.research.deepagent.runtime.mission_runner import run_mission

    activity.logger.info("Starting deep-research mission %s", mission_id)

    progress_callback = create_progress_callback(mission_id)

    result = await run_mission(mission_id, progress_callback=progress_callback)

    status = result.get("mission_status", "unknown")
    completed = len(result.get("completed_task_ids", []))
    failed = len(result.get("failed_task_ids", []))

    activity.logger.info(
        "Mission %s finished — status=%s completed=%d failed=%d",
        mission_id,
        status,
        completed,
        failed,
    )

    return {
        "mission_id": mission_id,
        "status": status,
        "completed_tasks": completed,
        "failed_tasks": failed,
    }
