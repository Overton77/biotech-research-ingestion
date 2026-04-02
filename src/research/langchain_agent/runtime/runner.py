"""Single entrypoint: run a compiled mission with injected runtime dependencies."""

from __future__ import annotations

from typing import Any

from src.research.langchain_agent.models.mission import ResearchMission
from src.research.langchain_agent.runtime.deps import ResearchMissionRuntime
from src.research.langchain_agent.workflow.run_mission import run_mission as _run_mission_workflow


async def run_compiled_mission(
    mission: ResearchMission,
    runtime: ResearchMissionRuntime,
) -> list[dict[str, Any]]:
    """Execute all stages of ``mission`` using the given LangGraph / memory stack.

    This is the function to call from FastAPI routes and workers: pass a
    ``ResearchMission`` built in memory or loaded via ``mission_loader``,
    plus a ``ResearchMissionRuntime`` built from your app's persistence layer.
    """
    return await _run_mission_workflow(
        mission,
        store=runtime.store,
        checkpointer=runtime.checkpointer,
        memory_manager=runtime.memory_manager,
        root_filesystem=runtime.root_filesystem,
        snapshot_output_dir=runtime.snapshot_output_dir,
    )
