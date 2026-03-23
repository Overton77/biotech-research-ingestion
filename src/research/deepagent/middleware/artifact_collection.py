"""ArtifactCollectionMiddleware — collects filesystem artifacts and writes
a per-task manifest to the LangGraph store after agent completion.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class ArtifactCollectionMiddleware(AgentMiddleware):
    """Writes a per-task manifest to the store after agent completion."""

    def __init__(self, mission_id: str, task_id: str):
        super().__init__()
        self.mission_id = mission_id
        self.task_id = task_id

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict | None:
        """Scan state for collected data and write a task-level manifest to the store."""
        if not runtime.store:
            return None

        manifest: dict[str, Any] = {
            "task_id": self.task_id,
            "status": "completed",
            "sources_collected": state.get("sources_collected", []),
            "quality_assessment": state.get("quality_assessment"),
        }

        try:
            runtime.store.put(
                ("mission", self.mission_id, "task_manifests"),
                self.task_id,
                manifest,
            )
            logger.debug(
                "Task manifest stored for mission=%s task=%s",
                self.mission_id, self.task_id,
            )
        except Exception:
            logger.debug("Failed to write task manifest to store", exc_info=True)

        return None
