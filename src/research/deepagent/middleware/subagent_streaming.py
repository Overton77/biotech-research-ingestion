"""SubagentStreamingMiddleware — emits subagent completion events via stream_writer.

Watches for the DeepAgents 'task' tool (subagent spawner) and emits
subagent_completed events when it returns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)


class SubagentStreamingMiddleware(AgentMiddleware):
    """Emits subagent_completed events when the task tool returns."""

    def __init__(self, mission_id: str, task_id: str):
        super().__init__()
        self.mission_id = mission_id
        self.task_id = task_id

    def _emit_subagent_event(self, request: Any, result: Any, subagent_name: str) -> None:
        """Emit a subagent_completed event via runtime.stream_writer."""
        runtime = getattr(request, "runtime", None)
        if runtime and hasattr(runtime, "stream_writer") and runtime.stream_writer:
            content_preview = ""
            if isinstance(result, ToolMessage):
                content_preview = (
                    result.content[:200] if isinstance(result.content, str)
                    else str(result.content)[:200]
                )
            try:
                runtime.stream_writer({
                    "event_type": "subagent_completed",
                    "mission_id": self.mission_id,
                    "task_id": self.task_id,
                    "subagent_name": subagent_name,
                    "output_preview": content_preview,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                logger.debug("Failed to emit subagent_completed event", exc_info=True)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        tool_name = request.tool_call.get("name", "") if isinstance(request.tool_call, dict) else ""
        if tool_name == "task":
            subagent_name = request.tool_call.get("args", {}).get("name", "unknown")
            self._emit_subagent_event(request, result, subagent_name)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        tool_name = request.tool_call.get("name", "") if isinstance(request.tool_call, dict) else ""
        if tool_name == "task":
            subagent_name = request.tool_call.get("args", {}).get("name", "unknown")
            self._emit_subagent_event(request, result, subagent_name)
        return result
