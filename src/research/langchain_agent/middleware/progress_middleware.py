"""Agent progress middleware for the active LangChain research pipeline."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _safe_fire(coro: Any) -> None:
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        pass


class ResearchProgressMiddleware(AgentMiddleware):
    """Emit main-agent progress events to an async callback."""

    def __init__(
        self,
        *,
        mission_id: str,
        task_id: str,
        progress_callback: ProgressCallback | None,
        subagent_name: str | None = None,
    ) -> None:
        super().__init__()
        self.mission_id = mission_id
        self.task_id = task_id
        self.progress_callback = progress_callback
        self.subagent_name = subagent_name

    def _base_payload(self) -> dict[str, Any]:
        payload = {
            "mission_id": self.mission_id,
            "task_id": self.task_id,
            "timestamp": _utc_iso(),
            "agent_role": "subagent" if self.subagent_name else "main",
        }
        if self.subagent_name:
            payload["subagent_name"] = self.subagent_name
        return payload

    def _emit(self, event_type: str, extra: dict[str, Any] | None = None) -> None:
        if not self.progress_callback:
            return
        payload = self._base_payload()
        if extra:
            payload.update(extra)
        _safe_fire(self.progress_callback(event_type, payload))

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._emit("agent_started", {"message_count": len(state.get("messages", []))})
        return None

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._emit("agent_started", {"message_count": len(state.get("messages", []))})
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", "")
        if not isinstance(content, str):
            content = str(content)
        self._emit(
            "model_response",
            {
                "content_preview": _truncate(content),
                "message_type": getattr(last, "type", "unknown"),
                "message_count": len(messages),
            },
        )
        return None

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._emit("agent_completed", {"message_count": len(state.get("messages", []))})
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._emit("agent_completed", {"message_count": len(state.get("messages", []))})
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else str(tool_call)
        args_raw = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
        try:
            args_summary = _truncate(json.dumps(args_raw, default=str), 200)
        except Exception:
            args_summary = _truncate(str(args_raw), 200)
        self._emit("tool_start", {"tool_name": tool_name, "args_summary": args_summary})

        result = handler(request)

        result_preview = ""
        if isinstance(result, ToolMessage):
            result_preview = _truncate(
                result.content if isinstance(result.content, str) else str(result.content),
                200,
            )
        self._emit("tool_end", {"tool_name": tool_name, "result_summary": result_preview})
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else str(tool_call)
        args_raw = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
        try:
            args_summary = _truncate(json.dumps(args_raw, default=str), 200)
        except Exception:
            args_summary = _truncate(str(args_raw), 200)
        self._emit("tool_start", {"tool_name": tool_name, "args_summary": args_summary})

        result = await handler(request)

        result_preview = ""
        if isinstance(result, ToolMessage):
            result_preview = _truncate(
                result.content if isinstance(result.content, str) else str(result.content),
                200,
            )
        self._emit("tool_end", {"tool_name": tool_name, "result_summary": result_preview})
        return result
