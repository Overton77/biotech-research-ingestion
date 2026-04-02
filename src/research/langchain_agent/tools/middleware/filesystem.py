from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from src.research.langchain_agent.tools.middleware.filesystem_helpers import (
    append_list_field,
    extract_file_event,
    normalize_sandbox_path as _normalize_sandbox_path,
    normalize_tool_call_paths,
)

logger = logging.getLogger(__name__)

FILESYSTEM_TOOL_NAMES = {
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
}


@wrap_tool_call
async def monitor_filesystem_tools(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    tool_name = request.tool_call["name"]

    if tool_name not in FILESYSTEM_TOOL_NAMES:
        return await handler(request)

    normalize_tool_call_paths(request)
    event = extract_file_event(request)
    path = event.get("path") or ""
    logger.debug("filesystem middleware tool=%s path=%s", tool_name, path or "(none)")

    result = await handler(request)

    state = request.state or {}

    filesystem_events = append_list_field(
        state.get("filesystem_events"),
        {
            **event,
            "tool_call_id": request.tool_call.get("id"),
        },
    )

    update: Dict[str, Any] = {
        "filesystem_events": filesystem_events,
    }

    if path:
        if tool_name == "read_file":
            update["read_file_paths"] = list(state.get("read_file_paths") or []) + [path]
        elif tool_name == "write_file":
            update["written_file_paths"] = list(state.get("written_file_paths") or []) + [path]
        elif tool_name == "edit_file":
            update["edited_file_paths"] = list(state.get("edited_file_paths") or []) + [path]

            if str(path).startswith("reports/") and str(path).endswith(".md"):
                update["final_report_path"] = path
                update["final_report_ready"] = True

    if isinstance(result, Command):
        existing_update = dict(getattr(result, "update", {}) or {})
        merged_update = {**existing_update, **update}

        for key in (
            "filesystem_events",
            "read_file_paths",
            "written_file_paths",
            "edited_file_paths",
        ):
            if key in existing_update and key in update:
                merged_update[key] = list(existing_update.get(key) or []) + list(update.get(key) or [])

        return Command(update=merged_update)

    return Command(
        update={
            **update,
            "messages": [result],
        }
    )
