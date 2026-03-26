from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM


FILESYSTEM_TOOL_NAMES = {
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
}


def _append_list_field(existing: Any, value: Dict[str, Any], max_items: int = 200) -> List[Dict[str, Any]]:
    cur = list(existing or [])
    cur.append(value)
    return cur[-max_items:]


def _extract_file_event(request: ToolCallRequest) -> Dict[str, Any]:
    tool_name = request.tool_call["name"]
    args = dict(request.tool_call.get("args") or {})

    # Common deepagents filesystem args are usually path/file_path/content-like.
    path = (
        args.get("path")
        or args.get("file_path")
        or args.get("filepath")
        or args.get("target_path")
        or ""
    )

    return {
        "tool_name": tool_name,
        "path": path,
        "args": args,
    }


def _normalize_sandbox_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path

    cleaned = raw_path.replace("\\\\?\\", "")
    candidate = Path(cleaned)
    if not candidate.is_absolute():
        return raw_path

    try:
        relative = candidate.resolve().relative_to(ROOT_FILESYSTEM.resolve())
    except ValueError:
        return raw_path

    return relative.as_posix()


def _normalize_tool_call_paths(request: ToolCallRequest) -> None:
    args = request.tool_call.get("args") or {}
    for key in ("path", "file_path", "filepath", "target_path"):
        value = args.get(key)
        if isinstance(value, str):
            normalized = _normalize_sandbox_path(value)
            if normalized != value:
                args[key] = normalized


def _tool_result_to_content(result: ToolMessage | Command) -> str:
    """
    Best-effort extraction for logging / state metadata.
    Do not depend on this for correctness.
    """
    if isinstance(result, ToolMessage):
        return str(result.content)

    # For Command results, there may or may not be a ToolMessage embedded.
    update = getattr(result, "update", None) or {}
    messages = update.get("messages") or []
    if messages:
        msg0 = messages[0]
        content = getattr(msg0, "content", "")
        return str(content)
    return ""


@wrap_tool_call
async def monitor_filesystem_tools(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    tool_name = request.tool_call["name"]

    # Pass through for non-filesystem tools
    if tool_name not in FILESYSTEM_TOOL_NAMES:
        return await handler(request)

    _normalize_tool_call_paths(request)
    event = _extract_file_event(request)

    result = await handler(request)

    # Pull prior state if available
    state = request.state or {}

    filesystem_events = _append_list_field(
        state.get("filesystem_events"),
        {
            **event,
            "tool_call_id": request.tool_call.get("id"),
        },
    )

    update: Dict[str, Any] = {
        "filesystem_events": filesystem_events,
    }

    # Add path-specific convenience fields
    path = event.get("path")
    if path:
        if tool_name == "read_file":
            update["read_file_paths"] = list(state.get("read_file_paths") or []) + [path]
        elif tool_name == "write_file":
            update["written_file_paths"] = list(state.get("written_file_paths") or []) + [path]
        elif tool_name == "edit_file":
            update["edited_file_paths"] = list(state.get("edited_file_paths") or []) + [path]

            # Optional: treat edits to report files specially
            if str(path).startswith("reports/") and str(path).endswith(".md"):
                update["final_report_path"] = path
                update["final_report_ready"] = True

    # If the handler already returned a Command, merge updates
    if isinstance(result, Command):
        existing_update = dict(getattr(result, "update", {}) or {})
        merged_update = {**existing_update, **update}

        # Merge overlapping list fields explicitly
        for key in (
            "filesystem_events",
            "read_file_paths",
            "written_file_paths",
            "edited_file_paths",
        ):
            if key in existing_update and key in update:
                merged_update[key] = list(existing_update.get(key) or []) + list(update.get(key) or [])

        return Command(update=merged_update)

    # If the handler returned a ToolMessage, wrap it into Command so we can inject state
    return Command(
        update={
            **update,
            "messages": [result],
        }
    )