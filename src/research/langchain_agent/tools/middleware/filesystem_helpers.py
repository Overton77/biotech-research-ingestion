"""Path normalization and event extraction for filesystem tool middleware."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain.tools.tool_node import ToolCallRequest

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM


def append_list_field(
    existing: Any, value: Dict[str, Any], max_items: int = 200
) -> List[Dict[str, Any]]:
    cur = list(existing or [])
    cur.append(value)
    return cur[-max_items:]


def extract_file_event(request: ToolCallRequest) -> Dict[str, Any]:
    tool_name = request.tool_call["name"]
    args = dict(request.tool_call.get("args") or {})

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


def normalize_sandbox_path(raw_path: str) -> str:
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


def normalize_tool_call_paths(request: ToolCallRequest) -> None:
    args = request.tool_call.get("args") or {}
    for key in ("path", "file_path", "filepath", "target_path"):
        value = args.get(key)
        if isinstance(value, str):
            normalized = normalize_sandbox_path(value)
            if normalized != value:
                args[key] = normalized
