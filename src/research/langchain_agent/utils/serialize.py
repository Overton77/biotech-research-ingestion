from typing import Any, Dict
from langchain_core.messages import BaseMessage
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def _serialize_message(msg: Any) -> Dict[str, Any]:
    """Convert a LangChain message to a JSON-safe dict."""
    if isinstance(msg, BaseMessage):
        d: Dict[str, Any] = {
            "type": msg.type,
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            d["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name") and msg.name:
            d["name"] = msg.name
        return d
    if isinstance(msg, dict):
        return msg
    return {"raw": str(msg)}


def _write_state_snapshot(
    *,
    output_dir: Path,
    task_slug: str,
    iteration: int | None,
    phase: str,
    data: Dict[str, Any],
) -> None:
    """Write a JSON state snapshot to the output directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        iter_suffix = f"_iter{iteration:02d}" if iteration is not None else ""
        filename = f"state_{task_slug}{iter_suffix}_{phase}.json"
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write state snapshot %s: %s", phase, exc)


def write_graph_state_snapshots(
    *,
    output_dir: Path | None,
    task_slug: str,
    iteration: int | None,
    agent_input: Dict[str, Any],
    result: Dict[str, Any],
    recalled_memories: Dict[str, str],
    user_message: str,
    run_thread_id: str,
) -> None:
    """Write comprehensive state snapshots for debugging and inspection."""
    if output_dir is None:
        return

    messages = result.get("messages", [])
    serialized_messages = [_serialize_message(m) for m in messages]

    tool_calls_log = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    "tool_name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })
        if hasattr(msg, "type") and msg.type == "tool":
            tool_calls_log.append({
                "tool_response": True,
                "tool_call_id": getattr(msg, "tool_call_id", ""),
                "name": getattr(msg, "name", ""),
                "content_preview": (msg.content if isinstance(msg.content, str) else str(msg.content))[:500],
            })

    _write_state_snapshot(
        output_dir=output_dir,
        task_slug=task_slug,
        iteration=iteration,
        phase="01_input",
        data={
            "run_thread_id": run_thread_id,
            "task_slug": task_slug,
            "iteration": iteration,
            "user_message": user_message,
            "recalled_memories": recalled_memories,
            "agent_input_keys": list(agent_input.keys()),
            "targets": agent_input.get("targets", []),
            "selected_subagent_names": agent_input.get("selected_subagent_names", []),
            "report_required_sections": agent_input.get("report_required_sections", []),
            "report_path": agent_input.get("report_path", ""),
            "max_step_budget": agent_input.get("max_step_budget", 0),
        },
    )

    _write_state_snapshot(
        output_dir=output_dir,
        task_slug=task_slug,
        iteration=iteration,
        phase="02_messages",
        data={
            "message_count": len(serialized_messages),
            "messages": serialized_messages,
        },
    )

    _write_state_snapshot(
        output_dir=output_dir,
        task_slug=task_slug,
        iteration=iteration,
        phase="03_tool_calls",
        data={
            "tool_call_count": len(tool_calls_log),
            "tool_calls": tool_calls_log,
        },
    )

    state_keys_to_capture = [
        "visited_urls", "written_file_paths", "read_file_paths",
        "edited_file_paths", "official_domains", "findings",
        "open_questions", "step_count", "final_report_ready",
        "tavily_search_events", "tavily_extract_events", "tavily_map_events",
        "tavily_crawl_events",
        "filesystem_events",
    ]
    final_state = {k: result.get(k) for k in state_keys_to_capture if result.get(k) is not None}

    _write_state_snapshot(
        output_dir=output_dir,
        task_slug=task_slug,
        iteration=iteration,
        phase="04_final_state",
        data=final_state,
    )