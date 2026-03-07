"""Socket.IO event handlers — send_message, plan_approved, plan_rejected."""

import logging
from typing import Any

from beanie.odm.fields import PydanticObjectId

from src.models.plan import ResearchPlan, ResearchTask
from src.services.coordinator_service import (
    stream_coordinator_response,
    persist_messages,
    register_pending_interrupt,
    get_thread_id_for_interrupt,
    clear_pending_interrupt,
    resume_coordinator_after_approval,
)

logger = logging.getLogger(__name__)


def _extract_plan_from_interrupt(interrupt_payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Extract (plan_dict, interrupt_id) from __interrupt__ payload."""
    raw = interrupt_payload.get("__interrupt__")
    interrupt_id = ""
    plan = {}
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        first = raw[0]
        if hasattr(first, "value"):
            val = first.value
            interrupt_id = getattr(first, "id", "") or ""
            if isinstance(val, dict):
                plan = val.get("plan") or val
                interrupt_id = val.get("interrupt_id") or interrupt_id
        elif isinstance(first, dict):
            plan = first.get("plan") or first
            interrupt_id = first.get("interrupt_id") or ""
    return plan, interrupt_id


async def handle_send_message(
    emit_fn: Any,
    thread_id: str,
    content: str,
) -> None:
    """Handle send_message: stream Coordinator response and persist messages."""
    room = f"thread:{thread_id}"
    run_id: str | None = None  # LangSmith run_id could be set from config later

    async def on_token(token: str) -> None:
        await emit_fn(
            "coordinator_token",
            {"token": token, "thread_id": thread_id, "run_id": run_id or ""},
            room=room,
        )

    async def on_tool_start(name: str, args_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_start",
            {
                "tool_name": name,
                "args_summary": args_summary,
                "thread_id": thread_id,
                "run_id": run_id or "",
            },
            room=room,
        )

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_end",
            {
                "tool_name": name,
                "result_summary": result_summary,
                "thread_id": thread_id,
                "run_id": run_id or "",
            },
            room=room,
        )

    try:
        assistant_content, interrupt_payload = await stream_coordinator_response(
            thread_id,
            content,
            on_token=on_token,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        tid = PydanticObjectId(thread_id)
        await persist_messages(tid, content, assistant_content, run_id=run_id)
        await emit_fn("coordinator_stream_end", {"thread_id": thread_id}, room=room)

        if interrupt_payload:
            plan_dict, interrupt_id = _extract_plan_from_interrupt(interrupt_payload)
            if interrupt_id:
                register_pending_interrupt(interrupt_id, thread_id)
            # Persist plan as pending_approval so frontend can GET/PATCH it
            try:
                tid = PydanticObjectId(thread_id)
                tasks = [ResearchTask.model_validate(t) for t in plan_dict.get("tasks", [])]
                doc = ResearchPlan(
                    thread_id=tid,
                    title=plan_dict.get("title") or "Research Plan",
                    objective=plan_dict.get("objective") or "",
                    stages=plan_dict.get("stages") or [],
                    tasks=tasks,
                    status="pending_approval",
                )
                await doc.insert()
                plan_dict["id"] = str(doc.id)
                plan_dict["thread_id"] = thread_id
                plan_dict["status"] = "pending_approval"
                plan_dict["created_at"] = doc.created_at.isoformat()
                plan_dict["updated_at"] = doc.updated_at.isoformat()
                plan_dict["version"] = doc.version
            except Exception as e:
                logger.warning("Failed to persist plan: %s", e)
            await emit_fn(
                "plan_ready",
                {"plan": plan_dict, "thread_id": thread_id, "interrupt_id": interrupt_id or str(id(interrupt_payload))},
                room=room,
            )
    except Exception as e:
        logger.exception("send_message failed for thread_id=%s: %s", thread_id, e)
        await emit_fn(
            "error",
            {"message": str(e), "code": "coordinator_error"},
            room=room,
        )


async def handle_plan_approved(
    emit_fn: Any,
    thread_id: str,
    interrupt_id: str,
    plan: dict[str, Any],
) -> None:
    """Resume Coordinator with approved plan; stream response and persist."""
    if get_thread_id_for_interrupt(interrupt_id) != thread_id:
        await emit_fn(
            "error",
            {"message": "Invalid or expired interrupt_id", "code": "bad_request"},
            room=f"thread:{thread_id}",
        )
        return
    clear_pending_interrupt(interrupt_id)
    room = f"thread:{thread_id}"

    async def on_token(token: str) -> None:
        await emit_fn("coordinator_token", {"token": token, "thread_id": thread_id, "run_id": ""}, room=room)

    async def on_tool_start(name: str, args_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_start",
            {"tool_name": name, "args_summary": args_summary, "thread_id": thread_id, "run_id": ""},
            room=room,
        )

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_end",
            {"tool_name": name, "result_summary": result_summary, "thread_id": thread_id, "run_id": ""},
            room=room,
        )

    try:
        resume_value = {"plan": plan, "approved": True}
        assistant_content, _ = await resume_coordinator_after_approval(
            thread_id,
            resume_value,
            on_token=on_token,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        tid = PydanticObjectId(thread_id)
        await persist_messages(tid, "[Plan approved]", assistant_content, run_id=None)
        await emit_fn("coordinator_stream_end", {"thread_id": thread_id}, room=room)
    except Exception as e:
        logger.exception("plan_approved failed for thread_id=%s: %s", thread_id, e)
        await emit_fn("error", {"message": str(e), "code": "coordinator_error"}, room=room)


async def handle_plan_rejected(
    emit_fn: Any,
    thread_id: str,
    interrupt_id: str,
    notes: str,
) -> None:
    """Resume Coordinator with rejection; stream response and persist."""
    if get_thread_id_for_interrupt(interrupt_id) != thread_id:
        await emit_fn(
            "error",
            {"message": "Invalid or expired interrupt_id", "code": "bad_request"},
            room=f"thread:{thread_id}",
        )
        return
    clear_pending_interrupt(interrupt_id)
    room = f"thread:{thread_id}"

    async def on_token(token: str) -> None:
        await emit_fn("coordinator_token", {"token": token, "thread_id": thread_id, "run_id": ""}, room=room)

    async def on_tool_start(name: str, args_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_start",
            {"tool_name": name, "args_summary": args_summary, "thread_id": thread_id, "run_id": ""},
            room=room,
        )

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_end",
            {"tool_name": name, "result_summary": result_summary, "thread_id": thread_id, "run_id": ""},
            room=room,
        )

    try:
        resume_value = {"approved": False, "notes": notes}
        assistant_content, _ = await resume_coordinator_after_approval(
            thread_id,
            resume_value,
            on_token=on_token,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        tid = PydanticObjectId(thread_id)
        await persist_messages(tid, "[Plan rejected]", assistant_content, run_id=None)
        await emit_fn("coordinator_stream_end", {"thread_id": thread_id}, room=room)
    except Exception as e:
        logger.exception("plan_rejected failed for thread_id=%s: %s", thread_id, e)
        await emit_fn("error", {"message": str(e), "code": "coordinator_error"}, room=room)
