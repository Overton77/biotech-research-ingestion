"""Socket.IO event handlers — send_message, plan_approved, plan_rejected."""

import logging
from datetime import datetime
from typing import Any

from beanie.odm.fields import PydanticObjectId

from src.models.plan import AgentConfig, ResearchPlan, ResearchTask
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
    """Extract (plan_dict, interrupt_id) from a HITL __interrupt__ payload.

    HumanInTheLoopMiddleware interrupt value is an HITLRequest:
        {
            "action_requests": [
                {"name": "create_research_plan", "args": {...}, "description": "..."}
            ],
            "review_configs": [...]
        }

    The plan fields live inside action_requests[0]["args"].
    """
    raw = interrupt_payload.get("__interrupt__")
    plan: dict[str, Any] = {}
    interrupt_id: str = ""

    if not isinstance(raw, (list, tuple)) or not raw:
        return plan, interrupt_id

    first = raw[0]

    val = first.value if hasattr(first, "value") else (first if isinstance(first, dict) else {})

    if isinstance(val, dict):
        action_requests = val.get("action_requests")
        if action_requests and isinstance(action_requests, list):
            args = action_requests[0].get("args", {})
            plan = {
                "title": args.get("title", "Research Plan"),
                "objective": args.get("objective", ""),
                "stages": args.get("stages", []),
                "tasks": args.get("tasks", []),
                "context": args.get("context", ""),
                "status": "pending_approval",
                "version": 1,
            }
        elif val.get("plan"):
            plan = val["plan"]
            interrupt_id = val.get("interrupt_id", "")

    if not interrupt_id and hasattr(first, "id") and first.id:
        interrupt_id = str(first.id)

    return plan, interrupt_id


def _coerce_tasks_to_models(raw_tasks: list[dict[str, Any]]) -> list[ResearchTask]:
    """Convert lightweight coordinator task dicts into ResearchTask models.

    The coordinator agent only provides id/title/description/stage/dependencies.
    We fill in sensible defaults for agent_config, inputs, outputs so the plan
    document is structurally valid for the mission compiler later.
    """
    result: list[ResearchTask] = []
    for t in raw_tasks:
        try:
            result.append(
                ResearchTask(
                    id=t.get("id", ""),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    stage=t.get("stage", ""),
                    dependencies=t.get("dependencies", []),
                    estimated_duration_minutes=t.get("estimated_duration_minutes"),
                    agent_config=AgentConfig(),
                    inputs=[],
                    outputs=[],
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed task dict: %s — %s", t, exc)
    return result


async def handle_send_message(
    emit_fn: Any,
    thread_id: str,
    content: str,
) -> None:
    """Handle send_message: stream Coordinator response and persist messages."""
    room = f"thread:{thread_id}"
    run_id: str | None = None

    async def on_token(token: str) -> None:
        await emit_fn(
            "coordinator_token",
            {"token": token, "thread_id": thread_id, "run_id": run_id or ""},
            room=room,
        )

    async def on_tool_start(name: str, args_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_start",
            {"tool_name": name, "args_summary": args_summary, "thread_id": thread_id, "run_id": run_id or ""},
            room=room,
        )

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn(
            "coordinator_tool_end",
            {"tool_name": name, "result_summary": result_summary, "thread_id": thread_id, "run_id": run_id or ""},
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

            if not interrupt_id:
                interrupt_id = thread_id

            register_pending_interrupt(interrupt_id, thread_id)

            try:
                raw_tasks = plan_dict.get("tasks") or []
                task_models = _coerce_tasks_to_models(raw_tasks)

                doc = ResearchPlan(
                    thread_id=tid,
                    title=plan_dict.get("title") or "Research Plan",
                    objective=plan_dict.get("objective") or "",
                    stages=plan_dict.get("stages") or [],
                    tasks=task_models,
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
                {
                    "plan": plan_dict,
                    "thread_id": thread_id,
                    "interrupt_id": interrupt_id,
                },
                room=room,
            )
    except Exception as e:
        logger.exception("send_message failed for thread_id=%s: %s", thread_id, e)
        await emit_fn(
            "error",
            {"message": str(e), "code": "coordinator_error"},
            room=room,
        )


async def _launch_mission_for_plan(
    emit_fn: Any,
    plan_id: str,
    thread_id: str,
) -> None:
    """Compile a ResearchMission from the approved plan and start a Temporal workflow."""
    from src.infrastructure.temporal.client import get_temporal_client
    from src.infrastructure.temporal.worker import DEEP_RESEARCH_TASK_QUEUE
    from src.infrastructure.temporal.workflows.deep_research import DeepResearchMissionWorkflow
    from src.research.compiler.mission_creator import (
        MissionCompilationError,
        UnapprovedPlanError,
        create_mission_from_plan,
    )

    room = f"thread:{thread_id}"

    plan_doc = await ResearchPlan.get(PydanticObjectId(plan_id))
    if not plan_doc:
        await emit_fn(
            "mission_launch_error",
            {"message": "Plan not found", "plan_id": plan_id, "thread_id": thread_id},
            room=room,
        )
        return

    if plan_doc.status != "approved":
        logger.warning("Plan %s status is '%s', expected 'approved'", plan_id, plan_doc.status)
        await emit_fn(
            "mission_launch_error",
            {"message": f"Plan status is '{plan_doc.status}', expected 'approved'", "plan_id": plan_id, "thread_id": thread_id},
            room=room,
        )
        return

    try:
        await emit_fn(
            "mission_compiling",
            {"plan_id": plan_id, "thread_id": thread_id},
            room=room,
        )

        mission = await create_mission_from_plan(plan_doc)

        temporal_client = await get_temporal_client()
        workflow_id = f"deep-research-{mission.id}"
        handle = await temporal_client.start_workflow(
            DeepResearchMissionWorkflow.run,
            str(mission.id),
            id=workflow_id,
            task_queue=DEEP_RESEARCH_TASK_QUEUE,
        )

        plan_doc.status = "executing"
        plan_doc.updated_at = datetime.utcnow()
        await plan_doc.save()

        logger.info(
            "Mission %s launched via Temporal workflow %s for plan %s",
            mission.id,
            handle.id,
            plan_id,
        )

        await emit_fn(
            "mission_launched",
            {
                "mission_id": str(mission.id),
                "plan_id": plan_id,
                "thread_id": thread_id,
                "workflow_id": workflow_id,
            },
            room=room,
        )

    except (UnapprovedPlanError, MissionCompilationError) as e:
        logger.exception("Mission compilation failed for plan %s: %s", plan_id, e)
        await emit_fn(
            "mission_launch_error",
            {"message": str(e), "plan_id": plan_id, "thread_id": thread_id},
            room=room,
        )
    except Exception as e:
        logger.exception("Failed to launch mission for plan %s: %s", plan_id, e)
        await emit_fn(
            "mission_launch_error",
            {"message": str(e), "plan_id": plan_id, "thread_id": thread_id},
            room=room,
        )


async def handle_plan_approved(
    emit_fn: Any,
    thread_id: str,
    interrupt_id: str,
    plan: dict[str, Any],
) -> None:
    """Resume Coordinator with an approved plan, then compile + launch the mission."""
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
        await emit_fn("coordinator_tool_start", {"tool_name": name, "args_summary": args_summary, "thread_id": thread_id, "run_id": ""}, room=room)

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn("coordinator_tool_end", {"tool_name": name, "result_summary": result_summary, "thread_id": thread_id, "run_id": ""}, room=room)

    try:
        if plan:
            resume_value = {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "create_research_plan",
                            "args": {
                                "objective": plan.get("objective", ""),
                                "title": plan.get("title", "Research Plan"),
                                "stages": plan.get("stages", []),
                                "tasks": plan.get("tasks", []),
                                "context": plan.get("context", ""),
                            },
                        },
                    }
                ]
            }
        else:
            resume_value = {"decisions": [{"type": "approve"}]}

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

        # ---- Auto-launch mission from the approved plan ----
        plan_id = plan.get("id") if plan else None
        if plan_id:
            plan_doc = await ResearchPlan.get(PydanticObjectId(plan_id))
            if plan_doc and plan_doc.status == "pending_approval":
                plan_doc.status = "approved"
                plan_doc.approved_at = datetime.utcnow()
                plan_doc.updated_at = datetime.utcnow()

                if plan and plan.get("tasks") and not plan_doc.tasks:
                    plan_doc.tasks = _coerce_tasks_to_models(plan["tasks"])

                await plan_doc.save()
                logger.info("Plan %s marked as approved", plan_id)

            await _launch_mission_for_plan(emit_fn, plan_id, thread_id)
        else:
            logger.warning(
                "No plan_id in approval payload for thread %s — cannot auto-launch mission",
                thread_id,
            )

    except Exception as e:
        logger.exception("plan_approved failed for thread_id=%s: %s", thread_id, e)
        await emit_fn("error", {"message": str(e), "code": "coordinator_error"}, room=room)


async def handle_plan_rejected(
    emit_fn: Any,
    thread_id: str,
    interrupt_id: str,
    notes: str,
) -> None:
    """Resume Coordinator with a rejected plan."""
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
        await emit_fn("coordinator_tool_start", {"tool_name": name, "args_summary": args_summary, "thread_id": thread_id, "run_id": ""}, room=room)

    async def on_tool_end(name: str, result_summary: str) -> None:
        await emit_fn("coordinator_tool_end", {"tool_name": name, "result_summary": result_summary, "thread_id": thread_id, "run_id": ""}, room=room)

    try:
        resume_value = {
            "decisions": [
                {
                    "type": "reject",
                    "message": notes or "Plan rejected by user.",
                }
            ]
        }
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
