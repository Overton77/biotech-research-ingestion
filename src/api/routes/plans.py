"""Plan REST routes — GET, PATCH, POST approve, POST launch."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel

from src.api.routes.langchain_dtos import plan_to_dict
from src.api.schemas.common import envelope
from src.research.langchain_agent.models.plan import ResearchPlan, ResearchPlanTask, StarterSource

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plans", tags=["plans"])


# ---------------------------------------------------------------------------
# GET /plans — paginated list
# ---------------------------------------------------------------------------

@router.get("")
async def list_plans(
    skip: int = 0,
    limit: int = 20,
    thread_id: str | None = None,
    status_filter: str | None = None,
) -> dict:
    """List research plans with pagination."""
    limit = min(limit, 100)
    query: dict = {}
    if thread_id:
        query["thread_id"] = PydanticObjectId(thread_id)
    if status_filter:
        query["status"] = status_filter

    total = await ResearchPlan.find(query).count()
    plans = await (
        ResearchPlan.find(query)
        .sort("-created_at")
        .skip(skip)
        .limit(limit)
        .to_list()
    )
    return envelope(
        {
            "items": [plan_to_dict(p) for p in plans],
            "total": total,
            "skip": skip,
            "limit": limit,
        }
    )


class PlanPatch(BaseModel):
    """Partial update for plan."""

    title: str | None = None
    objective: str | None = None
    stages: list[str] | None = None
    tasks: list[dict] | None = None
    starter_sources: list[dict] | None = None
    context: str | None = None
    approver_notes: str | None = None
    status: str | None = None


@router.get("/{plan_id}")
async def get_plan(plan_id: str) -> dict:
    """Get a plan by ID."""
    try:
        oid = PydanticObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    plan = await ResearchPlan.get(oid)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    return envelope(plan_to_dict(plan))


@router.patch("/{plan_id}")
async def update_plan(plan_id: str, body: PlanPatch) -> dict:
    """Update plan (pre-approval)."""
    try:
        oid = PydanticObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    plan = await ResearchPlan.get(oid)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    if plan.status not in ("draft", "pending_approval"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Plan no longer editable")
    if body.title is not None:
        plan.title = body.title
    if body.objective is not None:
        plan.objective = body.objective
    if body.stages is not None:
        plan.stages = body.stages
    if body.tasks is not None:
        plan.tasks = [ResearchPlanTask.model_validate(t) for t in body.tasks]
    if body.starter_sources is not None:
        plan.starter_sources = [StarterSource.model_validate(s) for s in body.starter_sources]
    if body.context is not None:
        plan.context = body.context
    if body.approver_notes is not None:
        plan.approver_notes = body.approver_notes
    if body.status is not None:
        plan.status = body.status
    plan.updated_at = datetime.utcnow()
    await plan.save()
    return envelope(plan_to_dict(plan))


class PlanApproveBody(BaseModel):
    """Optional body for approve."""

    notes: str | None = None


@router.post("/{plan_id}/approve", status_code=status.HTTP_200_OK)
async def approve_plan(plan_id: str, body: PlanApproveBody | None = None) -> dict:
    """REST fallback for plan approval (when WS unavailable)."""
    try:
        oid = PydanticObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    plan = await ResearchPlan.get(oid)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    plan.status = "approved"
    plan.approved_at = datetime.utcnow()
    plan.approver_notes = body.notes if body else None
    plan.updated_at = datetime.utcnow()
    await plan.save()
    return envelope(plan_to_dict(plan))


# ---------------------------------------------------------------------------
# POST /plans/{plan_id}/launch — compile mission + start execution
# ---------------------------------------------------------------------------

async def _get_plan_or_404(plan_id: str) -> ResearchPlan:
    try:
        oid = PydanticObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    plan = await ResearchPlan.get(oid)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    return plan


@router.post("/{plan_id}/launch", status_code=status.HTTP_202_ACCEPTED)
async def launch_plan(plan_id: str) -> dict:
    """
    Compile a ResearchMission from an approved plan and start a Temporal workflow.
    Returns mission_id + workflow_id immediately.
    """
    from src.infrastructure.temporal.client import get_temporal_client
    from src.infrastructure.temporal.models import MissionWorkflowInput
    from src.infrastructure.temporal.worker import DEEP_RESEARCH_TASK_QUEUE
    from src.infrastructure.temporal.workflows.research_mission import ResearchMissionWorkflow
    from src.research.langchain_agent.compiler.mission_compiler import (
        MissionCompilationError,
        UnapprovedPlanError,
        create_mission_from_plan,
    )

    plan = await _get_plan_or_404(plan_id)

    if plan.status != "approved":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plan must be approved before launch",
        )

    try:
        mission = await create_mission_from_plan(plan)
    except UnapprovedPlanError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except MissionCompilationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    temporal_client = await get_temporal_client()
    workflow_id = f"research-mission-{mission.mission_id}"
    await temporal_client.start_workflow(
        ResearchMissionWorkflow.run,
        MissionWorkflowInput(
            mission_json=mission.model_dump(mode="json"),
            plan_id=str(plan.id),
            thread_id=str(plan.thread_id),
            workflow_id=workflow_id,
            run_kg=mission.run_kg,
            output_dir=None,
        ),
        id=workflow_id,
        task_queue=DEEP_RESEARCH_TASK_QUEUE,
    )

    plan.mission_id = mission.mission_id
    plan.workflow_id = workflow_id
    plan.mission_status = "running"
    plan.status = "executing"
    plan.updated_at = datetime.utcnow()
    await plan.save()

    return envelope({
        "mission_id": mission.mission_id,
        "workflow_id": workflow_id,
        "status": "accepted",
    })
