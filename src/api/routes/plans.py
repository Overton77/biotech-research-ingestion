"""Plan REST routes — GET, PATCH, POST approve."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field

from src.models.plan import ResearchPlan
from src.api.schemas.common import envelope

router = APIRouter(prefix="/plans", tags=["plans"])


def _plan_to_dict(p: ResearchPlan) -> dict:
    return {
        "id": str(p.id),
        "thread_id": str(p.thread_id),
        "title": p.title,
        "objective": p.objective,
        "stages": p.stages,
        "tasks": [t.model_dump() for t in p.tasks],
        "status": p.status,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
        "approved_at": p.approved_at.isoformat() if p.approved_at else None,
        "approver_notes": p.approver_notes,
        "version": p.version,
    }


class PlanPatch(BaseModel):
    """Partial update for plan."""

    title: str | None = None
    objective: str | None = None
    stages: list[str] | None = None
    tasks: list[dict] | None = None
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
    return envelope(_plan_to_dict(plan))


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
        from src.models.plan import ResearchTask
        plan.tasks = [ResearchTask.model_validate(t) for t in body.tasks]
    if body.status is not None:
        plan.status = body.status
    plan.updated_at = datetime.utcnow()
    await plan.save()
    return envelope(_plan_to_dict(plan))


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
    return envelope(_plan_to_dict(plan))
