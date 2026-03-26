"""ResearchRun REST routes — paginated list, single-resource GET."""

from __future__ import annotations

import logging

from beanie.odm.fields import PydanticObjectId
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.common import envelope
from src.research.deepagent.models.mission import ResearchRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


def _run_to_dict(r: ResearchRun) -> dict:
    return {
        "id": str(r.id),
        "mission_id": str(r.mission_id),
        "task_id": r.task_id,
        "attempt_number": r.attempt_number,
        "status": r.status,
        "resolved_inputs_snapshot": r.resolved_inputs_snapshot,
        "outputs_snapshot": r.outputs_snapshot,
        "artifacts": [a.model_dump() for a in r.artifacts],
        "error_message": r.error_message,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "created_at": r.created_at.isoformat(),
    }


@router.get("")
async def list_runs(
    skip: int = 0,
    limit: int = 20,
    mission_id: str | None = None,
) -> dict:
    """List research runs with pagination, optionally filtered by mission_id."""
    limit = min(limit, 100)
    query: dict = {}
    if mission_id:
        query["mission_id"] = PydanticObjectId(mission_id)

    total = await ResearchRun.find(query).count()
    runs = await (
        ResearchRun.find(query)
        .sort("-created_at")
        .skip(skip)
        .limit(limit)
        .to_list()
    )
    return envelope({
        "items": [_run_to_dict(r) for r in runs],
        "total": total,
        "skip": skip,
        "limit": limit,
    })


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict:
    """Return a single ResearchRun by ID."""
    try:
        oid = PydanticObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    run = await ResearchRun.get(oid)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return envelope(_run_to_dict(run))
