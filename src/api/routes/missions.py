"""Mission REST routes — GET mission state, GET runs for a mission."""

from __future__ import annotations

import logging

from beanie.odm.fields import PydanticObjectId
from fastapi import APIRouter, HTTPException, status

from src.api.schemas.common import envelope
from src.research.models.mission import ResearchMission, ResearchRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/missions", tags=["missions"])


def _mission_to_dict(m: ResearchMission) -> dict:
    return {
        "id": str(m.id),
        "research_plan_id": str(m.research_plan_id),
        "thread_id": str(m.thread_id),
        "title": m.title,
        "goal": m.goal,
        "global_context": m.global_context,
        "global_constraints": m.global_constraints,
        "success_criteria": m.success_criteria,
        "task_defs": [td.model_dump() for td in m.task_defs],
        "dependency_map": m.dependency_map,
        "status": m.status,
        "created_at": m.created_at.isoformat(),
        "updated_at": m.updated_at.isoformat(),
    }


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


@router.get("/{mission_id}")
async def get_mission(mission_id: str) -> dict:
    """Return current mission state."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    mission = await ResearchMission.get(oid)
    if not mission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    return envelope(_mission_to_dict(mission))


@router.get("/{mission_id}/runs")
async def get_mission_runs(mission_id: str) -> dict:
    """Return all ResearchRun documents for a mission."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    runs = await ResearchRun.find(
        ResearchRun.mission_id == oid,
    ).sort("+created_at").to_list()

    return envelope([_run_to_dict(r) for r in runs])


@router.get("/{mission_id}/status")
async def get_mission_status(mission_id: str) -> dict:
    """Return a lightweight mission status summary."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    mission = await ResearchMission.get(oid)
    if not mission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    runs = await ResearchRun.find(
        ResearchRun.mission_id == oid,
    ).to_list()

    completed_tasks = [r.task_id for r in runs if r.status == "completed"]
    failed_tasks = [r.task_id for r in runs if r.status == "failed"]
    total_tasks = len(mission.task_defs)

    return envelope({
        "mission_id": str(mission.id),
        "status": mission.status,
        "total_tasks": total_tasks,
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(failed_tasks),
        "pending_tasks": total_tasks - len(completed_tasks) - len(failed_tasks),
        "completed_task_ids": completed_tasks,
        "failed_task_ids": failed_tasks,
    })
