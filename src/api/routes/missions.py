"""Mission REST routes — list, detail, status, runs, S3 outputs, artifacts."""

from __future__ import annotations

import logging
from typing import Any

from beanie.odm.fields import PydanticObjectId
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.common import envelope
from src.research.models.mission import ResearchMission, ResearchRun
from src.research.persistence.runs_s3 import (
    ResearchRunS3Paths,
    get_research_runs_s3_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/missions", tags=["missions"])


# ---------------------------------------------------------------------------
# GET /missions — paginated list
# ---------------------------------------------------------------------------

@router.get("")
async def list_missions(
    skip: int = 0,
    limit: int = 20,
    research_plan_id: str | None = None,
    thread_id: str | None = None,
    status_filter: str | None = None,
) -> dict:
    """List research missions with pagination."""
    limit = min(limit, 100)
    query: dict = {}
    if research_plan_id:
        query["research_plan_id"] = PydanticObjectId(research_plan_id)
    if thread_id:
        query["thread_id"] = PydanticObjectId(thread_id)
    if status_filter:
        query["status"] = status_filter

    total = await ResearchMission.find(query).count()
    missions = await (
        ResearchMission.find(query)
        .sort("-created_at")
        .skip(skip)
        .limit(limit)
        .to_list()
    )
    return envelope({
        "items": [_mission_to_dict(m) for m in missions],
        "total": total,
        "skip": skip,
        "limit": limit,
    })


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
        "task_defs": [td.model_dump(mode="json") for td in m.task_defs],
        "dependency_map": m.dependency_map,
        "reverse_dependency_map": m.reverse_dependency_map,
        "status": m.status,
        "summary": m.summary.model_dump(mode="json") if m.summary else None,
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
        "artifacts": [a.model_dump(mode="json") for a in r.artifacts],
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


@router.get("/{mission_id}/manifest")
async def get_mission_manifest(mission_id: str) -> dict:
    """Fetch the mission manifest (comprehensive task results + sources + quality)."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    mission = await ResearchMission.get(oid)
    if not mission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    s3_store = get_research_runs_s3_store()
    try:
        manifest = await s3_store.get_manifest(mission)
        return envelope(manifest)
    except Exception:
        logger.debug("Manifest not in S3 for mission %s, building from MongoDB", mission_id)
        return envelope(_build_basic_manifest_from_mongo(mission))


def _build_basic_manifest_from_mongo(mission: ResearchMission) -> dict:
    """Fallback manifest built from the MongoDB document when S3 is unavailable."""
    return {
        "mission_id": str(mission.id),
        "title": mission.title,
        "status": mission.status,
        "completed_at": mission.updated_at.isoformat() if mission.updated_at else None,
        "tasks": [
            {
                "task_id": td.task_id,
                "name": td.name,
                "status": "unknown",
            }
            for td in mission.task_defs
        ],
        "summary": mission.summary.model_dump(mode="json") if mission.summary else None,
        "source": "mongodb_fallback",
    }


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


# ---------------------------------------------------------------------------
# S3 Output Retrieval
# ---------------------------------------------------------------------------

async def _safe_s3_get_json(key: str) -> dict[str, Any] | None:
    try:
        s3 = get_research_runs_s3_store().s3
        return await s3.get_json(key)
    except Exception:
        return None


async def _safe_s3_get_text(key: str) -> str | None:
    try:
        s3 = get_research_runs_s3_store().s3
        return await s3.get_text(key)
    except Exception:
        return None


@router.get("/{mission_id}/outputs")
async def get_mission_outputs(mission_id: str) -> dict:
    """Return full mission-level outputs from S3."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    mission = await ResearchMission.get(oid)
    if not mission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    paths = ResearchRunS3Paths(mission_id=mission_id)

    mission_json = await _safe_s3_get_json(paths.mission_json_key())
    mission_draft = await _safe_s3_get_json(paths.mission_draft_json_key())
    final_report_md = await _safe_s3_get_text(paths.final_report_markdown_key())
    final_report_json = await _safe_s3_get_json(paths.final_report_json_key())
    summary = await _safe_s3_get_json(paths.summary_json_key())
    task_runs_index = await _safe_s3_get_json(paths.task_runs_index_key())

    return envelope({
        "mission": mission_json,
        "mission_draft": mission_draft,
        "final_report_markdown": final_report_md,
        "final_report_json": final_report_json,
        "summary": summary,
        "task_runs_index": task_runs_index,
    })


@router.get("/{mission_id}/runs/{task_id}/outputs")
async def get_task_run_outputs(
    mission_id: str,
    task_id: str,
    attempt_number: int = Query(default=1, ge=1),
) -> dict:
    """Return full task-level outputs from S3, with MongoDB fallback."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    paths = ResearchRunS3Paths(mission_id=mission_id)

    run_json = await _safe_s3_get_json(paths.run_json_key(task_id, attempt_number))
    resolved_inputs = await _safe_s3_get_json(paths.resolved_inputs_key(task_id, attempt_number))
    outputs = await _safe_s3_get_json(paths.outputs_key(task_id, attempt_number))
    events = await _safe_s3_get_json(paths.events_key(task_id, attempt_number))

    # Fallback to MongoDB if S3 has no data
    if run_json is None:
        run_doc = await ResearchRun.find_one({
            "mission_id": oid,
            "task_id": task_id,
            "attempt_number": attempt_number,
        })
        if not run_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

        return envelope({
            "run": _run_to_dict(run_doc),
            "resolved_inputs": run_doc.resolved_inputs_snapshot,
            "outputs": run_doc.outputs_snapshot,
            "events": None,
            "source": "mongodb",
        })

    return envelope({
        "run": run_json,
        "resolved_inputs": resolved_inputs,
        "outputs": outputs,
        "events": events,
        "source": "s3",
    })


@router.get("/{mission_id}/runs/{task_id}/artifacts")
async def get_task_artifacts(
    mission_id: str,
    task_id: str,
    attempt_number: int = Query(default=1, ge=1),
) -> dict:
    """Return artifact metadata for a task run."""
    try:
        oid = PydanticObjectId(mission_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    run_doc = await ResearchRun.find_one({
        "mission_id": oid,
        "task_id": task_id,
        "attempt_number": attempt_number,
    })
    if not run_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return envelope([a.model_dump(mode="json") for a in run_doc.artifacts])


@router.get("/{mission_id}/runs/{task_id}/artifacts/{artifact_name}/content")
async def get_artifact_content(
    mission_id: str,
    task_id: str,
    artifact_name: str,
    attempt_number: int = Query(default=1, ge=1),
    artifact_type: str = Query(default="report"),
) -> dict:
    """Return artifact content from S3 or a presigned URL for large files."""
    paths = ResearchRunS3Paths(mission_id=mission_id)
    key = paths.artifact_key(task_id, attempt_number, artifact_type, artifact_name)

    content = await _safe_s3_get_text(key)
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found in S3",
        )

    return envelope({
        "artifact_name": artifact_name,
        "artifact_type": artifact_type,
        "content": content,
    })
