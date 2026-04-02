"""Mission REST routes backed by LangChain mission run documents."""

from __future__ import annotations

from typing import Any

from beanie.odm.fields import PydanticObjectId
from fastapi import APIRouter, HTTPException, Query, status

from src.api.routes.langchain_dtos import (
    build_outputs_payload,
    compute_status_summary,
    flatten_stage_runs,
    mission_to_dict,
)
from src.api.schemas.common import envelope
from src.infrastructure.aws.async_s3 import AsyncS3Client
from src.research.langchain_agent.models.mission import MissionRunDocument
from src.research.langchain_agent.models.plan import ResearchPlan

router = APIRouter(prefix="/missions", tags=["missions"])


async def _get_mission_doc_or_404(mission_id: str) -> MissionRunDocument:
    doc = await MissionRunDocument.find_one({"mission_id": mission_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")
    return doc


async def _get_plan_for_doc(doc: MissionRunDocument) -> ResearchPlan | None:
    if not doc.research_plan_id:
        return None
    try:
        return await ResearchPlan.get(PydanticObjectId(doc.research_plan_id))
    except Exception:
        return None


def _artifact_entries_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = run["artifacts"]
    items: list[dict[str, Any]] = []
    if artifacts["final_report"]:
        items.append(artifacts["final_report"])
    items.extend(artifacts["intermediate_files"])
    if artifacts["memory_report_json"]:
        items.append(artifacts["memory_report_json"])
    if artifacts["agent_state_json"]:
        items.append(artifacts["agent_state_json"])
    return items


def _get_run_for_task(runs: list[dict[str, Any]], task_id: str, attempt_number: int) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if run["task_id"] == task_id and (run["iteration"] or 1) == attempt_number
    ]
    return candidates[0] if candidates else None


@router.get("")
async def list_missions(
    skip: int = 0,
    limit: int = 20,
    research_plan_id: str | None = None,
    thread_id: str | None = None,
    status_filter: str | None = None,
) -> dict:
    limit = min(limit, 100)
    query: dict[str, Any] = {}
    if research_plan_id:
        query["research_plan_id"] = research_plan_id
    if thread_id:
        query["thread_id"] = thread_id
    if status_filter:
        query["status"] = status_filter

    total = await MissionRunDocument.find(query).count()
    docs = await (
        MissionRunDocument.find(query)
        .sort("-created_at")
        .skip(skip)
        .limit(limit)
        .to_list()
    )
    items = []
    for doc in docs:
        items.append(mission_to_dict(doc, plan=await _get_plan_for_doc(doc)))
    return envelope({"items": items, "total": total, "skip": skip, "limit": limit})


@router.get("/{mission_id}")
async def get_mission(mission_id: str) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    return envelope(mission_to_dict(doc, plan=await _get_plan_for_doc(doc)))


@router.get("/{mission_id}/manifest")
async def get_mission_manifest(mission_id: str) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    plan = await _get_plan_for_doc(doc)
    return envelope(build_outputs_payload(doc, plan=plan))


@router.get("/{mission_id}/runs")
async def get_mission_runs(mission_id: str) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    return envelope(flatten_stage_runs(doc))


@router.get("/{mission_id}/status")
async def get_mission_status(mission_id: str) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    plan = await _get_plan_for_doc(doc)
    expected_task_ids = [task.id for task in plan.tasks] if plan else None
    return envelope(compute_status_summary(doc, expected_task_ids=expected_task_ids))


@router.get("/{mission_id}/outputs")
async def get_mission_outputs(mission_id: str) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    plan = await _get_plan_for_doc(doc)
    return envelope(build_outputs_payload(doc, plan=plan))


@router.get("/{mission_id}/runs/{task_id}/outputs")
async def get_task_run_outputs(
    mission_id: str,
    task_id: str,
    attempt_number: int = Query(default=1, ge=1),
) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    runs = flatten_stage_runs(doc)
    run = _get_run_for_task(runs, task_id, attempt_number)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return envelope(
        {
            "run": run,
            "resolved_inputs": None,
            "outputs": {"final_report_text": run["final_report_text"]},
            "events": None,
            "source": "mongodb",
        }
    )


@router.get("/{mission_id}/runs/{task_id}/artifacts")
async def get_task_artifacts(
    mission_id: str,
    task_id: str,
    attempt_number: int = Query(default=1, ge=1),
) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    runs = flatten_stage_runs(doc)
    run = _get_run_for_task(runs, task_id, attempt_number)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return envelope(_artifact_entries_for_run(run))


@router.get("/{mission_id}/runs/{task_id}/artifacts/{artifact_name}/content")
async def get_artifact_content(
    mission_id: str,
    task_id: str,
    artifact_name: str,
    attempt_number: int = Query(default=1, ge=1),
    artifact_type: str = Query(default="final_report"),
) -> dict:
    doc = await _get_mission_doc_or_404(mission_id)
    runs = flatten_stage_runs(doc)
    run = _get_run_for_task(runs, task_id, attempt_number)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    artifact = next(
        (
            item
            for item in _artifact_entries_for_run(run)
            if item["filename"] == artifact_name and item["artifact_type"] == artifact_type
        ),
        None,
    )
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    content = await AsyncS3Client().get_text(artifact["s3_key"])
    return envelope(
        {
            "artifact_name": artifact_name,
            "artifact_type": artifact_type,
            "content": content,
        }
    )
