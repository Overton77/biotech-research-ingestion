"""Flattened stage-run REST routes for LangChain mission runs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.routes.langchain_dtos import flatten_stage_runs
from src.api.schemas.common import envelope
from src.research.langchain_agent.models.mission import MissionRunDocument

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
async def list_runs(
    skip: int = 0,
    limit: int = 20,
    mission_id: str | None = None,
) -> dict:
    limit = min(limit, 100)
    query: dict[str, str] = {}
    if mission_id:
        query["mission_id"] = mission_id

    docs = await MissionRunDocument.find(query).sort("-created_at").to_list()
    runs = [run for doc in docs for run in flatten_stage_runs(doc)]
    runs.sort(key=lambda item: item["started_at"] or "", reverse=True)
    paged = runs[skip : skip + limit]
    return envelope({"items": paged, "total": len(runs), "skip": skip, "limit": limit})


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict:
    parts = run_id.split(":")
    if len(parts) != 4:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    mission_id, task_slug, iteration_raw, ordinal_raw = parts
    try:
        iteration = int(iteration_raw)
        ordinal = int(ordinal_raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from exc

    doc = await MissionRunDocument.find_one({"mission_id": mission_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    run = next(
        (
            item
            for item in flatten_stage_runs(doc)
            if item["task_slug"] == task_slug
            and (item["iteration"] or 1) == iteration
            and item["id"] == f"{mission_id}:{task_slug}:{iteration}:{ordinal}"
        ),
        None,
    )
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return envelope(run)
