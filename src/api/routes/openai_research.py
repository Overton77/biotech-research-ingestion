# src/api/routes/openai_research.py
from __future__ import annotations

from uuid import uuid4

from beanie.odm.fields import PydanticObjectId
from fastapi import APIRouter, HTTPException

from src.api.schemas.openai_research import (
    CreateOpenAIResearchRunRequest,
    CreateOpenAIResearchRunResponse,
)
from src.infrastructure.temporal.workflows.openai_research import OpenAIResearchWorkflow
from src.infrastructure.temporal.client import get_temporal_client
from src.models.openai_research import OpenAIResearchPlan, OpenAIResearchRun
from src.utils.now import utc_now

router = APIRouter(prefix="/openai-research", tags=["openai-research"])


@router.post("/runs", response_model=CreateOpenAIResearchRunResponse)
async def create_openai_research_run(
    payload: CreateOpenAIResearchRunRequest,
) -> CreateOpenAIResearchRunResponse:
    if payload.thread_id:
        try:
            thread_id = PydanticObjectId(payload.thread_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid thread_id") from exc
    else:
        thread_id = PydanticObjectId()

    plan = OpenAIResearchPlan(
        thread_id=thread_id,
        title=payload.title,
        objective=payload.objective,
        user_prompt=payload.user_prompt,
        model=payload.model,
        coordinator_notes=payload.coordinator_notes,
        system_instructions=payload.system_instructions,
        expected_output_format=payload.expected_output_format,
        seeded_sources=[s.model_dump() for s in payload.seeded_sources],
        tools=payload.tools,
        status="approved",
        approved_at=utc_now(),
        approver_notes=payload.approver_notes,
    )
    await plan.insert()

    run = OpenAIResearchRun(
        thread_id=thread_id,
        openai_research_plan_id=plan.id,
        model=payload.model,
        request_input="",
        request_tools=[],
        request_metadata={},
        status="queued",
        status_history=[
            {
                "source": "internal",
                "status": "queued",
                "at": utc_now().isoformat(),
                "details": {"reason": "run created from approved plan"},
            }
        ],
    )
    await run.insert()

    workflow_id = f"openai-research-run-{run.id}-{uuid4().hex[:8]}"

    temporal_client = await get_temporal_client()
    await temporal_client.start_workflow(
        OpenAIResearchWorkflow.run,
        str(run.id),
        id=workflow_id,
        task_queue="openai-research",
    )

    return CreateOpenAIResearchRunResponse(
        plan_id=str(plan.id),
        run_id=str(run.id),
        workflow_id=workflow_id,
        status=run.status,
    )