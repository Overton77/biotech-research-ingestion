"""Temporal worker — runs both OpenAI-research and deep-research task queues."""

from __future__ import annotations 
from src.infrastructure.runtime.windows_asyncio import configure_windows_asyncio 
configure_windows_asyncio()

import asyncio
import logging

from beanie import init_beanie
from pymongo import AsyncMongoClient
from temporalio.worker import Worker

from src.config import get_settings
from src.infrastructure.temporal.client import get_temporal_client

from src.infrastructure.temporal.activities.openai_research import (
    launch_openai_research_run,
    persist_openai_research_result,
    poll_openai_research_run,
)
from src.infrastructure.temporal.activities.deep_research import (
    run_deep_research_mission,
)
from src.infrastructure.temporal.activities.research_mission import (
    emit_mission_progress,
    execute_research_stage,
    finalize_mission_run,
    ingest_kg_from_report,
    ingest_unstructured_documents,
    initialize_mission_run,
    persist_stage_activity_result,
)
from src.infrastructure.temporal.workflows.openai_research import OpenAIResearchWorkflow
from src.infrastructure.temporal.workflows.deep_research import DeepResearchMissionWorkflow
from src.infrastructure.temporal.workflows.research_mission import ResearchMissionWorkflow

logger = logging.getLogger(__name__)

DEEP_RESEARCH_TASK_QUEUE = "deep-research-mission"


async def _init_beanie() -> None:
    """Initialise MongoDB + Beanie so activities can use Beanie documents."""
    from src.models import Message, Thread
    from src.research.langchain_agent.models import MissionRunDocument, ResearchPlan

    settings = get_settings()
    client = AsyncMongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB]
    await init_beanie(
        database=db,
        document_models=[Thread, Message, ResearchPlan, MissionRunDocument],
    )
    logger.info("Beanie initialised for Temporal worker")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    await _init_beanie()

    client = await get_temporal_client()

    openai_worker = Worker(
        client,
        task_queue="openai-research",
        workflows=[OpenAIResearchWorkflow],
        activities=[
            launch_openai_research_run,
            poll_openai_research_run,
            persist_openai_research_result,
        ],
    )

    deep_research_worker = Worker(
        client,
        task_queue=DEEP_RESEARCH_TASK_QUEUE,
        workflows=[DeepResearchMissionWorkflow, ResearchMissionWorkflow],
        activities=[
            run_deep_research_mission,
            emit_mission_progress,
            execute_research_stage,
            finalize_mission_run,
            ingest_kg_from_report,
            ingest_unstructured_documents,
            initialize_mission_run,
            persist_stage_activity_result,
        ],
    )

    logger.info(
        "Starting Temporal workers (queues: openai-research, %s)",
        DEEP_RESEARCH_TASK_QUEUE,
    )
    await asyncio.gather(
        openai_worker.run(),
        deep_research_worker.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
