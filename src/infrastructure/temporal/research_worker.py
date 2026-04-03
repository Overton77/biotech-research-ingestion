"""Temporal worker — research mission queue only.

Standalone worker for the research agent pipeline.  Does not depend on the
API-layer models (src.models, src.agents, etc.), so it can start without
the full backend.

Usage:
    uv run python -m src.infrastructure.temporal.research_worker

Registers:
    - ResearchMissionWorkflow and every activity it schedules (same surface as
      the deep-research worker in worker.py, minus DeepResearchMissionWorkflow).
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import logging

from dotenv import load_dotenv
from temporalio.worker import Worker

load_dotenv()

from src.infrastructure.temporal.client import get_temporal_client
from src.infrastructure.temporal.activities.research_mission import (
    emit_mission_progress,
    execute_research_stage,
    finalize_mission_run,
    ingest_kg_from_report,
    ingest_unstructured_documents,
    initialize_mission_run,
    persist_stage_activity_result,
)
from src.infrastructure.temporal.workflows.research_mission import (
    ResearchMissionWorkflow,
)

logger = logging.getLogger(__name__)

TASK_QUEUE = "deep-research-mission"


async def _init_research_beanie() -> None:
    """Initialize Beanie with research-agent-specific document models only."""
    try:
        from src.research.langchain_agent.storage.models import init_research_agent_beanie
        await init_research_agent_beanie()
        logger.info("Research agent Beanie models initialized")
    except Exception:
        logger.warning(
            "Beanie init skipped — MongoDB may not be configured. "
            "Mission runs will still work but MongoDB persistence will be best-effort.",
            exc_info=True,
        )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    await _init_research_beanie()

    client = await get_temporal_client()

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchMissionWorkflow],
        activities=[
            emit_mission_progress,
            execute_research_stage,
            finalize_mission_run,
            ingest_kg_from_report,
            ingest_unstructured_documents,
            initialize_mission_run,
            persist_stage_activity_result,
        ],
    )

    logger.info("Research mission worker started (queue=%s)", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
