"""FastAPI application with lifespan-managed MongoDB and LangGraph persistence."""
from __future__ import annotations  
import src.main 
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from beanie import init_beanie
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient

from src.agents.coordinator import get_coordinator_graph, reset_coordinator_graph
from src.api.routes import health, plans, threads, openai_research
from src.api.routes.internal import router as internal_router
from src.api.routes.missions import router as missions_router
from src.api.routes.runs import router as runs_router
from src.api.socketio.server import get_sio_mount_app
from src.config import get_settings
from src.models import Message, Thread
from src.models.plan import ResearchPlan 
from src.models.openai_research import OpenAIResearchPlan, OpenAIResearchRun
from src.research.models.mission import ResearchMission, ResearchRun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app resources on startup and cleanly close them on shutdown."""
    settings = get_settings()
    mongo_client: AsyncMongoClient | None = None

    try:
        await get_coordinator_graph()
        logger.info("Coordinator graph initialized")

        mongo_client = AsyncMongoClient(settings.MONGODB_URI)
        app.state.mongo_client = mongo_client
        app.state.mongo_db = mongo_client[settings.MONGODB_DB]

        await init_beanie(
            database=app.state.mongo_db,
            document_models=[
                Thread, Message, ResearchPlan,
                OpenAIResearchPlan, OpenAIResearchRun,
                ResearchMission, ResearchRun,
            ],
        )
        logger.info("Beanie initialized (Thread, Message, ResearchPlan, OpenAI*, ResearchMission, ResearchRun)")

        yield

    finally:
        await reset_coordinator_graph()  
        # add deep agents later  


        if mongo_client is not None:
            try:
                await mongo_client.aclose()
                logger.info("MongoDB client closed")
            except Exception:
                logger.exception("Failed to close MongoDB client cleanly")

        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create the FastAPI application with CORS, routers, and Socket.IO mount."""
    settings = get_settings()

    app = FastAPI(
        title="Biotech Research Ingestion",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.WEB_ORIGIN, "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(threads.router, prefix="/api/v1")
    app.include_router(plans.router, prefix="/api/v1")
    app.include_router(openai_research.router, prefix="/api/v1")
    app.include_router(missions_router, prefix="/api/v1")
    app.include_router(runs_router, prefix="/api/v1")
    app.include_router(internal_router, prefix="/api/v1")

    app.mount("/socket.io", get_sio_mount_app())

    return app


app = create_app()
