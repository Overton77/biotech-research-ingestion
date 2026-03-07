"""FastAPI application with lifespan-managed MongoDB and LangGraph checkpointer.

Windows / psycopg3 note
-----------------------
psycopg3 requires ``asyncio.SelectorEventLoop`` on Windows; the default
``ProactorEventLoop`` is incompatible.  The policy **must** be set before
uvicorn calls ``asyncio.run()``, which happens before this module is imported.
Use the root ``main.py`` runner (``python main.py``) instead of invoking
uvicorn directly so the policy is applied in time.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient
from beanie import init_beanie

from src.config import get_settings
from src.models import Thread, Message
from src.models.plan import ResearchPlan
from src.api.routes import health, threads, plans
from src.api.socketio.server import get_sio_mount_app
from src.agents.coordinator import setup_postgres_checkpointer, close_postgres_checkpointer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_mongo_client: AsyncMongoClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up: initialise MongoDB/Beanie and the LangGraph checkpointer.
    Shut-down: close all connections cleanly."""
    global _mongo_client
    settings = get_settings()

    # ── MongoDB / Beanie ────────────────────────────────────────────────────
    _mongo_client = AsyncMongoClient(settings.MONGODB_URI)
    await init_beanie(
        database=_mongo_client[settings.MONGODB_DB],
        document_models=[Thread, Message, ResearchPlan],
    )
    logger.info("Beanie initialised (Thread, Message, ResearchPlan)")

    # ── LangGraph Postgres checkpointer (optional) ─────────────────────────
    if settings.LANGGRAPH_CHECKPOINTER == "postgres" and settings.POSTGRES_URL:
        await setup_postgres_checkpointer(settings.POSTGRES_URL)

    yield

    # ── Shutdown ────────────────────────────────────────────────────────────
    await close_postgres_checkpointer()
    if _mongo_client:
        await _mongo_client.close()
        _mongo_client = None
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

    # Socket.IO mounted at /socket.io — matches the client's default path.
    app.mount("/socket.io", get_sio_mount_app())

    return app


app = create_app()
