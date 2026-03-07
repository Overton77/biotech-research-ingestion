"""Coordinator agent — ChatOpenAI with web search tools and AsyncPostgresSaver checkpointer."""

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.config import get_settings
from src.agents.tools.web_search import get_web_search_tool
from src.agents.tools.create_plan import create_research_plan

logger = logging.getLogger(__name__)

COORDINATOR_SYSTEM_PROMPT = """You are the Coordinator for a Deep Biotech Research system. Your role is to:
- Accept research objectives from the user and clarify scope when needed.
- Use web search to gather initial context on the topic.
- When the user is ready, use the create_research_plan tool to generate a structured research plan (stages and tasks). The plan will be sent for human approval before execution.
- Be concise and focused."""

# Set by setup_postgres_checkpointer() during lifespan startup.
_checkpointer: Any = None
_postgres_conn: Any = None  # Held open for the lifetime of the process.

# Lazy graph instance — reset to None whenever the checkpointer changes.
_coordinator_graph: Any = None


async def setup_postgres_checkpointer(postgres_url: str) -> None:
    """Create an AsyncPostgresSaver from a persistent psycopg connection.

    Called once from the FastAPI lifespan after the event loop is running.
    Falls back to MemorySaver if the connection fails.
    """
    global _checkpointer, _postgres_conn, _coordinator_graph
    try:
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        conn = await psycopg.AsyncConnection.connect(
            postgres_url,
            autocommit=True,
        )
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        _checkpointer = saver
        _postgres_conn = conn
        _coordinator_graph = None  # Force recreation with the new checkpointer.
        logger.info("AsyncPostgresSaver initialised (postgres checkpointer)")
    except Exception as exc:
        logger.warning(
            "AsyncPostgresSaver setup failed — falling back to MemorySaver: %s", exc
        )
        _checkpointer = MemorySaver()


async def close_postgres_checkpointer() -> None:
    """Close the postgres connection. Called from lifespan shutdown."""
    global _postgres_conn
    if _postgres_conn is not None:
        try:
            await _postgres_conn.close()
        except Exception:
            pass
        _postgres_conn = None
        logger.info("Postgres checkpointer connection closed")


def _get_checkpointer() -> Any:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def create_coordinator_graph(tools: list[Any] | None = None) -> Any:
    """Build and return the compiled coordinator graph."""
    settings = get_settings()
    model = ChatOpenAI(
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY or "not-set",
        temperature=0,
    )
    if tools is None:
        tools = [get_web_search_tool(), create_research_plan]
    return create_react_agent(
        model,
        tools,
        prompt=COORDINATOR_SYSTEM_PROMPT,
        checkpointer=_get_checkpointer(),
    )


def get_coordinator_graph(tools: list[Any] | None = None) -> Any:
    """Return the coordinator graph, creating it lazily on first call."""
    global _coordinator_graph
    if _coordinator_graph is None:
        _coordinator_graph = create_coordinator_graph(tools=tools)
    return _coordinator_graph
