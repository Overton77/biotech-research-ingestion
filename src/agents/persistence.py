"""Shared LangGraph persistence: Postgres store + checkpointer."""

from __future__ import annotations

import asyncio
import logging
import os 
from langgraph.checkpoint.memory import MemorySaver
from contextlib import AsyncExitStack
from typing import Optional, Tuple

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

logger = logging.getLogger(__name__)

load_dotenv()

_stack: Optional[AsyncExitStack] = None
_store: Optional[AsyncPostgresStore | InMemoryStore] = None
_checkpointer: Optional[AsyncPostgresSaver | InMemorySaver] = None
_lock = asyncio.Lock()


def _resolve_db_uri(explicit_uri: str | None = None) -> str | None:
    return (
        explicit_uri
        or os.environ.get("DEEP_AGENTS_POSTGRES_URI")
        or os.environ.get("POSTGRES_URL")
        or os.environ.get("POSTGRES_URI")
    )


async def get_persistence(
    uri: str | None = None,
) -> Tuple[AsyncPostgresStore | InMemoryStore, AsyncPostgresSaver | InMemorySaver]:
    """
    Initialize once per process and return (store, checkpointer).

    - store: long-term durable memory across threads / sessions
    - checkpointer: thread-scoped persistence for graph state, interrupts, resume
    """
    global _stack, _store, _checkpointer

    if _store is not None and _checkpointer is not None:
        return _store, _checkpointer

    async with _lock:
        if _store is not None and _checkpointer is not None:
            return _store, _checkpointer

        db_uri = _resolve_db_uri(uri)
        if not db_uri:
            logger.warning(
                "No Postgres URI configured; falling back to in-memory persistence"
            )
            _store = InMemoryStore()
            _checkpointer = InMemorySaver()
            return _store, _checkpointer

        try:
            stack = AsyncExitStack()

            store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(db_uri)
            )
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(db_uri)
            )

            # Must be called the first time these are used.
            await store.setup()
            await checkpointer.setup()

            _stack = stack
            _store = store
            _checkpointer = checkpointer

            logger.info("Initialized AsyncPostgresStore + AsyncPostgresSaver")
            return _store, _checkpointer

        except Exception:
            logger.exception(
                "Failed to initialize Postgres persistence; falling back to in-memory persistence"
            )
            if stack is not None:
                await stack.aclose()

            _stack = None
            _store = InMemoryStore()
            _checkpointer = InMemorySaver()
            return _store, _checkpointer


async def close_persistence() -> None:
    global _stack, _store, _checkpointer

    if _stack is not None:
        await _stack.aclose()

    _stack = None
    _store = None
    _checkpointer = None

    logger.info("Closed LangGraph persistence resources")  

# In Memory store and checkpointer 

def _get_checkpointer() -> MemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def _get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store