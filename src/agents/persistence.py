"""Shared LangGraph persistence: Postgres store + checkpointer."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Optional, Tuple

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
import selectors

logger = logging.getLogger(__name__)

load_dotenv()

_stack: Optional[AsyncExitStack] = None
_store: Optional[AsyncPostgresStore | InMemoryStore] = None
_checkpointer: Optional[AsyncPostgresSaver | InMemorySaver] = None
_lock = asyncio.Lock()


async def get_persistence() -> Tuple[AsyncPostgresStore | InMemoryStore, AsyncPostgresSaver | InMemorySaver]:
    """
    Initialize once per process and return (store, checkpointer).

    - store: long-term memory across threads / sessions
    - checkpointer: thread-scoped persistence for graph state, interrupts, resume
    """
    global _stack, _store, _checkpointer

    if _store is not None and _checkpointer is not None:
        return _store, _checkpointer

    async with _lock:
        if _store is not None and _checkpointer is not None:
            return _store, _checkpointer

        db_uri = os.environ.get("POSTGRES_URL") or os.environ.get("POSTGRES_URI")
        if not db_uri:
            logger.warning("POSTGRES_URL/POSTGRES_URI not set; falling back to in-memory persistence")
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

            # Safe to call repeatedly; creates tables/migrations if needed.
            asyncio.run(store.setup(), loop_factory=asyncio.SelectorEventLoop(selectors.SelectSelector()))
            asyncio.run(checkpointer.setup(), loop_factory=asyncio.SelectorEventLoop(selectors.SelectSelector()))

            _stack = stack
            _store = store
            _checkpointer = checkpointer

            logger.info("Initialized AsyncPostgresStore + AsyncPostgresSaver")
            return _store, _checkpointer

        except Exception:
            logger.exception(
                "Failed to initialize Postgres persistence; falling back to in-memory persistence"
            )
            if _stack is not None:
                await _stack.aclose()
                _stack = None

            _store = InMemoryStore()
            _checkpointer = InMemorySaver()
            return _store, _checkpointer


async def close_persistence() -> None:
    """Close shared persistence resources on application shutdown."""
    global _stack, _store, _checkpointer

    if _stack is not None:
        await _stack.aclose()

    _stack = None
    _store = None
    _checkpointer = None

    logger.info("Closed LangGraph persistence resources")
