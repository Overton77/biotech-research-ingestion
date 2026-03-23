import os
import asyncio
from contextlib import AsyncExitStack 
from typing import Optional, Tuple

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore 
from dotenv import load_dotenv

_stack: Optional[AsyncExitStack] = None
_store: Optional[AsyncPostgresStore] = None
_checkpointer: Optional[AsyncPostgresSaver] = None
_lock = asyncio.Lock() 

load_dotenv()


async def get_persistence() -> Tuple[AsyncPostgresStore, AsyncPostgresSaver]:
    """
    Initialize (once per process) and return (store, checkpointer).
    Keeps the underlying async contexts open for reuse across graph rebuilds.
    """
    global _stack, _store, _checkpointer

    if _store is not None and _checkpointer is not None:
        return _store, _checkpointer

    async with _lock:
        if _store is not None and _checkpointer is not None:
            return _store, _checkpointer

        db_uri = os.environ["POSTGRES_URI"] 

        # enter_async_context on the AsyncExitStack of the AsyncPostgresStore and AsyncPostgresSaver 

        _stack = AsyncExitStack()
        _store = await _stack.enter_async_context(AsyncPostgresStore.from_conn_string(db_uri))
        _checkpointer = await _stack.enter_async_context(AsyncPostgresSaver.from_conn_string(db_uri))

        # Safe to call repeatedly; creates tables/migrations if needed 
        # Set them up 
        await _store.setup()
        await _checkpointer.setup()

        return _store, _checkpointer


async def close_persistence() -> None:
    """Optional: call on shutdown if you have a lifecycle hook."""
    global _stack, _store, _checkpointer
    if _stack is not None:
        await _stack.aclose()
    _stack = None
    _store = None
    _checkpointer = None



