"""Shared LangGraph persistence: Postgres store + checkpointer.

Process-global persistence registry keyed by Postgres connection URI.

Use this when different parts of the app need distinct LangGraph persistence
backends, e.g. one for coordinator workflows and one for deep-agent research.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class PersistenceBundle:
    """One initialized LangGraph persistence bundle for a specific DB URI."""

    uri: str
    stack: AsyncExitStack
    store: AsyncPostgresStore
    checkpointer: AsyncPostgresSaver


# Registry of initialized persistence bundles, keyed by normalized URI.
_bundles: dict[str, PersistenceBundle] = {}

# Prevent concurrent double-init for the same URI / registry mutations.
_lock = asyncio.Lock()

# Environment variable names we support for defaults.
ENV_DEEP_AGENTS_URI: Final[str] = "DEEP_AGENTS_POSTGRES_URI"
ENV_COORDINATOR_URI: Final[str] = "COORDINATOR_POSTGRES_URI"
ENV_POSTGRES_URL: Final[str] = "POSTGRES_URL"
ENV_POSTGRES_URI: Final[str] = "POSTGRES_URI"


def _normalize_uri(uri: str) -> str:
    """Normalize a Postgres URI for stable cache keys."""
    normalized = uri.strip()

    # Remove matching quotes if someone wrapped the full URI in quotes in .env.
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"'}
    ):
        normalized = normalized[1:-1].strip()

    return normalized


def _require_uri(uri: str | None) -> str:
    """Resolve and validate a direct Postgres URI."""
    resolved = (uri or "").strip()
    if not resolved:
        raise ValueError(
            "A Postgres URI is required. Pass uri=... explicitly or use one of the "
            "named helpers such as get_deep_agents_persistence()."
        )

    resolved = _normalize_uri(resolved)

    if not resolved.startswith("postgresql://"):
        raise ValueError(
            "LangGraph AsyncPostgresStore/AsyncPostgresSaver expect a SQLAlchemy-style "
            "Postgres URI such as "
            "'postgresql://user:password@host:5432/dbname'. "
            f"Got: {resolved!r}"
        )

    return resolved


def _get_env_uri(env_var_name: str) -> str:
    """Load and validate a required Postgres URI from a specific env var."""
    raw = os.environ.get(env_var_name)
    if not raw:
        raise ValueError(
            f"Environment variable {env_var_name} is not set."
        )
    return _require_uri(raw)


async def _create_bundle(uri: str) -> PersistenceBundle:
    """Create and fully initialize one persistence bundle."""
    stack = AsyncExitStack()
    try:
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(uri)
        )
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(uri)
        )

        # Safe to call more than once, but should only run on first init per URI.
        await store.setup()
        await checkpointer.setup()

        bundle = PersistenceBundle(
            uri=uri,
            stack=stack,
            store=store,
            checkpointer=checkpointer,
        )

        logger.info("Initialized LangGraph persistence for uri=%s", uri)
        return bundle

    except Exception:
        logger.exception("Failed to initialize LangGraph persistence for uri=%s", uri)
        await stack.aclose()
        raise


async def get_persistence(
    uri: str,
) -> tuple[AsyncPostgresStore, AsyncPostgresSaver]:
    """Return the shared (store, checkpointer) pair for a specific Postgres URI.

    This is process-global and cached per normalized URI.
    """
    normalized_uri = _require_uri(uri)

    existing = _bundles.get(normalized_uri)
    if existing is not None:
        return existing.store, existing.checkpointer

    async with _lock:
        existing = _bundles.get(normalized_uri)
        if existing is not None:
            return existing.store, existing.checkpointer

        bundle = await _create_bundle(normalized_uri)
        _bundles[normalized_uri] = bundle
        return bundle.store, bundle.checkpointer


async def get_deep_agents_persistence() -> tuple[AsyncPostgresStore, AsyncPostgresSaver]:
    """Return the shared persistence pair for deep-agent research workflows."""
    uri = os.environ.get(ENV_DEEP_AGENTS_URI) or os.environ.get(ENV_POSTGRES_URL) or os.environ.get(ENV_POSTGRES_URI)
    return await get_persistence(_require_uri(uri))


async def get_coordinator_persistence() -> tuple[AsyncPostgresStore, AsyncPostgresSaver]:
    """Return the shared persistence pair for coordinator workflows."""
    uri = os.environ.get(ENV_COORDINATOR_URI) or os.environ.get(ENV_POSTGRES_URL) or os.environ.get(ENV_POSTGRES_URI)
    return await get_persistence(_require_uri(uri))


async def close_persistence(uri: str) -> None:
    """Close and remove the cached persistence bundle for a specific URI."""
    normalized_uri = _require_uri(uri)

    async with _lock:
        bundle = _bundles.pop(normalized_uri, None)

    if bundle is not None:
        await bundle.stack.aclose()
        logger.info("Closed LangGraph persistence for uri=%s", normalized_uri)


async def close_all_persistence() -> None:
    """Close all cached persistence bundles for all configured URIs."""
    async with _lock:
        bundles = list(_bundles.values())
        _bundles.clear()

    for bundle in bundles:
        try:
            await bundle.stack.aclose()
            logger.info("Closed LangGraph persistence for uri=%s", bundle.uri)
        except Exception:
            logger.exception(
                "Error while closing LangGraph persistence for uri=%s",
                bundle.uri,
            )