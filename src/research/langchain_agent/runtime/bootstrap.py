"""Default wiring for local runs and tests: Postgres checkpointer, Beanie, LangMem."""

from __future__ import annotations

from pathlib import Path

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.memory.langmem_manager import build_langmem_manager
from src.research.langchain_agent.runtime.deps import ResearchMissionRuntime
from src.research.langchain_agent.storage.langgraph_persistence import get_persistence
from src.research.langchain_agent.storage.models import init_research_agent_beanie


async def build_default_mission_runtime(
    *,
    root_filesystem: Path | None = None,
    snapshot_output_dir: Path | None = None,
    init_beanie: bool = True,
) -> ResearchMissionRuntime:
    """Create ``ResearchMissionRuntime`` using shared persistence (same as the CLI local path).

    For production FastAPI, prefer constructing ``ResearchMissionRuntime`` from
    application-scoped singletons (store/checkpointer/memory) and set
    ``init_beanie=False`` if Beanie is initialized at app startup.
    """
    store, checkpointer = await get_persistence()
    if init_beanie:
        await init_research_agent_beanie()
    memory_manager = await build_langmem_manager(store=store)
    root = root_filesystem if root_filesystem is not None else ROOT_FILESYSTEM
    return ResearchMissionRuntime(
        store=store,
        checkpointer=checkpointer,
        memory_manager=memory_manager,
        root_filesystem=root,
        snapshot_output_dir=snapshot_output_dir,
    )
