"""
Shared pytest fixtures for langchain_agent tests.

Fixtures:
  mini_mission_from_file — loads elysium_mini.json from test_runs/missions/
  memory_saver_pair      — (InMemoryStore, MemorySaver) — no Postgres needed
  stub_memory_manager    — lightweight no-op manager for unit tests
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.research.langchain_agent.models.mission import ResearchMission
from src.research.langchain_agent.mission_loader import load_mission_from_file

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "missions"


@pytest.fixture
def mini_mission_from_file() -> ResearchMission:
    """Load the elysium_mini.json test mission from test_runs/missions/."""
    path = MISSIONS_DIR / "elysium_mini.json"
    return load_mission_from_file(path)


@pytest.fixture
def memory_saver_pair():
    """Return (InMemoryStore, MemorySaver) — no external services required."""
    store = InMemoryStore()
    checkpointer = MemorySaver()
    return store, checkpointer


@pytest.fixture
def stub_memory_manager():
    """A no-op memory manager for unit tests (no LangMem calls)."""
    mgr = MagicMock()
    mgr.search_memories = AsyncMock(return_value=[])
    mgr.add_memories = AsyncMock(return_value=None)
    return mgr
