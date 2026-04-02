"""Injectable runtime dependencies for executing a compiled ``ResearchMission``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchMissionRuntime:
    """Everything the workflow needs besides the mission object itself.

    Construct this in a FastAPI dependency (or Temporal activity) from app state:
    LangGraph ``store``, ``checkpointer``, LangMem ``memory_manager``, and optional
    filesystem roots for artifacts and LangGraph state snapshots.
    """

    store: Any
    checkpointer: Any
    memory_manager: Any
    root_filesystem: Path | None = None
    snapshot_output_dir: Path | None = None
