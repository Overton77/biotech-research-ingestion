"""
Runtime API for executing compiled research missions (FastAPI, workers, CLI).

- ``ResearchMissionRuntime``: inject store, checkpointer, memory manager, paths.
- ``run_compiled_mission``: the single async entrypoint for a ``ResearchMission``.
- ``build_default_mission_runtime``: local/test wiring (CLI ``--local``).
"""

from src.research.langchain_agent.runtime.bootstrap import build_default_mission_runtime
from src.research.langchain_agent.runtime.deps import ResearchMissionRuntime
from src.research.langchain_agent.runtime.runner import run_compiled_mission

__all__ = [
    "ResearchMissionRuntime",
    "build_default_mission_runtime",
    "run_compiled_mission",
]
