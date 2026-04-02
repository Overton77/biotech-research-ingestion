"""
LangGraph-based research missions: stages, unstructured ingestion, KG writes.

Public symbols are loaded lazily so ``import src.research.langchain_agent.models`` does not
pull the full runtime / agent stack.
"""

from __future__ import annotations

__all__ = [
    "ResearchMissionRuntime",
    "build_default_mission_runtime",
    "load_mission_from_file",
    "run_compiled_mission",
]


def __getattr__(name: str):
    if name == "load_mission_from_file":
        from src.research.langchain_agent.mission_loader import load_mission_from_file

        return load_mission_from_file
    if name == "ResearchMissionRuntime":
        from src.research.langchain_agent.runtime.deps import ResearchMissionRuntime

        return ResearchMissionRuntime
    if name == "build_default_mission_runtime":
        from src.research.langchain_agent.runtime.bootstrap import build_default_mission_runtime

        return build_default_mission_runtime
    if name == "run_compiled_mission":
        from src.research.langchain_agent.runtime.runner import run_compiled_mission

        return run_compiled_mission
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
