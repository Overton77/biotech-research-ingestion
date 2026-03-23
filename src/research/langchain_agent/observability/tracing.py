"""
LangSmith tracing utilities for the research mission pipeline.

Provides @traceable wrappers that create a hierarchical trace structure:

    Mission (run_mission)
      ├── Stage (run_single_mission_slice)
      │   ├── memory_recall
      │   ├── research_agent.ainvoke  ← auto-traced by LangChain
      │   ├── memory_report_agent.ainvoke  ← auto-traced
      │   └── memory_ingestion
      ├── Iterative Stage (run_iterative_stage)
      │   ├── Iteration 1 (run_single_mission_slice)
      │   ├── next_steps_extraction
      │   ├── Iteration 2 ...
      │   └── ...
      └── Stage ...

All wrappers use langsmith.traceable which:
- Nests automatically via Python contextvars (3.12+)
- Works with async functions natively
- Captures inputs/outputs for inspection in the LangSmith UI
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import langsmith as ls
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# ---------------------------------------------------------------------------
# Project names
# ---------------------------------------------------------------------------

MISSION_PROJECT = os.getenv("LANGSMITH_PROJECT", "biotech_research")
EVAL_PROJECT = "biotech-research-evals"
DEV_PROJECT = "biotech-research-dev"


# ---------------------------------------------------------------------------
# Tracing context managers for project routing
# ---------------------------------------------------------------------------


@contextmanager
def mission_tracing_context(
    mission_id: str,
    mission_name: str,
    *,
    mission_type: str = "stage_based",
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
):
    """Scope all nested traces to the missions project with mission-level metadata."""
    tags = [f"mission-type:{mission_type}"]
    if extra_tags:
        tags.extend(extra_tags)

    metadata: Dict[str, Any] = {
        "mission_id": mission_id,
        "mission_name": mission_name,
        "mission_type": mission_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with ls.tracing_context(
        project_name=MISSION_PROJECT,
        tags=tags,
        metadata=metadata,
    ):
        yield


@contextmanager
def eval_tracing_context(
    dataset_name: str,
    experiment_name: str,
    *,
    extra_metadata: dict[str, Any] | None = None,
):
    """Scope all nested traces to the evaluation project."""
    metadata: Dict[str, Any] = {
        "dataset": dataset_name,
        "experiment": experiment_name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with ls.tracing_context(
        project_name=EVAL_PROJECT,
        tags=["eval"],
        metadata=metadata,
    ):
        yield


# ---------------------------------------------------------------------------
# Trace ID capture
# ---------------------------------------------------------------------------


def get_current_trace_id() -> Optional[str]:
    """Return the current LangSmith run/trace ID, or None if not in a traced context."""
    try:
        rt = get_current_run_tree()
        if rt is not None:
            return str(rt.id)
    except Exception:
        pass
    return None


def _get_trace_url() -> Optional[str]:
    """Return the LangSmith URL for the current trace, if available."""
    try:
        rt = get_current_run_tree()
        if rt is not None:
            return rt.get_url()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Traceable wrappers — these are decorators applied to the orchestration layer
# ---------------------------------------------------------------------------


traced_mission = traceable(
    name="research_mission",
    run_type="chain",
    tags=["mission"],
)

traced_stage = traceable(
    name="mission_stage",
    run_type="chain",
    tags=["stage"],
)

traced_iterative_stage = traceable(
    name="iterative_stage",
    run_type="chain",
    tags=["iterative-stage"],
)

traced_next_steps_extraction = traceable(
    name="next_steps_extraction",
    run_type="chain",
    tags=["next-steps"],
)

traced_memory_recall = traceable(
    name="memory_recall",
    run_type="retriever",
    tags=["memory"],
)

traced_memory_ingestion = traceable(
    name="memory_ingestion",
    run_type="chain",
    tags=["memory"],
)


# ---------------------------------------------------------------------------
# Metadata injection helpers
# ---------------------------------------------------------------------------


def inject_stage_metadata(
    *,
    mission_id: str,
    task_slug: str,
    stage_type: str,
    iteration: int | None = None,
    targets: list[str] | None = None,
) -> None:
    """Inject stage-level metadata into the current trace span.

    Call this at the start of a traced stage function to enrich the span
    with searchable metadata. Safe to call outside a traced context (no-op).
    """
    try:
        rt = get_current_run_tree()
        if rt is None:
            return
        rt.metadata = rt.metadata or {}
        rt.metadata.update({
            "mission_id": mission_id,
            "task_slug": task_slug,
            "stage_type": stage_type,
        })
        if iteration is not None:
            rt.metadata["iteration"] = iteration
        if targets:
            rt.metadata["targets"] = targets

        rt.tags = rt.tags or []
        rt.tags.append(f"stage:{task_slug}")
        if iteration is not None:
            rt.tags.append(f"iteration:{iteration}")
    except Exception:
        pass
