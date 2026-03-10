"""MissionRunner — LangGraph StateGraph that drives sequential mission execution.

Graph flow:
  load_mission → initialize_runtime_state
    → compute_ready_queue → select_next_task
      → execute_task → merge_task_result → persist_research_run
        → check_completion ──→ finalize_mission
                          └── loops via compute_ready_queue
"""

from __future__ import annotations

import logging
from datetime import datetime
from operator import add
from typing import Annotated, Any

from beanie.odm.fields import PydanticObjectId
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from src.research.compiler.agent_compiler import RuntimeContext
from src.research.models.mission import (
    ArtifactRef,
    ResearchEvent,
    ResearchMission,
    TaskDef,
    TaskResult,
)
from src.research.persistence.research_run_writer import ResearchRunWriter
from src.research.runtime.task_executor import execute_task as execute_task_def

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_store: InMemoryStore | None = None
_runner: CompiledStateGraph | None = None


def _get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class MissionRunnerState(TypedDict):
    # Static mission data (set once during initialization)
    mission_id: str
    mission: Any  # ResearchMission (Beanie doc — not TypedDict-compatible)
    task_defs_by_id: dict[str, Any]
    dependency_map: dict[str, list[str]]
    reverse_dependency_map: dict[str, list[str]]

    # Mutable execution state
    task_statuses: dict[str, str]
    task_outputs: dict[str, dict[str, Any]]

    # Append-only via reducers
    task_results: Annotated[list, add]
    artifacts: Annotated[list, add]
    events: Annotated[list, add]
    completed_task_ids: Annotated[list, add]
    failed_task_ids: Annotated[list, add]

    # Overwrite semantics (last write wins)
    ready_queue: list[str]
    current_task_id: str | None
    mission_status: str
    final_outputs: dict[str, Any]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def load_mission(state: MissionRunnerState) -> dict:
    """Load the ResearchMission document from MongoDB."""
    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    if not mission:
        raise ValueError(f"Mission not found: {state['mission_id']}")

    mission.status = "running"
    mission.updated_at = datetime.utcnow()
    await mission.save()

    return {"mission": mission}


async def initialize_runtime_state(state: MissionRunnerState) -> dict:
    """Set up runtime tracking dictionaries from the loaded mission."""
    mission = state["mission"]
    task_defs_by_id = {td.task_id: td for td in mission.task_defs}
    task_statuses = {td.task_id: "pending" for td in mission.task_defs}

    return {
        "task_defs_by_id": task_defs_by_id,
        "dependency_map": mission.dependency_map,
        "reverse_dependency_map": mission.reverse_dependency_map,
        "task_statuses": task_statuses,
        "task_outputs": {},
        "task_results": [],
        "artifacts": [],
        "events": [],
        "completed_task_ids": [],
        "failed_task_ids": [],
        "ready_queue": [],
        "current_task_id": None,
        "mission_status": "running",
        "final_outputs": {},
    }


def _all_required_inputs_resolvable(
    task_def: Any,
    task_outputs: dict[str, dict[str, Any]],
) -> bool:
    """Check if all required input bindings can be resolved."""
    for _local_name, binding in task_def.input_bindings.items():
        if not binding.required:
            continue
        source_data = task_outputs.get(binding.source_task_id, {})
        if binding.source_key not in source_data:
            return False
    return True


async def compute_ready_queue(state: MissionRunnerState) -> dict:
    """Find all tasks whose dependencies are met and inputs are resolvable."""
    ready: list[str] = []
    for task_id, status in state["task_statuses"].items():
        if status != "pending":
            continue
        deps = state["dependency_map"].get(task_id, [])
        if not all(state["task_statuses"].get(d) == "completed" for d in deps):
            continue
        task_def = state["task_defs_by_id"][task_id]
        if not _all_required_inputs_resolvable(task_def, state["task_outputs"]):
            continue
        ready.append(task_id)

    # Stable order: preserve declaration order from mission.task_defs
    mission = state["mission"]
    ordered = [td.task_id for td in mission.task_defs if td.task_id in ready]
    return {"ready_queue": ordered}


async def select_next_task(state: MissionRunnerState) -> dict:
    """Pick the first task from the ready queue."""
    queue = state["ready_queue"]
    if not queue:
        return {"current_task_id": None}
    return {"current_task_id": queue[0]}


async def run_task(state: MissionRunnerState) -> dict:
    """Execute the current task using the Deep Agent pipeline."""
    task_id = state["current_task_id"]
    if not task_id:
        return {"task_results": [], "events": []}

    task_def = state["task_defs_by_id"][task_id]
    store = _get_store()
    ctx = RuntimeContext(
        mission_id=state["mission_id"],
        task_id=task_id,
        store=store,
    )

    mission = state["mission"]
    result = await execute_task_def(
        task_def,
        state["task_outputs"],
        ctx,
        global_context=mission.global_context if hasattr(mission, "global_context") else None,
    )

    return {
        "task_results": [result],
        "events": result.events,
    }


async def merge_task_result(state: MissionRunnerState) -> dict:
    """Merge the latest task result into execution state."""
    if not state["task_results"]:
        return {}

    result = state["task_results"][-1]
    task_id = result.task_id

    new_statuses = dict(state["task_statuses"])
    new_outputs = dict(state["task_outputs"])

    if result.status == "completed":
        new_statuses[task_id] = "completed"
        new_outputs[task_id] = result.outputs
        return {
            "task_statuses": new_statuses,
            "task_outputs": new_outputs,
            "artifacts": result.artifacts,
            "completed_task_ids": [task_id],
        }
    else:
        new_statuses[task_id] = "failed"
        return {
            "task_statuses": new_statuses,
            "task_outputs": new_outputs,
            "failed_task_ids": [task_id],
        }


async def persist_research_run(state: MissionRunnerState) -> dict:
    """Write a ResearchRun document to MongoDB."""
    if not state["task_results"]:
        return {}

    result = state["task_results"][-1]
    writer = ResearchRunWriter()
    await writer.upsert_run(
        mission_id=state["mission_id"],
        task_result=result,
        resolved_inputs=state["task_outputs"].get(result.task_id, {}),
    )
    return {}


def check_completion(state: MissionRunnerState) -> str:
    """Routing function: decide whether to loop or finalize."""
    total = len(state["task_defs_by_id"])
    done = len(state["completed_task_ids"]) + len(state["failed_task_ids"])

    if done >= total:
        return "finalize_mission"

    # Check for stall: no progress possible
    if not state["ready_queue"] and state["current_task_id"] is None:
        return "finalize_mission"

    return "compute_ready_queue"


async def finalize_mission(state: MissionRunnerState) -> dict:
    """Update mission status in MongoDB and collect final outputs."""
    failed = state["failed_task_ids"]
    final_status = "failed" if failed else "completed"

    mission = state["mission"]
    mission.status = final_status
    mission.updated_at = datetime.utcnow()
    await mission.save()

    final_outputs = {
        tid: state["task_outputs"].get(tid, {})
        for tid in state["completed_task_ids"]
    }

    logger.info(
        "Mission %s finalized: status=%s, completed=%d, failed=%d",
        state["mission_id"],
        final_status,
        len(state["completed_task_ids"]),
        len(state["failed_task_ids"]),
    )

    return {"mission_status": final_status, "final_outputs": final_outputs}


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_mission_runner() -> CompiledStateGraph:
    """Build and compile the MissionRunner LangGraph StateGraph."""
    builder = StateGraph(MissionRunnerState)

    builder.add_node("load_mission", load_mission)
    builder.add_node("initialize_runtime_state", initialize_runtime_state)
    builder.add_node("compute_ready_queue", compute_ready_queue)
    builder.add_node("select_next_task", select_next_task)
    builder.add_node("run_task", run_task)
    builder.add_node("merge_task_result", merge_task_result)
    builder.add_node("persist_research_run", persist_research_run)
    builder.add_node("finalize_mission", finalize_mission)

    builder.set_entry_point("load_mission")
    builder.add_edge("load_mission", "initialize_runtime_state")
    builder.add_edge("initialize_runtime_state", "compute_ready_queue")
    builder.add_edge("compute_ready_queue", "select_next_task")
    builder.add_edge("select_next_task", "run_task")
    builder.add_edge("run_task", "merge_task_result")
    builder.add_edge("merge_task_result", "persist_research_run")

    builder.add_conditional_edges(
        "persist_research_run",
        check_completion,
        {
            "compute_ready_queue": "compute_ready_queue",
            "finalize_mission": "finalize_mission",
        },
    )
    builder.add_edge("finalize_mission", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def get_mission_runner() -> CompiledStateGraph:
    """Return the singleton MissionRunner graph."""
    global _runner
    if _runner is None:
        _runner = build_mission_runner()
    return _runner


async def run_mission(mission_id: str) -> dict:
    """Top-level entry point: run an entire mission to completion."""
    runner = get_mission_runner()
    config = {"configurable": {"thread_id": mission_id}}
    result = await runner.ainvoke(
        {"mission_id": mission_id},
        config=config,
    )
    return result
