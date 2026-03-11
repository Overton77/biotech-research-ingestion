"""MissionRunner — LangGraph StateGraph that drives sequential mission execution.

Graph flow:
  load_mission → initialize_runtime_state
    → compute_ready_queue → select_next_task
      → execute_task → merge_task_result → persist_research_run
        → check_completion ──→ finalize_mission
                          └── loops via compute_ready_queue
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from operator import add
from typing import Annotated, Any

from beanie.odm.fields import PydanticObjectId
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from src.agents.persistence import get_deep_agents_persistence

from src.research.compiler.agent_compiler import RuntimeContext
from src.research.models.mission import (
    ArtifactRef,
    ResearchEvent,
    ResearchMission,
    ResearchRun,
    TaskDef,
    TaskResult,
)
from src.research.persistence.research_run_writer import ResearchRunWriter
from src.research.persistence.runs_s3 import get_research_runs_s3_store
from src.research.runtime.task_executor import execute_task as execute_task_def

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_runner: CompiledStateGraph | None = None 
_runner_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class MissionRunnerState(TypedDict):
    # Static mission data (set once during initialization; all serializable for checkpointing)
    mission_id: str
    task_defs_by_id: dict[str, Any]  # task_id -> TaskDef.model_dump()
    task_def_order: list[str]  # stable order of task_ids from mission.task_defs
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

async def load_mission(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Load the ResearchMission from MongoDB; return only serializable state for checkpointing."""
    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    if not mission:
        raise ValueError(f"Mission not found: {state['mission_id']}")

    mission.status = "running"
    mission.updated_at = datetime.utcnow()
    await mission.save()

    # Store serializable data only (no Beanie/Pydantic in state for Postgres checkpointer)
    task_defs_by_id = {td.task_id: td.model_dump() for td in mission.task_defs}
    task_def_order = [td.task_id for td in mission.task_defs]

    return {
        "task_defs_by_id": task_defs_by_id,
        "task_def_order": task_def_order,
        "dependency_map": mission.dependency_map,
        "reverse_dependency_map": mission.reverse_dependency_map,
    }


async def initialize_runtime_state(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Set up runtime tracking dictionaries from the loaded mission."""
    task_defs_by_id = state["task_defs_by_id"]
    task_statuses = {tid: "pending" for tid in task_defs_by_id}

    return {
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


async def compute_ready_queue(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Find all tasks whose dependencies are met and inputs are resolvable."""
    ready: list[str] = []
    for task_id, status in state["task_statuses"].items():
        if status != "pending":
            continue
        deps = state["dependency_map"].get(task_id, [])
        if not all(state["task_statuses"].get(d) == "completed" for d in deps):
            continue
        task_def = TaskDef.model_validate(state["task_defs_by_id"][task_id])
        if not _all_required_inputs_resolvable(task_def, state["task_outputs"]):
            continue
        ready.append(task_id)

    # Stable order: preserve declaration order from task_def_order
    ordered = [tid for tid in state["task_def_order"] if tid in ready]
    return {"ready_queue": ordered}


async def select_next_task(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Pick the first task from the ready queue."""
    queue = state["ready_queue"]
    if not queue:
        return {"current_task_id": None}
    return {"current_task_id": queue[0]}


async def run_task(state: MissionRunnerState, config: RunnableConfig, runtime: Runtime) -> MissionRunnerState:
    """Execute the current task using the Deep Agent pipeline."""
    task_id = state["current_task_id"]
    if not task_id:
        return {"task_results": [], "events": []}

    task_def = TaskDef.model_validate(state["task_defs_by_id"][task_id])
    # store comes from runtime (graph compiled with deep-agents Postgres store).
    # checkpointer is fetched from the same cached bundle so sub-agents share it.
    _, checkpointer = await get_deep_agents_persistence()

    # Extract progress_callback from config if provided by the Temporal activity
    progress_callback = config.get("configurable", {}).get("progress_callback")

    ctx = RuntimeContext(
        mission_id=state["mission_id"],
        task_id=task_id,
        store=runtime.store,
        checkpointer=checkpointer,
        progress_callback=progress_callback,
    )

    # Emit task_started via progress callback
    if progress_callback:
        try:
            await progress_callback("task_started", {
                "mission_id": state["mission_id"],
                "task_id": task_id,
                "task_name": task_def.name,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception:
            logger.debug("Progress callback failed for task_started", exc_info=True)

    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    result = await execute_task_def(
        task_def,
        state["task_outputs"],
        ctx,
        global_context=mission.global_context if hasattr(mission, "global_context") else None,
    )

    # Emit task_completed/task_failed via progress callback
    if progress_callback:
        event_type = "task_completed" if result.status == "completed" else "task_failed"
        try:
            await progress_callback(event_type, {
                "mission_id": state["mission_id"],
                "task_id": task_id,
                "task_name": task_def.name,
                "status": result.status,
                "error": result.error_message[:200] if result.error_message else None,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception:
            logger.debug("Progress callback failed for %s", event_type, exc_info=True)

    return {
        "task_results": [result],
        "events": result.events,
    }


async def merge_task_result(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
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


async def persist_research_run(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Write a ResearchRun document to MongoDB and S3."""
    if not state["task_results"]:
        return {}

    result: TaskResult = state["task_results"][-1]
    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    task_def_data = state["task_defs_by_id"].get(result.task_id)
    task_def = TaskDef.model_validate(task_def_data) if task_def_data else None

    s3_store = get_research_runs_s3_store()

    # Upload local filesystem artifacts to S3 before persisting
    try:
        result.artifacts = await s3_store.upload_local_artifacts_to_s3(
            mission=mission,
            artifacts=result.artifacts,
            task_def=task_def,
        )
    except Exception:
        logger.exception("S3 artifact upload failed for task %s; continuing with local refs", result.task_id)

    # Persist to MongoDB
    writer = ResearchRunWriter()
    run_doc = await writer.upsert_run(
        mission_id=state["mission_id"],
        task_result=result,
        resolved_inputs=state["task_outputs"].get(result.task_id, {}),
    )

    # Persist to S3 (best-effort)
    try:
        await s3_store.write_research_run(
            mission=mission,
            research_run=run_doc,
            task_def=task_def,
        )
        await s3_store.write_task_result(
            mission=mission,
            task_result=result,
            task_def=task_def,
        )
    except Exception:
        logger.exception("S3 write failed for task %s; MongoDB has the authoritative record", result.task_id)

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


async def finalize_mission(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Update mission status in MongoDB and S3, build task-runs index."""
    failed = state["failed_task_ids"]
    final_status = "failed" if failed else "completed"

    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    mission.status = final_status
    mission.updated_at = datetime.utcnow()
    await mission.save()

    final_outputs = {
        tid: state["task_outputs"].get(tid, {})
        for tid in state["completed_task_ids"]
    }

    # Persist mission and task-runs index to S3 (best-effort)
    try:
        s3_store = get_research_runs_s3_store()
        await s3_store.write_mission(mission)

        runs = await ResearchRun.find(
            ResearchRun.mission_id == mission.id,
        ).to_list()
        await s3_store.build_task_runs_index(mission=mission, research_runs=runs)
    except Exception:
        logger.exception("S3 finalization writes failed for mission %s", state["mission_id"])

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

async def build_mission_runner() -> CompiledStateGraph:
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

    store, checkpointer = await get_deep_agents_persistence()
    return builder.compile(checkpointer=checkpointer, store=store)




async def get_mission_runner() -> CompiledStateGraph:
    global _runner
    if _runner is not None:
        return _runner

    async with _runner_lock:
        if _runner is None:
            _runner = await build_mission_runner()
        return _runner


async def run_mission(
    mission_id: str,
    progress_callback: Any | None = None,
) -> dict:
    runner: CompiledStateGraph = await get_mission_runner()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": mission_id,
            "progress_callback": progress_callback,
        },
    }
    return await runner.ainvoke({"mission_id": mission_id}, config=config)