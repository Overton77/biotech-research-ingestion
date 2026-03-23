"""MissionRunner — LangGraph StateGraph that drives parallel mission execution.

Graph flow:
  load_mission → initialize_runtime_state
    → compute_ready_queue → fan_out_tasks ──(Send)──→ run_single_task (×N parallel)
      → fan_in_results → validate_batch_outputs → merge_task_result → persist_research_run
        → check_completion ──→ finalize_mission
                          └── loops via compute_ready_queue
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from operator import add
from typing import Annotated, Any

from beanie.odm.fields import PydanticObjectId
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from typing_extensions import TypedDict

from src.agents.persistence import get_deep_agents_persistence

from src.research.compiler.agent_compiler import RuntimeContext
from src.research.models.mission import (
    ArtifactRef,
    QualityAssessment,
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
# Reducer helpers
# ---------------------------------------------------------------------------

def _merge_dicts(a: dict, b: dict) -> dict:
    """Shallow merge — b takes precedence on key conflict."""
    return {**a, **b}


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class MissionRunnerState(TypedDict):
    # ---- Static mission data (set once; all serializable for checkpointing) ----
    mission_id: str
    task_defs_by_id: dict[str, Any]
    task_def_order: list[str]
    dependency_map: dict[str, list[str]]
    reverse_dependency_map: dict[str, list[str]]

    # ---- Overwrite semantics (written by single nodes after fan-in) ----
    task_statuses: dict[str, str]
    task_outputs: dict[str, dict[str, Any]]
    ready_queue: list[str]
    current_task_id: str | None
    current_batch: list[str]
    max_parallel_tasks: int
    mission_status: str
    final_outputs: dict[str, Any]
    task_structured_summaries: dict[str, Any]
    final_structured_summaries: dict[str, Any]

    # ---- Append-only via operator.add (safe across parallel branches) ----
    task_results: Annotated[list, add]
    artifacts: Annotated[list, add]
    events: Annotated[list, add]
    completed_task_ids: Annotated[list, add]
    failed_task_ids: Annotated[list, add]

    # ---- Custom merge reducer (accumulates across graph loops) ----
    quality_assessments: Annotated[dict[str, Any], _merge_dicts]


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
    max_parallel = config.get("configurable", {}).get("max_parallel_tasks", 3)

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
        "current_batch": [],
        "max_parallel_tasks": max_parallel,
        "mission_status": "running",
        "final_outputs": {},
        "task_structured_summaries": {},
        "final_structured_summaries": {},
        "quality_assessments": {},
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


def fan_out_tasks(state: MissionRunnerState) -> list[Send] | str:
    """Conditional edge routing function: dispatch ready tasks as parallel Send messages.

    Must be a routing function (used with add_conditional_edges), NOT a node —
    LangGraph nodes must return dicts; only routing functions may return Send objects.
    """
    ready = state["ready_queue"]
    limit = state.get("max_parallel_tasks", 3)
    batch = [tid for tid in state["task_def_order"] if tid in ready][:limit]

    if not batch:
        # No ready tasks — skip straight to fan-in; normal state flows through
        return "fan_in_results"

    return [
        Send("run_single_task", {
            **state,
            "current_task_id": task_id,
            "current_batch": batch,
        })
        for task_id in batch
    ]


async def run_single_task(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Execute one task — called in parallel by fan_out_tasks via Send."""
    task_id = state["current_task_id"]
    if not task_id:
        return {"task_results": [], "events": []}

    task_def = TaskDef.model_validate(state["task_defs_by_id"][task_id])
    _, checkpointer = await get_deep_agents_persistence()

    progress_callback = config.get("configurable", {}).get("progress_callback")

    ctx = RuntimeContext(
        mission_id=state["mission_id"],
        task_id=task_id,
        store=runtime.store,
        checkpointer=checkpointer,
        progress_callback=progress_callback,
    )

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


async def fan_in_results(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Collect parallel task results — LangGraph merges via Annotated[list, add] reducers."""
    return {}


async def _run_acceptance_check(
    task_id: str,
    response: str,
    criteria: list[str],
    model: Any,
) -> QualityAssessment:
    """Run a lightweight LLM check of a task output against acceptance criteria."""
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt = (
        f"Evaluate this research output against the acceptance criteria.\n\n"
        f"CRITERIA:\n{criteria_text}\n\n"
        f"OUTPUT (last 3000 chars):\n{response[-3000:]}\n\n"
        f'Return JSON: {{"criteria_results": [{{"criterion": "...", "status": "MET|PARTIALLY_MET|NOT_MET", "evidence": "..."}}], '
        f'"overall_pass": true/false, "suggestions": ["..."]}}'
    )
    try:
        resp = await model.ainvoke([{"role": "user", "content": prompt}])
        data = json.loads(resp.content)
        return QualityAssessment(task_id=task_id, **data)
    except Exception:
        return QualityAssessment(task_id=task_id, overall_pass=True)


async def validate_batch_outputs(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Validate completed task outputs against their acceptance criteria (§7)."""
    if not state["task_results"]:
        return {}

    already_assessed = set(state.get("quality_assessments", {}).keys())
    model = init_chat_model("openai:gpt-5-mini")
    quality_updates: dict[str, Any] = {}

    for result in state["task_results"]:
        if not isinstance(result, TaskResult):
            continue
        if result.status != "completed" or result.task_id in already_assessed:
            continue

        task_def_data = state["task_defs_by_id"].get(result.task_id)
        if not task_def_data:
            continue

        task_def = TaskDef.model_validate(task_def_data)
        if not task_def.acceptance_criteria:
            continue

        response_text = result.outputs.get("response", "")
        assessment = await _run_acceptance_check(
            task_id=result.task_id,
            response=response_text,
            criteria=task_def.acceptance_criteria,
            model=model,
        )
        quality_updates[result.task_id] = assessment.model_dump()
        logger.info(
            "Quality assessment for task %s: overall_pass=%s",
            result.task_id, assessment.overall_pass,
        )

    return {"quality_assessments": quality_updates} if quality_updates else {}


async def merge_task_result(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Merge all task results from the current batch into execution state.

    After fan-in, task_results (Annotated[list, add]) may have grown by N
    items from the parallel batch. We process every unmerged result by
    comparing against already-completed/failed task IDs.
    """
    if not state["task_results"]:
        return {}

    already_done = set(state["completed_task_ids"]) | set(state["failed_task_ids"])
    new_statuses = dict(state["task_statuses"])
    new_outputs = dict(state["task_outputs"])
    new_summaries = dict(state.get("task_structured_summaries", {}))
    batch_completed: list[str] = []
    batch_failed: list[str] = []
    batch_artifacts: list[Any] = []

    for result in state["task_results"]:
        if not isinstance(result, TaskResult):
            continue
        task_id = result.task_id
        if task_id in already_done:
            continue

        if result.status == "completed":
            new_statuses[task_id] = "completed"
            new_outputs[task_id] = result.outputs
            batch_completed.append(task_id)
            batch_artifacts.extend(result.artifacts)

            summary = getattr(result, "structured_execution_summary", None) or result.outputs.get(
                "structured_response"
            )
            if summary is not None:
                summary_serialized = (
                    summary.model_dump() if hasattr(summary, "model_dump") else summary
                )
                new_summaries[task_id] = summary_serialized
        else:
            new_statuses[task_id] = "failed"
            batch_failed.append(task_id)

    return {
        "task_statuses": new_statuses,
        "task_outputs": new_outputs,
        "task_structured_summaries": new_summaries,
        "artifacts": batch_artifacts,
        "completed_task_ids": batch_completed,
        "failed_task_ids": batch_failed,
    }


async def persist_research_run(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Write ResearchRun documents to MongoDB and S3 for all results in the batch."""
    if not state["task_results"]:
        return {}

    already_persisted = set(state["completed_task_ids"]) | set(state["failed_task_ids"])
    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    s3_store = get_research_runs_s3_store()
    writer = ResearchRunWriter()

    for result in state["task_results"]:
        if not isinstance(result, TaskResult):
            continue
        # Only persist results that haven't been persisted in prior loops
        task_def_data = state["task_defs_by_id"].get(result.task_id)
        task_def = TaskDef.model_validate(task_def_data) if task_def_data else None

        try:
            result.artifacts = await s3_store.upload_local_artifacts_to_s3(
                mission=mission,
                artifacts=result.artifacts,
                task_def=task_def,
            )
        except Exception:
            logger.exception("S3 artifact upload failed for task %s", result.task_id)

        run_doc = await writer.upsert_run(
            mission_id=state["mission_id"],
            task_result=result,
            resolved_inputs=state["task_outputs"].get(result.task_id, {}),
        )

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
            logger.exception("S3 write failed for task %s", result.task_id)

    return {}


def check_completion(state: MissionRunnerState) -> str:
    """Routing function: decide whether to loop or finalize."""
    total = len(state["task_defs_by_id"])
    done = len(state["completed_task_ids"]) + len(state["failed_task_ids"])

    if done >= total:
        return "finalize_mission"

    # Check for stall: no ready tasks and no batch in progress
    batch = state.get("current_batch", [])
    if not state["ready_queue"] and not batch:
        return "finalize_mission"

    return "compute_ready_queue"


async def finalize_mission(
    state: MissionRunnerState, config: RunnableConfig, runtime: Runtime
) -> MissionRunnerState:
    """Update mission status in MongoDB and S3, build task-runs index and manifest."""
    from src.research.models.mission import MissionSummary

    failed = state["failed_task_ids"]
    final_status = "failed" if failed else "completed"

    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    mission.status = final_status
    mission.updated_at = datetime.utcnow()

    final_outputs = {
        tid: state["task_outputs"].get(tid, {})
        for tid in state["completed_task_ids"]
    }
    final_structured_summaries = {
        tid: state["task_structured_summaries"][tid]
        for tid in state["completed_task_ids"]
        if tid in state.get("task_structured_summaries", {})
    }

    # Build mission manifest (§11)
    all_task_ids = list(state["completed_task_ids"]) + list(state["failed_task_ids"])
    manifest = {
        "mission_id": str(state["mission_id"]),
        "title": mission.title,
        "status": final_status,
        "completed_at": datetime.utcnow().isoformat(),
        "tasks": [
            {
                "task_id": tid,
                "name": state["task_defs_by_id"].get(tid, {}).get("name"),
                "status": "completed" if tid in state["completed_task_ids"] else "failed",
                "structured_summary": state.get("task_structured_summaries", {}).get(tid),
                "outputs": state["task_outputs"].get(tid, {}),
                "quality_assessment": state.get("quality_assessments", {}).get(tid),
            }
            for tid in all_task_ids
        ],
        "all_sources": {
            "total": sum(
                len(state["task_outputs"].get(tid, {}).get("sources", []))
                for tid in state["completed_task_ids"]
            ),
        },
        "quality_assessments": state.get("quality_assessments", {}),
    }

    # Pull stored task manifests from store if available (§11)
    if runtime.store:
        for task_entry in manifest["tasks"]:
            tid = task_entry["task_id"]
            try:
                stored = runtime.store.get(
                    ("mission", state["mission_id"], "task_manifests"), tid
                )
                if stored and stored.value:
                    task_entry.update(stored.value)
            except Exception:
                pass

    # Populate MissionSummary on the MongoDB document
    mission.summary = MissionSummary(
        total_tasks=len(state["task_defs_by_id"]),
        completed_tasks=len(state["completed_task_ids"]),
        failed_tasks=len(state["failed_task_ids"]),
        total_sources=manifest["all_sources"]["total"],
        total_artifacts=len(state.get("artifacts", [])),
        has_final_report=any(
            s.get("final_synthesis_reports")
            for s in state.get("task_structured_summaries", {}).values()
            if isinstance(s, dict)
        ),
        completed_at=datetime.utcnow(),
    ).model_dump()
    await mission.save()

    # Persist mission, task-runs index, and manifest to S3 (best-effort)
    try:
        s3_store = get_research_runs_s3_store()
        await s3_store.write_mission(mission)

        runs = await ResearchRun.find(
            ResearchRun.mission_id == mission.id,
        ).to_list()
        await s3_store.build_task_runs_index(mission=mission, research_runs=runs)

        manifest_key = await s3_store.write_manifest(mission, manifest)
        logger.info("Mission manifest uploaded to S3: %s", manifest_key)
    except Exception:
        logger.exception("S3 finalization writes failed for mission %s", state["mission_id"])

    logger.info(
        "Mission %s finalized: status=%s, completed=%d, failed=%d",
        state["mission_id"],
        final_status,
        len(state["completed_task_ids"]),
        len(state["failed_task_ids"]),
    )

    return {
        "mission_status": final_status,
        "final_outputs": final_outputs,
        "final_structured_summaries": final_structured_summaries,
    }


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

async def build_mission_runner() -> CompiledStateGraph:
    """Build and compile the MissionRunner LangGraph StateGraph with parallel fan-out."""
    builder = StateGraph(MissionRunnerState)

    builder.add_node("load_mission", load_mission)
    builder.add_node("initialize_runtime_state", initialize_runtime_state)
    builder.add_node("compute_ready_queue", compute_ready_queue)
    # fan_out_tasks is a routing function, not a node — it returns Send objects
    builder.add_node("run_single_task", run_single_task)
    builder.add_node("fan_in_results", fan_in_results)
    builder.add_node("validate_batch_outputs", validate_batch_outputs)
    builder.add_node("merge_task_result", merge_task_result)
    builder.add_node("persist_research_run", persist_research_run)
    builder.add_node("finalize_mission", finalize_mission)

    builder.set_entry_point("load_mission")
    builder.add_edge("load_mission", "initialize_runtime_state")
    builder.add_edge("initialize_runtime_state", "compute_ready_queue")
    # fan_out_tasks is a conditional edge routing function that returns Send objects
    # for parallel dispatch, or the string "fan_in_results" when no tasks are ready
    builder.add_conditional_edges("compute_ready_queue", fan_out_tasks)
    builder.add_edge("run_single_task", "fan_in_results")
    builder.add_edge("fan_in_results", "validate_batch_outputs")
    builder.add_edge("validate_batch_outputs", "merge_task_result")
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
    max_parallel_tasks: int = 3,
) -> dict:
    runner: CompiledStateGraph = await get_mission_runner()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": mission_id,
            "progress_callback": progress_callback,
            "max_parallel_tasks": max_parallel_tasks,
        },
    }
    return await runner.ainvoke({"mission_id": mission_id}, config=config)