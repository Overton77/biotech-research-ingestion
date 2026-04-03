"""Compile LangChain ``ResearchPlan`` documents into executable ``ResearchMission`` objects."""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.research.langchain_agent.models.mission import (
    ResearchMission,
    ResearchMissionDraft,
    draft_to_research_mission,
)
from src.research.langchain_agent.models.plan import ResearchPlan

logger = logging.getLogger(__name__)


class MissionCompilationError(Exception):
    """Raised when mission compilation or structural validation fails."""


class UnapprovedPlanError(MissionCompilationError):
    """Plan is not approved."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _detect_cycles(dependency_map: dict[str, list[str]]) -> None:
    """Raise if the dependency graph has a cycle (Kahn-style reachability)."""
    if not dependency_map:
        return
    in_deg: dict[str, int] = {node: len(deps) for node, deps in dependency_map.items()}
    queue: deque[str] = deque()
    for node, deg in in_deg.items():
        if deg == 0:
            queue.append(node)
    adj: dict[str, list[str]] = {k: [] for k in dependency_map}
    for node, deps in dependency_map.items():
        for d in deps:
            if d in adj:
                adj[d].append(node)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in adj.get(current, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
    if visited != len(dependency_map):
        raise MissionCompilationError(
            f"Cyclic dependencies in mission stages: processed {visited}/{len(dependency_map)} nodes."
        )


def _validate_draft_against_plan(draft: ResearchMissionDraft, plan: ResearchPlan) -> None:
    """Ensure draft stages align with the approved plan (ids + dependency edges)."""
    plan_tasks = {t.id: t for t in plan.tasks}
    if not plan_tasks:
        raise MissionCompilationError("Plan has no tasks.")

    slugs = [s.slice_input.task_slug for s in draft.stages]
    if len(slugs) != len(set(slugs)):
        raise MissionCompilationError(f"Duplicate task_slug in draft stages: {slugs}")

    plan_ids = set(plan_tasks.keys())
    draft_ids = set(slugs)
    if plan_ids != draft_ids:
        raise MissionCompilationError(
            f"Stage task_slug set must match plan task ids. Plan: {sorted(plan_ids)} "
            f"Draft: {sorted(draft_ids)}"
        )

    for s in draft.stages:
        tid = s.slice_input.task_slug
        pt = plan_tasks[tid]
        plan_deps = set(pt.dependencies)
        draft_deps = set(s.dependencies)
        if plan_deps != draft_deps:
            raise MissionCompilationError(
                f"Dependencies for task {tid!r} must match the plan. "
                f"Plan: {sorted(plan_deps)} Draft: {sorted(draft_deps)}"
            )
        if s.slice_input.mission_id != "pending":
            raise MissionCompilationError(
                f"slice_input.mission_id must be 'pending' for task {tid!r}, "
                f"got {s.slice_input.mission_id!r}"
            )

    dep_map = {s.slice_input.task_slug: list(s.dependencies) for s in draft.stages}
    _detect_cycles(dep_map)


def _apply_plan_mission_settings(draft: ResearchMissionDraft, plan: ResearchPlan) -> ResearchMissionDraft:
    """Plan is authoritative for mission-level KG and unstructured ingestion settings."""
    return draft.model_copy(
        update={
            "run_kg": plan.run_kg,
            "unstructured_ingestion": plan.unstructured_ingestion.model_copy(deep=True),
        }
    )


def _build_compiler_system_prompt() -> str:
    return """You are a Mission Compiler for a LangChain biotech research pipeline.

Read an **approved research plan JSON** (see PLAN JSON in the user message) and output a single
**ResearchMissionDraft** executable as a DAG of research stages.

## Plan JSON fields you must respect
- **title**, **objective**, **stages**, **tasks**, **context**, **starter_sources**: drive stage design and prompts.
- **run_kg** (bool): Authoritative. When true, the mission will run **structured** knowledge-graph ingestion on completed stage reports (extract typed entities/relationships into Neo4j). The plan JSON states the user's intent; your draft should mirror the same boolean so validation is easy (the runtime will still apply the stored plan value).
- **unstructured_ingestion** (object): Authoritative configuration for the **unstructured** document pipeline after stages (candidate manifests → parse/chunk/embed; optional Neo4j relationship writes). Includes nested settings such as:
  - Top-level: enabled, validate_in_isolation, write_to_neo4j, max_relationship_chunks, parser_backend (docling | llamaparse), llama_parse_tier, summary_policy, embedding_config, chunk_cleaning, chunk_enhancement.
  - Mirror this structure in your draft's `unstructured_ingestion` field so it matches the plan; the runtime applies the plan document as the source of truth after compilation.

## ResearchMissionDraft (top-level)
- **mission_name**: Short label (often same as plan title).
- **base_domain**: Optional focus string (therapeutic area, company, or modality) when inferrable from the plan.
- **stages**: Exactly one **MissionStageDraft** per entry in plan.tasks — same count, same task ids as slugs. Order stages in a valid topological order respecting dependencies.
- **run_kg**: Copy from plan JSON `run_kg` (same boolean).
- **unstructured_ingestion**: Copy from plan JSON `unstructured_ingestion` (same nested object; use defaults only if the plan omits the object).

## MissionStageDraft (one per plan task)
- **slice_input** (MissionSliceInputDraft):
  - **mission_id**: MUST be the literal string `"pending"`.
  - **task_id** and **task_slug**: MUST equal the plan task `id` string.
  - **user_objective**: Concrete instructions for the agent; synthesize plan objective + this task's title/description + relevant context.
  - **targets**: Entity/topic strings when helpful (companies, drugs, trials).
  - **selected_tool_names** / **selected_subagent_names**: Copy from the plan task; only use allowed registry names.
  - **stage_type**, **max_step_budget**, **temporal_scope**, **research_date**: Set per task; **temporal_scope.mode** MUST be one of: current | as_of_date | date_range | unknown.
  - Use **as_of_date** for point-in-time snapshots and **date_range** for bounded windows.
- **prompt_spec** (**ResearchPromptSpecModel**): Specialize per task — agent_identity, domain_scope, workflow, tool_guidance, subagent_guidance, practical_limits, filesystem_rules, intermediate_files.
- **dependencies**: MUST exactly match the plan task's `dependencies` (list of other task ids).
- **execution_reminders** / **iterative_config**: Optional.

## Hard rules
- Exactly one stage per plan task; do not invent, merge, or drop tasks.
- Do not change dependency edges from the plan.
- **slice_input.mission_id** must always be `"pending"` for every stage.
- Prefer copying **run_kg** and **unstructured_ingestion** from the plan JSON rather than inventing new mission-level ingestion policy.
"""


def _plan_to_user_payload(plan: ResearchPlan) -> str:
    tasks_out: list[dict] = []
    for t in plan.tasks:
        item: dict = {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "stage": t.stage,
            "dependencies": list(t.dependencies),
        }
        if t.estimated_duration_minutes is not None:
            item["estimated_duration_minutes"] = t.estimated_duration_minutes
        if t.selected_tool_names:
            item["selected_tool_names"] = t.selected_tool_names
        if t.selected_subagent_names:
            item["selected_subagent_names"] = t.selected_subagent_names
        if t.stage_type is not None:
            item["stage_type"] = t.stage_type
        tasks_out.append(item)

    payload = {
        "title": plan.title,
        "objective": plan.objective,
        "stages": list(plan.stages),
        "tasks": tasks_out,
        "context": plan.context,
        "starter_sources": [s.model_dump() for s in plan.starter_sources],
        "run_kg": plan.run_kg,
        "unstructured_ingestion": plan.unstructured_ingestion.model_dump(mode="json"),
    }
    return json.dumps(payload, indent=2)


async def compile_mission_draft(
    plan: ResearchPlan,
    model_name: str = "gpt-5-mini",
) -> ResearchMissionDraft:
    """LLM: plan JSON → validated ``ResearchMissionDraft``."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise MissionCompilationError("OPENAI_API_KEY is not configured.")

    system = _build_compiler_system_prompt()
    user = f"""Compile the following approved research plan into a ResearchMissionDraft.

PLAN JSON:
```json
{_plan_to_user_payload(plan)}
```

Return structured output only."""

    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    structured = llm.with_structured_output(ResearchMissionDraft)

    try:
        result = await structured.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
        )
    except Exception as e:
        logger.exception("Mission compiler LLM failed")
        raise MissionCompilationError(f"LLM mission compilation failed: {e}") from e

    if not isinstance(result, ResearchMissionDraft):
        raise MissionCompilationError("LLM returned an unexpected structured type.")

    return result


async def create_mission_from_plan(
    plan: ResearchPlan,
    model_name: str = "gpt-5-mini",
) -> ResearchMission:
    """
    Approve-check → LLM compile → validate → assign mission_id → ``ResearchMission``.
    """
    if plan.status != "approved":
        raise UnapprovedPlanError(
            f"Plan status is '{plan.status}', expected 'approved'"
        )

    draft = await compile_mission_draft(plan, model_name=model_name)
    draft = _apply_plan_mission_settings(draft, plan)
    _validate_draft_against_plan(draft, plan)

    mission_id = str(uuid.uuid4())
    mission = draft_to_research_mission(draft, mission_id)
    logger.info(
        "Compiled LangChain ResearchMission %s with %d stages for plan %s",
        mission_id,
        len(mission.stages),
        plan.id,
    )
    return mission
