"""Mission Creator — orchestrates LLM compilation + structural validation + Mongo save.

Pipeline: approve-check → LLM compile → validate → topology build → save.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from beanie.odm.fields import PydanticObjectId

from src.research.compiler.mission_compiler_agent import (
    MissionDraftValidationError,
    compile_mission_draft,
)
from src.research.models.mission import ResearchMission, ResearchMissionDraft

if TYPE_CHECKING:
    from src.models.plan import ResearchPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class MissionCompilationError(Exception):
    """Base class for all mission compilation errors."""


class UnapprovedPlanError(MissionCompilationError):
    """Plan is not in approved status."""


class HallucinatedTaskIdError(MissionCompilationError):
    """LLM produced a task_id not present in the original plan."""


class CyclicDependencyError(MissionCompilationError):
    """LLM introduced or preserved a cyclic dependency in TaskDefs."""


class MissingInputError(MissionCompilationError):
    """A required InputBinding references a non-existent task/output key."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_task_ids(draft: ResearchMissionDraft, plan_task_ids: set[str]) -> None:
    """Ensure all draft task_ids exist in the original plan."""
    draft_ids = {td.task_id for td in draft.task_defs}
    hallucinated = draft_ids - plan_task_ids
    if hallucinated:
        raise HallucinatedTaskIdError(
            f"LLM produced task_ids not in the original plan: {hallucinated}"
        )

    missing = plan_task_ids - draft_ids
    if missing:
        logger.warning("Plan task_ids not present in draft (may be intentional): %s", missing)


def _validate_dependencies(draft: ResearchMissionDraft) -> None:
    """Ensure all depends_on references are valid task_ids within the draft."""
    valid_ids = {td.task_id for td in draft.task_defs}
    for td in draft.task_defs:
        for dep in td.depends_on:
            if dep not in valid_ids:
                raise HallucinatedTaskIdError(
                    f"Task '{td.task_id}' depends on '{dep}' which is not a valid task_id"
                )


def _detect_cycles(dependency_map: dict[str, list[str]]) -> None:
    """Kahn's algorithm for topological sort — raises if cycles exist."""
    in_degree: dict[str, int] = {node: 0 for node in dependency_map}
    for deps in dependency_map.values():
        for d in deps:
            if d in in_degree:
                in_degree[d] = in_degree.get(d, 0)

    adj: dict[str, list[str]] = {node: [] for node in dependency_map}
    for node, deps in dependency_map.items():
        for d in deps:
            if d in adj:
                adj[d].append(node)

    in_deg: dict[str, int] = {node: 0 for node in dependency_map}
    for node, deps in dependency_map.items():
        in_deg[node] = len(deps)

    queue: deque[str] = deque()
    for node, deg in in_deg.items():
        if deg == 0:
            queue.append(node)

    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in adj.get(current, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(dependency_map):
        raise CyclicDependencyError(
            f"Cyclic dependency detected. Processed {visited}/{len(dependency_map)} tasks."
        )


def _validate_input_bindings(draft: ResearchMissionDraft) -> None:
    """Ensure all required InputBindings reference existing tasks."""
    valid_ids = {td.task_id for td in draft.task_defs}
    for td in draft.task_defs:
        for local_name, binding in td.input_bindings.items():
            if binding.source_task_id not in valid_ids:
                raise MissingInputError(
                    f"Task '{td.task_id}' input '{local_name}' references "
                    f"non-existent source task '{binding.source_task_id}'"
                )


def _build_dependency_map(draft: ResearchMissionDraft) -> dict[str, list[str]]:
    return {td.task_id: list(td.depends_on) for td in draft.task_defs}


def _invert_dependency_map(dep_map: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {k: [] for k in dep_map}
    for task_id, deps in dep_map.items():
        for d in deps:
            if d not in reverse:
                reverse[d] = []
            reverse[d].append(task_id)
    return reverse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_mission_from_plan(
    plan: ResearchPlan,
    model_name: str = "openai:gpt-5-mini",
) -> ResearchMission:
    """
    Full pipeline: approve-check → LLM compile → validate → topology build → save.
    """
    # 1. Guard: plan must be approved
    if plan.status != "approved":
        raise UnapprovedPlanError(
            f"Plan status is '{plan.status}', expected 'approved'"
        )

    # 2. LLM compilation
    logger.info("Compiling mission draft for plan %s", plan.id)
    try:
        draft = await compile_mission_draft(plan, model_name=model_name)
    except MissionDraftValidationError:
        raise
    except Exception as e:
        raise MissionCompilationError(f"Unexpected error during LLM compilation: {e}") from e

    # 3. Structural validation
    plan_task_ids = {t.id for t in plan.tasks}
    _validate_task_ids(draft, plan_task_ids)
    _validate_dependencies(draft)
    _validate_input_bindings(draft)

    # 4. Build topology
    dependency_map = _build_dependency_map(draft)
    _detect_cycles(dependency_map)
    reverse_dependency_map = _invert_dependency_map(dependency_map)

    # 5. Construct ResearchMission document
    mission = ResearchMission(
        research_plan_id=PydanticObjectId(plan.id),
        thread_id=PydanticObjectId(plan.thread_id),
        title=draft.title,
        goal=draft.goal,
        global_context=draft.global_context,
        global_constraints=draft.global_constraints,
        success_criteria=draft.success_criteria,
        task_defs=draft.task_defs,
        dependency_map=dependency_map,
        reverse_dependency_map=reverse_dependency_map,
        status="pending",
    )

    # 6. Save to Mongo
    await mission.insert()
    logger.info("Mission %s created with %d tasks", mission.id, len(mission.task_defs))

    return mission
