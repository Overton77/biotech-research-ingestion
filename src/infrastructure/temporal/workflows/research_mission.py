"""Temporal workflow — DAG-aware research mission with parallel stage execution.

Stages are grouped into DAG levels based on their ``dependencies`` field.
All stages within a level execute as parallel activities.  Once a level
completes, the next level starts — each stage receives ``final_report_text``
from its declared dependencies.

After all stages complete, if ``run_kg`` is true, KG ingestion activities
run in parallel (one per completed stage report).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.infrastructure.temporal.activities.research_mission import (
        execute_research_stage,
        ingest_kg_from_report,
    )
    from src.infrastructure.temporal.models import (
        KGIngestionInput,
        MissionWorkflowInput,
        MissionWorkflowOutput,
        StageActivityInput,
        StageActivityOutput,
    )


# ---------------------------------------------------------------------------
# DAG level computation
# ---------------------------------------------------------------------------


def _compute_dag_levels(stages: list[dict]) -> list[list[dict]]:
    """Group stages into execution levels respecting dependency order.

    Each level contains stages whose dependencies are all satisfied by
    stages in previous levels.  Stages within a level can run in parallel.

    Returns a list of levels, where each level is a list of stage dicts.
    """
    slug_to_stage: dict[str, dict] = {}
    for s in stages:
        slug = s.get("slice_input", {}).get("task_slug", "")
        if slug:
            slug_to_stage[slug] = s

    placed: set[str] = set()
    levels: list[list[dict]] = []

    while len(placed) < len(stages):
        current_level: list[dict] = []
        for s in stages:
            slug = s.get("slice_input", {}).get("task_slug", "")
            if slug in placed:
                continue
            deps = s.get("dependencies", [])
            if all(d in placed for d in deps):
                current_level.append(s)

        if not current_level:
            # Remaining stages have unresolvable deps — force them into a final level
            for s in stages:
                slug = s.get("slice_input", {}).get("task_slug", "")
                if slug not in placed:
                    current_level.append(s)
            levels.append(current_level)
            break

        levels.append(current_level)
        for s in current_level:
            slug = s.get("slice_input", {}).get("task_slug", "")
            placed.add(slug)

    return levels


def _find_stage(stages: list[dict], task_slug: str) -> dict | None:
    for s in stages:
        if s.get("slice_input", {}).get("task_slug") == task_slug:
            return s
    return None


# ---------------------------------------------------------------------------
# Research stage retry policy
# ---------------------------------------------------------------------------

_STAGE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=10),
    maximum_attempts=2,
)

_KG_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=2,
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow.defn
class ResearchMissionWorkflow:
    """Durable, DAG-aware research mission workflow.

    Stages are grouped into levels; independent stages within a level
    execute as parallel Temporal activities.  After all stages complete,
    an optional KG ingestion phase runs in parallel across all reports.
    """

    @workflow.run
    async def run(self, input: MissionWorkflowInput) -> MissionWorkflowOutput:
        mission = input.mission_json
        mission_id: str = mission.get("mission_id", "unknown")
        stages: list[dict] = mission.get("stages", [])

        workflow.logger.info(
            "ResearchMissionWorkflow started: mission_id=%s, stages=%d, run_kg=%s",
            mission_id,
            len(stages),
            input.run_kg,
        )

        levels = _compute_dag_levels(stages)
        workflow.logger.info(
            "DAG levels computed: %d level(s), stages per level: %s",
            len(levels),
            [len(lvl) for lvl in levels],
        )

        report_by_slug: dict[str, str] = {}
        all_results: list[StageActivityOutput] = []

        # ----- Execute stages level by level -----
        for level_idx, level_stages in enumerate(levels):
            workflow.logger.info(
                "Executing level %d: %d stage(s)",
                level_idx,
                len(level_stages),
            )

            tasks: list[Any] = []
            for stage in level_stages:
                slug = stage.get("slice_input", {}).get("task_slug", "")
                deps = stage.get("dependencies", [])
                dep_reports = {d: report_by_slug[d] for d in deps if d in report_by_slug}

                tasks.append(
                    workflow.execute_activity(
                        execute_research_stage,
                        StageActivityInput(
                            mission_id=mission_id,
                            stage_json=stage,
                            dependency_reports=dep_reports,
                            root_filesystem=input.output_dir,
                        ),
                        start_to_close_timeout=timedelta(hours=3),
                        heartbeat_timeout=timedelta(minutes=15),
                        retry_policy=_STAGE_RETRY,
                    )
                )

            results: list[StageActivityOutput] = await asyncio.gather(*tasks)

            for result in results:
                all_results.append(result)
                if result.status == "completed" and result.final_report_text:
                    report_by_slug[result.task_slug] = result.final_report_text
                    workflow.logger.info("Stage completed: %s", result.task_slug)
                else:
                    workflow.logger.warning(
                        "Stage %s finished with status=%s error=%s",
                        result.task_slug,
                        result.status,
                        result.error,
                    )

        # ----- Conditional KG ingestion -----
        kg_results = []
        if input.run_kg:
            kg_tasks = []
            for result in all_results:
                if result.status != "completed" or not result.final_report_text:
                    continue
                stage = _find_stage(stages, result.task_slug)
                if stage is None:
                    continue

                slice_input = stage.get("slice_input", {})
                kg_tasks.append(
                    workflow.execute_activity(
                        ingest_kg_from_report,
                        KGIngestionInput(
                            report_text=result.final_report_text,
                            source_report=result.task_slug,
                            targets=slice_input.get("targets", []),
                            stage_type=slice_input.get("stage_type", "targeted_extraction"),
                            research_date=slice_input.get("research_date"),
                            temporal_scope=slice_input.get("temporal_scope"),
                            context=f"Mission {mission_id}, stage {result.task_slug}",
                        ),
                        start_to_close_timeout=timedelta(minutes=30),
                        heartbeat_timeout=timedelta(minutes=10),
                        retry_policy=_KG_RETRY,
                    )
                )

            if kg_tasks:
                workflow.logger.info("Running KG ingestion for %d report(s)", len(kg_tasks))
                kg_results = await asyncio.gather(*kg_tasks)
                workflow.logger.info(
                    "KG ingestion complete: %d succeeded, %d failed",
                    sum(1 for r in kg_results if r.status == "completed"),
                    sum(1 for r in kg_results if r.status == "failed"),
                )

        stages_completed = sum(1 for r in all_results if r.status == "completed")
        stages_failed = sum(1 for r in all_results if r.status == "failed")
        kg_completed = sum(1 for r in kg_results if r.status == "completed")

        workflow.logger.info(
            "Mission %s finished: %d/%d stages completed, %d KG ingestions",
            mission_id,
            stages_completed,
            len(all_results),
            kg_completed,
        )

        return MissionWorkflowOutput(
            mission_id=mission_id,
            status="completed" if stages_failed == 0 else "partial",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            kg_ingestions_completed=kg_completed,
            stage_results=all_results,
            kg_results=list(kg_results),
        )
