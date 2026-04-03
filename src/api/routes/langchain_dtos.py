from __future__ import annotations

from typing import Any

from src.research.langchain_agent.models.mission import (
    IterativeStageRecord,
    MissionRunDocument,
    StageRunRecord,
)
from src.research.langchain_agent.models.plan import ResearchPlan


def stage_run_iteration(record: StageRunRecord) -> int:
    return record.iteration if record.iteration is not None else 1


def stage_run_id(mission_id: str, record: StageRunRecord, ordinal: int) -> str:
    return f"{mission_id}:{record.task_slug}:{stage_run_iteration(record)}:{ordinal}"


def artifact_to_dict(artifact: Any) -> dict[str, Any]:
    if hasattr(artifact, "model_dump"):
        return artifact.model_dump(mode="json")
    return dict(artifact)


def plan_to_dict(plan: ResearchPlan) -> dict[str, Any]:
    return {
        "id": str(plan.id),
        "thread_id": str(plan.thread_id),
        "title": plan.title,
        "objective": plan.objective,
        "stages": list(plan.stages),
        "tasks": [task.model_dump(mode="json") for task in plan.tasks],
        "starter_sources": [source.model_dump(mode="json") for source in plan.starter_sources],
        "context": plan.context,
        "run_kg": plan.run_kg,
        "unstructured_ingestion": plan.unstructured_ingestion.model_dump(mode="json"),
        "status": plan.status,
        "mission_id": plan.mission_id,
        "workflow_id": plan.workflow_id,
        "mission_status": plan.mission_status,
        "created_at": plan.created_at.isoformat(),
        "updated_at": plan.updated_at.isoformat(),
        "approved_at": plan.approved_at.isoformat() if plan.approved_at else None,
        "approver_notes": plan.approver_notes,
        "version": plan.version,
    }


def stage_run_to_dict(
    mission_id: str,
    record: StageRunRecord,
    *,
    ordinal: int,
    parent_task_slug: str | None = None,
) -> dict[str, Any]:
    return {
        "id": stage_run_id(mission_id, record, ordinal),
        "mission_id": mission_id,
        "task_id": record.task_id,
        "task_slug": record.task_slug,
        "parent_task_slug": parent_task_slug,
        "stage_type": record.stage_type,
        "targets": list(record.targets),
        "dependencies": list(record.dependencies),
        "iteration": record.iteration,
        "status": record.status,
        "error": record.error,
        "final_report_text": record.final_report_text,
        "artifacts": {
            "final_report": artifact_to_dict(record.artifacts.final_report)
            if record.artifacts.final_report
            else None,
            "intermediate_files": [artifact_to_dict(a) for a in record.artifacts.intermediate_files],
            "memory_report_json": artifact_to_dict(record.artifacts.memory_report_json)
            if record.artifacts.memory_report_json
            else None,
            "agent_state_json": artifact_to_dict(record.artifacts.agent_state_json)
            if record.artifacts.agent_state_json
            else None,
        },
        "memory_report": record.memory_report.model_dump(mode="json")
        if record.memory_report
        else None,
        "langsmith_run_id": record.langsmith_run_id,
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }


def flatten_stage_runs(doc: MissionRunDocument) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    ordinal = 0
    for record in doc.stages:
        ordinal += 1
        runs.append(stage_run_to_dict(doc.mission_id, record, ordinal=ordinal))
    for iterative in doc.iterative_stages:
        for record in iterative.iterations:
            ordinal += 1
            runs.append(
                stage_run_to_dict(
                    doc.mission_id,
                    record,
                    ordinal=ordinal,
                    parent_task_slug=iterative.task_slug,
                )
            )
    runs.sort(key=lambda item: (item["started_at"] or "", item["task_slug"], item["iteration"] or 0))
    return runs


def task_ids_for_doc(doc: MissionRunDocument) -> list[str]:
    task_ids = {record.task_id for record in doc.stages}
    task_ids.update(iterative.task_slug for iterative in doc.iterative_stages)
    return sorted(task_ids)


def compute_status_summary(
    doc: MissionRunDocument,
    expected_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    completed_ids: set[str] = set()
    failed_ids: set[str] = set()
    running_ids: set[str] = set()

    for record in doc.stages:
        if record.status == "completed":
            completed_ids.add(record.task_id)
        elif record.status == "failed":
            failed_ids.add(record.task_id)
        else:
            running_ids.add(record.task_id)

    for iterative in doc.iterative_stages:
        if not iterative.iterations:
            running_ids.add(iterative.task_slug)
            continue
        last = iterative.iterations[-1]
        if last.status == "completed":
            completed_ids.add(iterative.task_slug)
        elif last.status == "failed":
            failed_ids.add(iterative.task_slug)
        else:
            running_ids.add(iterative.task_slug)

    task_ids = expected_task_ids or sorted(completed_ids | failed_ids | running_ids)
    pending_count = max(
        0,
        len(task_ids) - len(completed_ids) - len(failed_ids) - len(running_ids),
    )
    return {
        "mission_id": doc.mission_id,
        "status": doc.status,
        "total_tasks": len(task_ids),
        "completed_tasks": len(completed_ids),
        "failed_tasks": len(failed_ids),
        "running_tasks": len(running_ids),
        "pending_tasks": pending_count,
        "completed_task_ids": sorted(completed_ids),
        "failed_task_ids": sorted(failed_ids),
        "running_task_ids": sorted(running_ids),
    }


def mission_to_dict(
    doc: MissionRunDocument,
    *,
    plan: ResearchPlan | None = None,
) -> dict[str, Any]:
    task_ids = [task.id for task in plan.tasks] if plan else task_ids_for_doc(doc)
    summary = compute_status_summary(doc, expected_task_ids=task_ids)
    return {
        "id": doc.mission_id,
        "mission_id": doc.mission_id,
        "plan_id": doc.research_plan_id,
        "thread_id": doc.thread_id,
        "workflow_id": doc.workflow_id,
        "title": plan.title if plan else (doc.mission_name or doc.mission_id),
        "objective": plan.objective if plan else "",
        "mission_name": doc.mission_name,
        "base_domain": doc.base_domain,
        "mission_type": doc.mission_type,
        "targets": list(doc.targets),
        "status": doc.status,
        "error": doc.error,
        "task_count": summary["total_tasks"],
        "completed_task_count": summary["completed_tasks"],
        "failed_task_count": summary["failed_tasks"],
        "running_task_count": summary["running_tasks"],
        "tasks": [
            {
                "task_id": task.id,
                "task_slug": task.id,
                "title": task.title,
                "stage": task.stage,
                "dependencies": list(task.dependencies),
                "selected_tool_names": list(task.selected_tool_names),
                "selected_subagent_names": list(task.selected_subagent_names),
                "stage_type": task.stage_type,
            }
            for task in (plan.tasks if plan else [])
        ],
        "stages": flatten_stage_runs(doc),
        "langsmith_run_id": doc.langsmith_run_id,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
        "completed_at": doc.completed_at.isoformat() if doc.completed_at else None,
    }


def build_outputs_payload(
    doc: MissionRunDocument,
    *,
    plan: ResearchPlan | None = None,
) -> dict[str, Any]:
    runs = flatten_stage_runs(doc)
    final_report = None
    preferred = [
        run
        for run in runs
        if run["status"] == "completed" and run["stage_type"] == "report_synthesis" and run["final_report_text"]
    ]
    if preferred:
        final_report = preferred[-1]["final_report_text"]
    else:
        completed = [run for run in runs if run["status"] == "completed" and run["final_report_text"]]
        if completed:
            final_report = completed[-1]["final_report_text"]

    return {
        "mission": mission_to_dict(doc, plan=plan),
        "summary": compute_status_summary(
            doc,
            expected_task_ids=[task.id for task in plan.tasks] if plan else None,
        ),
        "final_report_markdown": final_report,
        "stage_reports": [
            {
                "run_id": run["id"],
                "task_id": run["task_id"],
                "task_slug": run["task_slug"],
                "iteration": run["iteration"],
                "stage_type": run["stage_type"],
                "status": run["status"],
                "final_report_text": run["final_report_text"],
            }
            for run in runs
            if run["final_report_text"]
        ],
        "artifacts": [
            {
                "run_id": run["id"],
                "task_id": run["task_id"],
                "task_slug": run["task_slug"],
                "iteration": run["iteration"],
                "artifact": artifact,
            }
            for run in runs
            for artifact in (
                ([run["artifacts"]["final_report"]] if run["artifacts"]["final_report"] else [])
                + run["artifacts"]["intermediate_files"]
                + ([run["artifacts"]["memory_report_json"]] if run["artifacts"]["memory_report_json"] else [])
                + ([run["artifacts"]["agent_state_json"]] if run["artifacts"]["agent_state_json"] else [])
            )
        ],
        "task_runs_index": runs,
    }


def build_iterative_stage_record(
    parent_task_slug: str,
    stage_run_records: list[StageRunRecord],
) -> IterativeStageRecord:
    iterations_completed = len(stage_run_records)
    stop_reason = "completed" if stage_run_records and stage_run_records[-1].status == "completed" else "failed"
    combined_report = "\n\n".join(
        record.final_report_text.strip()
        for record in stage_run_records
        if record.final_report_text.strip()
    )
    return IterativeStageRecord(
        task_slug=parent_task_slug,
        max_iterations=iterations_completed,
        iterations_completed=iterations_completed,
        stop_reason=stop_reason,
        iterations=stage_run_records,
        combined_report_text=combined_report,
        next_steps_history=[],
    )
