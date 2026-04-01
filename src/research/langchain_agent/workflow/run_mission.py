"""
Run a full ResearchMission with dependency ordering and report injection.
Stages with dependencies receive the final_report of each depended-on stage.

Stages may be iterative: when ``MissionStage.iterative_config`` is set, the
runner delegates to ``run_iterative_stage()`` which executes bounded passes
in a loop.  Non-iterative stages run exactly as before via
``run_single_mission_slice()``.

Post-run: each completed stage's StageRunRecord is appended to a
MissionRunDocument (MongoDB / Beanie) so the full run is persisted.
MongoDB persistence failures are logged and never interrupt execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.research.langchain_agent.models.mission import MissionStage, ResearchMission
from src.research.langchain_agent.workflow.run_slice import run_single_mission_slice 
from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient, Neo4jAuraSettings 
import json 
from src.research.langchain_agent.workflow.run_iterative_stage import run_iterative_stage
from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.observability.tracing import (
    get_current_trace_id,
    traced_mission,
)
from src.research.langchain_agent.storage.models import (
    IterativeStageRecord,
    MissionRunDocument,
) 
from src.research.langchain_agent.unstructured.candidate_collection import (
    gather_mission_candidates,
)
from src.research.langchain_agent.unstructured.paths import mission_unstructured_dir
from src.research.langchain_agent.unstructured.models import CandidateDocument
from src.research.langchain_agent.unstructured.run_unstructured_ingestion import (
    run_unstructured_ingestion,
)

logger = logging.getLogger(__name__) 

async def _run_staged_unstructured_ingestion(
    *,
    mission: ResearchMission,
    mission_candidate_manifest_path: str,
    root: Path,
) -> dict[str, Any]:
    manifest_abs = root / mission_candidate_manifest_path
    payload = json.loads(manifest_abs.read_text(encoding="utf-8"))
    final_candidates = payload.get("candidates", [])
    if not final_candidates:
        return {"status": "skipped", "reason": "no_candidates"}

    output_dir = mission_unstructured_dir(mission.mission_id, root=root) / "executions"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Neo4jAuraSettings.from_env()
    results: list[dict[str, Any]] = []
    async with Neo4jAuraClient(settings) as client:
        for candidate_payload in final_candidates:
            if not candidate_payload.get("local_path"):
                results.append(
                    {
                        "candidate_id": candidate_payload.get("candidate_id", ""),
                        "status": "skipped",
                        "reason": "candidate_missing_local_path",
                    }
                )
                continue

            candidate_dir = output_dir / candidate_payload["candidate_id"]
            try:
                candidate = CandidateDocument.model_validate(candidate_payload)
                ingestion_result = await run_unstructured_ingestion(
                    candidate=candidate,
                    output_dir=candidate_dir,
                    neo4j_client=client,
                    config=mission.unstructured_ingestion,
                )
                results.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "completed",
                        "document_id": ingestion_result.document.document_id,
                        "chunk_count": len(ingestion_result.chunks),
                        "relationship_count": len(ingestion_result.relationship_decisions),
                    }
                )
            except Exception as exc:
                logger.exception(
                    "Unstructured ingestion failed for candidate %s",
                    candidate_payload.get("candidate_id", ""),
                )
                results.append(
                    {
                        "candidate_id": candidate_payload.get("candidate_id", ""),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    return {"status": "completed", "results": results}


def _topological_stage_order(mission: ResearchMission) -> List[int]:
    """
    Return stage indices in an order that respects dependencies.
    For each stage, every stage in its dependencies list (by task_slug) must appear before it.
    """
    slug_to_index: Dict[str, int] = {
        stage.slice_input.task_slug: i for i, stage in enumerate(mission.stages)
    }
    ordered: List[int] = []

    while len(ordered) < len(mission.stages):
        added = False
        for i, stage in enumerate(mission.stages):
            if i in ordered:
                continue
            deps_met = True
            for dep_slug in stage.dependencies:
                if dep_slug in slug_to_index:
                    if slug_to_index[dep_slug] not in ordered:
                        deps_met = False
                        break
            if deps_met:
                ordered.append(i)
                added = True
                break
        if not added:
            # Cycle or missing dependency; include remaining in current order
            for i in range(len(mission.stages)):
                if i not in ordered:
                    ordered.append(i)
            break

    return ordered


@traced_mission
async def run_mission(
    mission: ResearchMission,
    *,
    store: Any,
    checkpointer: Any,
    memory_manager: Any,
    root_filesystem: Path | None = None,
    snapshot_output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Run all stages of the mission in dependency order. When a stage has dependencies,
    its slice_input receives those stages' final reports in dependency_reports (task_slug -> markdown).

    Creates a MissionRunDocument in MongoDB at the start, appends a StageRunRecord
    after each stage, and marks the document completed (or failed) at the end.
    MongoDB failures are logged and never raise — research output is always returned.

    Returns a list of per-stage result dicts (same shape as run_single_mission_slice return value).
    Each dict now also contains 'stage_run_record'.
    """
    root = root_filesystem or ROOT_FILESYSTEM
    ordered_indices = _topological_stage_order(mission)
    report_by_slug: Dict[str, str] = {}
    outputs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Create the MissionRunDocument — non-blocking on failure
    # ------------------------------------------------------------------
    has_iterative = any(s.iterative_config is not None for s in mission.stages)
    mission_type = "iterative" if has_iterative else "stage_based"

    mission_doc: MissionRunDocument | None = None
    try:
        mission_doc = MissionRunDocument(
            mission_id=mission.mission_id,
            mission_name=mission.mission_name,
            mission_type=mission_type,
            base_domain=mission.base_domain,
            targets=list({
                t
                for stage in mission.stages
                for t in stage.slice_input.targets
            }),
            langsmith_run_id=get_current_trace_id(),
        )
        await mission_doc.insert()
        logger.info(
            "MissionRunDocument created: mission_id=%s doc_id=%s langsmith_run_id=%s",
            mission.mission_id, mission_doc.id, mission_doc.langsmith_run_id,
        )
    except Exception:
        logger.exception(
            "Failed to create MissionRunDocument for mission %s — "
            "execution continues without MongoDB persistence",
            mission.mission_id,
        )

    # ------------------------------------------------------------------
    # Execute stages
    # ------------------------------------------------------------------
    try:
        for idx in ordered_indices:
            stage: MissionStage = mission.stages[idx]
            slug = stage.slice_input.task_slug

            dep_reports: Dict[str, str] = {}
            if stage.dependencies:
                dep_reports = {
                    dep_slug: report_by_slug[dep_slug]
                    for dep_slug in stage.dependencies
                    if dep_slug in report_by_slug
                }

            # ----- Iterative stage path -----
            if stage.iterative_config is not None:
                iter_result = await run_iterative_stage(
                    stage,
                    dependency_reports=dep_reports,
                    store=store,
                    checkpointer=checkpointer,
                    memory_manager=memory_manager,
                    root_filesystem=root,
                    snapshot_output_dir=snapshot_output_dir,
                )

                outputs.extend(iter_result.iteration_outputs)
                report_by_slug[slug] = iter_result.combined_report

                if mission_doc is not None:
                    try:
                        iter_record = IterativeStageRecord(
                            task_slug=slug,
                            max_iterations=stage.iterative_config.max_iterations,
                            iterations_completed=iter_result.iterations_completed,
                            stop_reason=iter_result.stop_reason,
                            iterations=[
                                out.get("stage_run_record")
                                for out in iter_result.iteration_outputs
                                if out.get("stage_run_record") is not None
                            ],
                            combined_report_text=iter_result.combined_report,
                            next_steps_history=iter_result.next_steps_history,
                        )
                        mission_doc.append_iterative_stage(iter_record)
                        await mission_doc.save()
                    except Exception:
                        logger.exception(
                            "Failed to append IterativeStageRecord for %s to MongoDB",
                            slug,
                        )

            # ----- Single-pass stage path (unchanged) -----
            else:
                run_input = stage.slice_input.model_copy(deep=True)
                run_input.dependency_reports = dep_reports

                out = await run_single_mission_slice(
                    run_input=run_input,
                    prompt_spec=stage.prompt_spec,
                    store=store,
                    checkpointer=checkpointer,
                    memory_manager=memory_manager,
                    execution_reminders=stage.execution_reminders,
                    root_filesystem=root,
                    snapshot_output_dir=snapshot_output_dir,
                )

                outputs.append(out)
                report_by_slug[slug] = out.get("final_report_text") or ""

                if mission_doc is not None:
                    try:
                        stage_record = out.get("stage_run_record")
                        if stage_record is not None:
                            mission_doc.append_stage(stage_record)
                            await mission_doc.save()
                    except Exception:
                        logger.exception(
                            "Failed to append StageRunRecord for %s to MongoDB",
                            slug,
                        )

    except Exception as exc:
        # Mark the mission document as failed before re-raising
        if mission_doc is not None:
            try:
                mission_doc.mark_failed(str(exc))
                await mission_doc.save()
            except Exception:
                logger.exception("Failed to mark MissionRunDocument as failed")
        raise  
    stage_manifest_paths = [
        out.get("stage_candidate_manifest_path", "")
        for out in outputs
        if out.get("stage_candidate_manifest_path")
    ]
    if stage_manifest_paths:
        try:
            mission_manifest, mission_manifest_path = gather_mission_candidates(
                mission_id=mission.mission_id,
                stage_manifest_paths=stage_manifest_paths,
                root=root,
            )
            logger.info(
                "Mission candidate manifest written: %s (%d candidates)",
                mission_manifest_path,
                mission_manifest.final_candidate_count,
            )
            for out in outputs:
                out["mission_candidate_manifest_path"] = mission_manifest_path

            if mission.unstructured_ingestion.enabled:
                unstructured_summary = await _run_staged_unstructured_ingestion(
                    mission=mission,
                    mission_candidate_manifest_path=mission_manifest_path,
                    root=root,
                )
                summary_path = mission_unstructured_dir(mission.mission_id, root=root) / "mission_unstructured_summary.json"
                summary_path.write_text(
                    json.dumps(unstructured_summary, indent=2, default=str),
                    encoding="utf-8",
                )
                for out in outputs:
                    out["mission_unstructured_summary_path"] = str(
                        summary_path.relative_to(root).as_posix()
                    )
        except Exception:
            logger.exception(
                "Failed to gather or execute staged unstructured ingestion for mission %s",
                mission.mission_id,
            )


    # ------------------------------------------------------------------
    # Mark mission complete
    # ------------------------------------------------------------------
    if mission_doc is not None:
        try:
            mission_doc.mark_completed()
            await mission_doc.save()
            logger.info(
                "MissionRunDocument completed: mission_id=%s doc_id=%s stages=%d",
                mission.mission_id, mission_doc.id, len(outputs),
            )
        except Exception:
            logger.exception(
                "Failed to mark MissionRunDocument as completed for mission %s",
                mission.mission_id,
            )

    return outputs
