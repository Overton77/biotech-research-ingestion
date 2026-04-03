"""Temporal activities for the research mission workflow.

Two activities:
  1. execute_research_stage — run a single MissionStage (single-pass or iterative)
  2. ingest_kg_from_report  — run KG ingestion on a completed stage report
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from temporalio import activity

from src.infrastructure.temporal.models import (
    KGIngestionInput,
    KGIngestionOutput,
    MissionFinalizeInput,
    MissionProgressEventInput,
    MissionStagePersistInput,
    MissionWorkflowInput,
    StageActivityInput,
    StageActivityOutput,
    UnstructuredIngestionInput,
    UnstructuredIngestionOutput,
)
from src.research.deepagent.middleware.progress_callback import create_progress_callback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_stage(stage_json: dict):
    """Rebuild a MissionStage from a serialized dict."""
    from src.research.langchain_agent.agent.config import (
        MissionSliceInput,
        ResearchPromptSpec,
    )
    from src.research.langchain_agent.models.mission import (
        IterativeStageConfig,
        MissionStage,
    )

    spec_data = stage_json.get("prompt_spec", {})
    prompt_spec = ResearchPromptSpec(
        agent_identity=spec_data.get("agent_identity", "You are a biotech research agent."),
        domain_scope=spec_data.get("domain_scope", []),
        workflow=spec_data.get("workflow", []),
        tool_guidance=spec_data.get("tool_guidance", []),
        subagent_guidance=spec_data.get("subagent_guidance", []),
        practical_limits=spec_data.get("practical_limits", []),
        filesystem_rules=spec_data.get("filesystem_rules", []),
        intermediate_files=spec_data.get("intermediate_files", []),
    )

    iter_cfg_data = stage_json.get("iterative_config")
    iterative_config = IterativeStageConfig(**iter_cfg_data) if iter_cfg_data else None

    return MissionStage(
        slice_input=MissionSliceInput(**stage_json["slice_input"]),
        prompt_spec=prompt_spec,
        execution_reminders=stage_json.get("execution_reminders", []),
        dependencies=stage_json.get("dependencies", []),
        iterative_config=iterative_config,
    )


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _emit_progress(mission_id: str, event_type: str, payload: dict) -> None:
    callback = create_progress_callback(mission_id)
    body = dict(payload)
    body.setdefault("timestamp", _utc_iso())
    await callback(event_type, body)


@activity.defn
async def emit_mission_progress(input: MissionProgressEventInput) -> None:
    """Broadcast a workflow-level progress event through the internal relay."""
    await _emit_progress(input.mission_id, input.event_type, input.payload)


async def _expected_task_count_for_mission(mission, plan_id: str | None) -> int:
    """Match list filtering to mission_to_dict task_count when a plan exists."""
    if plan_id:
        from beanie.odm.fields import PydanticObjectId

        from src.research.langchain_agent.models.plan import ResearchPlan

        try:
            plan = await ResearchPlan.get(PydanticObjectId(plan_id))
        except Exception:
            plan = None
        if plan is not None and plan.tasks:
            return len(plan.tasks)
    return len(mission.stages)


@activity.defn
async def initialize_mission_run(input: MissionWorkflowInput) -> None:
    """Create or refresh the canonical MissionRunDocument and linked plan."""
    from beanie.odm.fields import PydanticObjectId
    from src.research.langchain_agent.models.mission import MissionRunDocument, ResearchMission
    from src.research.langchain_agent.models.plan import ResearchPlan
    from src.research.langchain_agent.storage.models import init_research_agent_beanie

    await init_research_agent_beanie()
    mission = ResearchMission.model_validate(input.mission_json)
    expected_tc = await _expected_task_count_for_mission(mission, input.plan_id)

    existing = await MissionRunDocument.find_one({"mission_id": mission.mission_id})
    if existing is None:
        doc = MissionRunDocument(
            mission_id=mission.mission_id,
            mission_name=mission.mission_name or mission.mission_id,
            research_plan_id=input.plan_id,
            thread_id=input.thread_id,
            workflow_id=input.workflow_id,
            mission_type="iterative" if any(stage.iterative_config for stage in mission.stages) else "stage_based",
            base_domain=mission.base_domain,
            targets=sorted({target for stage in mission.stages for target in stage.slice_input.targets}),
            max_cycles=max(
                (stage.iterative_config.max_iterations for stage in mission.stages if stage.iterative_config),
                default=None,
            ),
            expected_task_count=expected_tc,
        )
        await doc.insert()
    else:
        existing.mission_name = mission.mission_name or mission.mission_id
        existing.research_plan_id = input.plan_id
        existing.thread_id = input.thread_id
        existing.workflow_id = input.workflow_id
        existing.base_domain = mission.base_domain
        existing.targets = sorted({target for stage in mission.stages for target in stage.slice_input.targets})
        existing.expected_task_count = expected_tc
        existing.status = "running"
        existing.error = None
        existing.updated_at = datetime.now(timezone.utc)
        await existing.save()

    if input.plan_id:
        plan = await ResearchPlan.get(PydanticObjectId(input.plan_id))
        if plan is not None:
            plan.mission_id = mission.mission_id
            plan.workflow_id = input.workflow_id
            plan.mission_status = "running"
            plan.status = "executing"
            plan.updated_at = datetime.utcnow()
            await plan.save()


@activity.defn
async def persist_stage_activity_result(input: MissionStagePersistInput) -> None:
    """Append completed stage records to the canonical MissionRunDocument."""
    from src.api.routes.langchain_dtos import build_iterative_stage_record
    from src.research.langchain_agent.models.mission import MissionRunDocument, StageRunRecord
    from src.research.langchain_agent.storage.models import init_research_agent_beanie

    await init_research_agent_beanie()
    doc = await MissionRunDocument.find_one({"mission_id": input.mission_id})
    if doc is None:
        raise ValueError(f"MissionRunDocument not found for mission_id={input.mission_id}")

    stage_records = [StageRunRecord.model_validate(record) for record in input.stage_run_records]
    if not stage_records:
        doc.updated_at = datetime.now(timezone.utc)
        await doc.save()
        return

    if len(stage_records) == 1 and stage_records[0].iteration is None:
        doc.append_stage(stage_records[0])
    else:
        doc.append_iterative_stage(build_iterative_stage_record(input.task_slug, stage_records))
        doc.cycles_completed += len(stage_records)

    await doc.save()


@activity.defn
async def finalize_mission_run(input: MissionFinalizeInput) -> None:
    """Mark the mission run and linked plan as completed or failed."""
    from beanie.odm.fields import PydanticObjectId
    from src.research.langchain_agent.models.mission import MissionRunDocument
    from src.research.langchain_agent.models.plan import ResearchPlan
    from src.research.langchain_agent.storage.models import init_research_agent_beanie

    await init_research_agent_beanie()
    doc = await MissionRunDocument.find_one({"mission_id": input.mission_id})
    if doc is not None:
        if input.status == "failed":
            doc.mark_failed(input.error or "Mission failed")
        else:
            doc.mark_completed()
            if input.status != "completed":
                doc.status = input.status
        await doc.save()

    if input.plan_id:
        plan = await ResearchPlan.get(PydanticObjectId(input.plan_id))
        if plan is not None:
            plan.status = "failed" if input.status == "failed" else "complete"
            plan.mission_status = input.status
            plan.updated_at = datetime.utcnow()
            await plan.save()


# ---------------------------------------------------------------------------
# Activity 1: execute a research stage
# ---------------------------------------------------------------------------


@activity.defn
async def execute_research_stage(input: StageActivityInput) -> StageActivityOutput:
    """Run a single research stage (single-pass or iterative).

    Initializes LangGraph persistence inside the activity so that
    non-serializable objects never cross the Temporal boundary.
    """
    from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
    from src.research.langchain_agent.memory.langmem_manager import build_langmem_manager
    from src.research.langchain_agent.storage.langgraph_persistence import get_persistence
    from src.research.langchain_agent.storage.models import init_research_agent_beanie
    from src.research.langchain_agent.workflow.run_iterative_stage import run_iterative_stage
    from src.research.langchain_agent.workflow.run_slice import run_single_mission_slice

    stage = _reconstruct_stage(input.stage_json)
    slug = stage.slice_input.task_slug
    started_at = datetime.now(timezone.utc)

    activity.logger.info("Starting research stage: %s (mission=%s)", slug, input.mission_id)
    await _emit_progress(
        input.mission_id,
        "task_started",
        {
            "task_id": stage.slice_input.task_id,
            "task_name": slug,
            "task_slug": slug,
            "stage_type": stage.slice_input.stage_type,
        },
    )

    try:
        store, checkpointer = await get_persistence()
        await init_research_agent_beanie()
        memory_manager = await build_langmem_manager(store=store)
        progress_callback = create_progress_callback(input.mission_id)

        root = Path(input.root_filesystem) if input.root_filesystem else ROOT_FILESYSTEM
        snapshot_dir = Path(input.snapshot_output_dir) if input.snapshot_output_dir else None

        run_input = stage.slice_input.model_copy(deep=True)
        run_input.dependency_reports = input.dependency_reports

        manifest_path = ""
        stage_run_records: list[dict] = []
        if stage.iterative_config is not None:
            activity.logger.info("Running iterative stage: %s", slug)
            iter_result = await run_iterative_stage(
                stage,
                dependency_reports=input.dependency_reports,
                store=store,
                checkpointer=checkpointer,
                memory_manager=memory_manager,
                root_filesystem=root,
                snapshot_output_dir=snapshot_dir,
                progress_callback=progress_callback,
            )
            report_text = iter_result.combined_report
            stage_run_records = [
                record.model_dump(mode="json")
                for record in (
                    output.get("stage_run_record")
                    for output in iter_result.iteration_outputs
                )
                if record is not None
            ]
            manifest_path = next(
                (
                    output.get("stage_candidate_manifest_path", "")
                    for output in reversed(iter_result.iteration_outputs)
                    if output.get("stage_candidate_manifest_path")
                ),
                "",
            )
        else:
            activity.logger.info("Running single-pass stage: %s", slug)
            out = await run_single_mission_slice(
                run_input=run_input,
                prompt_spec=stage.prompt_spec,
                store=store,
                checkpointer=checkpointer,
                memory_manager=memory_manager,
                execution_reminders=stage.execution_reminders,
                root_filesystem=root,
                snapshot_output_dir=snapshot_dir,
                progress_callback=progress_callback,
            )
            report_text = out.get("final_report_text") or ""
            inner_result = out.get("result") or out.get("agent_state") or {}
            manifest_path = (
                inner_result.get("stage_candidate_manifest_path", "")
                or out.get("stage_candidate_manifest_path", "")
            )
            activity.logger.info(
                "Stage %s: report_len=%d, manifest_path=%s, out_keys=%s",
                slug, len(report_text), manifest_path, list(out.keys()),
            )
            if out.get("stage_run_record") is not None:
                stage_run_records = [out["stage_run_record"].model_dump(mode="json")]

        if not report_text:
            report_path = root / "reports" / f"{slug}.md"
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8")
                activity.logger.info(
                    "Stage %s: read report from disk fallback (%d chars)", slug, len(report_text),
                )

        activity.logger.info(
            "Stage %s completed — report length: %d chars", slug, len(report_text),
        )
        completed_at = datetime.now(timezone.utc)
        await _emit_progress(
            input.mission_id,
            "task_completed",
            {
                "task_id": stage.slice_input.task_id,
                "task_name": slug,
                "task_slug": slug,
                "stage_type": stage.slice_input.stage_type,
                "duration_seconds": max(0.0, (completed_at - started_at).total_seconds()),
                "artifact_count": sum(
                    1
                    for record in stage_run_records
                    for artifact in (
                        ([record["artifacts"]["final_report"]] if record.get("artifacts", {}).get("final_report") else [])
                        + record.get("artifacts", {}).get("intermediate_files", [])
                        + ([record["artifacts"]["memory_report_json"]] if record.get("artifacts", {}).get("memory_report_json") else [])
                        + ([record["artifacts"]["agent_state_json"]] if record.get("artifacts", {}).get("agent_state_json") else [])
                    )
                    if artifact
                ),
            },
        )
        return StageActivityOutput(
            task_slug=slug,
            final_report_text=report_text,
            status="completed",
            stage_candidate_manifest_path=manifest_path,
            stage_run_records=stage_run_records,
        )

    except Exception as exc:
        activity.logger.error("Stage %s failed: %s", slug, exc, exc_info=True)
        await _emit_progress(
            input.mission_id,
            "task_failed",
            {
                "task_id": stage.slice_input.task_id,
                "task_name": slug,
                "task_slug": slug,
                "stage_type": stage.slice_input.stage_type,
                "error": str(exc),
            },
        )
        return StageActivityOutput(
            task_slug=slug,
            final_report_text="",
            status="failed",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Activity 2: KG ingestion from a completed report
# ---------------------------------------------------------------------------


@activity.defn
async def ingest_kg_from_report(input: KGIngestionInput) -> KGIngestionOutput:
    """Run KG ingestion on a single completed stage report."""
    from src.research.langchain_agent.kg.extraction_models import TemporalScope
    from src.research.langchain_agent.kg.run_kg_ingestion import run_kg_ingestion
    from src.research.langchain_agent.kg.schema_selector import load_schema_index
    from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient, Neo4jAuraSettings

    activity.logger.info(
        "Starting KG ingestion for report: %s (targets=%s)",
        input.source_report,
        input.targets,
    )

    try:
        from datetime import datetime as dt, timezone as tz

        research_date = None
        if input.research_date:
            research_date = dt.fromisoformat(input.research_date).replace(tzinfo=tz.utc)

        temporal_scope = None
        if input.temporal_scope:
            temporal_scope = TemporalScope(
                mode=input.temporal_scope,
                description=f"Temporal scope: {input.temporal_scope}",
            )

        schema_index = load_schema_index()

        settings = Neo4jAuraSettings.from_env()
        async with Neo4jAuraClient(settings) as client:
            result = await run_kg_ingestion(
                report_text=input.report_text,
                source_report=input.source_report,
                targets=input.targets,
                stage_type=input.stage_type,
                neo4j_client=client,
                context=input.context or f"Research targets: {', '.join(input.targets)}",
                schema_index=schema_index,
                research_date=research_date,
                temporal_scope=temporal_scope,
            )

        activity.logger.info(
            "KG ingestion complete for %s — nodes=%d, rels=%d, states=%d",
            input.source_report,
            result["total_nodes"],
            result["total_rels_written"],
            result["states_created"],
        )
        return KGIngestionOutput(
            source_report=input.source_report,
            total_nodes=result["total_nodes"],
            total_rels_written=result["total_rels_written"],
            states_created=result["states_created"],
            status="completed",
        )

    except Exception as exc:
        activity.logger.error(
            "KG ingestion failed for %s: %s", input.source_report, exc, exc_info=True,
        )
        return KGIngestionOutput(
            source_report=input.source_report,
            status="failed",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Activity 3: unstructured document ingestion
# ---------------------------------------------------------------------------


@activity.defn
async def ingest_unstructured_documents(input: UnstructuredIngestionInput) -> UnstructuredIngestionOutput:
    """Gather mission candidates and run unstructured ingestion for each.

    Performs candidate manifest gathering, document parsing (Docling/LlamaParse),
    schema-based entity extraction, and Neo4j writes within a single activity.
    """
    import json as _json
    from pathlib import Path as _Path

    from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
    from src.research.langchain_agent.models.mission import ResearchMission
    from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient, Neo4jAuraSettings
    from src.research.langchain_agent.unstructured.candidate_collection import gather_mission_candidates
    from src.research.langchain_agent.unstructured.models import CandidateDocument, UnstructuredIngestionConfig
    from src.research.langchain_agent.unstructured.paths import mission_unstructured_dir
    from src.research.langchain_agent.unstructured.run_unstructured_ingestion import run_unstructured_ingestion

    activity.logger.info(
        "Starting unstructured ingestion — %d stage manifests",
        len(input.stage_manifest_paths),
    )

    if not input.stage_manifest_paths:
        return UnstructuredIngestionOutput(status="skipped", candidates_found=0)

    try:
        mission_data = input.mission_json
        mission_id = mission_data.get("mission_id", "unknown")
        root = ROOT_FILESYSTEM

        ui_raw = mission_data.get("unstructured_ingestion", {})
        ui_config = UnstructuredIngestionConfig(**ui_raw) if ui_raw else UnstructuredIngestionConfig()

        if not ui_config.enabled:
            return UnstructuredIngestionOutput(status="skipped", candidates_found=0)

        mission_manifest, mission_manifest_path = gather_mission_candidates(
            mission_id=mission_id,
            stage_manifest_paths=input.stage_manifest_paths,
            root=root,
        )

        candidates = mission_manifest.candidates
        activity.logger.info(
            "Mission candidate manifest: %d candidates from %d stages",
            len(candidates),
            len(input.stage_manifest_paths),
        )

        if not candidates:
            return UnstructuredIngestionOutput(status="completed", candidates_found=0)

        output_base = mission_unstructured_dir(mission_id, root=root) / "executions"
        output_base.mkdir(parents=True, exist_ok=True)

        settings = Neo4jAuraSettings.from_env()
        ingested = 0
        failed = 0

        async with Neo4jAuraClient(settings) as client:
            for candidate_payload in candidates:
                if not candidate_payload.local_path:
                    failed += 1
                    continue

                candidate_dir = output_base / candidate_payload.candidate_id
                try:
                    result = await run_unstructured_ingestion(
                        candidate=candidate_payload,
                        output_dir=candidate_dir,
                        neo4j_client=client,
                        config=ui_config,
                    )
                    ingested += 1
                    activity.logger.info(
                        "Ingested candidate %s — doc_id=%s, chunks=%d, rels=%d",
                        candidate_payload.candidate_id,
                        result.document.document_id,
                        len(result.chunks),
                        len(result.relationship_decisions),
                    )
                except Exception as exc:
                    failed += 1
                    activity.logger.error(
                        "Failed to ingest candidate %s: %s",
                        candidate_payload.candidate_id,
                        exc,
                        exc_info=True,
                    )

        summary = {
            "mission_id": mission_id,
            "candidates_found": len(candidates),
            "candidates_ingested": ingested,
            "candidates_failed": failed,
        }
        summary_path = mission_unstructured_dir(mission_id, root=root) / "mission_unstructured_summary.json"
        summary_path.write_text(_json.dumps(summary, indent=2, default=str), encoding="utf-8")

        return UnstructuredIngestionOutput(
            status="completed",
            candidates_found=len(candidates),
            candidates_ingested=ingested,
            candidates_failed=failed,
        )

    except Exception as exc:
        activity.logger.error("Unstructured ingestion failed: %s", exc, exc_info=True)
        return UnstructuredIngestionOutput(
            status="failed",
            error=str(exc),
        )
