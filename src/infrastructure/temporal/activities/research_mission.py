"""Temporal activities for the research mission workflow.

Two activities:
  1. execute_research_stage — run a single MissionStage (single-pass or iterative)
  2. ingest_kg_from_report  — run KG ingestion on a completed stage report
"""

from __future__ import annotations

import logging
from pathlib import Path

from temporalio import activity

from src.infrastructure.temporal.models import (
    KGIngestionInput,
    KGIngestionOutput,
    StageActivityInput,
    StageActivityOutput,
)

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

    activity.logger.info("Starting research stage: %s (mission=%s)", slug, input.mission_id)

    try:
        store, checkpointer = await get_persistence()
        await init_research_agent_beanie()
        memory_manager = await build_langmem_manager(store=store)

        root = Path(input.root_filesystem) if input.root_filesystem else ROOT_FILESYSTEM
        snapshot_dir = Path(input.snapshot_output_dir) if input.snapshot_output_dir else None

        run_input = stage.slice_input.model_copy(deep=True)
        run_input.dependency_reports = input.dependency_reports

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
            )
            report_text = iter_result.combined_report
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
            )
            report_text = out.get("final_report_text") or ""

        # Fallback: if the slice didn't return report text, read it from disk
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
        return StageActivityOutput(
            task_slug=slug,
            final_report_text=report_text,
            status="completed",
        )

    except Exception as exc:
        activity.logger.error("Stage %s failed: %s", slug, exc, exc_info=True)
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
    from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient, Neo4jAuraSettings

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
