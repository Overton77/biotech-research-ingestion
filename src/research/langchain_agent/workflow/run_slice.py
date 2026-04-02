"""
Run one bounded mission slice: memory recall, agent run, memory ingestion, persistence.
Supports dependency_reports on MissionSliceInput (injected into user message).

Post-run: uploads all artifacts to S3 and returns a StageRunRecord for the caller
(run_mission.py) to embed into the MissionRunDocument. S3/MongoDB failures never
raise — they are logged and the research output is always returned regardless.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.research.langchain_agent.agent.config import (
    ROOT_FILESYSTEM,
    MissionSliceInput,
    ResearchPromptSpec,
    ResearchTaskMemoryReport,
    build_memory_ingestion_prompt,
    ensure_dirs,
    input_to_agent_state,
    load_memories_for_prompt,
    read_file_text,
)
from src.research.langchain_agent.agent.config import _safe_json_dumps
from src.research.langchain_agent.agent.factory import (
    build_memory_report_agent,
    build_research_agent,
)
from src.research.langchain_agent.observability.tracing import (
    get_current_trace_id,
    inject_stage_metadata,
    traced_memory_ingestion,
    traced_memory_recall,
    traced_stage,
)
from src.research.langchain_agent.storage.models import StageRunRecord
from src.research.langchain_agent.storage.s3_store import persist_slice_artifacts 
from src.research.langchain_agent.unstructured.candidate_collection import (
    build_stage_candidate_manifest,
)
from src.research.langchain_agent.utils.serialize import write_graph_state_snapshots


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State snapshot writer — dumps full graph state to filesystem for inspection
# ---------------------------------------------------------------------------





def _build_user_message(run_input: MissionSliceInput) -> str:
    """Build the initial user message; include dependency reports, temporal context, and report template."""
    parts = []

    # Temporal context header
    parts.append(f"Current date: {run_input.effective_current_date}")
    parts.append(f"Research date: {run_input.effective_research_date}")
    if run_input.temporal_scope.mode != "current":
        parts.append(f"Temporal scope: {run_input.temporal_scope.mode} — {run_input.temporal_scope.description}")
    parts.append("")

    if run_input.dependency_reports:
        parts.append("Reports from prior stages (use for context):")
        parts.append("")
        for slug, text in run_input.dependency_reports.items():
            parts.append(f"## Report: {slug}")
            parts.append("")
            parts.append(text.strip())
            parts.append("")
            parts.append("---")
            parts.append("")
        parts.append("")
        parts.append("Your objective for this stage:")
        parts.append("")

    parts.append(run_input.user_objective)
    parts.append("")
    parts.append(
        "IMPORTANT: When reporting facts, be explicit about temporal context. "
        "State when facts were verified, note dates for roles/positions, and "
        "flag anything that may be outdated or time-sensitive."
    )

    if run_input.report_required_sections:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("MANDATORY REPORT STRUCTURE — your final report MUST use this exact skeleton:")
        parts.append("")
        parts.append(f"# {run_input.task_slug.replace('-', ' ').title()}")
        parts.append("")
        for section in run_input.report_required_sections:
            parts.append(f"## {section}")
            if section == "Sources":
                parts.append("")
                parts.append("(List every URL you used as markdown links: `- [Title](url) — description`)")
            elif section == "Executive Summary":
                parts.append("")
                parts.append("(2-3 paragraph overview of key findings)")
            else:
                parts.append("")
                parts.append(f"(Content for {section})")
            parts.append("")
        parts.append("Every ## heading above MUST appear in your final report. Missing sections = evaluation failure.")
        parts.append("The ## Sources section MUST contain actual URLs from your research, not placeholders.")

    if run_input.guidance_notes:
        parts.append("")
        parts.append("Guidance notes from prior iterations:")
        for note in run_input.guidance_notes:
            parts.append(f"- {note}")

    return "\n".join(parts).strip()


@traced_stage
async def run_single_mission_slice(
    run_input: MissionSliceInput,
    prompt_spec: ResearchPromptSpec,
    *,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any],
    memory_manager: Any,
    execution_reminders: Sequence[str] | None = None,
    root_filesystem: Path | None = None,
    iteration: int | None = None,
    snapshot_output_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run one bounded mission slice: pre-run memory recall, agent execution,
    memory ingestion, memory persistence, and artifact upload.

    If run_input.dependency_reports is set, those reports are included in the
    user message so the agent can use prior stages' outputs.

    iteration — cycle number for iterative missions (1-based); None for stage-based.

    Returns a dict with keys:
      result, final_agent_response, final_report_text,
      structured_memory_report, memory_updates, agent_state, file_results,
      stage_run_record, langsmith_run_id
    """
    inject_stage_metadata(
        mission_id=run_input.mission_id,
        task_slug=run_input.task_slug,
        stage_type=run_input.stage_type,
        iteration=iteration,
        targets=run_input.targets,
    )

    root = root_filesystem or ROOT_FILESYSTEM
    await ensure_dirs()

    reminders = execution_reminders or [
        "Use runs/, reports/, scratch/ as main folders.",
        "Save intermediate findings; write final report to reports/.",
        "Use recalled memories as hints, not unquestioned truth.",
    ]

    run_thread_id = f"{run_input.task_id}-{uuid.uuid4().hex[:8]}"
    runtime_config: RunnableConfig = {
        "configurable": {
            "thread_id": run_thread_id,
            "mission_id": run_input.mission_id,
        },
        "recursion_limit": max(40, run_input.max_step_budget * 8),
    }

    recalled = await _traced_memory_recall(
        manager=memory_manager,
        run_input=run_input,
        config=runtime_config,
    )

    agent_state = input_to_agent_state(run_input)
    agent_state.update(recalled)

    research_agent = await build_research_agent(
        prompt_spec=prompt_spec,
        execution_reminders=reminders,
        selected_tool_names=run_input.selected_tool_names,
        selected_subagent_names=run_input.selected_subagent_names,
        store=store,
        checkpointer=checkpointer,
    )

    user_message = _build_user_message(run_input)
    result = await research_agent.ainvoke(
        {
            "messages": [{"role": "user", "content": user_message}],
            **agent_state,
        },
        config=runtime_config,
    )  

    
    stage_candidate_manifest, stage_candidate_manifest_path = await build_stage_candidate_manifest(
        run_input=run_input,
        result=result,
        root=root,
    )
    written_file_paths = list(result.get("written_file_paths", []) or [])
    if stage_candidate_manifest_path not in written_file_paths:
        written_file_paths.append(stage_candidate_manifest_path)
    result["written_file_paths"] = written_file_paths
    result["stage_candidate_manifest"] = stage_candidate_manifest.model_dump(mode="json")
    result["stage_candidate_manifest_path"] = stage_candidate_manifest_path




    write_graph_state_snapshots(
        output_dir=snapshot_output_dir,
        task_slug=run_input.task_slug,
        iteration=iteration,
        agent_input=agent_state,
        result=result,
        recalled_memories=recalled,
        user_message=user_message,
        run_thread_id=run_thread_id,
    )

    final_agent_response = result["messages"][-1].content if result.get("messages") else ""
    final_report_text = await read_file_text(agent_state["report_path"], root=root)

    memory_prompt = build_memory_ingestion_prompt(
        run_input=run_input,
        final_agent_response=str(final_agent_response),
        final_report_text=final_report_text,
        final_report_path=agent_state["report_path"],
        visited_urls=result.get("visited_urls", []),
        tavily_search_events=result.get("tavily_search_events", []),
        tavily_extract_events=result.get("tavily_extract_events", []),
        tavily_map_events=result.get("tavily_map_events", []),
        tavily_crawl_events=result.get("tavily_crawl_events", []),
        filesystem_events=result.get("filesystem_events", []),
        read_file_paths=result.get("read_file_paths", []),
        written_file_paths=result.get("written_file_paths", []),
        edited_file_paths=result.get("edited_file_paths", []),
    )

    memory_report_agent = build_memory_report_agent(store=store, checkpointer=checkpointer)
    memory_report_result = await memory_report_agent.ainvoke(
        {"messages": [{"role": "user", "content": memory_prompt}]},
        config=runtime_config,
    )

    structured_memory_report: ResearchTaskMemoryReport = memory_report_result["structured_response"]

    logger.info("Structured memory report: %s", structured_memory_report)

    memory_updates = await _traced_memory_ingestion(
        memory_manager=memory_manager,
        run_input=run_input,
        final_agent_response=final_agent_response,
        agent_state=agent_state,
        final_report_text=final_report_text,
        structured_memory_report=structured_memory_report,
        runtime_config=runtime_config,
    )

    logger.info("Memory updates: %s", memory_updates)

    stage_run_record = await _build_stage_run_record(
        run_input=run_input,
        agent_state=agent_state,
        result=result,
        final_report_text=final_report_text,
        structured_memory_report=structured_memory_report,
        iteration=iteration,
        root=root,
    )

    stage_run_record.langsmith_run_id = get_current_trace_id()

    return {
        "result": result,
        "final_agent_response": final_agent_response,
        "final_report_text": final_report_text,
        "structured_memory_report": structured_memory_report,
        "memory_updates": memory_updates,
        "agent_state": result,
        "file_results": result.get("file_results", []),
        "stage_run_record": stage_run_record,
        "langsmith_run_id": stage_run_record.langsmith_run_id,
    }


@traced_memory_recall
async def _traced_memory_recall(
    *,
    manager: Any,
    run_input: MissionSliceInput,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Traced wrapper around memory recall for LangSmith visibility."""
    return await load_memories_for_prompt(
        manager=manager,
        run_input=run_input,
        config=config,
    )


@traced_memory_ingestion
async def _traced_memory_ingestion(
    *,
    memory_manager: Any,
    run_input: MissionSliceInput,
    final_agent_response: str,
    agent_state: Dict[str, Any],
    final_report_text: str,
    structured_memory_report: ResearchTaskMemoryReport,
    runtime_config: RunnableConfig,
) -> List[Dict[str, Any]]:
    """Traced wrapper around memory persistence for LangSmith visibility."""
    memory_messages = [
        {
            "role": "user",
            "content": (
                f"Mission objective:\n{run_input.user_objective}\n\n"
                f"Targets:\n{', '.join(run_input.targets)}\n\n"
                f"Task slug:\n{run_input.task_slug}\n\n"
                f"Structured memory report request:\n"
                f"{_safe_json_dumps(structured_memory_report.model_dump())}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"Final agent response:\n{final_agent_response}\n\n"
                f"Final report path:\n{agent_state['report_path']}\n\n"
                f"Final report text:\n{final_report_text}"
            ),
        },
    ]

    return await memory_manager.ainvoke(
        {"messages": memory_messages},
        config=runtime_config,
    )


async def _build_stage_run_record(
    *,
    run_input: MissionSliceInput,
    agent_state: Dict[str, Any],
    result: Dict[str, Any],
    final_report_text: str,
    structured_memory_report: ResearchTaskMemoryReport,
    iteration: int | None,
    root: Path,
) -> StageRunRecord:
    """
    Upload all artifacts to S3 concurrently and assemble a StageRunRecord.
    Wrapped in a broad try/except so persistence failures never surface as
    exceptions to the caller.
    """
    started_at = agent_state.get("_started_at") or datetime.now(timezone.utc)

    try:
        artifacts, memory_record = await persist_slice_artifacts(
            run_input=run_input,
            final_report_text=final_report_text,
            written_file_paths=result.get("written_file_paths", []),
            agent_state=result,  # use final state so visited_urls/written_file_paths are populated in S3 snapshot
            structured_memory_report=structured_memory_report,
            report_local_path=result.get("report_path") or agent_state.get("report_path", ""),
            iteration=iteration,
            root=root,
        )
    except Exception:
        logger.exception(
            "persist_slice_artifacts failed for %s — StageRunRecord will have empty artifacts",
            run_input.task_slug,
        )
        from src.research.langchain_agent.storage.models import (
            StageArtifacts,
            MemoryReportRecord,
        )
        artifacts = StageArtifacts()
        memory_record = MemoryReportRecord(
            mission_id=run_input.mission_id,
            summary=structured_memory_report.summary,
            file_paths=structured_memory_report.file_paths,
        )

    return StageRunRecord(
        task_id=run_input.task_id,
        task_slug=run_input.task_slug,
        stage_type=run_input.stage_type,
        targets=run_input.targets,
        dependencies=list(run_input.dependency_reports.keys()),
        iteration=iteration,
        status="completed",
        final_report_text=final_report_text,
        artifacts=artifacts,
        memory_report=memory_record,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
