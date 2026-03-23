"""
Run a single MissionStage iteratively: execute bounded passes in a loop,
extract structured next-steps after each pass, and decide whether to continue.

The iteration loop wraps ``run_single_mission_slice()`` — the research agent
itself is unchanged.  Iteration control is purely a runner-level concern.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.research.langchain_agent.agent.config import (
    MissionSliceInput,
    NextStepsArtifact,
    _truncate_text,
)
from src.research.langchain_agent.agent.factory import build_next_steps_agent
from src.research.langchain_agent.models.mission import (
    IterativeStageConfig,
    MissionStage,
)
from src.research.langchain_agent.observability.tracing import (
    inject_stage_metadata,
    traced_iterative_stage,
    traced_next_steps_extraction,
)
from src.research.langchain_agent.workflow.iteration_control import (
    build_iteration_context,
    evaluate_stop_condition,
    synthesize_iteration_reports,
)
from src.research.langchain_agent.workflow.run_slice import run_single_mission_slice

logger = logging.getLogger(__name__)


@dataclass
class IterativeStageResult:
    """Aggregated result of all iterations for one iterative stage."""

    task_slug: str
    iterations_completed: int = 0
    stop_reason: str = ""
    combined_report: str = ""
    iteration_outputs: List[Dict[str, Any]] = dc_field(default_factory=list)
    next_steps_history: List[Dict[str, Any]] = dc_field(default_factory=list)
    iteration_reports: List[str] = dc_field(default_factory=list)


@traced_next_steps_extraction
async def _extract_next_steps(
    *,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any],
    run_input: MissionSliceInput,
    final_report_text: str,
    final_agent_response: str,
    iteration: int,
    completion_criteria: str,
) -> NextStepsArtifact:
    """Call the next-steps extraction agent and return a NextStepsArtifact."""
    agent = build_next_steps_agent(store=store, checkpointer=checkpointer)

    eval_thread_id = f"next-steps-{run_input.task_slug}-iter{iteration}-{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {
            "thread_id": eval_thread_id,
            "mission_id": run_input.mission_id,
        },
    }

    user_content = (
        f"Iteration: {iteration}\n\n"
        f"Original objective:\n{run_input.user_objective}\n\n"
        f"Completion criteria:\n{completion_criteria or '(none specified — use your judgment)'}\n\n"
        f"Final report for this iteration:\n{_truncate_text(final_report_text, max_chars=10000)}\n\n"
        f"Agent's last response:\n{_truncate_text(str(final_agent_response), max_chars=3000)}"
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    artifact: NextStepsArtifact = result["structured_response"]
    return artifact


@traced_iterative_stage
async def run_iterative_stage(
    stage: MissionStage,
    *,
    dependency_reports: Dict[str, str],
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any],
    memory_manager: Any,
    root_filesystem: Path,
    snapshot_output_dir: Path | None = None,
) -> IterativeStageResult:
    """Execute a stage iteratively until a stop condition is met.

    Each iteration:
    1. Runs ``run_single_mission_slice`` with the current input
    2. Extracts a ``NextStepsArtifact`` via the evaluator agent
    3. Evaluates deterministic stop conditions
    4. If continuing, builds updated context for the next iteration

    Returns an ``IterativeStageResult`` with all per-iteration outputs,
    the combined report, and the stop reason.
    """
    inject_stage_metadata(
        mission_id=stage.slice_input.mission_id,
        task_slug=stage.slice_input.task_slug,
        stage_type=stage.slice_input.stage_type,
        targets=stage.slice_input.targets,
    )

    config = stage.iterative_config
    if config is None:
        raise ValueError(
            f"run_iterative_stage called on non-iterative stage {stage.slice_input.task_slug}"
        )

    result = IterativeStageResult(task_slug=stage.slice_input.task_slug)
    current_input = stage.slice_input.model_copy(deep=True)
    current_input.dependency_reports = dict(dependency_reports)

    for iteration in range(1, config.max_iterations + 1):
        logger.info(
            "=== Iterative stage %s — iteration %d/%d ===",
            stage.slice_input.task_slug, iteration, config.max_iterations,
        )
        print(
            f"\n{'='*60}\n"
            f"  Iterative stage: {stage.slice_input.task_slug}\n"
            f"  Iteration: {iteration}/{config.max_iterations}\n"
            f"{'='*60}\n"
        )

        out = await run_single_mission_slice(
            run_input=current_input,
            prompt_spec=stage.prompt_spec,
            store=store,
            checkpointer=checkpointer,
            memory_manager=memory_manager,
            execution_reminders=stage.execution_reminders,
            root_filesystem=root_filesystem,
            iteration=iteration,
            snapshot_output_dir=snapshot_output_dir,
        )

        result.iteration_outputs.append(out)
        report_text = out.get("final_report_text") or ""
        result.iteration_reports.append(report_text)
        final_agent_response = out.get("final_agent_response") or ""

        next_steps = await _extract_next_steps(
            store=store,
            checkpointer=checkpointer,
            run_input=current_input,
            final_report_text=report_text,
            final_agent_response=final_agent_response,
            iteration=iteration,
            completion_criteria=config.completion_criteria,
        )

        result.next_steps_history.append(next_steps.model_dump(mode="json"))

        logger.info(
            "Iteration %d next-steps: complete=%s confidence=%.2f open_questions=%d",
            iteration,
            next_steps.stage_complete,
            next_steps.confidence,
            len(next_steps.open_questions),
        )
        print(
            f"  Iteration {iteration} result: "
            f"complete={next_steps.stage_complete}, "
            f"confidence={next_steps.confidence:.2f}, "
            f"open_questions={len(next_steps.open_questions)}"
        )
        if next_steps.suggested_focus:
            print(f"  Suggested focus: {next_steps.suggested_focus}")

        should_stop, reason = evaluate_stop_condition(next_steps, iteration, config)

        if should_stop:
            result.iterations_completed = iteration
            result.stop_reason = reason
            logger.info(
                "Iterative stage %s stopping after iteration %d: %s",
                stage.slice_input.task_slug, iteration, reason,
            )
            print(f"  -> Stopping: {reason}")
            break

        current_input = build_iteration_context(
            original_input=stage.slice_input,
            prior_reports=result.iteration_reports,
            next_steps=next_steps,
            iteration=iteration,
            config=config,
        )
        current_input.dependency_reports = dict(dependency_reports)

    else:
        result.iterations_completed = config.max_iterations
        result.stop_reason = "max_iterations_reached"

    result.combined_report = synthesize_iteration_reports(
        reports=result.iteration_reports,
        task_slug=stage.slice_input.task_slug,
        user_objective=stage.slice_input.user_objective,
    )

    return result
