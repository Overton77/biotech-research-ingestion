"""Task Executor — resolves inputs, invokes the Deep Agent, normalizes results.

The only agentic module: compiles a TaskDef into a live agent, invokes it,
and returns a structured TaskResult regardless of success or failure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime  
from langchain.chat_models import BaseChatModel 
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from src.research.compiler.agent_compiler import RuntimeContext, compile_main_task_agent
from src.research.models.mission import (
    ArtifactRef,
    FileReference,
    InputBinding,
    ResearchEvent,
    TaskDef,
    TaskExecutionStructuredOutput,
    TaskResult,
)
from src.research.runtime.backends import subagent_root, task_root

logger = logging.getLogger(__name__)


class InputResolutionError(Exception):
    """A required input binding could not be resolved."""


async def _resolve_inputs(
    bindings: dict[str, InputBinding],
    task_outputs: dict[str, dict[str, Any]],
    model: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Resolve input bindings from prior task outputs.

    Supports max_tokens truncation and LLM-based summarization (transform="summarize").
    """
    resolved: dict[str, Any] = {}
    for local_name, binding in bindings.items():
        source_data = task_outputs.get(binding.source_task_id, {})
        value = source_data.get(binding.source_key)
        if value is None and binding.required:
            raise InputResolutionError(
                f"Required input '{local_name}' could not be resolved from "
                f"task '{binding.source_task_id}' key '{binding.source_key}'"
            )

        if binding.max_tokens and value is not None:
            text = str(value)
            char_limit = binding.max_tokens * 4  # rough chars-to-tokens ratio
            if len(text) > char_limit:
                if binding.transform == "summarize" and model:
                    try:
                        response = await model.ainvoke([
                            {"role": "system", "content": "Summarize the following for a downstream research task. Be concise but preserve key facts."},
                            {"role": "user", "content": text[:char_limit * 2]},
                        ])
                        value = response.content
                    except Exception:
                        logger.warning("LLM summarization failed for input '%s', falling back to truncation", local_name)
                        value = text[:char_limit]
                else:
                    value = text[:char_limit]

        resolved[local_name] = value
    return resolved


def _build_invocation_message(
    task_def: TaskDef,
    resolved_inputs: dict[str, Any],
    global_context: dict[str, Any] | None = None,
) -> str:
    """Build the user message to send to the task agent."""
    parts: list[str] = []

    parts.append(f"# Task: {task_def.name}")
    parts.append(f"\n## Description\n{task_def.description}")

    if task_def.acceptance_criteria:
        parts.append("\n## Acceptance Criteria")
        for i, criterion in enumerate(task_def.acceptance_criteria, 1):
            parts.append(f"{i}. {criterion}")

    if resolved_inputs:
        parts.append("\n## Inputs from Prior Tasks")
        for key, value in resolved_inputs.items():
            if isinstance(value, str) and len(value) > 2000:
                parts.append(f"\n### {key}\n{value[:2000]}...\n[truncated — full content available in filesystem]")
            elif isinstance(value, (dict, list)):
                serialized = json.dumps(value, indent=2, default=str)
                if len(serialized) > 2000:
                    parts.append(f"\n### {key}\n```json\n{serialized[:2000]}...\n```\n[truncated]")
                else:
                    parts.append(f"\n### {key}\n```json\n{serialized}\n```")
            else:
                parts.append(f"\n### {key}\n{value}")

    if global_context:
        parts.append("\n## Global Research Context")
        parts.append(json.dumps(global_context, indent=2, default=str))

    parts.append(
        "\n## Instructions\n"
        "Complete this task thoroughly. Use your subagents for specialized work. "
        "Write key outputs to /outputs/ in your workspace. "
        "Your final message should contain a concise summary of what you accomplished and key findings."
    )

    return "\n".join(parts)


def _collect_artifacts(mission_id: str, task_id: str) -> list[ArtifactRef]:
    """Scan the task's outputs/ directory for artifacts produced by the agent."""
    outputs_dir = task_root(mission_id, task_id) / "outputs"
    if not outputs_dir.exists():
        return []

    artifacts: list[ArtifactRef] = []
    for filepath in sorted(outputs_dir.iterdir()):
        if not filepath.is_file():
            continue

        suffix = filepath.suffix.lower()
        type_map = {
            ".md": ("report", "text/markdown"),
            ".json": ("json", "application/json"),
            ".txt": ("document", "text/plain"),
            ".log": ("log", "text/plain"),
        }
        artifact_type, content_type = type_map.get(suffix, ("document", "text/plain"))

        artifacts.append(ArtifactRef(
            task_id=task_id,
            name=filepath.name,
            artifact_type=artifact_type,
            storage="filesystem",
            path=str(filepath),
            content_type=content_type,
        ))

    return artifacts


def _build_structured_summary_from_workspace(
    task_def: TaskDef,
    mission_id: str,
    task_id: str,
) -> TaskExecutionStructuredOutput:
    """
    Build TaskExecutionStructuredOutput from the task workspace (fallback when
    the deep agent does not return structured_response or it is incomplete).
    """
    subagent_locations: list[FileReference] = []
    for sub in task_def.compiled_subagents:
        if not sub.expected_output_path:
            continue
        root = subagent_root(mission_id, task_id, sub.name)
        # expected_output_path is relative to subagent workspace (e.g. outputs/foo.md)
        full_path = (root / sub.expected_output_path).resolve()
        if full_path.is_file():
            subagent_locations.append(
                FileReference(
                    path=str(full_path),
                    name=full_path.name,
                    description=sub.expected_output_format,
                )
            )

    synthesis_reports: list[FileReference] = []
    outputs_dir = task_root(mission_id, task_id) / "outputs"
    if outputs_dir.exists():
        for filepath in sorted(outputs_dir.iterdir()):
            if filepath.is_file():
                synthesis_reports.append(
                    FileReference(path=str(filepath), name=filepath.name, description=None)
                )

    return TaskExecutionStructuredOutput(
        subagent_final_output_locations=subagent_locations,
        final_synthesis_reports=synthesis_reports,
    )


def _normalize_structured_summary(
    result: dict[str, Any],
    task_def: TaskDef,
    mission_id: str,
    task_id: str,
) -> TaskExecutionStructuredOutput:
    """Extract or build TaskExecutionStructuredOutput from agent result."""
    raw = result.get("structured_response")
    if raw is None:
        return _build_structured_summary_from_workspace(task_def, mission_id, task_id)
    if isinstance(raw, TaskExecutionStructuredOutput):
        summary = raw
    elif isinstance(raw, dict):
        try:
            summary = TaskExecutionStructuredOutput.model_validate(raw)
        except Exception:
            return _build_structured_summary_from_workspace(task_def, mission_id, task_id)
    else:
        return _build_structured_summary_from_workspace(task_def, mission_id, task_id)
    # Use fallback if both lists are empty (incomplete LLM response)
    if not summary.subagent_final_output_locations and not summary.final_synthesis_reports:
        return _build_structured_summary_from_workspace(task_def, mission_id, task_id)
    return summary


async def execute_task(
    task_def: TaskDef,
    task_outputs: dict[str, dict[str, Any]],
    ctx: RuntimeContext,
    global_context: dict[str, Any] | None = None,
) -> TaskResult:
    """
    Full task execution pipeline:
    1. Resolve inputs from task_outputs using task_def.input_bindings
    2. Compile main task agent
    3. Build invocation message from resolved inputs
    4. Invoke agent (async)
    5. Normalize into TaskResult
    """
    started_at = datetime.utcnow()
    events: list[ResearchEvent] = [
        ResearchEvent(
            event_type="task_started",
            task_id=task_def.task_id,
            payload={"name": task_def.name},
        )
    ]

    try:
        # 1. Resolve inputs (async — supports LLM-based summarization)
        summarize_model = init_chat_model("openai:gpt-5-mini") if any(
            b.transform == "summarize" for b in task_def.input_bindings.values()
        ) else None
        resolved_inputs = await _resolve_inputs(
            task_def.input_bindings, task_outputs, model=summarize_model,
        )

        # 2. Compile agent
        agent = await compile_main_task_agent(task_def, ctx)

        # 3. Build invocation message
        message = _build_invocation_message(task_def, resolved_inputs, global_context)

        # 4. Invoke agent
        logger.info("Executing task '%s' (%s)", task_def.task_id, task_def.name)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": f"task-{ctx.task_id}"}},
        )

        # 5. Extract the last assistant message
        messages = result.get("messages", [])
        last_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                last_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        # 6. Collect artifacts from the filesystem
        artifacts = _collect_artifacts(ctx.mission_id, ctx.task_id)

        # 7. Structured execution summary (from agent response or workspace fallback)
        structured_summary = _normalize_structured_summary(
            result, task_def, ctx.mission_id, ctx.task_id
        )

        completed_at = datetime.utcnow()
        events.append(ResearchEvent(
            event_type="task_completed",
            task_id=task_def.task_id,
            payload={
                "name": task_def.name,
                "duration_seconds": (completed_at - started_at).total_seconds(),
                "artifact_count": len(artifacts),
            },
        ))

        return TaskResult(
            task_id=task_def.task_id,
            status="completed",
            outputs={
                "response": last_content,
                "files": [a.name for a in artifacts],
            },
            artifacts=artifacts,
            events=events,
            started_at=started_at,
            completed_at=completed_at,
            structured_execution_summary=structured_summary,
        )

    except Exception as e:
        completed_at = datetime.utcnow()
        logger.exception("Task '%s' failed: %s", task_def.task_id, e)

        events.append(ResearchEvent(
            event_type="task_failed",
            task_id=task_def.task_id,
            payload={"error": str(e)[:500]},
        ))

        return TaskResult(
            task_id=task_def.task_id,
            status="failed",
            error_message=str(e)[:2000],
            events=events,
            started_at=started_at,
            completed_at=completed_at,
            structured_execution_summary=None,
        )
