"""
Iteration control logic for stage-level iterative research missions.

Pure-function evaluator (no LLM calls) and context-builder for carrying
forward state between iterations.
"""

from __future__ import annotations

from typing import List

from src.research.langchain_agent.agent.config import (
    MissionSliceInput,
    NextStepsArtifact,
    _truncate_text,
)
from src.research.langchain_agent.models.mission import IterativeStageConfig


# ---------------------------------------------------------------------------
# Stop-condition evaluator
# ---------------------------------------------------------------------------


def evaluate_stop_condition(
    next_steps: NextStepsArtifact,
    iteration: int,
    config: IterativeStageConfig,
) -> tuple[bool, str]:
    """Decide whether to stop iterating.

    Returns ``(should_stop, reason)`` where *reason* is a short machine-readable
    label suitable for logging and persistence.
    """
    if iteration >= config.max_iterations:
        return True, "max_iterations_reached"
    if next_steps.stage_complete:
        return True, "stage_signaled_complete"
    if config.stop_on_no_next_steps and not next_steps.open_questions:
        return True, "no_material_next_steps"
    if next_steps.confidence >= 0.9:
        return True, "high_confidence_threshold"
    return False, "continue"


# ---------------------------------------------------------------------------
# Context builder for the next iteration
# ---------------------------------------------------------------------------


def build_iteration_context(
    original_input: MissionSliceInput,
    prior_reports: List[str],
    next_steps: NextStepsArtifact,
    iteration: int,
    config: IterativeStageConfig,
) -> MissionSliceInput:
    """Build a new ``MissionSliceInput`` for the next iteration.

    The new input carries forward:
    - the original targets, tools, and stage_type
    - a refined ``user_objective`` that includes the suggested focus
    - ``guidance_notes`` with summaries of prior iteration reports
    - ``dependency_reports`` from the original input (inter-stage deps)
    """
    recent_reports = prior_reports[-config.carry_forward_reports:]

    prior_context_parts: List[str] = []
    for idx, report_text in enumerate(recent_reports, start=1):
        label = f"Iteration {iteration - len(recent_reports) + idx}"
        truncated = _truncate_text(report_text, max_chars=6000)
        prior_context_parts.append(f"--- {label} report ---\n{truncated}")

    high_priority_qs = [
        q for q in next_steps.open_questions if q.priority == "high"
    ]
    medium_priority_qs = [
        q for q in next_steps.open_questions if q.priority == "medium"
    ]

    focus_lines: List[str] = []
    if next_steps.suggested_focus:
        focus_lines.append(f"Primary focus: {next_steps.suggested_focus}")
    if high_priority_qs:
        focus_lines.append("High-priority open questions:")
        for q in high_priority_qs:
            focus_lines.append(f"  - {q.question}")
    if medium_priority_qs:
        focus_lines.append("Medium-priority open questions:")
        for q in medium_priority_qs:
            focus_lines.append(f"  - {q.question}")

    refined_objective = (
        f"This is iteration {iteration + 1} of an iterative research stage.\n\n"
        f"Original objective:\n{original_input.user_objective}\n\n"
    )
    if focus_lines:
        refined_objective += "Focus for this iteration:\n" + "\n".join(focus_lines) + "\n\n"
    if next_steps.key_findings_this_iteration:
        refined_objective += (
            "Key findings from the previous iteration:\n"
            + "\n".join(f"- {f}" for f in next_steps.key_findings_this_iteration)
            + "\n\n"
        )
    refined_objective += (
        "Build on prior work. Do NOT repeat searches or extractions that have "
        "already been completed. Focus on resolving the open questions above."
    )

    guidance = list(original_input.guidance_notes)
    if prior_context_parts:
        guidance.append(
            "Prior iteration reports (use as context, do not repeat their work):\n\n"
            + "\n\n".join(prior_context_parts)
        )

    return original_input.model_copy(
        update={
            "user_objective": refined_objective,
            "guidance_notes": guidance,
        },
        deep=True,
    )


# ---------------------------------------------------------------------------
# Report synthesis
# ---------------------------------------------------------------------------


def synthesize_iteration_reports(
    reports: List[str],
    task_slug: str,
    user_objective: str,
) -> str:
    """Produce a combined markdown report from all iteration reports.

    This is a deterministic concatenation (no LLM call). The combined report
    is what downstream dependent stages receive via ``dependency_reports``.
    """
    parts = [
        f"# Combined Iterative Report: {task_slug}",
        "",
        f"**Objective:** {user_objective}",
        f"**Iterations completed:** {len(reports)}",
        "",
        "---",
        "",
    ]
    for idx, report_text in enumerate(reports, start=1):
        parts.append(f"## Iteration {idx}")
        parts.append("")
        parts.append(report_text.strip() if report_text else "(no report produced)")
        parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts)
