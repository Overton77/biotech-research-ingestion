"""
Evaluator functions for research mission quality assessment.

Two categories:
  - LLM-as-judge evaluators: use an LLM to score subjective quality dimensions
  - Code evaluators: deterministic checks on structure, sources, and efficiency

All evaluators follow the LangSmith convention:
  def evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None) -> dict
  Returns: {"key": str, "score": float, "comment": str}

Fallback pattern for static vs live mode:
  - In live mode: `outputs` contains the agent's freshly produced report.
  - In static mode: `static_target` returns `inputs` (no "report" key), so
    evaluators fall back to `reference_outputs` (the stored dataset outputs)
    to retrieve the report text. This is handled transparently by
    `_extract_report()` and `_extract_visited_urls()`.

Usage with langsmith.evaluate():
  results = await aevaluate(
      target_fn,
      data="biotech-research-reports-v1",
      evaluators=[report_completeness, report_structure, source_quality],
  )
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langsmith import traceable, wrappers

from src.research.langchain_agent.eval.rubrics import (
    DIMINISHING_RETURNS_THRESHOLD,
    MIN_CONFIDENCE_IMPROVEMENT_PER_ITERATION,
    REPORT_ACCURACY_RUBRIC,
    REPORT_COMPLETENESS_RUBRIC,
)


# ---------------------------------------------------------------------------
# Data-extraction helpers (live mode vs static mode)
# ---------------------------------------------------------------------------


def _extract_report(
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]],
) -> str:
    """Return the report text, falling back to reference_outputs in static mode.

    In live mode `outputs` contains the freshly produced report.
    In static mode `static_target` returns the dataset inputs (no "report" key),
    so we fall back to the stored reference_outputs which do contain it.
    """
    return (
        outputs.get("report")
        or outputs.get("final_report_text")
        or (reference_outputs or {}).get("report")
        or (reference_outputs or {}).get("final_report_text", "")
    )


def _extract_visited_urls(
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]],
) -> list:
    """Return visited_urls, falling back to reference_outputs in static mode."""
    urls = outputs.get("visited_urls")
    if urls is None:
        urls = (reference_outputs or {}).get("visited_urls", [])
    return urls or []


# ---------------------------------------------------------------------------
# LLM-as-judge evaluators
# ---------------------------------------------------------------------------


def _get_judge_client():
    """Lazy-init a wrapped OpenAI client for LLM-as-judge evaluators."""
    from openai import OpenAI

    return wrappers.wrap_openai(OpenAI())


@traceable(name="eval:report_completeness", run_type="chain")
def report_completeness(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LLM-as-judge: score report completeness against required sections and depth."""
    client = _get_judge_client()

    report_text = _extract_report(outputs, reference_outputs)
    required_sections = inputs.get("report_required_sections", [])
    objective = inputs.get("user_objective", "")

    prompt = (
        f"{REPORT_COMPLETENESS_RUBRIC}\n\n"
        f"---\n\n"
        f"Research objective:\n{objective}\n\n"
        f"Required sections:\n{json.dumps(required_sections)}\n\n"
        f"Report to evaluate:\n{report_text}"
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        temperature=1, 
        # 0 was not supported with this model 
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a research quality evaluator. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return {"key": "report_completeness", "score": 0.0, "comment": "Failed to parse judge response"}

    score = result.get("total_score", 0) / 10.0
    return {
        "key": "report_completeness",
        "score": score,
        "comment": result.get("reasoning", ""),
    }


@traceable(name="eval:report_accuracy", run_type="chain")
def report_accuracy(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LLM-as-judge: score report factual accuracy and source alignment."""
    client = _get_judge_client()

    report_text = _extract_report(outputs, reference_outputs)
    objective = inputs.get("user_objective", "")
    targets = inputs.get("targets", [])

    prompt = (
        f"{REPORT_ACCURACY_RUBRIC}\n\n"
        f"---\n\n"
        f"Research objective:\n{objective}\n\n"
        f"Research targets:\n{json.dumps(targets)}\n\n"
        f"Report to evaluate:\n{report_text}"
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a research accuracy evaluator. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return {"key": "report_accuracy", "score": 0.0, "comment": "Failed to parse judge response"}

    score = result.get("total_score", 0) / 10.0
    return {
        "key": "report_accuracy",
        "score": score,
        "comment": result.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Code-based evaluators
# ---------------------------------------------------------------------------


def report_structure(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check whether all required sections appear as markdown headers in the report."""
    report_text = _extract_report(outputs, reference_outputs)
    required_sections = inputs.get("report_required_sections", [])

    if not required_sections:
        return {"key": "report_structure", "score": 1.0, "comment": "No required sections specified"}

    report_lower = report_text.lower()
    headers_in_report = set(re.findall(r"^#{1,3}\s+(.+)$", report_text, re.MULTILINE))
    headers_lower = {h.strip().lower() for h in headers_in_report}

    found = []
    missing = []
    for section in required_sections:
        section_lower = section.lower().strip()
        if section_lower in headers_lower or section_lower in report_lower:
            found.append(section)
        else:
            missing.append(section)

    score = len(found) / len(required_sections) if required_sections else 1.0
    comment = f"Found {len(found)}/{len(required_sections)} required sections."
    if missing:
        comment += f" Missing: {', '.join(missing)}"

    return {"key": "report_structure", "score": score, "comment": comment}


def source_quality(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate source diversity and count from the report and metadata."""
    report_text = _extract_report(outputs, reference_outputs)
    visited_urls = _extract_visited_urls(outputs, reference_outputs)

    url_pattern = re.compile(r"https?://[^\s\)\"'>]+")
    urls_in_report = set(url_pattern.findall(report_text))
    all_urls = set(visited_urls) | urls_in_report

    domains = set()
    for url in all_urls:
        match = re.match(r"https?://([^/]+)", url)
        if match:
            domains.add(match.group(1).lower())

    url_count = len(all_urls)
    domain_count = len(domains)

    if url_count == 0:
        score = 0.0
        comment = "No sources found in report or metadata"
    elif url_count < 3:
        score = 0.3
        comment = f"{url_count} sources from {domain_count} domains — minimal"
    elif domain_count < 2:
        score = 0.5
        comment = f"{url_count} sources but only {domain_count} domain — low diversity"
    elif url_count < 6:
        score = 0.7
        comment = f"{url_count} sources from {domain_count} domains — adequate"
    else:
        score = min(1.0, 0.7 + (domain_count * 0.05))
        comment = f"{url_count} sources from {domain_count} domains — good diversity"

    return {"key": "source_quality", "score": score, "comment": comment}


def tool_efficiency(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate tool usage efficiency: steps used vs. budget."""
    max_budget = inputs.get("max_step_budget", 12)
    messages = outputs.get("messages", [])

    tool_calls = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls += len(msg.tool_calls)

    if tool_calls == 0:
        return {"key": "tool_efficiency", "score": 0.0, "comment": "No tool calls detected"}

    utilization = tool_calls / max_budget if max_budget > 0 else 0
    if utilization > 1.0:
        score = max(0.0, 1.0 - (utilization - 1.0) * 0.5)
        comment = f"Over budget: {tool_calls}/{max_budget} steps"
    elif utilization < 0.3:
        score = 0.5
        comment = f"Under-utilized: {tool_calls}/{max_budget} steps — may indicate insufficient research"
    else:
        score = 1.0 - abs(utilization - 0.7) * 0.5
        comment = f"Used {tool_calls}/{max_budget} steps ({utilization:.0%} utilization)"

    return {"key": "tool_efficiency", "score": max(0.0, min(1.0, score)), "comment": comment}


def iteration_convergence(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """For iterative stages: evaluate whether iterations showed meaningful convergence."""
    next_steps_history = outputs.get("next_steps_history", [])

    if len(next_steps_history) < 2:
        return {
            "key": "iteration_convergence",
            "score": 0.5,
            "comment": "Insufficient iterations for convergence analysis",
        }

    confidences = [h.get("confidence", 0.0) for h in next_steps_history]
    improvements = [
        confidences[i + 1] - confidences[i]
        for i in range(len(confidences) - 1)
    ]

    total_improvement = confidences[-1] - confidences[0]
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0

    diminishing = sum(
        1 for imp in improvements if imp < DIMINISHING_RETURNS_THRESHOLD
    )
    diminishing_ratio = diminishing / len(improvements) if improvements else 0

    if total_improvement < 0:
        score = 0.2
        comment = f"Regression: confidence dropped from {confidences[0]:.2f} to {confidences[-1]:.2f}"
    elif avg_improvement < MIN_CONFIDENCE_IMPROVEMENT_PER_ITERATION:
        score = 0.4
        comment = f"Stagnant: avg improvement {avg_improvement:.3f} per iteration"
    elif diminishing_ratio > 0.5:
        score = 0.6
        comment = f"Diminishing returns in {diminishing}/{len(improvements)} transitions"
    else:
        score = min(1.0, 0.6 + total_improvement)
        comment = (
            f"Good convergence: {confidences[0]:.2f} → {confidences[-1]:.2f} "
            f"(+{total_improvement:.2f} over {len(confidences)} iterations)"
        )

    return {"key": "iteration_convergence", "score": score, "comment": comment}
