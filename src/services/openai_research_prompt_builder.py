# src/services/openai_research_prompt_builder.py
from __future__ import annotations

from src.models.openai_research import OpenAIResearchPlan


def build_openai_research_input(plan: OpenAIResearchPlan) -> str:
    seeded_sources_block = ""
    if plan.seeded_sources:
        lines = []
        for i, source in enumerate(plan.seeded_sources, start=1):
            line = f"{i}. [{source.type}] {source.value}"
            if source.label:
                line += f" | label={source.label}"
            if source.description:
                line += f" | description={source.description}"
            if source.required:
                line += " | REQUIRED"
            if source.metadata:
                line += f" | metadata={source.metadata}"
            lines.append(line)
        seeded_sources_block = "\n".join(lines)
    else:
        seeded_sources_block = "None provided."

    system_instructions = plan.system_instructions or (
        "You are conducting a deep research mission. "
        "Use web research extensively, reason carefully, synthesize findings, "
        "and produce a well-structured, citation-rich report."
    )

    expected_output_format = plan.expected_output_format or (
        "Return a comprehensive markdown report with:\n"
        "1. Executive summary\n"
        "2. Key findings\n"
        "3. Detailed analysis\n"
        "4. Risks / uncertainties\n"
        "5. Recommended next steps\n"
        "6. Source-backed citations throughout"
    )

    coordinator_notes = plan.coordinator_notes or "None."

    return f"""# OpenAI Deep Research Mission

## System Instructions
{system_instructions}

## Research Objective
{plan.objective}

## User Request
{plan.user_prompt}

## Coordinator Notes
{coordinator_notes}

## Seeded Sources / Hints
{seeded_sources_block}

## Output Requirements
{expected_output_format}

## Important Instructions
- Work in the first person where appropriate if that helps match the user's request.
- Prefer primary and authoritative sources where possible.
- Be explicit about uncertainty when evidence is incomplete.
- Use citations throughout the report.
- If comparisons are useful, include tables.
"""