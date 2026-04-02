"""
Entity extraction agent.

Uses create_agent with response_format=KGExtractionResult so the agent
runs in a tool-calling loop (tools can be added later) and always returns
a typed KGExtractionResult via result["structured_response"].

Temporal-aware: the extraction prompt includes the current date and asks
the LLM to populate temporal qualifiers when evidence exists in the report.
"""

from __future__ import annotations

from datetime import date

from langchain.agents import create_agent

from src.research.langchain_agent.kg.extraction_models import KGExtractionResult
from src.research.langchain_agent.kg.prompts.extraction_prompts import _EXTRACTION_SYSTEM_PROMPT



def build_extraction_agent(llm, tools: list | None = None):
    """
    Build a reusable extraction agent.

    Pass tools=[schema_search_tool, ...] to enable progressive schema lookup.
    Pass tools=None (default) for a single-pass extraction run.

    Build once at startup; reuse across all ingestion runs in a session.
    """
    system_prompt = _EXTRACTION_SYSTEM_PROMPT.format(
        current_date=date.today().isoformat(),
    )
    return create_agent(
        model=llm,
        tools=tools or [],
        system_prompt=system_prompt,
        response_format=KGExtractionResult,
    )


async def extract_kg_entities(
    report_text: str,
    selected_schema_text: str,
    agent,
    source_report: str = "",
    *,
    current_date: str | None = None,
    temporal_scope_description: str = "",
) -> KGExtractionResult:
    """
    Run the extraction agent on a single report.
    """
    temporal_block = ""
    if current_date or temporal_scope_description:
        temporal_block = f"\n\nTemporal context for this extraction:\n"
        if current_date:
            temporal_block += f"- Current date: {current_date}\n"
        if temporal_scope_description:
            temporal_block += f"- Temporal scope: {temporal_scope_description}\n"
        temporal_block += (
            "Use this context to ground your temporal reasoning. "
            "If the report describes 'current' facts, they are current as of the date above."
        )

    user_message = (
        f"Extraction contract (JSON):\n{selected_schema_text}\n\n"
        f"Research report:\n{report_text}\n\n"
        f'Extract all nodes and relationships. Set source_report = "{source_report}".'
        f"{temporal_block}"
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    extraction: KGExtractionResult = result["structured_response"]

    entity_counts = []
    for field_name, field_info in KGExtractionResult.model_fields.items():
        if field_name == "source_report":
            continue
        entities = getattr(extraction, field_name, [])
        if entities:
            entity_counts.append(f"{len(entities)} {field_name}")

    print(f"[extractor] Extracted: {', '.join(entity_counts) or 'nothing'}")
    return extraction
