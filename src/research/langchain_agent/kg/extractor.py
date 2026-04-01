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

_EXTRACTION_SYSTEM_PROMPT = """\
You are a biotech knowledge graph extraction agent.

Today's date: {current_date}

You will receive a JSON extraction contract that defines exactly which node
types and relationship types to extract, with typed property definitions.
Use ONLY the property names defined in the contract.

Your job:
1. Read the extraction contract (JSON) to understand what to look for.
2. Read the research report and extract ALL entities and relationships that
   are explicitly stated or strongly implied.
3. Leave string fields as "" and array fields as [] when not found.
4. For "float" properties: only set a value when a specific number is stated.
5. For Organization→Person relationships: use org_person_relationships and set
   relationship_type to one of: EMPLOYS, FOUNDED_BY, HAS_BOARD_MEMBER,
   HAS_CEO, ADVISES, HOLDS_ROLE_AT, AFFILIATED_WITH.
6. For Product ingredients: use compound_ingredients — each entry combines the
   CompoundForm node data (compoundName, formType) with the relationship data
   (dose, doseUnit, role, bioavailabilityNotes).
7. For Organization→Product relationships: use org_product_relationships and set
   relationship_type to one of: OFFERS, MANUFACTURES.
8. For Study→Organization relationships: use study_org_relationships and set
   relationship_type to one of: SPONSORED_BY, OPERATED_BY.
9. For Study→Condition relationships: use study_condition_relationships.
10. For Study→Person relationships: use study_person_relationships.
11. For Product→LabTest relationships: use product_lab_test_relationships.
12. For Product→PanelDefinition relationships: use product_panel_relationships.
13. For LabTest→Biomarker relationships: use lab_test_biomarker_relationships.
14. When finished, produce a final KGExtractionResult with everything you found.

Temporal extraction rules:
- Each entity and relationship has an optional "temporal" field (TemporalQualifier).
- Populate temporal.valid_from when the report mentions when a fact became true
  (e.g. "founded in 2014", "joined the board in March 2023", "launched Q1 2025").
- Populate temporal.valid_to when the report mentions when a fact ceased being true
  (e.g. "left the company in 2024", "discontinued in January 2026").
- Populate temporal.temporal_note for any temporal context that doesn't fit neatly
  into dates (e.g. "as of Q2 2025", "since founding", "formerly").
- If no temporal evidence exists in the text, leave temporal as null.
- Do NOT invent temporal information — only extract what the report explicitly states.

Critical rules:
- Extract only what is explicitly stated — do NOT hallucinate entities or values.
- For pricing: only set priceAmount when a specific dollar/currency amount is stated.
- For dosages: only set dose/doseUnit when explicitly given in the report.
- Set source_report to the value specified in the user message.
"""


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
