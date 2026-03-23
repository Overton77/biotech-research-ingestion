"""
Entity extraction agent.

Uses create_agent with response_format=KGExtractionResult so the agent
runs in a tool-calling loop (tools can be added later) and always returns
a typed KGExtractionResult via result["structured_response"].
"""

from __future__ import annotations

from langchain.agents import create_agent

from src.research.langchain_agent.kg.extraction_models import KGExtractionResult

_EXTRACTION_SYSTEM_PROMPT = """\
You are a biotech knowledge graph extraction agent.

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
   HAS_SCIENTIFIC_ADVISOR, HAS_EXECUTIVE_ROLE.
6. For Product ingredients: use compound_ingredients — each entry combines the
   CompoundForm node data (compoundName, formType) with the relationship data
   (dose, doseUnit, role, bioavailabilityNotes).
7. When finished, produce a final KGExtractionResult with everything you found.

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
    return create_agent(
        model=llm,
        tools=tools or [],
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        response_format=KGExtractionResult,
    )


async def extract_kg_entities(
    report_text: str,
    selected_schema_text: str,
    agent,
    source_report: str = "",
) -> KGExtractionResult:
    """
    Run the extraction agent on a single report.

    Args:
        report_text:          Full text of the final research report.
        selected_schema_text: Concatenated schema .md files for selected chunks.
        agent:                Built with build_extraction_agent().
        source_report:        Identifier for the report (task_slug or file path).

    Returns:
        KGExtractionResult with all extracted nodes and relationships.
    """
    user_message = (
        f"Extraction contract (JSON):\n{selected_schema_text}\n\n"
        f"Research report:\n{report_text}\n\n"
        f'Extract all nodes and relationships. Set source_report = "{source_report}".'
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    extraction: KGExtractionResult = result["structured_response"]
    print(
        f"[extractor] Extracted {len(extraction.organizations)} org(s), "
        f"{len(extraction.persons)} person(s), "
        f"{len(extraction.products)} product(s), "
        f"{len(extraction.compound_ingredients)} compound ingredient(s), "
        f"{len(extraction.org_person_relationships)} org-person rel(s)."
    )
    return extraction
