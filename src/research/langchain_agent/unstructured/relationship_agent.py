from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient
from src.research.langchain_agent.unstructured.models import (
    ChunkRecord,
    DocumentRecord,
    RelationshipDecisionBatch,
)
from src.research.langchain_agent.unstructured.neo4j_tools import build_relationship_agent_tools

RELATIONSHIP_AGENT_PROMPT = """
You are a temporally-aware relationship extraction agent for document ingestion.

Your job is to attach document or chunk evidence to the existing structured Neo4j graph.

Rules:
- You MUST query the graph before naming a target. Do not invent targets.
- Prefer identity nodes when the text concerns an enduring entity.
- Prefer as-of snapshots when the chunk is tied to a historical filing date or clearly supports a time-bounded status, role, or stateful fact.
- `ABOUT` is for the main subject of the document or chunk.
- `MENTIONS` is for secondary referenced entities.
- `SUPPORTS` is evidentiary support and should only be emitted when the chunk contains factual backing for the target.
- `IS_PRIMARY_SOURCE` is reserved for authoritative origin material, such as issuer filings, primary reports, or direct disclosures.
- If there is no grounded target, return an empty decision list.
- Keep the number of decisions low and high precision.
- Use `search_existing_graph_targets` first, then inspect `fetch_state_snapshots_tool` and `fetch_graph_neighborhood_tool` when the attachment target is ambiguous or historical.
""".strip()


def build_relationship_extraction_agent(client: Neo4jAuraClient):
    tools = build_relationship_agent_tools(client)
    return create_agent(
        model="gpt-5.4-mini",
        tools=tools,
        system_prompt=RELATIONSHIP_AGENT_PROMPT,
        response_format=RelationshipDecisionBatch,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_existing_graph_targets",
                run_limit=3,
                exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(
                tool_name="fetch_state_snapshots_tool",
                run_limit=3,
                exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(
                tool_name="fetch_graph_neighborhood_tool",
                run_limit=2,
                exit_behavior="continue",
            ),
        ],
    )


async def decide_chunk_relationships(
    *,
    client: Neo4jAuraClient,
    document: DocumentRecord,
    chunk: ChunkRecord,
) -> RelationshipDecisionBatch:
    agent = build_relationship_extraction_agent(client)
    prompt = f"""
Document title: {document.title}
Issuer: {document.issuer_name} ({document.issuer_ticker})
Form type: {document.form_type}
Accession number: {document.accession_number}
Canonical source: {document.canonical_source_uri}

Chunk id: {chunk.chunk_id}
Chunk headings: {chunk.headings}
Chunk captions: {chunk.captions}
Document filing date / as-of anchor: {document.filing_date}
Chunk text:
{(chunk.contextualized_text or chunk.text)[:4000]}

Return a RelationshipDecisionBatch for this chunk.
""".strip()

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": 8},
    )
    return result["structured_response"]
