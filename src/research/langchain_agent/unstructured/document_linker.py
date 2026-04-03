"""
Document-level structured node linkage via schema selection + LLM extraction.

Replaces the per-chunk relationship agent with a single document-level pass:
  1. Schema selection   — pick relevant schema chunks from schema_index.json
  2. LLM extraction     — extract entity references (names + labels) from the
                          document text using the full schema contract
  3. Neo4j resolution   — look up each entity, prefer temporal state nodes for
                          entities that have HAS_STATE modeling
  4. RelationshipDecision — emit ABOUT / AUTHORED_BY decisions from Document to
                           resolved structured nodes
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.research.langchain_agent.kg.neo4j_resolver import resolve_node_id, FULLTEXT_INDEX_MAP
from src.research.langchain_agent.kg.schema_loader import load_schema_registry
from src.research.langchain_agent.kg.schema_selector import (
    SchemaSelectionResult,
    build_schema_selector_agent,
    load_schema_index,
    select_schema_chunks,
    load_schema_chunks,
)
from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient
from src.research.langchain_agent.unstructured.models import (
    DocumentRecord,
    GraphTargetRef,
    RelationshipDecision,
)
from src.research.langchain_agent.unstructured.neo4j_tools import (
    TARGET_CONFIG,
    fetch_state_snapshots,
)

logger = logging.getLogger(__name__)

_STATE_LABELS: dict[str, str] = {
    cfg["label"]: cfg["state_label"]
    for cfg in TARGET_CONFIG
    if cfg.get("state_label")
}

_LABEL_ID_PROPERTY: dict[str, str] = {
    cfg["label"]: cfg["id_property"]
    for cfg in TARGET_CONFIG
}

_RESOLVABLE_LABELS = set(FULLTEXT_INDEX_MAP.keys()) | set(_LABEL_ID_PROPERTY.keys())


class DocumentEntityRef(BaseModel):
    entity_label: str = Field(description="Node label from the schema contract, e.g. Organization, Product, Study.")
    entity_name: str = Field(description="Canonical name of the entity as stated or clearly implied in the document.")
    relationship_type: Literal["ABOUT", "AUTHORED_BY"] = Field(
        default="ABOUT",
        description="ABOUT for subjects the document discusses; AUTHORED_BY only for named authors.",
    )
    is_primary_subject: bool = Field(
        default=False,
        description="True if this entity is the primary subject of the entire document.",
    )
    rationale: str = ""


class DocumentLinkageExtraction(BaseModel):
    entities: list[DocumentEntityRef] = Field(default_factory=list)
    reasoning: str = ""


_LINKER_SYSTEM_PROMPT = """\
You extract structured entity references from a document for knowledge-graph linkage.

You will receive:
- The document metadata (title, issuer, form type, filing date).
- A truncated document text.
- A JSON schema contract describing extractable node types and their properties.

Your task: identify every entity of a type present in the schema contract that the \
document is clearly ABOUT, and any Person or Organization that AUTHORED it.

Rules:
- Only emit entity_label values that appear as node keys in the schema contract.
- Use the canonical entity name from the document (e.g. "Exact Sciences Corporation", not "EXAS").
- Mark the issuer / primary subject with is_primary_subject = true.
- For AUTHORED_BY, only emit when the document explicitly names an author or filer.
- Prefer specificity: if a Product name is stated, emit Product rather than just Organization.
- Keep the list concise — at most 12 entities.
"""


def _build_linker_agent():
    return create_agent(
        model=ChatOpenAI(model="gpt-5-mini", temperature=0.0, max_retries=2),
        tools=[],
        system_prompt=_LINKER_SYSTEM_PROMPT,
        response_format=DocumentLinkageExtraction,
    )


async def _extract_document_entities(
    *,
    document: DocumentRecord,
    document_text: str,
    schema_contract: str,
) -> DocumentLinkageExtraction:
    agent = _build_linker_agent()

    user_message = (
        f"## Document Metadata\n"
        f"Title: {document.title}\n"
        f"Issuer: {document.issuer_name} ({document.issuer_ticker})\n"
        f"Form type: {document.form_type}\n"
        f"Filing date: {document.filing_date}\n"
        f"Accession: {document.accession_number}\n"
        f"Source type: {document.source_type}\n\n"
        f"## Schema Contract\n"
        f"```json\n{schema_contract}\n```\n\n"
        f"## Document Text (truncated)\n"
        f"{document_text[:6000]}\n\n"
        f"Extract entity references for graph linkage."
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"recursion_limit": 4},
    )
    return result["structured_response"]


async def _resolve_entity_target(
    client: Neo4jAuraClient,
    entity: DocumentEntityRef,
    filing_date: str,
) -> GraphTargetRef | None:
    label = entity.entity_label
    name = entity.entity_name

    if label not in _RESOLVABLE_LABELS:
        logger.debug("[document_linker] Label %s is not resolvable, skipping %s", label, name)
        return None

    node_id = await resolve_node_id(client, label, name)
    if node_id is None:
        logger.info("[document_linker] No match for %s '%s' — skipping", label, name)
        return None

    id_property = _LABEL_ID_PROPERTY.get(label, "id")
    state_label = _STATE_LABELS.get(label, "")

    if state_label and filing_date:
        snapshots = await fetch_state_snapshots(
            client,
            label=label,
            id_property=id_property,
            node_id=node_id,
            as_of_date=filing_date,
            limit=1,
        )
        if snapshots:
            state_props = snapshots[0].get("state_props", {})
            state_id = state_props.get("id") or state_props.get("stateId", "")
            if state_id:
                return GraphTargetRef(
                    target_level="state",
                    target_label=state_label,
                    target_id_property="id",
                    target_id=node_id,
                    state_id=state_id,
                    display_name=name,
                    metadata={"resolved_via": "state_snapshot", "as_of_date": filing_date},
                )

        snapshots_latest = await fetch_state_snapshots(
            client,
            label=label,
            id_property=id_property,
            node_id=node_id,
            limit=1,
        )
        if snapshots_latest:
            state_props = snapshots_latest[0].get("state_props", {})
            state_id = state_props.get("id") or state_props.get("stateId", "")
            if state_id:
                return GraphTargetRef(
                    target_level="state",
                    target_label=state_label,
                    target_id_property="id",
                    target_id=node_id,
                    state_id=state_id,
                    display_name=name,
                    metadata={"resolved_via": "latest_state"},
                )

    return GraphTargetRef(
        target_level="identity",
        target_label=label,
        target_id_property=id_property,
        target_id=node_id,
        display_name=name,
        metadata={"resolved_via": "identity"},
    )


async def link_document_to_structured_nodes(
    *,
    client: Neo4jAuraClient,
    document: DocumentRecord,
    document_text: str,
    targets: list[str] | None = None,
    stage_type: str = "targeted_extraction",
) -> list[RelationshipDecision]:
    schema_index = load_schema_index()

    selector_agent = build_schema_selector_agent(
        ChatOpenAI(model="gpt-5-mini", temperature=0.0, max_retries=2),
        top_k=4,
    )

    doc_preview = (
        f"Title: {document.title}\n"
        f"Issuer: {document.issuer_name} ({document.issuer_ticker})\n"
        f"Form: {document.form_type}\n"
        f"Document type: {document.source_type}\n\n"
        f"{document_text[:800]}"
    )
    selected_chunks = await select_schema_chunks(
        report_text=doc_preview,
        stage_type=stage_type,
        targets=targets or [document.issuer_name, document.issuer_ticker],
        index=schema_index,
        selector_agent=selector_agent,
    )

    if not selected_chunks:
        logger.info("[document_linker] No schema chunks selected — no structured linkage")
        return []

    schema_contract = load_schema_chunks(selected_chunks)

    extraction = await _extract_document_entities(
        document=document,
        document_text=document_text,
        schema_contract=schema_contract,
    )

    logger.info(
        "[document_linker] Extracted %d entity references: %s",
        len(extraction.entities),
        [(e.entity_label, e.entity_name) for e in extraction.entities],
    )

    decisions: list[RelationshipDecision] = []
    filing_date = document.filing_date or ""

    for entity in extraction.entities:
        target = await _resolve_entity_target(client, entity, filing_date)
        if target is None:
            continue

        confidence = 0.90 if entity.is_primary_subject else 0.75
        temporal_note = (
            f"Document {entity.relationship_type} {entity.entity_label} "
            f"(filing_date={filing_date})" if filing_date
            else f"Document {entity.relationship_type} {entity.entity_label}"
        )

        decisions.append(
            RelationshipDecision(
                relationship_type=entity.relationship_type,
                source_scope="document",
                source_record_id=document.document_id,
                target=target,
                rationale=entity.rationale or f"Schema-based extraction: document is {entity.relationship_type} {entity.entity_name}",
                confidence=confidence,
                temporal_note=temporal_note,
                metadata={
                    "extraction_method": "schema_linker",
                    "is_primary_subject": entity.is_primary_subject,
                    "resolved_level": target.target_level,
                },
            )
        )

    logger.info(
        "[document_linker] Produced %d relationship decisions for document %s",
        len(decisions),
        document.document_id,
    )
    return decisions
