"""
KG ingestion orchestrator.

Ties together:
  1. Schema selection  (schema_selector)
  2. Entity extraction (extractor)
  3. searchText generation (searchtext)
  4. Batch embedding   (embedder)
  5. Neo4j write       (neo4j_writer)

Callable from run_mission.py as a post-step or from ingest_report.py as a
standalone CLI run.

Temporal context flows through the entire pipeline:
  - IngestionTemporalContext is built from the caller's parameters.
  - The extraction agent receives current_date and temporal_scope.
  - The neo4j_writer uses temporal context for bitemporal properties.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.research.langchain_agent.kg.embedder import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL, embed_batch
from src.research.langchain_agent.kg.extractor import build_extraction_agent, extract_kg_entities
from src.research.langchain_agent.kg.extraction_models import (
    IngestionTemporalContext,
    KGExtractionResult,
    TemporalScope,
)
from src.research.langchain_agent.kg.neo4j_writer import write_extraction_to_neo4j
from src.research.langchain_agent.kg.schema_selector import (
    build_schema_selector_agent,
    load_schema_chunks,
    load_schema_index,
    select_schema_chunks,
)
from src.research.langchain_agent.kg.searchtext import (
    build_searchtext_agent,
    build_searchtext_from_fields,
    get_searchtext_mode,
)
from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient 
from src.research.langchain_agent.constants import GPT_5, GPT_5_MINI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

_SELECTOR_MODEL = GPT_5_MINI
_EXTRACTION_MODEL = GPT_5 # use GPT-5 ! for extraction ! 
_SEARCHTEXT_MODEL = GPT_5_MINI


def build_selector_llm() -> ChatOpenAI:
    return ChatOpenAI(model=_SELECTOR_MODEL, temperature=0.0, max_retries=2)


def build_extraction_llm() -> ChatOpenAI:
    return ChatOpenAI(model=_EXTRACTION_MODEL, temperature=0.0, max_retries=2)


def build_searchtext_llm() -> ChatOpenAI:
    return ChatOpenAI(model=_SEARCHTEXT_MODEL, temperature=0.0, max_retries=2)


def build_default_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Entity type → node key prefix mapping (registry-driven)
# ---------------------------------------------------------------------------

# Maps KGExtractionResult field names to (key_prefix, entity_label, name_field)
# for building the node_key used in embeddings and searchtext.
_ENTITY_FIELD_CONFIG: dict[str, tuple[str, str, str]] = {
    "organizations":      ("org",       "Organization",    "name"),
    "persons":            ("person",    "Person",          "name"),
    "products":           ("product",   "Product",         "name"),
    "compounds":          ("compound",  "Compound",        "name"),
    "biomarkers":         ("biomarker", "Biomarker",       "name"),
    "conditions":         ("condition", "Condition",       "name"),
    "studies":            ("study",     "Study",           "name"),
    "labTests":           ("labtest",   "LabTest",         "name"),
    "panelDefinitions":   ("panel",     "PanelDefinition", "name"),
}


# ---------------------------------------------------------------------------
# searchText + embedding helpers
# ---------------------------------------------------------------------------


def _entity_pairs_from_extraction(
    extraction: KGExtractionResult,
) -> list[tuple[str, str, dict]]:
    """
    Return a flat list of (node_key, entity_type, entity_dict) tuples.

    Registry-driven: iterates over all entity list fields on KGExtractionResult
    using _ENTITY_FIELD_CONFIG to derive keys.
    """
    pairs: list[tuple[str, str, dict]] = []

    for field_name, (prefix, label, name_attr) in _ENTITY_FIELD_CONFIG.items():
        entities = getattr(extraction, field_name, [])
        seen: set[str] = set()
        for entity in entities:
            d = entity.model_dump()
            entity_name = d.get(name_attr, "")
            if not entity_name or entity_name in seen:
                continue
            seen.add(entity_name)
            pairs.append((f"{prefix}:{entity_name}", label, d))

    # Deduplicated compound forms from compound_ingredients
    seen_compounds: set[str] = set()
    for ingredient in extraction.compound_ingredients:
        cname = ingredient.compoundName
        if cname and cname not in seen_compounds:
            seen_compounds.add(cname)
            pairs.append(
                (f"compoundform:{cname}", "CompoundForm", ingredient.model_dump())
            )

    return pairs


async def _build_search_texts(
    pairs: list[tuple[str, str, dict]],
    context: str,
    searchtext_agent,
) -> dict[str, str]:
    """
    Generate searchText for every node. LLM nodes run concurrently.
    Returns a map: node_key -> searchText string.
    """
    llm_tasks: dict[str, asyncio.Task] = {}
    concat_results: dict[str, str] = {}

    for node_key, entity_type, entity_dict in pairs:
        mode = get_searchtext_mode(entity_type)
        if mode == "llm":
            from src.research.langchain_agent.kg.searchtext import generate_searchtext_llm

            llm_tasks[node_key] = asyncio.create_task(
                generate_searchtext_llm(
                    entity_type=entity_type,
                    entity_dict=entity_dict,
                    context=context,
                    searchtext_agent=searchtext_agent,
                )
            )
        else:
            search_fields = entity_dict.get(
                "searchFields", list(entity_dict.keys())
            )
            concat_results[node_key] = build_searchtext_from_fields(
                entity_dict, search_fields
            )

    search_texts: dict[str, str] = dict(concat_results)
    for node_key, task in llm_tasks.items():
        search_texts[node_key] = await task

    return search_texts


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_kg_ingestion(
    report_text: str,
    source_report: str,
    targets: list[str],
    stage_type: str,
    neo4j_client: Neo4jAuraClient,
    *,
    selector_llm: ChatOpenAI | None = None,
    extraction_llm: ChatOpenAI | None = None,
    searchtext_llm: ChatOpenAI | None = None,
    embedder: OpenAIEmbeddings | None = None,
    schema_index: list[dict] | None = None,
    schema_root: Path | None = None,
    context: str = "",
    research_date: datetime | None = None,
    temporal_scope: TemporalScope | None = None,
) -> dict[str, Any]:
    """
    Run the full KG ingestion pipeline for a single research report.

    Steps:
        1. Schema selection  -- lightweight LLM picks relevant schema chunks.
        2. Schema loading    -- selected chunks build extraction contract.
        3. Entity extraction -- extraction agent produces KGExtractionResult.
        4. searchText gen    -- LLM for Org/Person/Product; field-concat for rest.
        5. Batch embed       -- one OpenAI embedding call for all searchTexts.
        6. Neo4j write       -- identity/state nodes + temporal relationships.
    """
    ingestion_time = datetime.now(timezone.utc)
    scope = temporal_scope or TemporalScope()

    temporal_ctx = IngestionTemporalContext(
        research_date=research_date or ingestion_time,
        ingestion_time=ingestion_time,
        temporal_scope=scope,
        source_report=source_report,
    )

    sel_llm = selector_llm or build_selector_llm()
    ext_llm = extraction_llm or build_extraction_llm()
    st_llm = searchtext_llm or build_searchtext_llm()
    emb = embedder or build_default_embedder()

    selector_agent = build_schema_selector_agent(sel_llm)
    extraction_agent = build_extraction_agent(ext_llm)
    searchtext_agent = build_searchtext_agent(st_llm)

    index = schema_index or load_schema_index()

    # Step 1: Schema selection
    logger.info("[kg_ingestion] Step 1: schema selection")
    selected_chunks = await select_schema_chunks(
        report_text=report_text,
        stage_type=stage_type,
        targets=targets,
        index=index,
        selector_agent=selector_agent,
    )

    if not selected_chunks:
        logger.warning("[kg_ingestion] No schema chunks selected — aborting.")
        return {"extraction": None, "chunks_used": [], "node_counts": {}}

    # Step 2: Build extraction contract from registry
    logger.info("[kg_ingestion] Step 2: building extraction contract")
    selected_schema_text = load_schema_chunks(selected_chunks)

    # Step 3: Entity extraction
    logger.info("[kg_ingestion] Step 3: entity extraction")
    current_date_str = (research_date or ingestion_time).strftime("%Y-%m-%d")
    extraction: KGExtractionResult = await extract_kg_entities(
        report_text=report_text,
        selected_schema_text=selected_schema_text,
        agent=extraction_agent,
        source_report=source_report,
        current_date=current_date_str,
        temporal_scope_description=scope.description,
    )

    # Step 4: searchText generation
    logger.info("[kg_ingestion] Step 4: searchText generation")
    pairs = _entity_pairs_from_extraction(extraction)
    effective_context = context or f"Research targets: {', '.join(targets)}"
    search_texts = await _build_search_texts(pairs, effective_context, searchtext_agent)

    # Step 5: Batch embed
    logger.info("[kg_ingestion] Step 5: batch embedding (%d texts)", len(pairs))
    node_keys = [node_key for node_key, _, _ in pairs]
    texts_to_embed = [search_texts[k] for k in node_keys]

    embeddings_list = await embed_batch(texts_to_embed, embedder=emb)

    node_embeddings: dict[str, Any] = {}
    for node_key, embedding in zip(node_keys, embeddings_list):
        node_embeddings[node_key] = embedding
        node_embeddings[f"searchtext:{node_key}"] = search_texts[node_key]

    # Step 6: Neo4j write
    logger.info("[kg_ingestion] Step 6: writing to Neo4j (bitemporal mode)")
    counts = await write_extraction_to_neo4j(
        client=neo4j_client,
        extraction=extraction,
        node_embeddings=node_embeddings,
        temporal_ctx=temporal_ctx,
    )

    total_nodes = sum(
        counts.get(k, 0) for k in counts if k.endswith("_written") and k != "rels_written"
    )

    logger.info(
        "[kg_ingestion] Done. nodes=%d, rels_written=%d, rels_skipped=%d, "
        "states_created=%d, states_skipped=%d",
        total_nodes,
        counts.get("rels_written", 0),
        counts.get("rels_skipped", 0),
        counts.get("states_created", 0),
        counts.get("states_skipped", 0),
    )

    return {
        "extraction": extraction,
        "chunks_used": [c["chunk_id"] for c in selected_chunks],
        "node_counts": counts,
        "total_nodes": total_nodes,
        "total_rels_written": counts.get("rels_written", 0),
        "total_rels_skipped": counts.get("rels_skipped", 0),
        "states_created": counts.get("states_created", 0),
        "states_skipped": counts.get("states_skipped", 0),
    }
