"""
Run the full unstructured ingestion pipeline for a single candidate document.

Pipeline steps:
  1. Parse         — Docling or LlamaParse → Document, TextVersion, Segmentation, Chunks
  2. Embed         — searchText + vector embeddings for Document, TextVersions, Chunks
  3. Seed linkage  — attach Document to issuer Organization via direct graph search
  4. Schema linkage— schema-selection + LLM extraction → Document ABOUT/AUTHORED_BY to
                     structured nodes (prefers temporal state nodes for Organization/Product)
  5. Persist       — write Document, TextVersion, Segmentation, Chunks, and document-level
                     relationships to Neo4j (optional, gated by config.write_to_neo4j)

Chunk-level relationship extraction is intentionally omitted. Only Document-level
ABOUT / AUTHORED_BY / IS_PRIMARY_SOURCE relationships are created.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.research.langchain_agent.kg.embedder import build_embedder, embed_batch
from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient
from src.research.langchain_agent.unstructured.docling_pipeline import (
    materialize_candidate_with_docling,
    materialize_candidate_with_llamaparse,
)
from src.research.langchain_agent.unstructured.document_linker import (
    link_document_to_structured_nodes,
)
from src.research.langchain_agent.unstructured.graph_writer import write_unstructured_ingestion_result
from src.research.langchain_agent.unstructured.models import (
    CandidateDocument,
    GraphTargetRef,
    RelationshipDecision,
    UnstructuredIngestionConfig,
    UnstructuredIngestionResult,
)
from src.research.langchain_agent.unstructured.neo4j_tools import search_graph_targets

logger = logging.getLogger(__name__)


def _truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars].strip()


def _build_document_search_text(
    *,
    candidate: CandidateDocument,
    document,
    text_versions,
    chunks,
    max_chars: int,
) -> str:
    header = [
        f"Title: {document.title}",
        f"Issuer: {document.issuer_name} ({document.issuer_ticker})",
        f"Form: {document.form_type}",
        f"Accession: {document.accession_number}",
        f"Filing date: {document.filing_date}",
    ]
    top_chunks = []
    for chunk in chunks[:8]:
        heading_prefix = f"{' > '.join(chunk.headings)}\n" if chunk.headings else ""
        top_chunks.append(f"{heading_prefix}{chunk.contextualized_text or chunk.text}")
    body = "\n\n".join(top_chunks)
    raw_text = Path(text_versions[0].text_path).read_text(encoding="utf-8")
    combined = "\n".join(header) + "\n\n" + body + "\n\n" + raw_text[: max_chars // 2]
    return _truncate_text(combined, max_chars)


def _build_text_version_search_text(text_version, *, max_chars: int) -> str:
    raw_text = Path(text_version.text_path).read_text(encoding="utf-8")
    return _truncate_text(raw_text, max_chars)


def _build_chunk_embedding_text(chunk, *, source_field: str, max_chars: int) -> str:
    raw = chunk.contextualized_text if source_field == "contextualized_text" else chunk.text
    heading_prefix = f"{' > '.join(chunk.headings)}\n" if chunk.headings else ""
    return _truncate_text(f"{heading_prefix}{raw}", max_chars)


async def _apply_embeddings(
    *,
    candidate: CandidateDocument,
    document,
    text_versions,
    chunks,
    config: UnstructuredIngestionConfig,
) -> None:
    embedding_cfg = config.embedding_config
    if not embedding_cfg.enabled:
        return

    embedder = build_embedder(model=embedding_cfg.model)

    document_text = _build_document_search_text(
        candidate=candidate,
        document=document,
        text_versions=text_versions,
        chunks=chunks,
        max_chars=embedding_cfg.document_max_chars,
    )
    text_version_texts = [
        _build_text_version_search_text(tv, max_chars=embedding_cfg.text_version_max_chars)
        for tv in text_versions
    ]
    chunk_texts = [
        _build_chunk_embedding_text(
            chunk,
            source_field=embedding_cfg.chunk_source_field,
            max_chars=embedding_cfg.chunk_max_chars,
        )
        for chunk in chunks
    ]

    embeddings = await embed_batch(
        [document_text, *text_version_texts, *chunk_texts],
        embedder=embedder,
        model=embedding_cfg.model,
    )
    document_embedding = embeddings[0]
    text_version_embeddings = embeddings[1 : 1 + len(text_versions)]
    chunk_embeddings = embeddings[1 + len(text_versions) :]

    document.search_text = document_text
    document.search_text_embedding = document_embedding
    document.search_text_model = embedding_cfg.model
    document.search_text_dimensions = embedding_cfg.dimensions
    document.search_text_version = embedding_cfg.version

    for text_version, search_text, embedding in zip(text_versions, text_version_texts, text_version_embeddings):
        text_version.search_text = search_text
        text_version.search_text_embedding = embedding
        text_version.search_text_model = embedding_cfg.model
        text_version.search_text_dimensions = embedding_cfg.dimensions
        text_version.search_text_version = embedding_cfg.version

    for chunk, embedding_text, embedding in zip(chunks, chunk_texts, chunk_embeddings):
        chunk.embedding_text = embedding_text
        chunk.embedding = embedding
        chunk.embedding_model = embedding_cfg.model
        chunk.embedding_dimensions = embedding_cfg.dimensions
        chunk.embedding_version = embedding_cfg.version


async def _seed_document_relationships(
    *,
    client: Neo4jAuraClient,
    candidate: CandidateDocument,
    document_id: str,
) -> list[RelationshipDecision]:
    decisions: list[RelationshipDecision] = []
    issuer_query = candidate.issuer_name or candidate.issuer_ticker
    if not issuer_query:
        return decisions

    matches = await search_graph_targets(client, issuer_query, limit=5)
    match = next((row for row in matches if row.get("label") == "Organization"), None)
    if match is None:
        return decisions

    state_id = match.get("state_id")
    if state_id:
        target = GraphTargetRef(
            target_level="state",
            target_label="OrganizationState",
            target_id_property="id",
            target_id=match["id"],
            state_id=state_id,
            display_name=match.get("display_name", ""),
            metadata={"seeded": True, "resolved_via": "issuer_state"},
        )
    else:
        target = GraphTargetRef(
            target_level="identity",
            target_label=match["label"],
            target_id_property=match["id_property"],
            target_id=match["id"],
            display_name=match.get("display_name", ""),
            metadata={"seeded": True, "resolved_via": "issuer_identity"},
        )

    decisions.append(
        RelationshipDecision(
            relationship_type="ABOUT",
            source_scope="document",
            source_record_id=document_id,
            target=target,
            rationale="The candidate document issuer matches an existing organization in the graph.",
            confidence=0.95,
            temporal_note="Document-level issuer attachment",
            metadata={"seeded": True},
        )
    )
    if candidate.source_type == "edgar_filing":
        decisions.append(
            RelationshipDecision(
                relationship_type="IS_PRIMARY_SOURCE",
                source_scope="document",
                source_record_id=document_id,
                target=target,
                rationale="SEC filings are primary-source issuer disclosures.",
                confidence=0.99,
                temporal_note="Primary SEC filing source",
                metadata={"seeded": True, "edgar": True},
            )
        )
    return decisions


async def run_unstructured_ingestion(
    *,
    candidate: CandidateDocument,
    output_dir: Path,
    neo4j_client: Neo4jAuraClient,
    config: UnstructuredIngestionConfig | None = None,
) -> UnstructuredIngestionResult:
    cfg = config or UnstructuredIngestionConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Parse ----
    if cfg.parser_backend == "llamaparse":
        document, text_versions, segmentations, chunks, artifact_paths = await materialize_candidate_with_llamaparse(
            candidate=candidate,
            output_dir=output_dir,
            summary_policy=cfg.summary_policy,
            cleaning=cfg.chunk_cleaning,
            enhancement=cfg.chunk_enhancement,
            tier=cfg.llama_parse_tier,
        )
    else:
        document, text_versions, segmentations, chunks, artifact_paths = await materialize_candidate_with_docling(
            candidate=candidate,
            output_dir=output_dir,
            summary_policy=cfg.summary_policy,
            cleaning=cfg.chunk_cleaning,
            enhancement=cfg.chunk_enhancement,
        )

    # ---- Step 2: Embed ----
    await _apply_embeddings(
        candidate=candidate,
        document=document,
        text_versions=text_versions,
        chunks=chunks,
        config=cfg,
    )

    # ---- Persist parsed artifacts to disk ----
    (output_dir / "document_record.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8",
    )
    if text_versions:
        (output_dir / "raw_text_version.json").write_text(
            text_versions[0].model_dump_json(indent=2), encoding="utf-8",
        )
    if len(text_versions) > 1 and text_versions[1].version_kind == "summary":
        (output_dir / "summary_text_version.json").write_text(
            text_versions[1].model_dump_json(indent=2), encoding="utf-8",
        )
    (output_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2, default=str),
        encoding="utf-8",
    )

    # ---- Step 3: Seed issuer relationship ----
    relationship_decisions = await _seed_document_relationships(
        client=neo4j_client,
        candidate=candidate,
        document_id=document.document_id,
    )

    # ---- Step 4: Schema-based document linkage ----
    document_text = Path(text_versions[0].text_path).read_text(encoding="utf-8") if text_versions else ""
    try:
        schema_decisions = await link_document_to_structured_nodes(
            client=neo4j_client,
            document=document,
            document_text=document_text,
            targets=[candidate.issuer_name, candidate.issuer_ticker],
            stage_type="targeted_extraction",
        )
        seeded_target_ids = {d.target.target_id for d in relationship_decisions}
        for sd in schema_decisions:
            if sd.target.target_id not in seeded_target_ids:
                relationship_decisions.append(sd)
                seeded_target_ids.add(sd.target.target_id)
            else:
                logger.debug(
                    "[unstructured] Skipping duplicate schema linkage for %s (already seeded)",
                    sd.target.display_name,
                )
    except Exception:
        logger.exception("Schema-based document linkage failed — continuing with seeded relationships only")

    # ---- Step 5: Build result ----
    result = UnstructuredIngestionResult(
        document=document,
        text_versions=text_versions,
        segmentations=segmentations,
        chunks=chunks,
        claim_occurrences=[],
        relationship_decisions=relationship_decisions,
        artifact_paths=artifact_paths,
        metadata={"candidate_id": candidate.candidate_id},
    )

    result_path = output_dir / "unstructured_ingestion_result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # ---- Step 6: Write to Neo4j ----
    if cfg.write_to_neo4j:
        counts = await write_unstructured_ingestion_result(
            neo4j_client,
            document=document,
            text_versions=text_versions,
            segmentations=segmentations,
            chunks=chunks,
            claim_occurrences=[],
            relationship_decisions=relationship_decisions,
        )
        counts_path = output_dir / "neo4j_write_counts.json"
        counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        result.metadata["neo4j_counts"] = counts

    return result
