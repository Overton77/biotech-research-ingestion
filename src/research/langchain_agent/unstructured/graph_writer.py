from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.research.langchain_agent.kg.temporal import default_bitemporal_props

logger = logging.getLogger(__name__)
from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient
from src.research.langchain_agent.unstructured.models import (
    ClaimOccurrenceRecord,
    ChunkRecord,
    DocumentRecord,
    DocumentTextVersionRecord,
    RelationshipDecision,
    SegmentationRecord,
)


def _strip_none(props: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in props.items() if value is not None}


def _json_prop(value: dict[str, Any] | list[Any]) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _read_text(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _batched(rows: list[dict[str, Any]], batch_size: int = 50) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


async def _merge_document(client: Neo4jAuraClient, document: DocumentRecord) -> None:
    await client.execute_write(
        """
        MERGE (d:Document {documentKey: $documentKey})
        ON CREATE SET d.createdAt = datetime()
        SET d += $props
        SET d.updatedAt = datetime()
        """,
        {
            "documentKey": document.canonical_source_uri or document.document_id,
            "props": _strip_none(
                {
                    "documentId": document.document_id,
                    "sourceType": document.source_type,
                    "documentKey": document.canonical_source_uri or document.document_id,
                    "type": document.source_type.upper(),
                    "canonicalSourceUri": document.canonical_source_uri,
                    "title": document.title,
                    "url": document.canonical_source_uri if document.canonical_source_uri.startswith("http") else "",
                    "issuerName": document.issuer_name,
                    "issuerTicker": document.issuer_ticker,
                    "formType": document.form_type,
                    "accessionNumber": document.accession_number,
                    "filingDate": document.filing_date,
                    "localSourcePath": document.local_source_path,
                    "searchText": document.search_text,
                    "searchTextEmbedding": document.search_text_embedding,
                    "searchTextModel": document.search_text_model,
                    "searchTextDimensions": document.search_text_dimensions,
                    "searchTextVersion": document.search_text_version,
                    "searchTextUpdatedAt": datetime.utcnow().isoformat(),
                    "metadataJson": _json_prop(document.metadata),
                }
            ),
        },
    )


async def _create_text_version(
    client: Neo4jAuraClient,
    text_version: DocumentTextVersionRecord,
) -> None:
    labels = "DocumentTextVersion:SummaryVersion" if text_version.version_kind == "summary" else "DocumentTextVersion"
    text_payload = _read_text(text_version.text_path)
    await client.execute_write(
        f"""
        MATCH (d:Document {{documentKey: $documentKey}})
        MERGE (tv:{labels} {{textVersionHash: $textVersionHash}})
        ON CREATE SET tv.createdAt = datetime()
        SET tv += $props
        SET tv.updatedAt = datetime()
        MERGE (d)-[:HAS_TEXT_VERSION]->(tv)
        """,
        {
            "documentKey": text_version.source_file_path or text_version.text_path,
            "textVersionHash": text_version.content_hash,
            "props": _strip_none(
                {
                    "textVersionId": text_version.text_version_id,
                    "documentTextVersionId": text_version.text_version_id,
                    "documentId": text_version.document_id,
                    "versionKind": text_version.version_kind,
                    "parentTextVersionId": text_version.parent_text_version_id,
                    "source": text_version.source_file_path or text_version.text_path,
                    "sourceFilePath": text_version.source_file_path,
                    "markdownPath": text_version.markdown_path,
                    "jsonPath": text_version.json_path,
                    "textPath": text_version.text_path,
                    "text": text_payload,
                    "searchText": text_version.search_text,
                    "searchTextEmbedding": text_version.search_text_embedding,
                    "searchTextModel": text_version.search_text_model,
                    "searchTextDimensions": text_version.search_text_dimensions,
                    "searchTextVersion": text_version.search_text_version,
                    "searchTextUpdatedAt": datetime.utcnow().isoformat(),
                    "textVersionHash": text_version.content_hash,
                    "contentHash": text_version.content_hash,
                    "charCount": text_version.char_count,
                    "metadataJson": _json_prop(text_version.metadata),
                }
            ),
        },
    )
    if text_version.parent_text_version_id:
        await client.execute_write(
            """
            MATCH (child:DocumentTextVersion {textVersionHash: $childHash})
            MATCH (parent:DocumentTextVersion {textVersionHash: $parentHash})
            MERGE (child)-[:DERIVED_FROM]->(parent)
            """,
            {"childHash": text_version.content_hash, "parentHash": text_version.parent_text_version_id},
        )


async def _create_segmentation(client: Neo4jAuraClient, segmentation: SegmentationRecord) -> None:
    segmentation_hash = hashlib.sha256(
        f"{segmentation.text_version_id}|{segmentation.chunker_type}|{segmentation.chunk_count}".encode("utf-8")
    ).hexdigest()
    await client.execute_write(
        """
        MATCH (tv:DocumentTextVersion {documentTextVersionId: $textVersionId})
        MERGE (s:Segmentation {segmentationHash: $segmentationHash})
        ON CREATE SET s.createdAt = datetime()
        SET s += $props
        SET s.updatedAt = datetime()
        MERGE (tv)-[:HAS_SEGMENTATION]->(s)
        """,
        {
            "textVersionId": segmentation.text_version_id,
            "segmentationHash": segmentation_hash,
            "props": _strip_none(
                {
                    "segmentationId": segmentation.segmentation_id,
                    "chunkerType": segmentation.chunker_type,
                    "contextualizer": segmentation.contextualizer,
                    "chunkCount": segmentation.chunk_count,
                    "chunkSize": segmentation.metadata.get("chunk_size", 0),
                    "overlap": segmentation.metadata.get("overlap", 0),
                    "strategy": segmentation.chunker_type,
                    "segmentationHash": segmentation_hash,
                    "metadataJson": _json_prop(segmentation.metadata),
                }
            ),
        },
    )


async def _create_chunks(client: Neo4jAuraClient, chunks: list[ChunkRecord]) -> None:
    rows = [
        {
            "segmentationId": chunk.segmentation_id,
            "chunkId": chunk.chunk_id,
            "chunkIndex": chunk.chunk_index,
            "props": _strip_none(
                {
                    "chunkId": chunk.chunk_id,
                    "chunkKey": chunk.chunk_id,
                    "textVersionId": chunk.text_version_id,
                    "index": chunk.chunk_index,
                    "chunkIndex": chunk.chunk_index,
                    "text": chunk.text,
                    "contextualizedText": chunk.contextualized_text,
                    "tokenCount": chunk.token_count,
                    "embeddingText": chunk.embedding_text,
                    "embedding": chunk.embedding,
                    "embeddingModel": chunk.embedding_model,
                    "embeddingDimensions": chunk.embedding_dimensions,
                    "embeddingVersion": chunk.embedding_version,
                    "headings": chunk.headings,
                    "captions": chunk.captions,
                    "docItemRefs": chunk.doc_item_refs,
                    "metadataJson": _json_prop(chunk.metadata),
                }
            ),
        }
        for chunk in chunks
    ]
    for batch in _batched(rows, batch_size=32):
        await client.execute_write(
            """
            UNWIND $rows AS row
            MATCH (s:Segmentation {segmentationId: row.segmentationId})
            MERGE (c:Chunk {chunkKey: row.chunkId})
            ON CREATE SET c.createdAt = datetime()
            SET c += row.props
            SET c.updatedAt = datetime()
            MERGE (s)-[:HAS_CHUNK {chunkIndex: row.chunkIndex}]->(c)
            """,
            {"rows": batch},
        )

    next_rows = [
        {"leftId": left.chunk_id, "rightId": right.chunk_id}
        for left, right in zip(chunks, chunks[1:])
    ]
    if next_rows:
        for batch in _batched(next_rows, batch_size=128):
            await client.execute_write(
                """
                UNWIND $rows AS row
                MATCH (a:Chunk {chunkId: row.leftId})
                MATCH (b:Chunk {chunkId: row.rightId})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                """,
                {"rows": batch},
            )


async def _create_claim_occurrence(client: Neo4jAuraClient, claim: ClaimOccurrenceRecord) -> None:
    await client.execute_write(
        """
        MATCH (d:Document {documentId: $documentId})
        MATCH (c:Chunk {chunkId: $chunkId})
        MERGE (co:ClaimOccurrence {claimOccurrenceId: $claimOccurrenceId})
        ON CREATE SET co.createdAt = datetime()
        SET co += $props
        SET co.updatedAt = datetime()
        MERGE (c)-[:EVIDENCES]->(co)
        MERGE (d)-[:HAS_CLAIM_OCCURRENCE]->(co)
        """,
        {
            "documentId": claim.document_id,
            "chunkId": claim.chunk_id,
            "claimOccurrenceId": claim.claim_occurrence_id,
            "props": _strip_none(
                {
                    "claimText": claim.claim_text,
                    "rationale": claim.rationale,
                    "confidence": claim.confidence,
                    "metadataJson": _json_prop(claim.metadata),
                }
            ),
        },
    )


async def _write_relationship(
    client: Neo4jAuraClient,
    decision: RelationshipDecision,
    *,
    research_date: datetime | None = None,
    ingestion_time: datetime | None = None,
) -> None:
    temporal = default_bitemporal_props(
        research_date=research_date,
        ingestion_time=ingestion_time,
    )
    target_label = decision.target.target_label
    target_id_property = decision.target.target_id_property
    target_id_value = decision.target.target_id

    if decision.target.target_level == "state":
        target_label = target_label if target_label.endswith("State") else f"{target_label}State"
        target_id_property = "stateId"
        target_id_value = decision.target.state_id or decision.target.target_id

    source_match = {
        "document": ("Document", "documentId"),
        "chunk": ("Chunk", "chunkId"),
        "claim_occurrence": ("ClaimOccurrence", "claimOccurrenceId"),
    }[decision.source_scope]

    await client.execute_write(
        f"""
        MATCH (src:{source_match[0]} {{ {source_match[1]}: $sourceId }})
        MATCH (dst:{target_label} {{ {target_id_property}: $targetId }})
        CREATE (src)-[r:{decision.relationship_type} {{
          validFrom: $validFrom,
          validTo: $validTo,
          recordedFrom: $recordedFrom,
          recordedTo: $recordedTo,
          temporalNote: $temporalNote,
          rationale: $rationale,
          confidence: $confidence,
          createdAt: datetime(),
          metadataJson: $metadataJson
        }}]->(dst)
        """,
        {
            "sourceId": decision.source_record_id,
            "targetId": target_id_value,
            "validFrom": temporal["validFrom"],
            "validTo": temporal["validTo"],
            "recordedFrom": temporal["recordedFrom"],
            "recordedTo": temporal["recordedTo"],
            "temporalNote": decision.temporal_note,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
            "metadataJson": _json_prop(decision.metadata),
        },
    )


async def write_unstructured_ingestion_result(
    client: Neo4jAuraClient,
    *,
    document: DocumentRecord,
    text_versions: list[DocumentTextVersionRecord],
    segmentations: list[SegmentationRecord],
    chunks: list[ChunkRecord],
    claim_occurrences: list[ClaimOccurrenceRecord] | None = None,
    relationship_decisions: list[RelationshipDecision],
    research_date: datetime | None = None,
    ingestion_time: datetime | None = None,
) -> dict[str, int]:
    await _merge_document(client, document)
    for text_version in text_versions:
        await _create_text_version(client, text_version)
    for segmentation in segmentations:
        await _create_segmentation(client, segmentation)
    await _create_chunks(client, chunks)

    claims = claim_occurrences or []
    for claim in claims:
        await _create_claim_occurrence(client, claim)

    doc_decisions = [d for d in relationship_decisions if d.source_scope == "document"]
    for decision in doc_decisions:
        try:
            await _write_relationship(
                client,
                decision,
                research_date=research_date,
                ingestion_time=ingestion_time,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write document-level relationship %s -> %s: %s",
                decision.relationship_type,
                decision.target.display_name,
                exc,
            )

    return {
        "text_versions": len(text_versions),
        "segmentations": len(segmentations),
        "chunks": len(chunks),
        "claim_occurrences": len(claims),
        "document_relationships": len(doc_decisions),
    }
