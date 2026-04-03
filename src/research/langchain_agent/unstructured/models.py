from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SummaryPolicy(BaseModel):
    enabled: bool = False
    min_chunk_count: int = 80
    max_input_chars: int = 12000
    model: str = "gpt-5.4-mini"


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    version: str = "unstructured-v1"
    document_max_chars: int = 12000
    text_version_max_chars: int = 16000
    chunk_max_chars: int = 6000
    chunk_source_field: Literal["contextualized_text", "text"] = "contextualized_text"


class ChunkCleaningConfig(BaseModel):
    enabled: bool = True
    strip_html: bool = True
    strip_markdown: bool = True
    collapse_whitespace: bool = True
    drop_doc_markers: bool = True


class ChunkEnhancementConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-5-mini"
    max_chunks: int = 12
    max_input_chars: int = 3500


class UnstructuredIngestionConfig(BaseModel):
    enabled: bool = False
    validate_in_isolation: bool = True
    write_to_neo4j: bool = False
    max_relationship_chunks: int = 12
    parser_backend: Literal["docling", "llamaparse"] = "docling"
    llama_parse_tier: Literal["fast", "cost_effective", "agentic", "agentic_plus"] = "agentic"
    summary_policy: SummaryPolicy = Field(default_factory=SummaryPolicy)
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk_cleaning: ChunkCleaningConfig = Field(default_factory=ChunkCleaningConfig)
    chunk_enhancement: ChunkEnhancementConfig = Field(default_factory=ChunkEnhancementConfig)


class CandidateProvenance(BaseModel):
    mission_id: str
    task_slug: str
    discovered_at: datetime = Field(default_factory=utc_now)
    discovered_by: str = ""
    source_artifact_path: str = ""
    source_url: str = ""
    notes: list[str] = Field(default_factory=list)


class CandidateDocument(BaseModel):
    candidate_id: str
    dedupe_key: str
    source_type: Literal["local_file", "url", "edgar_filing", "report_artifact"] = "local_file"
    title: str = ""
    description: str = ""
    mime_type: str = ""
    uri: str = ""
    local_path: str = ""
    relative_path: str = ""
    issuer_name: str = ""
    issuer_ticker: str = ""
    form_type: str = ""
    accession_number: str = ""
    filing_date: str = ""
    should_ingest: bool = True
    priority: Literal["high", "medium", "low"] = "low"
    review_status: Literal["auto_kept", "keep_high_priority", "keep_if_needed", "reject", "provisional"] = "provisional"
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: CandidateProvenance


class StageCandidateManifest(BaseModel):
    mission_id: str
    task_slug: str
    generated_at: datetime = Field(default_factory=utc_now)
    candidate_dir: str = ""
    raw_candidate_count: int = 0
    candidate_count: int = 0
    candidates: list[CandidateDocument] = Field(default_factory=list)
    skipped_items: list[dict[str, Any]] = Field(default_factory=list)


class MissionCandidateManifest(BaseModel):
    mission_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    stage_manifest_paths: list[str] = Field(default_factory=list)
    total_stage_candidates: int = 0
    final_candidate_count: int = 0
    candidates: list[CandidateDocument] = Field(default_factory=list)
    dedupe_notes: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    document_id: str
    source_type: Literal["edgar_filing", "url", "local_file", "report_artifact"] = "local_file"
    canonical_source_uri: str = ""
    title: str = ""
    issuer_name: str = ""
    issuer_ticker: str = ""
    form_type: str = ""
    accession_number: str = ""
    filing_date: str = ""
    local_source_path: str = ""
    search_text: str = ""
    search_text_embedding: list[float] = Field(default_factory=list)
    search_text_model: str = ""
    search_text_dimensions: int = 0
    search_text_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DocumentTextVersionRecord(BaseModel):
    text_version_id: str
    document_id: str
    version_kind: Literal["raw", "summary"] = "raw"
    parent_text_version_id: str | None = None
    source_file_path: str = ""
    markdown_path: str = ""
    json_path: str = ""
    text_path: str = ""
    content_hash: str = ""
    char_count: int = 0
    search_text: str = ""
    search_text_embedding: list[float] = Field(default_factory=list)
    search_text_model: str = ""
    search_text_dimensions: int = 0
    search_text_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SummaryVersionRecord(DocumentTextVersionRecord):
    version_kind: Literal["summary"] = "summary"
    summary_model: str = ""


class SegmentationRecord(BaseModel):
    segmentation_id: str
    text_version_id: str
    chunker_type: Literal["hierarchical", "hybrid"] = "hybrid"
    contextualizer: str = "docling"
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ChunkRecord(BaseModel):
    chunk_id: str
    segmentation_id: str
    text_version_id: str
    chunk_index: int
    text: str
    contextualized_text: str = ""
    token_count: int | None = None
    headings: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    doc_item_refs: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    embedding: list[float] = Field(default_factory=list)
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimOccurrenceRecord(BaseModel):
    claim_occurrence_id: str
    chunk_id: str
    document_id: str
    claim_text: str
    rationale: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphTargetRef(BaseModel):
    target_level: Literal["identity", "state"] = "identity"
    target_label: str
    target_id_property: str
    target_id: str
    state_id: str | None = None
    display_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipDecision(BaseModel):
    relationship_type: Literal["ABOUT", "MENTIONS", "SUPPORTS", "IS_PRIMARY_SOURCE"]
    source_scope: Literal["document", "chunk", "claim_occurrence"] = "chunk"
    source_record_id: str
    target: GraphTargetRef
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_text: str = ""
    temporal_note: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipDecisionBatch(BaseModel):
    document_id: str
    chunk_id: str
    decisions: list[RelationshipDecision] = Field(default_factory=list)


class UnstructuredIngestionResult(BaseModel):
    document: DocumentRecord
    text_versions: list[DocumentTextVersionRecord] = Field(default_factory=list)
    segmentations: list[SegmentationRecord] = Field(default_factory=list)
    chunks: list[ChunkRecord] = Field(default_factory=list)
    claim_occurrences: list[ClaimOccurrenceRecord] = Field(default_factory=list)
    relationship_decisions: list[RelationshipDecision] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
