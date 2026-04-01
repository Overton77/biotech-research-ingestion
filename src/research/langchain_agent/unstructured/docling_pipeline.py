from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI
from llama_cloud import LlamaCloud
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from src.research.langchain_agent.unstructured.models import (
    CandidateDocument,
    ChunkCleaningConfig,
    ChunkEnhancementConfig,
    ChunkRecord,
    DocumentRecord,
    DocumentTextVersionRecord,
    SegmentationRecord,
    SummaryPolicy,
    SummaryVersionRecord,
)

load_dotenv()

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_DOC_MARKER_RE = re.compile(r"\{\d+\}-+")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_title(candidate: CandidateDocument) -> str:
    if candidate.title:
        return candidate.title
    if candidate.local_path:
        return Path(candidate.local_path).stem
    if candidate.uri:
        return Path(candidate.uri).name
    return candidate.candidate_id


def _serializer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _chunk_meta(chunk: Any) -> tuple[list[str], list[str], list[str]]:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return [], [], []

    doc_items = getattr(meta, "doc_items", None) or []
    doc_refs: list[str] = []
    captions: list[str] = []
    for item in doc_items:
        self_ref = getattr(item, "self_ref", None)
        if self_ref:
            doc_refs.append(str(self_ref))
        caption = getattr(item, "caption", None)
        if caption:
            captions.append(str(caption))

    headings = _serializer_list(getattr(meta, "headings", None))
    return headings, captions, doc_refs


def _build_chunker(chunker_type: Literal["hierarchical", "hybrid"] = "hybrid") -> tuple[Any, str]:
    from docling.chunking import HierarchicalChunker

    if chunker_type == "hybrid":
        try:
            from docling.chunking import HybridChunker

            return HybridChunker(), "hybrid"
        except Exception:
            pass

    return HierarchicalChunker(), "hierarchical"


def _convert_with_docling(source_path: Path) -> Any:
    converter = DocumentConverter()
    result = converter.convert(str(source_path))
    if result.status.name != "SUCCESS":
        errors = "; ".join(str(error) for error in result.errors)
        raise RuntimeError(f"Docling conversion failed for {source_path}: {errors}")
    return result


def _convert_with_llamaparse(source_path: Path, *, tier: str = "agentic") -> tuple[str, dict[str, Any]]:
    client = LlamaCloud()
    result = client.parsing.parse(
        upload_file=str(source_path),
        tier=tier,
        version="latest",
        expand=["markdown_full"],
    )
    markdown_text = result.markdown_full or ""
    if not markdown_text.strip():
        raise RuntimeError(f"LlamaParse returned empty markdown for {source_path}")
    return markdown_text, {
        "job_id": getattr(result.job, "id", ""),
        "job_status": str(getattr(result.job, "status", "")),
        "parser_backend": "llamaparse",
        "tier": tier,
    }


def _resolve_preferred_source_path(source_path: Path) -> Path:
    if source_path.suffix.lower() == ".html":
        markdown_sibling = source_path.with_suffix(".md")
        if markdown_sibling.exists():
            return markdown_sibling
    return source_path


def _clean_chunk_text(text: str, cleaning: ChunkCleaningConfig) -> str:
    cleaned = html.unescape(text)
    if cleaning.strip_html:
        cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    if cleaning.strip_markdown:
        cleaned = _MD_CODE_FENCE_RE.sub(" ", cleaned)
        cleaned = cleaned.replace("|", " ")
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"[*_`>#-]+", " ", cleaned)
    if cleaning.drop_doc_markers:
        cleaned = _DOC_MARKER_RE.sub(" ", cleaned)
    if cleaning.collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


async def _maybe_enhance_chunks(
    *,
    chunks: list[ChunkRecord],
    enhancement: ChunkEnhancementConfig,
) -> None:
    if not enhancement.enabled:
        return

    llm = ChatOpenAI(model=enhancement.model, temperature=0.0, max_retries=2)
    for chunk in chunks[: enhancement.max_chunks]:
        base_text = (chunk.contextualized_text or chunk.text)[: enhancement.max_input_chars]
        if not base_text.strip():
            continue
        response = await llm.ainvoke(
            "Clean and lightly enrich the following chunk for retrieval. Preserve factual meaning and temporal qualifiers. "
            "Remove formatting noise and keep the output concise.\n\n"
            f"{base_text}"
        )
        enhanced = str(response.content).strip()
        if enhanced:
            chunk.contextualized_text = enhanced
            chunk.metadata["enhanced_by_llm"] = True
            chunk.metadata["enhancement_model"] = enhancement.model


def _build_document_record(
    candidate: CandidateDocument,
    source_path: Path,
    canonical_source: str,
    metadata: dict[str, Any],
) -> DocumentRecord:
    document_id = _sha256_text(canonical_source)[:32]
    return DocumentRecord(
        document_id=document_id,
        source_type=candidate.source_type,
        canonical_source_uri=canonical_source,
        title=_safe_title(candidate),
        issuer_name=candidate.issuer_name,
        issuer_ticker=candidate.issuer_ticker,
        form_type=candidate.form_type,
        accession_number=candidate.accession_number,
        filing_date=candidate.filing_date,
        local_source_path=str(source_path),
        metadata=metadata,
    )


def _write_common_artifacts(
    output_dir: Path,
    document: DocumentRecord,
    markdown_text: str,
    json_payload: str,
    plain_text: str,
) -> tuple[Path, Path, Path, Path]:
    markdown_path = output_dir / "document.md"
    json_path = output_dir / "document.docling.json"
    text_path = output_dir / "document.txt"
    record_path = output_dir / "document_record.json"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    json_path.write_text(json_payload, encoding="utf-8")
    text_path.write_text(plain_text, encoding="utf-8")
    record_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return markdown_path, json_path, text_path, record_path


async def _maybe_create_summary_version(
    *,
    raw_version: DocumentTextVersionRecord,
    chunks: list[ChunkRecord],
    output_dir: Path,
    policy: SummaryPolicy,
) -> SummaryVersionRecord | None:
    if not policy.enabled or len(chunks) < policy.min_chunk_count:
        return None

    summary_source = "\n\n".join(chunk.contextualized_text or chunk.text for chunk in chunks[:12])
    summary_source = summary_source[: policy.max_input_chars]
    if not summary_source.strip():
        return None

    llm = ChatOpenAI(model=policy.model, temperature=0.0, max_retries=2)
    prompt = (
        "Summarize the following primary-source document excerpt for downstream KG ingestion.\n"
        "Keep it factual, preserve temporal cues, distinguish raw source from any interpretation, "
        "and do not invent claims.\n\n"
        f"{summary_source}"
    )
    response = await llm.ainvoke(prompt)
    summary_text = str(response.content).strip()
    if not summary_text:
        return None

    summary_path = output_dir / "summary_version.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    summary_hash = _sha256_text(summary_text)

    return SummaryVersionRecord(
        text_version_id=summary_hash,
        document_id=raw_version.document_id,
        parent_text_version_id=raw_version.text_version_id,
        source_file_path=raw_version.source_file_path,
        markdown_path="",
        json_path="",
        text_path=str(summary_path),
        content_hash=summary_hash,
        char_count=len(summary_text),
        summary_model=policy.model,
        metadata={"derived_from": raw_version.text_version_id},
    )


async def materialize_candidate_with_docling(
    *,
    candidate: CandidateDocument,
    output_dir: Path,
    chunker_type: Literal["hierarchical", "hybrid"] = "hybrid",
    summary_policy: SummaryPolicy | None = None,
    cleaning: ChunkCleaningConfig | None = None,
    enhancement: ChunkEnhancementConfig | None = None,
) -> tuple[DocumentRecord, list[DocumentTextVersionRecord], list[SegmentationRecord], list[ChunkRecord], list[str]]:
    source_path = _resolve_preferred_source_path(Path(candidate.local_path).expanduser().resolve())
    if not source_path.exists():
        raise FileNotFoundError(f"Candidate source file does not exist: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = _convert_with_docling(source_path)
    canonical_source = candidate.uri or candidate.local_path
    document = _build_document_record(
        candidate,
        source_path,
        canonical_source,
        {
            "candidate_id": candidate.candidate_id,
            "candidate_dedupe_key": candidate.dedupe_key,
            "candidate_metadata": candidate.metadata,
            "parser_backend": "docling",
        },
    )

    markdown_text = result.document.export_to_markdown()
    json_text = result.document.model_dump_json(indent=2)
    plain_text = source_path.read_text(encoding="utf-8") if source_path.suffix.lower() in {".md", ".txt"} else markdown_text
    markdown_path, json_path, text_path, record_path = _write_common_artifacts(
        output_dir,
        document,
        markdown_text,
        json_text,
        plain_text,
    )

    raw_hash = _sha256_text(plain_text)
    raw_version = DocumentTextVersionRecord(
        text_version_id=raw_hash,
        document_id=document.document_id,
        version_kind="raw",
        source_file_path=str(source_path),
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        text_path=str(text_path),
        content_hash=raw_hash,
        char_count=len(plain_text),
        metadata={
            "docling_status": result.status.name,
            "docling_document_hash": str(result.input.document_hash),
            "page_count": len(result.pages),
            "input_format": result.input.format.name if result.input.format else None,
            "parser_backend": "docling",
        },
    )

    chunker, realized_chunker_type = _build_chunker(chunker_type)
    segmentation = SegmentationRecord(
        segmentation_id=_sha256_text(f"{raw_version.text_version_id}:{realized_chunker_type}")[:32],
        text_version_id=raw_version.text_version_id,
        chunker_type=realized_chunker_type,
        metadata={
            "docling_document_hash": raw_version.metadata.get("docling_document_hash", ""),
            "parser_backend": "docling",
        },
    )

    cleaning_cfg = cleaning or ChunkCleaningConfig()
    chunks: list[ChunkRecord] = []
    for index, chunk in enumerate(chunker.chunk(dl_doc=result.document)):
        contextualized_text = chunker.contextualize(chunk=chunk)
        headings, captions, doc_refs = _chunk_meta(chunk)
        chunk_text = getattr(chunk, "text", None) or contextualized_text
        chunks.append(
            ChunkRecord(
                chunk_id=f"{raw_version.text_version_id}:{index:04d}",
                segmentation_id=segmentation.segmentation_id,
                text_version_id=raw_version.text_version_id,
                chunk_index=index,
                text=_clean_chunk_text(str(chunk_text), cleaning_cfg),
                contextualized_text=_clean_chunk_text(str(contextualized_text), cleaning_cfg),
                headings=headings,
                captions=captions,
                doc_item_refs=doc_refs,
                metadata={"serializer": "docling_contextualize", "parser_backend": "docling"},
            )
        )

    segmentation.chunk_count = len(chunks)
    await _maybe_enhance_chunks(chunks=chunks, enhancement=enhancement or ChunkEnhancementConfig())

    chunks_path = output_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2, default=str),
        encoding="utf-8",
    )
    segmentation_path = output_dir / "segmentation.json"
    segmentation_path.write_text(segmentation.model_dump_json(indent=2), encoding="utf-8")
    raw_version_path = output_dir / "raw_text_version.json"
    raw_version_path.write_text(raw_version.model_dump_json(indent=2), encoding="utf-8")

    text_versions: list[DocumentTextVersionRecord] = [raw_version]
    artifact_paths = [
        str(record_path),
        str(markdown_path),
        str(json_path),
        str(text_path),
        str(raw_version_path),
        str(segmentation_path),
        str(chunks_path),
    ]

    summary_version = await _maybe_create_summary_version(
        raw_version=raw_version,
        chunks=chunks,
        output_dir=output_dir,
        policy=summary_policy or SummaryPolicy(),
    )
    if summary_version is not None:
        summary_version_path = output_dir / "summary_text_version.json"
        summary_version_path.write_text(summary_version.model_dump_json(indent=2), encoding="utf-8")
        text_versions.append(summary_version)
        artifact_paths.append(str(summary_version_path))
        artifact_paths.append(summary_version.text_path)

    return document, text_versions, [segmentation], chunks, artifact_paths


async def materialize_candidate_with_llamaparse(
    *,
    candidate: CandidateDocument,
    output_dir: Path,
    summary_policy: SummaryPolicy | None = None,
    cleaning: ChunkCleaningConfig | None = None,
    enhancement: ChunkEnhancementConfig | None = None,
    tier: str = "agentic",
) -> tuple[DocumentRecord, list[DocumentTextVersionRecord], list[SegmentationRecord], list[ChunkRecord], list[str]]:
    source_path = Path(candidate.local_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Candidate source file does not exist: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_text, parser_metadata = _convert_with_llamaparse(source_path, tier=tier)
    canonical_source = candidate.uri or candidate.local_path
    document = _build_document_record(
        candidate,
        source_path,
        canonical_source,
        {
            "candidate_id": candidate.candidate_id,
            "candidate_dedupe_key": candidate.dedupe_key,
            "candidate_metadata": candidate.metadata,
            **parser_metadata,
        },
    )

    markdown_path, json_path, text_path, record_path = _write_common_artifacts(
        output_dir,
        document,
        markdown_text,
        json.dumps({"markdown_full": markdown_text, "metadata": parser_metadata}, indent=2),
        markdown_text,
    )

    raw_hash = _sha256_text(markdown_text)
    raw_version = DocumentTextVersionRecord(
        text_version_id=raw_hash,
        document_id=document.document_id,
        version_kind="raw",
        source_file_path=str(source_path),
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        text_path=str(text_path),
        content_hash=raw_hash,
        char_count=len(markdown_text),
        metadata={"parser_backend": "llamaparse", **parser_metadata},
    )

    llama_doc = LlamaDocument(text=markdown_text, metadata={"source": str(source_path)})
    markdown_parser = MarkdownNodeParser()
    nodes = markdown_parser.get_nodes_from_documents([llama_doc])
    split_nodes = SentenceSplitter(chunk_size=1200, chunk_overlap=150).get_nodes_from_documents(nodes)

    cleaning_cfg = cleaning or ChunkCleaningConfig()
    segmentation = SegmentationRecord(
        segmentation_id=_sha256_text(f"{raw_version.text_version_id}:llamaparse")[:32],
        text_version_id=raw_version.text_version_id,
        chunker_type="hierarchical",
        contextualizer="llamaparse_markdown",
        metadata={"parser_backend": "llamaparse", "tier": tier},
    )

    chunks: list[ChunkRecord] = []
    for index, node in enumerate(split_nodes):
        node_text = getattr(node, "text", "") or ""
        node_meta = getattr(node, "metadata", {}) or {}
        headings = [str(value) for key, value in node_meta.items() if "header" in key.lower() or "section" in key.lower()]
        cleaned = _clean_chunk_text(node_text, cleaning_cfg)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{raw_version.text_version_id}:{index:04d}",
                segmentation_id=segmentation.segmentation_id,
                text_version_id=raw_version.text_version_id,
                chunk_index=index,
                text=cleaned,
                contextualized_text=cleaned,
                headings=headings,
                captions=[],
                doc_item_refs=[],
                metadata={"parser_backend": "llamaparse", "node_metadata": node_meta},
            )
        )

    segmentation.chunk_count = len(chunks)
    await _maybe_enhance_chunks(chunks=chunks, enhancement=enhancement or ChunkEnhancementConfig())

    chunks_path = output_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2, default=str),
        encoding="utf-8",
    )
    segmentation_path = output_dir / "segmentation.json"
    segmentation_path.write_text(segmentation.model_dump_json(indent=2), encoding="utf-8")
    raw_version_path = output_dir / "raw_text_version.json"
    raw_version_path.write_text(raw_version.model_dump_json(indent=2), encoding="utf-8")

    text_versions: list[DocumentTextVersionRecord] = [raw_version]
    artifact_paths = [
        str(record_path),
        str(markdown_path),
        str(json_path),
        str(text_path),
        str(raw_version_path),
        str(segmentation_path),
        str(chunks_path),
    ]

    summary_version = await _maybe_create_summary_version(
        raw_version=raw_version,
        chunks=chunks,
        output_dir=output_dir,
        policy=summary_policy or SummaryPolicy(),
    )
    if summary_version is not None:
        summary_version_path = output_dir / "summary_text_version.json"
        summary_version_path.write_text(summary_version.model_dump_json(indent=2), encoding="utf-8")
        text_versions.append(summary_version)
        artifact_paths.append(str(summary_version_path))
        artifact_paths.append(summary_version.text_path)

    return document, text_versions, [segmentation], chunks, artifact_paths
