from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.kg.extraction_models import IngestionTemporalContext
from src.research.langchain_agent.kg.neo4j_resolver import resolve_organization_id
from src.research.langchain_agent.kg.neo4j_writer import upsert_entity_with_state
from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient, Neo4jAuraSettings
from src.research.langchain_agent.unstructured.edgar_tools import EdgarCollectionRequest, collect_company_filings
from src.research.langchain_agent.unstructured.models import CandidateDocument, CandidateProvenance, UnstructuredIngestionConfig
from src.research.langchain_agent.unstructured.paths import isolated_validation_dir
from src.research.langchain_agent.unstructured.run_unstructured_ingestion import run_unstructured_ingestion


def _pick_primary_local_file(ticker_dir: Path, filing: dict) -> Path:
    filing_dir = (
        ticker_dir
        / "filings"
        / f"{filing['filing_date']}_{filing['form'].replace('/', '_').replace(' ', '_')}_{filing['accession_no']}"
    )
    candidates = [
        filing_dir / "primary_document.md",
        filing_dir / "primary_document.txt",
        filing_dir / "primary_document.html",
        filing_dir / "full_submission.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No primary filing artifact found in {filing_dir}")


async def _ensure_structured_issuer_exists(
    client: Neo4jAuraClient,
    *,
    issuer_name: str,
    issuer_ticker: str,
) -> str:
    existing = await resolve_organization_id(client, issuer_name)
    if existing:
        return existing

    organization_id = f"org::{issuer_ticker.lower()}"
    now = datetime.now(timezone.utc)
    await upsert_entity_with_state(
        client,
        identity_label="Organization",
        identity_merge_key="organizationId",
        node_id=organization_id,
        identity_props={
            "organizationId": organization_id,
            "name": issuer_name,
            "aliases": [issuer_ticker],
        },
        state_label="OrganizationState",
        state_props={
            "orgType": "COMPANY",
            "businessModel": "PUBLIC_COMPANY",
            "description": f"Structured validation seed for {issuer_name}",
            "websiteUrl": "",
            "legalName": issuer_name,
            "primaryIndustryTags": ["biotech", "pharmaceutical"],
            "regionsServed": [],
            "headquartersCity": "",
            "headquartersCountry": "",
        },
        search_text=f"{issuer_name} {issuer_ticker}",
        search_fields=["name", "aliases", "description"],
        embedding=[],
        source_report="unstructured_validation_seed",
        temporal_ctx=IngestionTemporalContext(
            research_date=now,
            ingestion_time=now,
            source_report="unstructured_validation_seed",
        ),
    )
    return organization_id


async def _run(args: argparse.Namespace) -> None:
    validation_dir = isolated_validation_dir(args.validation_name, root=ROOT_FILESYSTEM)
    edgar_dir = validation_dir / "edgar"
    ingestion_dir = validation_dir / "ingestion"

    if args.local_file:
        local_file = Path(args.local_file).expanduser().resolve()
        candidate = CandidateDocument(
            candidate_id=f"validation-{args.validation_name}",
            dedupe_key=f"validation::{args.validation_name}::{local_file.name}",
            source_type="edgar_filing",
            title=local_file.stem,
            local_path=str(local_file),
            relative_path="",
            issuer_name=args.issuer_name,
            issuer_ticker=args.ticker.upper(),
            form_type=args.form,
            accession_number=args.accession_number,
            filing_date=args.filing_date,
            reasons=["Local-file SEC validation candidate."],
            metadata={"local_file": str(local_file)},
            provenance=CandidateProvenance(
                mission_id=args.validation_name,
                task_slug=args.validation_name,
                discovered_by="validate_edgar_docling_flow",
                source_artifact_path=str(local_file),
            ),
        )
    else:
        collection = collect_company_filings(
            EdgarCollectionRequest(
                ticker=args.ticker,
                forms=[args.form],
                latest_only=True,
                per_form_limit=1,
                include_markdown=True,
                include_full_submission=False,
                download_mode="all",
                output_dir=str(edgar_dir),
            )
        )
        if not collection.filings:
            raise RuntimeError(f"No filings were collected for {args.ticker} {args.form}")

        filing = collection.filings[0]
        local_file = _pick_primary_local_file(Path(collection.output_dir), filing)
        candidate = CandidateDocument(
            candidate_id=f"validation-{args.ticker.lower()}-{args.form.lower()}",
            dedupe_key=f"validation::{args.ticker.lower()}::{args.form.lower()}::{filing['accession_no']}",
            source_type="edgar_filing",
            title=local_file.stem,
            local_path=str(local_file),
            relative_path="",
            issuer_name=filing.get("company", ""),
            issuer_ticker=args.ticker.upper(),
            form_type=filing.get("form", ""),
            accession_number=filing.get("accession_no", ""),
            filing_date=filing.get("filing_date", ""),
            reasons=["Isolated SEC Edgar validation candidate."],
            metadata=filing,
            provenance=CandidateProvenance(
                mission_id=args.validation_name,
                task_slug=args.validation_name,
                discovered_by="validate_edgar_docling_flow",
                source_artifact_path=str(Path(collection.manifest_path)),
            ),
        )

    settings = Neo4jAuraSettings.from_env()
    async with Neo4jAuraClient(settings) as client:
        await _ensure_structured_issuer_exists(
            client,
            issuer_name=candidate.issuer_name,
            issuer_ticker=candidate.issuer_ticker,
        )
        result = await run_unstructured_ingestion(
            candidate=candidate,
            output_dir=ingestion_dir,
            neo4j_client=client,
            config=UnstructuredIngestionConfig(
                enabled=True,
                validate_in_isolation=True,
                write_to_neo4j=True,
                max_relationship_chunks=args.max_relationship_chunks,
                parser_backend=args.parser_backend,
                llama_parse_tier=args.llama_parse_tier,
            ),
        )
        verification = await client.execute_read(
            """
            MATCH (d:Document {documentId: $documentId})
            OPTIONAL MATCH (d)-[:HAS_TEXT_VERSION]->(tv:DocumentTextVersion)
            OPTIONAL MATCH (d)-[rel:ABOUT|IS_PRIMARY_SOURCE]->(target)
            RETURN d.documentId AS document_id,
                   count(DISTINCT tv) AS text_versions,
                   count(DISTINCT rel) AS document_relationships,
                   collect(DISTINCT labels(target)) AS target_labels
            """,
            {"documentId": result.document.document_id},
        )

    summary = {
        "validation_name": args.validation_name,
        "candidate": candidate.model_dump(mode="json"),
        "document_id": result.document.document_id,
        "chunk_count": len(result.chunks),
        "relationship_count": len(result.relationship_decisions),
        "artifact_paths": result.artifact_paths,
        "verification": verification,
    }
    summary_path = validation_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the SEC Edgar to Docling to Neo4j unstructured ingestion flow.")
    parser.add_argument("--ticker", default="VRTX")
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--validation-name", default="vrtx_unstructured_validation")
    parser.add_argument("--max-relationship-chunks", type=int, default=2)
    parser.add_argument("--parser-backend", choices=["docling", "llamaparse"], default="docling")
    parser.add_argument("--llama-parse-tier", choices=["fast", "cost_effective", "agentic", "agentic_plus"], default="agentic")
    parser.add_argument("--local-file", default="")
    parser.add_argument("--issuer-name", default="VERTEX PHARMACEUTICALS INC / MA")
    parser.add_argument("--filing-date", default="")
    parser.add_argument("--accession-number", default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
