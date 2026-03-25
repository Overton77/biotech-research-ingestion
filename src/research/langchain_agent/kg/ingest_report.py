"""
Standalone CLI for KG ingestion.

Processes a completed research report (markdown) without re-running the
research pipeline.  Useful for ingesting existing reports or re-ingesting
after schema updates.

Usage (from repo root):
    uv run python -m src.test_runs.kg.ingest_report \\
        --report src/test_runs/agent_outputs/reports/elysium-products-and-specs.md \\
        --targets "Elysium Health" \\
        --stage_type targeted_extraction

    # or using a path relative to the agent_outputs root:
    uv run python -m src.test_runs.kg.ingest_report \\
        --report reports/elysium-products-and-specs.md \\
        --targets "Elysium Health" "elysiumhealth.com"

Flags:
    --report        Path to the .md report.  Tries absolute, then relative to
                    agent_outputs/, then relative to cwd.
    --targets       One or more target strings (e.g. company names or domains).
    --stage_type    Stage type hint for schema selection (default: targeted_extraction).
    --context       Optional free-text context string for searchText generation.
    --dry_run       If set, runs extraction + searchText but skips the Neo4j write.
    --log_level     Python log level (default: INFO).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Resolve the agent_outputs root so that relative paths work regardless of cwd
_AGENT_OUTPUTS_ROOT = (
    Path(__file__).resolve().parent.parent / "agent_outputs"
)


def _resolve_report_path(raw: str) -> Path:
    """
    Try to resolve a report path in this order:
      1. Absolute path as-is.
      2. Relative to agent_outputs/.
      3. Relative to cwd.
    """
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_to_outputs = _AGENT_OUTPUTS_ROOT / raw
    if relative_to_outputs.exists():
        return relative_to_outputs

    relative_to_cwd = Path.cwd() / raw
    if relative_to_cwd.exists():
        return relative_to_cwd

    # Return the raw path anyway — caller will get a clear FileNotFoundError
    return candidate


async def _run(args: argparse.Namespace) -> None:
    from src.research.langchain_agent.kg.run_kg_ingestion import (
        build_default_embedder,
        build_extraction_llm,
        build_searchtext_llm,
        build_selector_llm,
        run_kg_ingestion,
    )
    from src.research.langchain_agent.kg.schema_selector import load_schema_index
    from src.research.langchain_agent.neo4j_aura import Neo4jAuraSettings, Neo4jAuraClient

    # Resolve and read the report
    report_path = _resolve_report_path(args.report)
    if not report_path.exists():
        print(f"ERROR: Report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    report_text = report_path.read_text(encoding="utf-8")
    source_report = report_path.stem  # e.g. "elysium-products-and-specs"
    print(f"[ingest_report] Report: {report_path}")
    print(f"[ingest_report] Source: {source_report}")
    print(f"[ingest_report] Targets: {args.targets}")
    print(f"[ingest_report] Stage type: {args.stage_type}")
    print(f"[ingest_report] Dry run: {args.dry_run}")

    schema_index = load_schema_index()
    print(f"[ingest_report] Schema index loaded: {len(schema_index)} chunk(s)")

    if args.dry_run:
        print("[ingest_report] DRY RUN — skipping Neo4j write.")
        # Run up to Neo4j step without writing
        from src.research.langchain_agent.kg.extractor import build_extraction_agent, extract_kg_entities
        from src.research.langchain_agent.kg.schema_selector import (
            build_schema_selector_agent,
            load_schema_chunks,
            select_schema_chunks,
        )
        from src.research.langchain_agent.kg.searchtext import build_searchtext_agent
        from src.research.langchain_agent.kg.embedder import embed_batch
        from src.research.langchain_agent.kg.run_kg_ingestion import (
            _build_search_texts,
            _entity_pairs_from_extraction,
        )

        sel_llm = build_selector_llm()
        ext_llm = build_extraction_llm()
        st_llm = build_searchtext_llm()
        emb = build_default_embedder()

        selector_agent = build_schema_selector_agent(sel_llm)
        extraction_agent = build_extraction_agent(ext_llm)
        searchtext_agent = build_searchtext_agent(st_llm)

        selected_chunks = await select_schema_chunks(
            report_text=report_text,
            stage_type=args.stage_type,
            targets=args.targets,
            index=schema_index,
            selector_agent=selector_agent,
        )
        selected_schema_text = load_schema_chunks(selected_chunks)

        extraction = await extract_kg_entities(
            report_text=report_text,
            selected_schema_text=selected_schema_text,
            agent=extraction_agent,
            source_report=source_report,
        )

        pairs = _entity_pairs_from_extraction(extraction)
        context = args.context or f"Research targets: {', '.join(args.targets)}"
        search_texts = await _build_search_texts(pairs, context, searchtext_agent)

        node_keys = [k for k, _, _ in pairs]
        texts_to_embed = [search_texts[k] for k in node_keys]
        embeddings_list = await embed_batch(texts_to_embed, embedder=emb)

        print("\n[ingest_report] Dry-run extraction summary:")
        print(f"  Chunks selected:  {[c['chunk_id'] for c in selected_chunks]}")
        print(f"  Organizations:    {len(extraction.organizations)}")
        print(f"  Persons:          {len(extraction.persons)}")
        print(f"  Products:         {len(extraction.products)}")
        print(f"  Compound ingr.:   {len(extraction.compound_ingredients)}")
        print(f"  Org-person rels:  {len(extraction.org_person_relationships)}")
        print(f"  Nodes embedded:   {len(embeddings_list)}")
        print("\n[ingest_report] searchTexts (first 3):")
        for key in node_keys[:3]:
            print(f"  {key}: {search_texts[key][:120]!r}")
        if args.output_json:
            out = Path(args.output_json)
            out.write_text(
                json.dumps(extraction.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n[ingest_report] Extraction JSON written to: {out}")
        return

    # Parse research_date if provided
    from datetime import datetime as dt, timezone as tz
    research_date = None
    if args.research_date:
        research_date = dt.fromisoformat(args.research_date).replace(tzinfo=tz.utc)

    temporal_scope = None
    if args.temporal_scope:
        from src.research.langchain_agent.kg.extraction_models import TemporalScope
        temporal_scope = TemporalScope(
            mode=args.temporal_scope,
            description=f"Temporal scope: {args.temporal_scope}",
        )

    # Full run — connect to Neo4j and write
    settings = Neo4jAuraSettings.from_env()
    async with Neo4jAuraClient(settings) as client:
        result = await run_kg_ingestion(
            report_text=report_text,
            source_report=source_report,
            targets=args.targets,
            stage_type=args.stage_type,
            neo4j_client=client,
            context=args.context or "",
            schema_index=schema_index,
            research_date=research_date,
            temporal_scope=temporal_scope,
        )

    print("\n[ingest_report] Ingestion complete (bitemporal mode).")
    print(f"  Chunks used:      {result['chunks_used']}")
    print(f"  Total nodes:      {result['total_nodes']}")
    print(f"  Rels written:     {result['total_rels_written']}")
    print(f"  Rels skipped:     {result['total_rels_skipped']}")
    print(f"  States created:   {result.get('states_created', 0)}")
    print(f"  States skipped:   {result.get('states_skipped', 0)}")
    counts = result.get("node_counts", {})
    if counts:
        print(
            f"  Breakdown:        "
            f"orgs={counts.get('orgs_written', 0)}, "
            f"persons={counts.get('persons_written', 0)}, "
            f"products={counts.get('products_written', 0)}, "
            f"compounds={counts.get('compounds_written', 0)}, "
            f"lab_tests={counts.get('lab_tests_written', 0)}, "
            f"panels={counts.get('panels_written', 0)}"
        )

    if args.output_json and result.get("extraction"):
        out = Path(args.output_json)
        out.write_text(
            json.dumps(result["extraction"].model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n[ingest_report] Extraction JSON written to: {out}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest_report",
        description="Ingest a completed research report into the Neo4j knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--report",
        required=True,
        help=(
            "Path to the .md report file.  "
            "Accepts absolute paths, paths relative to agent_outputs/, or cwd."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        metavar="TARGET",
        help="One or more target entity/domain strings.",
    )
    parser.add_argument(
        "--stage_type",
        default="targeted_extraction",
        help="Stage type hint for schema selection (default: targeted_extraction).",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional free-text context string for searchText generation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run extraction and searchText but skip the Neo4j write.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        metavar="PATH",
        help="If set, write the KGExtractionResult as JSON to this path.",
    )
    parser.add_argument(
        "--research_date",
        default="",
        metavar="YYYY-MM-DD",
        help="ISO date when the research is considered current (validFrom default). Defaults to today.",
    )
    parser.add_argument(
        "--temporal_scope",
        default="",
        choices=["", "current", "as_of_date", "date_range", "unknown"],
        help="Temporal scope mode for this ingestion run.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
