"""
One-time Neo4j vector index setup.

Creates vector indexes for all node labels that carry an embedding.
Run this script once before the first ingestion run.

IMPORTANT: The vector dimensions (1536) must match the embedding model
used in embedder.py (text-embedding-3-small).  If you switch models,
drop and recreate the indexes.

Usage:
    uv run python -m src.test_runs.kg.setup_indexes
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_VECTOR_DIMENSIONS = 1536
_SIMILARITY_FUNCTION = "cosine"

# (index_name, label)
_NODE_INDEXES: list[tuple[str, str]] = [
    ("organization_search_idx", "Organization"),
    ("person_search_idx", "Person"),
    ("product_search_idx", "Product"),
    ("compound_form_search_idx", "CompoundForm"),
    ("lab_test_search_idx", "LabTest"),
    ("panel_definition_search_idx", "PanelDefinition"),
]

# Fulltext indexes for non-vector keyword search
_FULLTEXT_INDEXES: list[tuple[str, str, list[str]]] = [
    ("organization_fulltext_idx", "Organization", ["name", "aliases", "description", "searchText"]),
    ("person_fulltext_idx", "Person", ["canonicalName", "bio", "searchText"]),
    ("product_fulltext_idx", "Product", ["name", "synonyms", "description", "searchText"]),
    ("compound_form_fulltext_idx", "CompoundForm", ["canonicalName", "searchText"]),
    ("lab_test_fulltext_idx", "LabTest", ["name", "synonyms", "whatItMeasures", "searchText"]),
    ("panel_definition_fulltext_idx", "PanelDefinition", ["canonicalName", "aliases", "description", "searchText"]),
]


def _vector_index_cypher(index_name: str, label: str) -> str:
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS\n"
        f"FOR (n:{label}) ON (n.embedding)\n"
        f"OPTIONS {{ indexConfig: {{ "
        f"`vector.dimensions`: {_VECTOR_DIMENSIONS}, "
        f"`vector.similarity_function`: '{_SIMILARITY_FUNCTION}' "
        f"}} }}"
    )


def _fulltext_index_cypher(
    index_name: str, label: str, properties: list[str]
) -> str:
    props = ", ".join(f"n.{p}" for p in properties)
    return (
        f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS\n"
        f"FOR (n:{label}) ON EACH [{props}]"
    )


async def _setup(dry_run: bool = False) -> None:
    from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient, Neo4jAuraSettings

    settings = Neo4jAuraSettings.from_env()

    print(f"Connecting to Neo4j: {settings.uri} / db={settings.database}")

    async with Neo4jAuraClient(settings) as client:
        # Vector indexes
        for index_name, label in _NODE_INDEXES:
            cypher = _vector_index_cypher(index_name, label)
            print(f"\n[vector] {index_name} ({label})")
            if dry_run:
                print(f"  DRY RUN — would execute:\n  {cypher.strip()}")
            else:
                try:
                    await client.execute_write(cypher)
                    print(f"  OK (created or already exists)")
                except Exception as exc:
                    logger.error("  FAILED: %s", exc)

        # Fulltext indexes
        for index_name, label, properties in _FULLTEXT_INDEXES:
            cypher = _fulltext_index_cypher(index_name, label, properties)
            print(f"\n[fulltext] {index_name} ({label})")
            if dry_run:
                print(f"  DRY RUN — would execute:\n  {cypher.strip()}")
            else:
                try:
                    await client.execute_write(cypher)
                    print(f"  OK (created or already exists)")
                except Exception as exc:
                    logger.error("  FAILED: %s", exc)

    print("\nIndex setup complete.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Neo4j vector and fulltext indexes for the KG pipeline."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the Cypher statements without executing them.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    asyncio.run(_setup(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
