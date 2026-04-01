"""
Entity resolution: find an existing Neo4j node before creating a new one.

Resolution order for each node type:
  1. Exact name match  — toLower comparison on the `name` property.
  2. Fulltext phrase   — quoted-phrase Lucene query against the fulltext index.
  3. Return None       — caller mints a new UUID and MERGE creates a new node.

All nodes use `id` as the merge key and `name` as the canonical name property.
Fulltext index names match those defined in the biotech-kg SDL:
  OrganizationName, PersonSearch, ProductSearch, CompoundSearch,
  LabTestSearch, BiomarkerSearch, ConditionSearch, StudySearch, etc.
"""

from __future__ import annotations

import logging
import re

from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient

logger = logging.getLogger(__name__)

DEFAULT_SCORE_THRESHOLD: float = 1.0

_LUCENE_SPECIAL = re.compile(r'([\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\\/])')


def _escape_lucene(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r'\\\1', text)


# Fulltext index names from the biotech-kg SDL (as deployed in Neo4j Aura)
FULLTEXT_INDEX_MAP: dict[str, str] = {
    "Organization": "OrganizationName",
    "Person": "PersonSearch",
    "Product": "ProductSearch",
    "Compound": "CompoundSearch",
    "CompoundForm": "CompoundSearch",
    "Biomarker": "BiomarkerSearch",
    "Condition": "ConditionSearch",
    "Study": "StudySearch",
    "LabTest": "LabTestSearch",
    "PanelDefinition": "PanelDefinition",
    "TechnologyPlatform": "TechnologyPlatformSearch",
    "Mechanism": "MechanismSearch",
    "Metric": "MetricSearch",
}


async def resolve_node_id(
    client: Neo4jAuraClient,
    label: str,
    name: str,
    *,
    fulltext_index: str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str | None:
    """
    Return the `id` of a node that matches *name*, or None.

    Args:
        client:           Connected Neo4jAuraClient.
        label:            Node label, e.g. "Organization".
        name:             The name to search for.
        fulltext_index:   Name of the fulltext index (auto-resolved if None).
        score_threshold:  Minimum Lucene score to accept a fulltext hit.
    """
    if not name or not name.strip():
        return None

    ft_index = fulltext_index or FULLTEXT_INDEX_MAP.get(label)

    # Step 1: exact match (case-insensitive)
    try:
        rows = await client.execute_read(
            f"MATCH (n:{label}) "
            f"WHERE toLower(n.name) = toLower($name) "
            f"RETURN n.id AS id "
            f"LIMIT 1",
            {"name": name.strip()},
        )
        if rows:
            existing_id: str = rows[0]["id"]
            logger.debug("[resolver] Exact match  %-20s %-40s → %s", label, name, existing_id)
            return existing_id
    except Exception as exc:
        logger.warning("[resolver] Exact-match query failed for %s '%s': %s", label, name, exc)

    # Step 2: fulltext phrase search
    if ft_index:
        phrase = f'"{_escape_lucene(name.strip())}"'
        try:
            rows = await client.execute_read(
                "CALL db.index.fulltext.queryNodes($index, $query) "
                "YIELD node, score "
                "WHERE score >= $threshold "
                "RETURN node.id AS id, score "
                "ORDER BY score DESC "
                "LIMIT 1",
                {
                    "index": ft_index,
                    "query": phrase,
                    "threshold": score_threshold,
                },
            )
            if rows:
                existing_id = rows[0]["id"]
                score: float = rows[0]["score"]
                logger.debug(
                    "[resolver] Fulltext hit %-20s %-40s → %s (score=%.2f)",
                    label, name, existing_id, score,
                )
                return existing_id
        except Exception as exc:
            logger.debug("[resolver] Fulltext search unavailable for %s '%s': %s", label, name, exc)

    logger.debug("[resolver] No match     %-20s %-40s → new node", label, name)
    return None


# ---------------------------------------------------------------------------
# Typed helpers — convenience wrappers
# ---------------------------------------------------------------------------


async def resolve_organization_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Organization", name)


async def resolve_person_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Person", name)


async def resolve_product_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Product", name)


async def resolve_compound_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Compound", name)


async def resolve_compound_form_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "CompoundForm", name)


async def resolve_study_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Study", name)


async def resolve_condition_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Condition", name)


async def resolve_biomarker_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "Biomarker", name)


async def resolve_lab_test_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "LabTest", name)


async def resolve_panel_definition_id(client: Neo4jAuraClient, name: str) -> str | None:
    return await resolve_node_id(client, "PanelDefinition", name)
