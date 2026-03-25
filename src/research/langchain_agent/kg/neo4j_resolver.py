"""
Entity resolution: find an existing Neo4j node before creating a new one.

Resolution order for each node type:
  1. Exact name match  — toLower comparison on the canonical name property.
  2. Fulltext phrase   — quoted-phrase Lucene query against the fulltext index.
  3. Return None       — caller mints a new UUID and MERGE creates a new node.

Why this stops duplicates:
  Each MERGE in neo4j_writer.py uses the *Id property as the match key.
  When the resolver returns an existing ID, MERGE hits the existing node and
  only updates its properties — it never creates a second node.
  When the resolver returns None a fresh UUID is used and a new node is created.

Fulltext index names must match what setup_indexes.py creates:
  organization_fulltext_idx, person_fulltext_idx,
  product_fulltext_idx, compound_form_fulltext_idx,
  lab_test_fulltext_idx, panel_definition_fulltext_idx
"""

from __future__ import annotations

import logging
import re

from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient

logger = logging.getLogger(__name__)

# Minimum Lucene phrase-match score to accept a fulltext hit.
# Exact phrase matches on short name fields typically score 1.0 – 5.0.
# Raise this if you see false-positive merges; lower it if valid matches are missed.
DEFAULT_SCORE_THRESHOLD: float = 1.0

# Lucene special characters that must be escaped in query strings.
_LUCENE_SPECIAL = re.compile(r'([\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\\/])')


def _escape_lucene(text: str) -> str:
    """Escape Lucene special characters so the name is treated as a literal."""
    return _LUCENE_SPECIAL.sub(r'\\\1', text)


async def resolve_node_id(
    client: Neo4jAuraClient,
    label: str,
    name: str,
    name_property: str,
    id_property: str,
    fulltext_index: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str | None:
    """
    Return the *id_property* value of a node that matches *name*, or None.

    Args:
        client:           Connected Neo4jAuraClient.
        label:            Node label, e.g. "Organization".
        name:             The canonical name to search for.
        name_property:    The property holding the name, e.g. "name" or "canonicalName".
        id_property:      The primary-key property, e.g. "organizationId".
        fulltext_index:   Name of the fulltext index to use as fallback.
        score_threshold:  Minimum Lucene score to accept a fulltext hit.

    Returns:
        The existing node's id string, or None if no match is found.
    """
    if not name or not name.strip():
        return None

    # ------------------------------------------------------------------
    # Step 1: exact match (case-insensitive)
    # ------------------------------------------------------------------
    try:
        rows = await client.execute_read(
            f"MATCH (n:{label}) "
            f"WHERE toLower(n.{name_property}) = toLower($name) "
            f"RETURN n.{id_property} AS id "
            f"LIMIT 1",
            {"name": name.strip()},
        )
        if rows:
            existing_id: str = rows[0]["id"]
            logger.debug(
                "[resolver] Exact match  %-20s %-40s → %s",
                label,
                name,
                existing_id,
            )
            return existing_id
    except Exception as exc:
        logger.warning("[resolver] Exact-match query failed for %s '%s': %s", label, name, exc)

    # ------------------------------------------------------------------
    # Step 2: fulltext phrase search
    # ------------------------------------------------------------------
    phrase = f'"{_escape_lucene(name.strip())}"'
    try:
        rows = await client.execute_read(
            "CALL db.index.fulltext.queryNodes($index, $query) "
            "YIELD node, score "
            "WHERE score >= $threshold "
            f"RETURN node.{id_property} AS id, score "
            "ORDER BY score DESC "
            "LIMIT 1",
            {
                "index": fulltext_index,
                "query": phrase,
                "threshold": score_threshold,
            },
        )
        if rows:
            existing_id = rows[0]["id"]
            score: float = rows[0]["score"]
            logger.debug(
                "[resolver] Fulltext hit %-20s %-40s → %s (score=%.2f)",
                label,
                name,
                existing_id,
                score,
            )
            return existing_id
    except Exception as exc:
        # Graceful fallback: fulltext index may not exist yet (pre-setup run).
        logger.debug(
            "[resolver] Fulltext search unavailable for %s '%s': %s", label, name, exc
        )

    logger.debug("[resolver] No match     %-20s %-40s → new node", label, name)
    return None


# ---------------------------------------------------------------------------
# Typed helpers — one per node label, using the correct properties
# ---------------------------------------------------------------------------


async def resolve_organization_id(
    client: Neo4jAuraClient, name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="Organization",
        name=name,
        name_property="name",
        id_property="organizationId",
        fulltext_index="organization_fulltext_idx",
    )


async def resolve_person_id(
    client: Neo4jAuraClient, canonical_name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="Person",
        name=canonical_name,
        name_property="canonicalName",
        id_property="personId",
        fulltext_index="person_fulltext_idx",
    )


async def resolve_product_id(
    client: Neo4jAuraClient, name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="Product",
        name=name,
        name_property="name",
        id_property="productId",
        fulltext_index="product_fulltext_idx",
    )


async def resolve_compound_form_id(
    client: Neo4jAuraClient, canonical_name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="CompoundForm",
        name=canonical_name,
        name_property="canonicalName",
        id_property="compoundFormId",
        fulltext_index="compound_form_fulltext_idx",
    )


async def resolve_lab_test_id(
    client: Neo4jAuraClient, name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="LabTest",
        name=name,
        name_property="name",
        id_property="labTestId",
        fulltext_index="lab_test_fulltext_idx",
    )


async def resolve_panel_definition_id(
    client: Neo4jAuraClient, canonical_name: str
) -> str | None:
    return await resolve_node_id(
        client=client,
        label="PanelDefinition",
        name=canonical_name,
        name_property="canonicalName",
        id_property="panelDefinitionId",
        fulltext_index="panel_definition_fulltext_idx",
    )
