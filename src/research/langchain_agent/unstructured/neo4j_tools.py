from __future__ import annotations

from typing import Any

from langchain.tools import tool

from src.research.langchain_agent.kg.embedder import build_embedder, embed_batch
from src.research.langchain_agent.kg.neo4j_resolver import _escape_lucene
from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient

TARGET_CONFIG = [
    {
        "label": "Organization",
        "state_label": "OrganizationState",
        "id_property": "organizationId",
        "display_property": "name",
        "fulltext_index": "organization_fulltext_idx",
        "vector_index": "org_embedding_index",
    },
    {
        "label": "Person",
        "state_label": "PersonState",
        "id_property": "personId",
        "display_property": "canonicalName",
        "fulltext_index": "person_fulltext_idx",
        "vector_index": "person_search_idx",
    },
    {
        "label": "Product",
        "state_label": "ProductState",
        "id_property": "productId",
        "display_property": "name",
        "fulltext_index": "product_fulltext_idx",
        "vector_index": "product_embedding_index",
    },
    {
        "label": "CompoundForm",
        "state_label": "",
        "id_property": "compoundFormId",
        "display_property": "canonicalName",
        "fulltext_index": "compound_form_fulltext_idx",
        "vector_index": "compound_form_search_idx",
    },
]


async def _run_fulltext_search(
    client: Neo4jAuraClient,
    *,
    index_name: str,
    label: str,
    id_property: str,
    display_property: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    phrase_query = f'"{_escape_lucene(query)}"'
    cypher = f"""
    CALL db.index.fulltext.queryNodes($indexName, $query) YIELD node, score
    RETURN '{label}' AS label,
           '{id_property}' AS id_property,
           node.{id_property} AS id,
           node.{display_property} AS display_name,
           score AS score,
           'fulltext' AS retrieval_mode
    LIMIT $limit
    """
    return await client.execute_read(cypher, {"indexName": index_name, "query": phrase_query, "limit": limit})


async def _run_vector_search(
    client: Neo4jAuraClient,
    *,
    index_name: str,
    label: str,
    id_property: str,
    display_property: str,
    embedding: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    cypher = f"""
    CALL db.index.vector.queryNodes($indexName, $limit, $embedding) YIELD node, score
    RETURN '{label}' AS label,
           '{id_property}' AS id_property,
           node.{id_property} AS id,
           node.{display_property} AS display_name,
           score AS score,
           'vector' AS retrieval_mode
    """
    try:
        return await client.execute_read(cypher, {"indexName": index_name, "embedding": embedding, "limit": limit})
    except Exception:
        return []


async def fetch_state_snapshots(
    client: Neo4jAuraClient,
    *,
    label: str,
    id_property: str,
    node_id: str,
    as_of_date: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    state_label = next((cfg["state_label"] for cfg in TARGET_CONFIG if cfg["label"] == label), f"{label}State")
    if not state_label:
        return []
    if as_of_date:
        cypher = f"""
        MATCH (n:{label} {{ {id_property}: $nodeId }})-[hs:HAS_STATE]->(s:{state_label})
        WHERE hs.validFrom <= $asOfDate AND (hs.validTo IS NULL OR hs.validTo >= $asOfDate)
        RETURN properties(n) AS identity_props, properties(s) AS state_props, properties(hs) AS rel_props
        ORDER BY hs.validFrom DESC
        LIMIT $limit
        """
        return await client.execute_read(cypher, {"nodeId": node_id, "asOfDate": as_of_date, "limit": limit})

    cypher = f"""
    MATCH (n:{label} {{ {id_property}: $nodeId }})-[hs:HAS_STATE]->(s:{state_label})
    RETURN properties(n) AS identity_props, properties(s) AS state_props, properties(hs) AS rel_props
    ORDER BY hs.recordedFrom DESC
    LIMIT $limit
    """
    return await client.execute_read(cypher, {"nodeId": node_id, "limit": limit})


async def search_graph_targets(
    client: Neo4jAuraClient,
    query: str,
    *,
    limit: int = 8,
    as_of_date: str = "",
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    query_embedding = (await embed_batch([query], embedder=build_embedder()))[0]
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for config in TARGET_CONFIG:
        fulltext_rows = await _run_fulltext_search(
            client,
            index_name=config["fulltext_index"],
            label=config["label"],
            id_property=config["id_property"],
            display_property=config["display_property"],
            query=query,
            limit=limit,
        )
        vector_rows = await _run_vector_search(
            client,
            index_name=config["vector_index"],
            label=config["label"],
            id_property=config["id_property"],
            display_property=config["display_property"],
            embedding=query_embedding,
            limit=limit,
        )
        for row in [*fulltext_rows, *vector_rows]:
            key = (row["label"], row["id"])
            existing = merged.get(key)
            if existing is None or row["score"] > existing["score"]:
                merged[key] = row

    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:limit]
    enriched: list[dict[str, Any]] = []
    for row in ranked:
        snapshots = await fetch_state_snapshots(
            client,
            label=row["label"],
            id_property=row["id_property"],
            node_id=row["id"],
            as_of_date=as_of_date,
            limit=3,
        )
        state_id = ""
        if snapshots:
            state_id = snapshots[0].get("state_props", {}).get("stateId", "")
        enriched.append(
            {
                **row,
                "state_id": state_id or None,
                "snapshots": snapshots,
                "as_of_date": as_of_date,
            }
        )
    return enriched


async def fetch_graph_neighborhood(
    client: Neo4jAuraClient,
    *,
    label: str,
    id_property: str,
    node_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    cypher = f"""
    MATCH (n:{label} {{ {id_property}: $nodeId }})-[r]-(m)
    RETURN type(r) AS rel_type, labels(m) AS neighbor_labels, properties(m) AS neighbor_props, properties(r) AS rel_props
    LIMIT $limit
    """
    return await client.execute_read(cypher, {"nodeId": node_id, "limit": limit})


def build_relationship_agent_tools(client: Neo4jAuraClient) -> list[Any]:
    @tool
    async def search_existing_graph_targets(query: str, limit: int = 8, as_of_date: str = "") -> dict[str, Any]:
        """Search structured graph targets with both fulltext and vector retrieval, optionally anchored to an as-of date."""
        return {"matches": await search_graph_targets(client, query, limit=limit, as_of_date=as_of_date)}

    @tool
    async def fetch_state_snapshots_tool(label: str, id_property: str, node_id: str, as_of_date: str = "", limit: int = 5) -> dict[str, Any]:
        """Fetch state snapshots for an identity node, either active/current or as of a historical date."""
        return {
            "rows": await fetch_state_snapshots(
                client,
                label=label,
                id_property=id_property,
                node_id=node_id,
                as_of_date=as_of_date,
                limit=limit,
            )
        }

    @tool
    async def fetch_graph_neighborhood_tool(label: str, id_property: str, node_id: str, limit: int = 20) -> dict[str, Any]:
        """Fetch nearby graph structure for a candidate node before making an attachment decision."""
        return {
            "rows": await fetch_graph_neighborhood(
                client,
                label=label,
                id_property=id_property,
                node_id=node_id,
                limit=limit,
            )
        }

    return [
        search_existing_graph_targets,
        fetch_state_snapshots_tool,
        fetch_graph_neighborhood_tool,
    ]
