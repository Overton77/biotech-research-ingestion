"""
Neo4j bitemporal writer — identity/state separation with temporal relationships.

Architecture:
  - Identity nodes are durable anchors with minimal stable properties.
  - State nodes are immutable snapshots of descriptive properties.
  - Identity → State is attached via HAS_STATE with bitemporal bounds.
  - Structural relationships carry bitemporal bounds.

All nodes use `id` as the merge key and `name` as the universal name property.
State change detection uses SHA-256 hashing of normalized state properties.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.research.langchain_agent.kg.extraction_models import (
    IngestionTemporalContext,
    TemporalQualifier,
)
from src.research.langchain_agent.kg.temporal import (
    MERGE_KEY,
    OPEN_ENDED,
    compute_state_hash,
    default_bitemporal_props,
    now_utc,
)
from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# State label mapping for nodes that have identity/state separation
STATE_LABELS: dict[str, str] = {
    "Organization": "OrganizationState",
    "Person": "PersonState",
    "Product": "ProductState",
}

# Allowed relationship types for upsert_temporal_relationship
ALLOWED_REL_TYPES = {
    "EMPLOYS", "FOUNDED_BY", "HAS_BOARD_MEMBER", "HAS_CEO",
    "ADVISES", "HOLDS_ROLE_AT", "AFFILIATED_WITH",
    "OFFERS", "MANUFACTURES",
    "CONTAINS_COMPOUND_FORM",
    "DELIVERS_LABTEST", "IMPLEMENTS_PANEL",
    "INCLUDES_LABTEST", "INCLUDES_BIOMARKER",
    "SPONSORED_BY", "OPERATED_BY",
    "INVESTIGATES", "INVESTIGATED_BY",
    "MEASURES", "EVALUATES",
}


# ---------------------------------------------------------------------------
# Bitemporal helpers
# ---------------------------------------------------------------------------


def _temporal_props_from_qualifier(
    qualifier: TemporalQualifier | None,
    temporal_ctx: IngestionTemporalContext,
) -> dict[str, Any]:
    base = default_bitemporal_props(
        research_date=temporal_ctx.research_date,
        ingestion_time=temporal_ctx.ingestion_time,
    )

    if qualifier:
        if qualifier.valid_from:
            base["validFrom"] = qualifier.valid_from
        if qualifier.valid_to:
            base["validTo"] = qualifier.valid_to
        if qualifier.temporal_note:
            base["temporalNote"] = qualifier.temporal_note

    return base


# ---------------------------------------------------------------------------
# Generic identity + state writer
# ---------------------------------------------------------------------------


async def upsert_identity_node(
    client: Neo4jAuraClient,
    label: str,
    node_id: str,
    identity_props: dict[str, Any],
) -> None:
    """MERGE a durable identity node using `id` as the merge key."""
    set_clauses = ", ".join(
        f"n.{k} = ${k}" for k in identity_props if k != MERGE_KEY
    )
    set_part = f"SET {set_clauses}" if set_clauses else ""
    cypher = f"""
        MERGE (n:{label} {{ {MERGE_KEY}: $nodeId }})
        ON CREATE SET n.createdAt = datetime()
        {set_part}
    """
    params = {"nodeId": node_id, **identity_props}
    await client.execute_write(cypher, parameters=params)


async def _get_active_state_hash(
    client: Neo4jAuraClient,
    identity_label: str,
    node_id: str,
    state_label: str,
) -> str | None:
    cypher = f"""
        MATCH (n:{identity_label} {{ {MERGE_KEY}: $nodeId }})
              -[r:HAS_STATE]->
              (s:{state_label})
        WHERE r.validTo IS NULL AND r.recordedTo IS NULL
        RETURN s.stateHash AS hash
        LIMIT 1
    """
    rows = await client.execute_read(cypher, {"nodeId": node_id})
    if rows:
        return rows[0]["hash"]
    return None


async def _close_active_state(
    client: Neo4jAuraClient,
    identity_label: str,
    node_id: str,
    state_label: str,
    close_time: str,
) -> None:
    cypher = f"""
        MATCH (n:{identity_label} {{ {MERGE_KEY}: $nodeId }})
              -[r:HAS_STATE]->
              (s:{state_label})
        WHERE r.validTo IS NULL AND r.recordedTo IS NULL
        SET r.validTo = $closeTime,
            r.recordedTo = $closeTime
    """
    await client.execute_write(
        cypher,
        parameters={"nodeId": node_id, "closeTime": close_time},
    )


async def create_state_node(
    client: Neo4jAuraClient,
    state_label: str,
    state_id: str,
    state_props: dict[str, Any],
    state_hash: str,
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
    source_report: str,
) -> None:
    set_clauses = ", ".join(
        f"s.{k} = ${k}" for k in state_props
    )
    extra_set = f"SET {set_clauses}" if set_clauses else ""
    cypher = f"""
        CREATE (s:{state_label} {{
            id: $stateId,
            stateHash: $stateHash,
            sourceReport: $sourceReport,
            searchText: $searchText,
            searchFields: $searchFields,
            searchEmbedding: $embedding,
            embeddingModel: $embeddingModel,
            embeddingDimensions: $embeddingDimensions,
            createdAt: datetime()
        }})
        {extra_set}
    """
    params = {
        "stateId": state_id,
        "stateHash": state_hash,
        "sourceReport": source_report,
        "searchText": search_text,
        "searchFields": search_fields,
        "embedding": embedding,
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingDimensions": EMBEDDING_DIMENSIONS,
        **state_props,
    }
    await client.execute_write(cypher, parameters=params)


async def create_has_state_rel(
    client: Neo4jAuraClient,
    identity_label: str,
    node_id: str,
    state_label: str,
    state_id: str,
    temporal_props: dict[str, Any],
) -> None:
    cypher = f"""
        MATCH (n:{identity_label} {{ {MERGE_KEY}: $nodeId }})
        MATCH (s:{state_label} {{ id: $stateId }})
        CREATE (n)-[r:HAS_STATE {{
            validFrom: $validFrom,
            validTo: $validTo,
            recordedFrom: $recordedFrom,
            recordedTo: $recordedTo
        }}]->(s)
    """
    await client.execute_write(
        cypher,
        parameters={
            "nodeId": node_id,
            "stateId": state_id,
            **temporal_props,
        },
    )


async def upsert_entity_with_state(
    client: Neo4jAuraClient,
    *,
    identity_label: str,
    node_id: str,
    identity_props: dict[str, Any],
    state_label: str,
    state_props: dict[str, Any],
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
    source_report: str,
    temporal_ctx: IngestionTemporalContext,
    temporal_qualifier: TemporalQualifier | None = None,
) -> bool:
    """
    Upsert an identity node and conditionally create a new state snapshot.
    Returns True if a new state was created, False if hash matched (no-op).
    """
    await upsert_identity_node(client, identity_label, node_id, identity_props)

    incoming_hash = compute_state_hash(state_props)

    active_hash = await _get_active_state_hash(
        client, identity_label, node_id, state_label,
    )

    if active_hash == incoming_hash:
        logger.debug(
            "[neo4j_writer] State unchanged for %s:%s — skipping.",
            identity_label, node_id,
        )
        return False

    if active_hash is not None:
        close_time = (temporal_ctx.ingestion_time or now_utc()).isoformat()
        await _close_active_state(
            client, identity_label, node_id, state_label, close_time,
        )

    state_id = str(uuid4())
    await create_state_node(
        client, state_label, state_id, state_props,
        incoming_hash, search_text, search_fields, embedding, source_report,
    )

    temporal_props = _temporal_props_from_qualifier(temporal_qualifier, temporal_ctx)
    await create_has_state_rel(
        client, identity_label, node_id, state_label, state_id, temporal_props,
    )

    logger.info(
        "[neo4j_writer] Created new state for %s:%s (hash=%s)",
        identity_label, node_id, incoming_hash[:12],
    )
    return True


async def upsert_simple_node(
    client: Neo4jAuraClient,
    *,
    label: str,
    node_id: str,
    props: dict[str, Any],
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
    source_report: str,
) -> None:
    """Upsert a node without identity/state separation (e.g. Compound, Condition)."""
    set_clauses = ", ".join(f"n.{k} = ${k}" for k in props if k != MERGE_KEY)
    extra_set = f", {set_clauses}" if set_clauses else ""
    cypher = f"""
        MERGE (n:{label} {{ {MERGE_KEY}: $nodeId }})
        ON CREATE SET n.createdAt = datetime()
        SET n.updatedAt = datetime(),
            n.searchText = $searchText,
            n.searchFields = $searchFields,
            n.searchEmbedding = $embedding,
            n.embeddingModel = $embeddingModel,
            n.embeddingDimensions = $embeddingDimensions,
            n.mongoResearchRunId = $sourceReport{extra_set}
    """
    params = {
        "nodeId": node_id,
        "searchText": search_text,
        "searchFields": search_fields,
        "embedding": embedding,
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingDimensions": EMBEDDING_DIMENSIONS,
        "sourceReport": source_report,
        **props,
    }
    await client.execute_write(cypher, parameters=params)


# ---------------------------------------------------------------------------
# Temporal structural relationship writer
# ---------------------------------------------------------------------------


async def upsert_temporal_relationship(
    client: Neo4jAuraClient,
    *,
    from_label: str,
    from_id: str,
    to_label: str,
    to_id: str,
    rel_type: str,
    rel_props: dict[str, Any],
    temporal_ctx: IngestionTemporalContext,
    temporal_qualifier: TemporalQualifier | None = None,
) -> bool:
    if rel_type not in ALLOWED_REL_TYPES:
        logger.warning("[neo4j_writer] Unknown rel_type %s — skipping.", rel_type)
        return False

    temporal_props = _temporal_props_from_qualifier(temporal_qualifier, temporal_ctx)

    check_cypher = f"""
        MATCH (a:{from_label} {{ {MERGE_KEY}: $fromId }})
              -[r:{rel_type}]->
              (b:{to_label} {{ {MERGE_KEY}: $toId }})
        WHERE r.validTo IS NULL AND r.recordedTo IS NULL
        RETURN r
        LIMIT 1
    """
    existing = await client.execute_read(
        check_cypher,
        {"fromId": from_id, "toId": to_id},
    )

    if existing:
        close_time = (temporal_ctx.ingestion_time or now_utc()).isoformat()
        close_cypher = f"""
            MATCH (a:{from_label} {{ {MERGE_KEY}: $fromId }})
                  -[r:{rel_type}]->
                  (b:{to_label} {{ {MERGE_KEY}: $toId }})
            WHERE r.validTo IS NULL AND r.recordedTo IS NULL
            SET r.validTo = $closeTime,
                r.recordedTo = $closeTime
        """
        await client.execute_write(
            close_cypher,
            parameters={"fromId": from_id, "toId": to_id, "closeTime": close_time},
        )

    prop_set_parts = []
    prop_params: dict[str, Any] = {
        "fromId": from_id,
        "toId": to_id,
        **temporal_props,
    }
    for k, v in rel_props.items():
        param_name = f"rp_{k}"
        prop_set_parts.append(f"r.{k} = ${param_name}")
        prop_params[param_name] = v

    prop_set_clause = ", ".join(prop_set_parts) if prop_set_parts else ""
    extra_set = f", {prop_set_clause}" if prop_set_clause else ""

    create_cypher = f"""
        MATCH (a:{from_label} {{ {MERGE_KEY}: $fromId }})
        MATCH (b:{to_label} {{ {MERGE_KEY}: $toId }})
        CREATE (a)-[r:{rel_type} {{
            validFrom: $validFrom,
            validTo: $validTo,
            recordedFrom: $recordedFrom,
            recordedTo: $recordedTo,
            createdAt: datetime()
        }}]->(b)
        SET r.dummy = null{extra_set}
    """
    await client.execute_write(create_cypher, parameters=prop_params)
    return True


# ---------------------------------------------------------------------------
# Name resolution helper
# ---------------------------------------------------------------------------


def resolve_name(
    name: str,
    name_map: dict[str, str],
    entity_type: str,
) -> str | None:
    key = name.strip().lower()
    for k, v in name_map.items():
        if k.strip().lower() == key:
            return v
    logger.warning(
        "[neo4j_writer] Could not resolve %s '%s' — skipping relationship.",
        entity_type, name,
    )
    return None


# ---------------------------------------------------------------------------
# Batch write entry point
# ---------------------------------------------------------------------------


async def write_extraction_to_neo4j(
    client: Neo4jAuraClient,
    extraction: Any,
    node_embeddings: dict[str, Any],
    temporal_ctx: IngestionTemporalContext | None = None,
) -> dict[str, int]:
    """
    Write all nodes and relationships from a KGExtractionResult to Neo4j
    using the identity/state separation pattern with bitemporal semantics.

    All nodes use `id` as the merge key and `name` as the canonical name.
    """
    from src.research.langchain_agent.kg.neo4j_resolver import resolve_node_id

    if temporal_ctx is None:
        temporal_ctx = IngestionTemporalContext(
            ingestion_time=now_utc(),
            source_report=extraction.source_report,
        )

    counts: dict[str, int] = {
        "orgs_written": 0,
        "persons_written": 0,
        "products_written": 0,
        "compounds_written": 0,
        "studies_written": 0,
        "conditions_written": 0,
        "biomarkers_written": 0,
        "lab_tests_written": 0,
        "panels_written": 0,
        "rels_written": 0,
        "rels_skipped": 0,
        "states_created": 0,
        "states_skipped": 0,
    }

    # Name → id maps for relationship resolution
    org_ids: dict[str, str] = {}
    person_ids: dict[str, str] = {}
    product_ids: dict[str, str] = {}
    compound_ids: dict[str, str] = {}
    study_ids: dict[str, str] = {}
    condition_ids: dict[str, str] = {}
    biomarker_ids: dict[str, str] = {}
    lab_test_ids: dict[str, str] = {}
    panel_ids: dict[str, str] = {}
    compound_form_ids: dict[str, str] = {}

    def _emb(prefix: str, name: str) -> tuple[list[float], str]:
        key = f"{prefix}:{name}"
        return (
            node_embeddings.get(key, []),
            node_embeddings.get(f"searchtext:{key}", ""),
        )

    # --- Organizations (with state) -----------------------------------------
    for org in extraction.organizations:
        existing_id = await resolve_node_id(client, "Organization", org.name)
        oid = existing_id or str(uuid4())
        emb, st = _emb("org", org.name)

        identity_props = {"id": oid, "name": org.name}
        state_props = {
            "name": org.name,
            "description": org.description,
            "canonicalTicker": getattr(org, "canonicalTicker", ""),
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Organization",
            node_id=oid,
            identity_props=identity_props,
            state_label="OrganizationState",
            state_props=state_props,
            search_text=st or "",
            search_fields=org.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=org.temporal,
        )
        org_ids[org.name] = oid
        counts["orgs_written"] += 1
        counts["states_created" if new_state else "states_skipped"] += 1

    # --- Persons (with state) -----------------------------------------------
    for person in extraction.persons:
        existing_id = await resolve_node_id(client, "Person", person.name)
        pid = existing_id or str(uuid4())
        emb, st = _emb("person", person.name)

        identity_props = {"id": pid, "name": person.name}
        state_props = {
            "name": person.name,
            "description": person.description,
            "title": getattr(person, "title", ""),
            "bio": getattr(person, "bio", ""),
            "linkedInUrl": getattr(person, "linkedInUrl", ""),
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Person",
            node_id=pid,
            identity_props=identity_props,
            state_label="PersonState",
            state_props=state_props,
            search_text=st or "",
            search_fields=person.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=person.temporal,
        )
        person_ids[person.name] = pid
        counts["persons_written"] += 1
        counts["states_created" if new_state else "states_skipped"] += 1

    # --- Products (with state) ----------------------------------------------
    for product in extraction.products:
        existing_id = await resolve_node_id(client, "Product", product.name)
        pid = existing_id or str(uuid4())
        emb, st = _emb("product", product.name)

        identity_props = {"id": pid, "name": product.name}
        state_props = {
            "name": product.name,
            "description": product.description,
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Product",
            node_id=pid,
            identity_props=identity_props,
            state_label="ProductState",
            state_props=state_props,
            search_text=st or "",
            search_fields=product.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=product.temporal,
        )
        product_ids[product.name] = pid
        counts["products_written"] += 1
        counts["states_created" if new_state else "states_skipped"] += 1

    # --- Simple nodes (no state separation) ---------------------------------

    for compound in extraction.compounds:
        existing_id = await resolve_node_id(client, "Compound", compound.name)
        cid = existing_id or str(uuid4())
        emb, st = _emb("compound", compound.name)
        props = {k: v for k, v in compound.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [])}
        props["id"] = cid
        await upsert_simple_node(
            client, label="Compound", node_id=cid, props=props,
            search_text=st or "", search_fields=compound.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        compound_ids[compound.name] = cid
        counts["compounds_written"] += 1

    for study in extraction.studies:
        existing_id = await resolve_node_id(client, "Study", study.name)
        sid = existing_id or str(uuid4())
        emb, st = _emb("study", study.name)
        props = {k: v for k, v in study.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [], False)}
        props["id"] = sid
        await upsert_simple_node(
            client, label="Study", node_id=sid, props=props,
            search_text=st or "", search_fields=study.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        study_ids[study.name] = sid
        counts["studies_written"] += 1

    for condition in extraction.conditions:
        existing_id = await resolve_node_id(client, "Condition", condition.name)
        cid = existing_id or str(uuid4())
        emb, st = _emb("condition", condition.name)
        props = {k: v for k, v in condition.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [])}
        props["id"] = cid
        await upsert_simple_node(
            client, label="Condition", node_id=cid, props=props,
            search_text=st or "", search_fields=condition.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        condition_ids[condition.name] = cid
        counts["conditions_written"] += 1

    for biomarker in extraction.biomarkers:
        existing_id = await resolve_node_id(client, "Biomarker", biomarker.name)
        bid = existing_id or str(uuid4())
        emb, st = _emb("biomarker", biomarker.name)
        props = {k: v for k, v in biomarker.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [])}
        props["id"] = bid
        await upsert_simple_node(
            client, label="Biomarker", node_id=bid, props=props,
            search_text=st or "", search_fields=biomarker.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        biomarker_ids[biomarker.name] = bid
        counts["biomarkers_written"] += 1

    for lt in extraction.labTests:
        existing_id = await resolve_node_id(client, "LabTest", lt.name)
        lid = existing_id or str(uuid4())
        emb, st = _emb("labtest", lt.name)
        props = {k: v for k, v in lt.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [])}
        props["id"] = lid
        await upsert_simple_node(
            client, label="LabTest", node_id=lid, props=props,
            search_text=st or "", search_fields=lt.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        lab_test_ids[lt.name] = lid
        counts["lab_tests_written"] += 1

    for panel in extraction.panelDefinitions:
        existing_id = await resolve_node_id(client, "PanelDefinition", panel.name)
        pid = existing_id or str(uuid4())
        emb, st = _emb("panel", panel.name)
        props = {k: v for k, v in panel.model_dump().items()
                 if k not in ("temporal", "searchFields") and v not in (None, "", [])}
        props["id"] = pid
        await upsert_simple_node(
            client, label="PanelDefinition", node_id=pid, props=props,
            search_text=st or "", search_fields=panel.searchFields,
            embedding=emb or [], source_report=extraction.source_report,
        )
        panel_ids[panel.name] = pid
        counts["panels_written"] += 1

    # CompoundForms from compound_ingredients
    seen_cf: set[str] = set()
    for ingredient in extraction.compound_ingredients:
        cname = ingredient.compoundName
        if cname in seen_cf:
            continue
        seen_cf.add(cname)
        existing_id = await resolve_node_id(client, "CompoundForm", cname)
        cfid = existing_id or str(uuid4())
        emb, st = _emb("compoundform", cname)
        props = {"id": cfid, "name": cname, "description": getattr(ingredient, "formType", "")}
        await upsert_simple_node(
            client, label="CompoundForm", node_id=cfid, props=props,
            search_text=st or "", search_fields=["name"],
            embedding=emb or [], source_report=extraction.source_report,
        )
        compound_form_ids[cname] = cfid

    # --- Relationships ------------------------------------------------------

    # Org → Person
    for rel in extraction.org_person_relationships:
        oid = resolve_name(rel.org_name, org_ids, "Organization")
        pid = resolve_name(rel.person_name, person_ids, "Person")
        if oid and pid:
            rel_props = {
                "role": getattr(rel, "roleTitle", ""),
                "title": getattr(rel, "roleTitle", ""),
                "notes": getattr(rel, "department", ""),
            }
            await upsert_temporal_relationship(
                client, from_label="Organization", from_id=oid,
                to_label="Person", to_id=pid,
                rel_type=rel.relationship_type, rel_props=rel_props,
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Org → Product
    for rel in extraction.org_product_relationships:
        oid = resolve_name(rel.org_name, org_ids, "Organization")
        pid = resolve_name(rel.product_name, product_ids, "Product")
        if oid and pid:
            await upsert_temporal_relationship(
                client, from_label="Organization", from_id=oid,
                to_label="Product", to_id=pid,
                rel_type=rel.relationship_type, rel_props={},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Product → CompoundForm
    for ingredient in extraction.compound_ingredients:
        pid = resolve_name(ingredient.product_name, product_ids, "Product")
        cid = compound_form_ids.get(ingredient.compoundName)
        if pid and cid:
            rel_props = {
                "dose": ingredient.dose,
                "doseUnit": ingredient.doseUnit,
                "role": ingredient.role,
            }
            await upsert_temporal_relationship(
                client, from_label="Product", from_id=pid,
                to_label="CompoundForm", to_id=cid,
                rel_type="CONTAINS_COMPOUND_FORM", rel_props=rel_props,
                temporal_ctx=temporal_ctx, temporal_qualifier=ingredient.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Study → Organization
    for rel in extraction.study_org_relationships:
        sid = resolve_name(rel.study_name, study_ids, "Study")
        oid = resolve_name(rel.org_name, org_ids, "Organization")
        if sid and oid:
            await upsert_temporal_relationship(
                client, from_label="Study", from_id=sid,
                to_label="Organization", to_id=oid,
                rel_type=rel.relationship_type,
                rel_props={"role": getattr(rel, "role", "")},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Study → Condition
    for rel in extraction.study_condition_relationships:
        sid = resolve_name(rel.study_name, study_ids, "Study")
        cid = resolve_name(rel.condition_name, condition_ids, "Condition")
        if sid and cid:
            await upsert_temporal_relationship(
                client, from_label="Study", from_id=sid,
                to_label="Condition", to_id=cid,
                rel_type="INVESTIGATES", rel_props={},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Study → Person
    for rel in extraction.study_person_relationships:
        sid = resolve_name(rel.study_name, study_ids, "Study")
        pid = resolve_name(rel.person_name, person_ids, "Person")
        if sid and pid:
            await upsert_temporal_relationship(
                client, from_label="Study", from_id=sid,
                to_label="Person", to_id=pid,
                rel_type="INVESTIGATED_BY",
                rel_props={"role": getattr(rel, "role", "")},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Product → LabTest
    for rel in extraction.product_lab_test_relationships:
        pid = resolve_name(rel.product_name, product_ids, "Product")
        lid = resolve_name(rel.lab_test_name, lab_test_ids, "LabTest")
        if pid and lid:
            await upsert_temporal_relationship(
                client, from_label="Product", from_id=pid,
                to_label="LabTest", to_id=lid,
                rel_type="DELIVERS_LABTEST", rel_props={},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Product → PanelDefinition
    for rel in extraction.product_panel_relationships:
        pid = resolve_name(rel.product_name, product_ids, "Product")
        panid = resolve_name(rel.panel_name, panel_ids, "PanelDefinition")
        if pid and panid:
            await upsert_temporal_relationship(
                client, from_label="Product", from_id=pid,
                to_label="PanelDefinition", to_id=panid,
                rel_type="IMPLEMENTS_PANEL", rel_props={},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # LabTest → Biomarker
    for rel in extraction.lab_test_biomarker_relationships:
        lid = resolve_name(rel.lab_test_name, lab_test_ids, "LabTest")
        bid = resolve_name(rel.biomarker_name, biomarker_ids, "Biomarker")
        if lid and bid:
            await upsert_temporal_relationship(
                client, from_label="LabTest", from_id=lid,
                to_label="Biomarker", to_id=bid,
                rel_type="MEASURES",
                rel_props={"role": getattr(rel, "role", "")},
                temporal_ctx=temporal_ctx, temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    return counts
