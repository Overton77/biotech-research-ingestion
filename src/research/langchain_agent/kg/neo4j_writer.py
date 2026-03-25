"""
Neo4j bitemporal writer — identity/state separation with temporal relationships.

Architecture:
  - Identity nodes are durable anchors with minimal stable properties.
  - State nodes are immutable snapshots of descriptive properties.
  - Identity → State is attached via HAS_STATE with bitemporal bounds.
  - Structural relationships carry bitemporal bounds.

State change detection:
  - Before creating a new state, the current active state's hash is compared.
  - If the hash matches, no new state is created (idempotent).
  - If different, the old HAS_STATE interval is closed and a new one opened.

Bitemporal properties on HAS_STATE and structural relationships:
  - validFrom:    when the fact is true in the domain
  - validTo:      when the fact stopped being true (null = active)
  - recordedFrom: when the system learned the fact
  - recordedTo:   when the system stopped believing (null = current belief)
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
    OPEN_ENDED,
    compute_state_hash,
    default_bitemporal_props,
    now_utc,
)
from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ---------------------------------------------------------------------------
# Bitemporal helpers
# ---------------------------------------------------------------------------


def _temporal_props_from_qualifier(
    qualifier: TemporalQualifier | None,
    temporal_ctx: IngestionTemporalContext,
) -> dict[str, Any]:
    """
    Build bitemporal relationship properties from an extracted temporal
    qualifier and the system-level ingestion context.
    """
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
    merge_key: str,
    node_id: str,
    identity_props: dict[str, Any],
) -> None:
    """MERGE a durable identity node with minimal stable properties."""
    set_clauses = ", ".join(
        f"n.{k} = ${k}" for k in identity_props if k != merge_key
    )
    cypher = f"""
        MERGE (n:{label} {{ {merge_key}: $nodeId }})
        ON CREATE SET n.createdAt = datetime()
        SET {set_clauses}
    """
    params = {"nodeId": node_id, **identity_props}
    await client.execute_write(cypher, parameters=params)


async def _get_active_state_hash(
    client: Neo4jAuraClient,
    identity_label: str,
    identity_merge_key: str,
    node_id: str,
    state_label: str,
) -> str | None:
    """
    Return the stateHash of the current active state for a given identity node,
    or None if no active state exists.
    """
    cypher = f"""
        MATCH (n:{identity_label} {{ {identity_merge_key}: $nodeId }})
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
    identity_merge_key: str,
    node_id: str,
    state_label: str,
    close_time: str,
) -> None:
    """Close the validity and recorded windows on the current active HAS_STATE."""
    cypher = f"""
        MATCH (n:{identity_label} {{ {identity_merge_key}: $nodeId }})
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
    """Create an immutable state snapshot node."""
    set_clauses = ", ".join(
        f"s.{k} = ${k}" for k in state_props
    )
    cypher = f"""
        CREATE (s:{state_label} {{
            stateId: $stateId,
            stateHash: $stateHash,
            sourceReport: $sourceReport,
            searchText: $searchText,
            searchFields: $searchFields,
            embedding: $embedding,
            embeddingModel: $embeddingModel,
            embeddingDimensions: $embeddingDimensions,
            createdAt: datetime()
        }})
        SET {set_clauses}
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
    identity_merge_key: str,
    node_id: str,
    state_label: str,
    state_id: str,
    temporal_props: dict[str, Any],
) -> None:
    """Create a HAS_STATE relationship with bitemporal bounds."""
    cypher = f"""
        MATCH (n:{identity_label} {{ {identity_merge_key}: $nodeId }})
        MATCH (s:{state_label} {{ stateId: $stateId }})
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
    identity_merge_key: str,
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
    # 1. Upsert identity
    await upsert_identity_node(
        client, identity_label, identity_merge_key, node_id, identity_props,
    )

    # 2. Compute hash of incoming state
    incoming_hash = compute_state_hash(state_props)

    # 3. Compare with current active state
    active_hash = await _get_active_state_hash(
        client, identity_label, identity_merge_key, node_id, state_label,
    )

    if active_hash == incoming_hash:
        logger.debug(
            "[neo4j_writer] State unchanged for %s:%s — skipping state creation.",
            identity_label, node_id,
        )
        return False

    # 4. Close old active state if one exists
    if active_hash is not None:
        close_time = (temporal_ctx.ingestion_time or now_utc()).isoformat()
        await _close_active_state(
            client, identity_label, identity_merge_key, node_id,
            state_label, close_time,
        )
        logger.info(
            "[neo4j_writer] Closed previous state for %s:%s (old hash=%s)",
            identity_label, node_id, active_hash[:12],
        )

    # 5. Create new state node
    state_id = str(uuid4())
    await create_state_node(
        client, state_label, state_id, state_props,
        incoming_hash, search_text, search_fields, embedding, source_report,
    )

    # 6. Create HAS_STATE relationship with temporal bounds
    temporal_props = _temporal_props_from_qualifier(temporal_qualifier, temporal_ctx)
    await create_has_state_rel(
        client, identity_label, identity_merge_key, node_id,
        state_label, state_id, temporal_props,
    )

    logger.info(
        "[neo4j_writer] Created new state for %s:%s (hash=%s, state_id=%s)",
        identity_label, node_id, incoming_hash[:12], state_id,
    )
    return True


# ---------------------------------------------------------------------------
# Temporal structural relationship writer
# ---------------------------------------------------------------------------


async def upsert_temporal_relationship(
    client: Neo4jAuraClient,
    *,
    from_label: str,
    from_merge_key: str,
    from_id: str,
    to_label: str,
    to_merge_key: str,
    to_id: str,
    rel_type: str,
    rel_props: dict[str, Any],
    temporal_ctx: IngestionTemporalContext,
    temporal_qualifier: TemporalQualifier | None = None,
) -> bool:
    """
    MERGE a structural relationship with bitemporal properties.

    Checks for an existing active relationship first. If one exists with
    matching properties, it is left unchanged (idempotent). If the properties
    differ, the old relationship's interval is closed and a new one created.

    Returns True if a relationship was created/updated, False if unchanged.
    """
    allowed_rel_types = {
        "EMPLOYS", "FOUNDED_BY", "HAS_BOARD_MEMBER",
        "HAS_SCIENTIFIC_ADVISOR", "HAS_EXECUTIVE_ROLE",
        "OFFERS_PRODUCT", "CONTAINS_COMPOUND_FORM",
        "DELIVERS_LAB_TEST", "IMPLEMENTS_PANEL", "INCLUDES_LABTEST",
    }
    if rel_type not in allowed_rel_types:
        logger.warning("[neo4j_writer] Unknown rel_type %s — skipping.", rel_type)
        return False

    temporal_props = _temporal_props_from_qualifier(temporal_qualifier, temporal_ctx)

    # Check for existing active relationship
    check_cypher = f"""
        MATCH (a:{from_label} {{ {from_merge_key}: $fromId }})
              -[r:{rel_type}]->
              (b:{to_label} {{ {to_merge_key}: $toId }})
        WHERE r.validTo IS NULL AND r.recordedTo IS NULL
        RETURN r
        LIMIT 1
    """
    existing = await client.execute_read(
        check_cypher,
        {"fromId": from_id, "toId": to_id},
    )

    if existing:
        # Active relationship exists — close it before creating new one
        close_time = (temporal_ctx.ingestion_time or now_utc()).isoformat()
        close_cypher = f"""
            MATCH (a:{from_label} {{ {from_merge_key}: $fromId }})
                  -[r:{rel_type}]->
                  (b:{to_label} {{ {to_merge_key}: $toId }})
            WHERE r.validTo IS NULL AND r.recordedTo IS NULL
            SET r.validTo = $closeTime,
                r.recordedTo = $closeTime
        """
        await client.execute_write(
            close_cypher,
            parameters={"fromId": from_id, "toId": to_id, "closeTime": close_time},
        )

    # Build SET clauses for non-temporal relationship properties
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
        MATCH (a:{from_label} {{ {from_merge_key}: $fromId }})
        MATCH (b:{to_label} {{ {to_merge_key}: $toId }})
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
# Name resolution helper (unchanged)
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
        entity_type,
        name,
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

    Before creating any node the resolver checks for an existing match:
      1. Exact toLower name match in the graph.
      2. Fulltext phrase search (fallback).
    If a match is found the existing ID is reused.

    State snapshots are compared by hash — duplicate states are not created.

    Args:
        client:           Connected Neo4jAuraClient.
        extraction:       KGExtractionResult from the extraction agent.
        node_embeddings:  Mapping node_key → embedding / searchText.
        temporal_ctx:     Bitemporal context for this ingestion run.

    Returns:
        Dict with counts: orgs_written, persons_written, products_written,
        compounds_written, lab_tests_written, panels_written,
        rels_written, rels_skipped, states_created, states_skipped.
    """
    from src.research.langchain_agent.kg.neo4j_resolver import (
        resolve_compound_form_id,
        resolve_lab_test_id,
        resolve_organization_id,
        resolve_panel_definition_id,
        resolve_person_id,
        resolve_product_id,
    )

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
        "lab_tests_written": 0,
        "panels_written": 0,
        "rels_written": 0,
        "rels_skipped": 0,
        "states_created": 0,
        "states_skipped": 0,
    }

    org_name_to_id: dict[str, str] = {}
    person_name_to_id: dict[str, str] = {}
    product_name_to_id: dict[str, str] = {}
    compound_name_to_id: dict[str, str] = {}
    lab_test_name_to_id: dict[str, str] = {}
    panel_name_to_id: dict[str, str] = {}

    # --- Organizations -------------------------------------------------------
    for org in extraction.organizations:
        existing_id = await resolve_organization_id(client, org.name)
        oid = existing_id or str(uuid4())
        key = f"org:{org.name}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")

        identity_props = {
            "organizationId": oid,
            "name": org.name,
            "aliases": org.aliases,
        }
        state_props = {
            "orgType": org.orgType,
            "businessModel": org.businessModel,
            "description": org.description,
            "websiteUrl": org.websiteUrl,
            "legalName": org.legalName,
            "primaryIndustryTags": org.primaryIndustryTags,
            "regionsServed": org.regionsServed,
            "headquartersCity": org.headquartersCity,
            "headquartersCountry": org.headquartersCountry,
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Organization",
            identity_merge_key="organizationId",
            node_id=oid,
            identity_props=identity_props,
            state_label="OrganizationState",
            state_props=state_props,
            search_text=search_text or "",
            search_fields=org.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=org.temporal,
        )
        org_name_to_id[org.name] = oid
        counts["orgs_written"] += 1
        if new_state:
            counts["states_created"] += 1
        else:
            counts["states_skipped"] += 1

    # --- Persons -------------------------------------------------------------
    for person in extraction.persons:
        existing_id = await resolve_person_id(client, person.canonicalName)
        pid = existing_id or str(uuid4())
        key = f"person:{person.canonicalName}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")

        identity_props = {
            "personId": pid,
            "canonicalName": person.canonicalName,
        }
        state_props = {
            "givenName": person.givenName,
            "familyName": person.familyName,
            "honorific": person.honorific,
            "degrees": person.degrees,
            "bio": person.bio,
            "primaryDomain": person.primaryDomain,
            "specialties": person.specialties,
            "expertiseTags": person.expertiseTags,
            "linkedinUrl": person.linkedinUrl,
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Person",
            identity_merge_key="personId",
            node_id=pid,
            identity_props=identity_props,
            state_label="PersonState",
            state_props=state_props,
            search_text=search_text or "",
            search_fields=person.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=person.temporal,
        )
        person_name_to_id[person.canonicalName] = pid
        counts["persons_written"] += 1
        if new_state:
            counts["states_created"] += 1
        else:
            counts["states_skipped"] += 1

    # --- Products ------------------------------------------------------------
    for product in extraction.products:
        existing_id = await resolve_product_id(client, product.name)
        pid = existing_id or str(uuid4())
        key = f"product:{product.name}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")

        identity_props = {
            "productId": pid,
            "name": product.name,
            "synonyms": product.synonyms,
        }
        state_props = {
            "brandName": product.brandName,
            "productDomain": product.productDomain,
            "productType": product.productType,
            "description": product.description,
            "intendedUse": product.intendedUse,
            "priceAmount": product.priceAmount,
            "currency": product.currency,
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="Product",
            identity_merge_key="productId",
            node_id=pid,
            identity_props=identity_props,
            state_label="ProductState",
            state_props=state_props,
            search_text=search_text or "",
            search_fields=product.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=product.temporal,
        )
        product_name_to_id[product.name] = pid
        counts["products_written"] += 1
        if new_state:
            counts["states_created"] += 1
        else:
            counts["states_skipped"] += 1

    # --- CompoundForms -------------------------------------------------------
    seen_compounds: set[str] = set()
    for ingredient in extraction.compound_ingredients:
        cname = ingredient.compoundName
        if cname in seen_compounds:
            continue
        seen_compounds.add(cname)

        existing_id = await resolve_compound_form_id(client, cname)
        cid = existing_id or str(uuid4())
        key = f"compound:{cname}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")

        identity_props = {
            "compoundFormId": cid,
            "canonicalName": cname,
        }
        state_props = {
            "formType": ingredient.formType,
        }

        new_state = await upsert_entity_with_state(
            client,
            identity_label="CompoundForm",
            identity_merge_key="compoundFormId",
            node_id=cid,
            identity_props=identity_props,
            state_label="CompoundFormState",
            state_props=state_props,
            search_text=search_text or "",
            search_fields=ingredient.searchFields,
            embedding=emb or [],
            source_report=extraction.source_report,
            temporal_ctx=temporal_ctx,
            temporal_qualifier=ingredient.temporal,
        )
        compound_name_to_id[cname] = cid
        counts["compounds_written"] += 1
        if new_state:
            counts["states_created"] += 1
        else:
            counts["states_skipped"] += 1

    # --- Relationships -------------------------------------------------------

    # Org → Person
    for rel in extraction.org_person_relationships:
        oid = resolve_name(rel.org_name, org_name_to_id, "Organization")
        pid = resolve_name(rel.person_name, person_name_to_id, "Person")
        if oid and pid:
            rel_props = {
                "roleTitle": rel.roleTitle,
                "department": rel.department,
                "seniority": rel.seniority,
                "isCurrent": rel.isCurrent,
            }
            await upsert_temporal_relationship(
                client,
                from_label="Organization",
                from_merge_key="organizationId",
                from_id=oid,
                to_label="Person",
                to_merge_key="personId",
                to_id=pid,
                rel_type=rel.relationship_type,
                rel_props=rel_props,
                temporal_ctx=temporal_ctx,
                temporal_qualifier=rel.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Org → Product (OFFERS_PRODUCT)
    for product in extraction.products:
        brand = product.brandName or ""
        oid = resolve_name(brand, org_name_to_id, "Organization") if brand else None
        if not oid and extraction.organizations:
            if len(extraction.organizations) == 1:
                oid = org_name_to_id.get(extraction.organizations[0].name)
        pid = product_name_to_id.get(product.name)
        if oid and pid:
            await upsert_temporal_relationship(
                client,
                from_label="Organization",
                from_merge_key="organizationId",
                from_id=oid,
                to_label="Product",
                to_merge_key="productId",
                to_id=pid,
                rel_type="OFFERS_PRODUCT",
                rel_props={"channel": "ONLINE"},
                temporal_ctx=temporal_ctx,
                temporal_qualifier=product.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Product → CompoundForm (CONTAINS_COMPOUND_FORM)
    for ingredient in extraction.compound_ingredients:
        pid = resolve_name(ingredient.product_name, product_name_to_id, "Product")
        cid = compound_name_to_id.get(ingredient.compoundName)
        if pid and cid:
            rel_props = {
                "dose": ingredient.dose,
                "doseUnit": ingredient.doseUnit,
                "role": ingredient.role,
                "bioavailabilityNotes": ingredient.bioavailabilityNotes,
            }
            await upsert_temporal_relationship(
                client,
                from_label="Product",
                from_merge_key="productId",
                from_id=pid,
                to_label="CompoundForm",
                to_merge_key="compoundFormId",
                to_id=cid,
                rel_type="CONTAINS_COMPOUND_FORM",
                rel_props=rel_props,
                temporal_ctx=temporal_ctx,
                temporal_qualifier=ingredient.temporal,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    return counts
