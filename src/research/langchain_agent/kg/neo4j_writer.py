"""
Neo4j MERGE helpers for each entity type.

Node design:
  - All nodes use MERGE on their UUID *Id field (generated client-side).
  - Nodes carry searchText + embedding for vector search.

Relationship design:
  - Relationships use MERGE on the pair of node IDs.
  - searchText and embedding are NOT stored on relationships (per schema spec).

Entity resolution:
  - The extraction LLM returns names, not UUIDs.
  - The orchestrator passes name→id maps built during the node-write phase.
  - Missing names are logged and skipped — never silently create orphan edges.
"""

from __future__ import annotations

import logging
from typing import Any

from src.research.langchain_agent.neo4j_aura import Neo4jAuraClient

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ---------------------------------------------------------------------------
# Node writers
# ---------------------------------------------------------------------------


async def upsert_organization(
    client: Neo4jAuraClient,
    organization_id: str,
    name: str,
    aliases: list[str],
    org_type: str,
    business_model: str,
    description: str,
    website_url: str,
    legal_name: str,
    primary_industry_tags: list[str],
    regions_served: list[str],
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
) -> None:
    await client.execute_write(
        """
        MERGE (o:Organization { organizationId: $organizationId })
        ON CREATE SET o.createdAt = datetime()
        SET o.name                 = coalesce($name, o.name),
            o.aliases              = $aliases,
            o.orgType              = coalesce($orgType, o.orgType),
            o.businessModel        = coalesce($businessModel, o.businessModel),
            o.description          = coalesce($description, o.description),
            o.websiteUrl           = coalesce($websiteUrl, o.websiteUrl),
            o.legalName            = coalesce($legalName, o.legalName),
            o.primaryIndustryTags  = $primaryIndustryTags,
            o.regionsServed        = $regionsServed,
            o.searchText           = $searchText,
            o.searchFields         = $searchFields,
            o.embedding            = $embedding,
            o.embeddingModel       = $embeddingModel,
            o.embeddingDimensions  = $embeddingDimensions,
            o.validAt              = datetime()
        """,
        parameters={
            "organizationId": organization_id,
            "name": name,
            "aliases": aliases,
            "orgType": org_type,
            "businessModel": business_model,
            "description": description,
            "websiteUrl": website_url,
            "legalName": legal_name,
            "primaryIndustryTags": primary_industry_tags,
            "regionsServed": regions_served,
            "searchText": search_text,
            "searchFields": search_fields,
            "embedding": embedding,
            "embeddingModel": EMBEDDING_MODEL,
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
        },
    )


async def upsert_person(
    client: Neo4jAuraClient,
    person_id: str,
    canonical_name: str,
    given_name: str,
    family_name: str,
    honorific: str,
    degrees: list[str],
    bio: str,
    primary_domain: str,
    specialties: list[str],
    expertise_tags: list[str],
    linkedin_url: str,
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
) -> None:
    await client.execute_write(
        """
        MERGE (p:Person { personId: $personId })
        ON CREATE SET p.createdAt = datetime()
        SET p.canonicalName       = coalesce($canonicalName, p.canonicalName),
            p.givenName           = coalesce($givenName, p.givenName),
            p.familyName          = coalesce($familyName, p.familyName),
            p.honorific           = coalesce($honorific, p.honorific),
            p.degrees             = $degrees,
            p.bio                 = coalesce($bio, p.bio),
            p.primaryDomain       = coalesce($primaryDomain, p.primaryDomain),
            p.specialties         = $specialties,
            p.expertiseTags       = $expertiseTags,
            p.linkedinUrl         = coalesce($linkedinUrl, p.linkedinUrl),
            p.searchText          = $searchText,
            p.searchFields        = $searchFields,
            p.embedding           = $embedding,
            p.embeddingModel      = $embeddingModel,
            p.embeddingDimensions = $embeddingDimensions,
            p.validAt             = datetime()
        """,
        parameters={
            "personId": person_id,
            "canonicalName": canonical_name,
            "givenName": given_name,
            "familyName": family_name,
            "honorific": honorific,
            "degrees": degrees,
            "bio": bio,
            "primaryDomain": primary_domain,
            "specialties": specialties,
            "expertiseTags": expertise_tags,
            "linkedinUrl": linkedin_url,
            "searchText": search_text,
            "searchFields": search_fields,
            "embedding": embedding,
            "embeddingModel": EMBEDDING_MODEL,
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
        },
    )


async def upsert_product(
    client: Neo4jAuraClient,
    product_id: str,
    name: str,
    brand_name: str,
    product_domain: str,
    product_type: str,
    description: str,
    intended_use: str,
    price_amount: float | None,
    currency: str,
    synonyms: list[str],
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
) -> None:
    await client.execute_write(
        """
        MERGE (p:Product { productId: $productId })
        ON CREATE SET p.createdAt = datetime()
        SET p.name                = coalesce($name, p.name),
            p.brandName           = coalesce($brandName, p.brandName),
            p.productDomain       = coalesce($productDomain, p.productDomain),
            p.productType         = coalesce($productType, p.productType),
            p.description         = coalesce($description, p.description),
            p.intendedUse         = coalesce($intendedUse, p.intendedUse),
            p.priceAmount         = $priceAmount,
            p.currency            = coalesce($currency, p.currency),
            p.synonyms            = $synonyms,
            p.searchText          = $searchText,
            p.searchFields        = $searchFields,
            p.embedding           = $embedding,
            p.embeddingModel      = $embeddingModel,
            p.embeddingDimensions = $embeddingDimensions,
            p.validAt             = datetime()
        """,
        parameters={
            "productId": product_id,
            "name": name,
            "brandName": brand_name,
            "productDomain": product_domain,
            "productType": product_type,
            "description": description,
            "intendedUse": intended_use,
            "priceAmount": price_amount,
            "currency": currency,
            "synonyms": synonyms,
            "searchText": search_text,
            "searchFields": search_fields,
            "embedding": embedding,
            "embeddingModel": EMBEDDING_MODEL,
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
        },
    )


async def upsert_compound_form(
    client: Neo4jAuraClient,
    compound_id: str,
    canonical_name: str,
    form_type: str,
    search_text: str,
    search_fields: list[str],
    embedding: list[float],
) -> None:
    await client.execute_write(
        """
        MERGE (c:CompoundForm { compoundFormId: $compoundFormId })
        ON CREATE SET c.createdAt = datetime()
        SET c.canonicalName       = coalesce($canonicalName, c.canonicalName),
            c.formType            = coalesce($formType, c.formType),
            c.searchText          = $searchText,
            c.searchFields        = $searchFields,
            c.embedding           = $embedding,
            c.embeddingModel      = $embeddingModel,
            c.embeddingDimensions = $embeddingDimensions,
            c.validAt             = datetime()
        """,
        parameters={
            "compoundFormId": compound_id,
            "canonicalName": canonical_name,
            "formType": form_type,
            "searchText": search_text,
            "searchFields": search_fields,
            "embedding": embedding,
            "embeddingModel": EMBEDDING_MODEL,
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
        },
    )


# ---------------------------------------------------------------------------
# Relationship writers  (no searchText / embedding on edges)
# ---------------------------------------------------------------------------


async def upsert_org_person_rel(
    client: Neo4jAuraClient,
    org_id: str,
    person_id: str,
    rel_type: str,
    role_title: str = "",
    department: str = "",
    seniority: str = "",
    is_current: bool = True,
) -> None:
    """
    MERGE an Organization→Person relationship.
    rel_type must be one of:
      EMPLOYS | FOUNDED_BY | HAS_BOARD_MEMBER | HAS_SCIENTIFIC_ADVISOR | HAS_EXECUTIVE_ROLE
    """
    allowed = {
        "EMPLOYS",
        "FOUNDED_BY",
        "HAS_BOARD_MEMBER",
        "HAS_SCIENTIFIC_ADVISOR",
        "HAS_EXECUTIVE_ROLE",
    }
    if rel_type not in allowed:
        logger.warning("[neo4j_writer] Unknown rel_type %s — skipping.", rel_type)
        return

    cypher = f"""
        MATCH (o:Organization {{ organizationId: $orgId }})
        MATCH (p:Person {{ personId: $personId }})
        MERGE (o)-[r:{rel_type}]->(p)
        ON CREATE SET r.createdAt = datetime()
        SET r.roleTitle  = coalesce($roleTitle, r.roleTitle),
            r.department = coalesce($department, r.department),
            r.seniority  = coalesce($seniority, r.seniority),
            r.isCurrent  = $isCurrent,
            r.validAt    = datetime()
    """
    await client.execute_write(
        cypher,
        parameters={
            "orgId": org_id,
            "personId": person_id,
            "roleTitle": role_title,
            "department": department,
            "seniority": seniority,
            "isCurrent": is_current,
        },
    )


async def upsert_offers_product_rel(
    client: Neo4jAuraClient,
    org_id: str,
    product_id: str,
    channel: str = "ONLINE",
) -> None:
    """MERGE an Organization-[:OFFERS_PRODUCT]->Product relationship."""
    await client.execute_write(
        """
        MATCH (o:Organization { organizationId: $orgId })
        MATCH (p:Product { productId: $productId })
        MERGE (o)-[r:OFFERS_PRODUCT]->(p)
        ON CREATE SET r.createdAt = datetime()
        SET r.channel = coalesce($channel, r.channel),
            r.validAt = datetime()
        """,
        parameters={
            "orgId": org_id,
            "productId": product_id,
            "channel": channel,
        },
    )


async def upsert_contains_compound_rel(
    client: Neo4jAuraClient,
    product_id: str,
    compound_id: str,
    dose: float | None,
    dose_unit: str,
    role: str,
    bioavailability_notes: str,
) -> None:
    """MERGE a Product-[:CONTAINS_COMPOUND_FORM]->CompoundForm relationship."""
    await client.execute_write(
        """
        MATCH (p:Product { productId: $productId })
        MATCH (c:CompoundForm { compoundFormId: $compoundId })
        MERGE (p)-[r:CONTAINS_COMPOUND_FORM]->(c)
        ON CREATE SET r.createdAt = datetime()
        SET r.dose                 = $dose,
            r.doseUnit             = coalesce($doseUnit, r.doseUnit),
            r.role                 = coalesce($role, r.role),
            r.bioavailabilityNotes = coalesce($bioavailabilityNotes, r.bioavailabilityNotes),
            r.validAt              = datetime()
        """,
        parameters={
            "productId": product_id,
            "compoundId": compound_id,
            "dose": dose,
            "doseUnit": dose_unit,
            "role": role,
            "bioavailabilityNotes": bioavailability_notes,
        },
    )


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_name(
    name: str,
    name_map: dict[str, str],
    entity_type: str,
) -> str | None:
    """
    Look up a name in the given map.  Returns the UUID or None.
    Normalise to lowercase + stripped for matching.
    """
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
) -> dict[str, int]:
    """
    Write all nodes and relationships from a KGExtractionResult to Neo4j.

    Before creating any node the resolver checks for an existing match:
      1. Exact toLower name match in the graph.
      2. Fulltext phrase search (fallback).
    If a match is found the existing ID is reused, so MERGE hits the existing
    node and updates its properties without creating a duplicate.

    Args:
        client:           Connected Neo4jAuraClient.
        extraction:       KGExtractionResult from the extraction agent.
        node_embeddings:  Mapping node_key → embedding / searchText.
                          Keys: "<type>:<name>" for embedding,
                                "searchtext:<type>:<name>" for text.

    Returns:
        Dict with counts: orgs_written, persons_written, products_written,
        compounds_written, rels_written, rels_skipped.
        *_written counts every upsert regardless of create vs. update.
    """
    from uuid import uuid4

    from src.research.langchain_agent.kg.neo4j_resolver import (
        resolve_compound_form_id,
        resolve_organization_id,
        resolve_person_id,
        resolve_product_id,
    )

    counts: dict[str, int] = {
        "orgs_written": 0,
        "persons_written": 0,
        "products_written": 0,
        "compounds_written": 0,
        "rels_written": 0,
        "rels_skipped": 0,
    }

    # --- name → ID maps for relationship resolution -------------------------
    org_name_to_id: dict[str, str] = {}
    person_name_to_id: dict[str, str] = {}
    product_name_to_id: dict[str, str] = {}
    compound_name_to_id: dict[str, str] = {}

    # --- Organizations -------------------------------------------------------
    for org in extraction.organizations:
        existing_id = await resolve_organization_id(client, org.name)
        oid = existing_id or str(uuid4())
        key = f"org:{org.name}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")
        await upsert_organization(
            client=client,
            organization_id=oid,
            name=org.name,
            aliases=org.aliases,
            org_type=org.orgType,
            business_model=org.businessModel,
            description=org.description,
            website_url=org.websiteUrl,
            legal_name=org.legalName,
            primary_industry_tags=org.primaryIndustryTags,
            regions_served=org.regionsServed,
            search_text=search_text or "",
            search_fields=org.searchFields,
            embedding=emb or [],
        )
        org_name_to_id[org.name] = oid
        counts["orgs_written"] += 1

    # --- Persons -------------------------------------------------------------
    for person in extraction.persons:
        existing_id = await resolve_person_id(client, person.canonicalName)
        pid = existing_id or str(uuid4())
        key = f"person:{person.canonicalName}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")
        await upsert_person(
            client=client,
            person_id=pid,
            canonical_name=person.canonicalName,
            given_name=person.givenName,
            family_name=person.familyName,
            honorific=person.honorific,
            degrees=person.degrees,
            bio=person.bio,
            primary_domain=person.primaryDomain,
            specialties=person.specialties,
            expertise_tags=person.expertiseTags,
            linkedin_url=person.linkedinUrl,
            search_text=search_text or "",
            search_fields=person.searchFields,
            embedding=emb or [],
        )
        person_name_to_id[person.canonicalName] = pid
        counts["persons_written"] += 1

    # --- Products ------------------------------------------------------------
    for product in extraction.products:
        existing_id = await resolve_product_id(client, product.name)
        pid = existing_id or str(uuid4())
        key = f"product:{product.name}"
        emb = node_embeddings.get(key, [])
        search_text = node_embeddings.get(f"searchtext:{key}", "")
        await upsert_product(
            client=client,
            product_id=pid,
            name=product.name,
            brand_name=product.brandName,
            product_domain=product.productDomain,
            product_type=product.productType,
            description=product.description,
            intended_use=product.intendedUse,
            price_amount=product.priceAmount,
            currency=product.currency,
            synonyms=product.synonyms,
            search_text=search_text or "",
            search_fields=product.searchFields,
            embedding=emb or [],
        )
        product_name_to_id[product.name] = pid
        counts["products_written"] += 1

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
        await upsert_compound_form(
            client=client,
            compound_id=cid,
            canonical_name=cname,
            form_type=ingredient.formType,
            search_text=search_text or "",
            search_fields=ingredient.searchFields,
            embedding=emb or [],
        )
        compound_name_to_id[cname] = cid
        counts["compounds_written"] += 1

    # --- Relationships -------------------------------------------------------

    # Org → Person
    for rel in extraction.org_person_relationships:
        oid = resolve_name(rel.org_name, org_name_to_id, "Organization")
        pid = resolve_name(rel.person_name, person_name_to_id, "Person")
        if oid and pid:
            await upsert_org_person_rel(
                client=client,
                org_id=oid,
                person_id=pid,
                rel_type=rel.relationship_type,
                role_title=rel.roleTitle,
                department=rel.department,
                seniority=rel.seniority,
                is_current=rel.isCurrent,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Org → Product (OFFERS_PRODUCT)
    # Infer from product.brandName matching org.name
    for product in extraction.products:
        brand = product.brandName or ""
        oid = resolve_name(brand, org_name_to_id, "Organization") if brand else None
        if not oid and extraction.organizations:
            # Fall back to the single organisation in the extraction if unique
            if len(extraction.organizations) == 1:
                oid = org_name_to_id.get(extraction.organizations[0].name)
        pid = product_name_to_id.get(product.name)
        if oid and pid:
            await upsert_offers_product_rel(client=client, org_id=oid, product_id=pid)
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    # Product → CompoundForm (CONTAINS_COMPOUND_FORM)
    for ingredient in extraction.compound_ingredients:
        pid = resolve_name(ingredient.product_name, product_name_to_id, "Product")
        cid = compound_name_to_id.get(ingredient.compoundName)
        if pid and cid:
            await upsert_contains_compound_rel(
                client=client,
                product_id=pid,
                compound_id=cid,
                dose=ingredient.dose,
                dose_unit=ingredient.doseUnit,
                role=ingredient.role,
                bioavailability_notes=ingredient.bioavailabilityNotes,
            )
            counts["rels_written"] += 1
        else:
            counts["rels_skipped"] += 1

    return counts
