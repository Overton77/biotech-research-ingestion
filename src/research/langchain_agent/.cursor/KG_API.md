# Knowledge Graph API Reference

> **Covers:** `biotech-kg` GraphQL API + Neo4j Aura direct writes + entity schema  
> **biotech-kg server:** `http://localhost:4002/graphql` (also `ws://localhost:4002/graphql` for subscriptions)  
> **Neo4j:** Aura instance — env vars `NEO4J_URI`, `NEO4J_AURA_USERNAME`, `NEO4J_AURA_PASSWORD`

---

## Two Write Paths

### Path 1 — Direct Neo4j Cypher (preferred for bulk ingestion)

Use for high-throughput batch writes during research ingestion. No HTTP overhead.  
Use the existing `Neo4jAuraClient` (see `src/research/langchain_agent/neo4j_aura.py` if it exists, or create one).

```python
import neo4j

driver = neo4j.driver(
    os.getenv("NEO4J_URI"),
    neo4j.auth.basic(os.getenv("NEO4J_AURA_USERNAME"), os.getenv("NEO4J_AURA_PASSWORD")),
)

async with driver.session() as session:
    await session.execute_write(lambda tx: tx.run(cypher, **params))
```

### Path 2 — GraphQL API (biotech-kg, for reads and app-layer writes)

Use for app-level queries, subscriptions, and writes that benefit from the auto-generated Neo4j GraphQL layer.

```python
import httpx

GQL_URL = "http://localhost:4002/graphql"

async def gql(query: str, variables: dict = None) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GQL_URL,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": f"Bearer {token}"},  # optional until auth is wired
        )
        resp.raise_for_status()
        return resp.json()
```

---

## Entity Schema (biotech-kg/src/schema/)

### Organization (`organization.ts`)

**Node label:** `Organization`

| Field | Type | Notes |
|---|---|---|
| `id` | ID! | Auto-generated |
| `name` | String! | |
| `description` | String | |
| `website` | String | |
| `sector` | String | |
| `founded` | Int | Year |
| `headquarters` | String | |
| `descriptionEmbedding` | [Float!] | OpenAI text-embedding-3-small vector |
| `createdAt` | DateTime! | Auto |
| `updatedAt` | DateTime! | Auto |

**Full-text indexes:** `OrganizationName` (name), `OrganizationDescription` (description, sector, headquarters)  
**Vector index:** `OrganizationDescriptionEmbedding` (OpenAI)

**Relationships from Organization:**
- `-[EMPLOYS]->` Person
- `-[HAS_BOARD_MEMBER]->` Person
- `-[FOUNDED_BY]->` Person
- `-[IS_SCIENTIFIC_ADVISOR_TO]<-` Person
- `-[OFFERS_PRODUCT]->` Product
- `-[MANUFACTURES]->` Product
- `-[IS_DIRECT_COMPETITOR]->` Organization
- `-[OWNS_OR_CONTROLS]->` Organization

### Person (`person.ts`)

**Node label:** `Person`

| Field | Type | Notes |
|---|---|---|
| `id` | ID! | Auto-generated |
| `name` | String! | |
| `title` | String | |
| `bio` | String | |
| `linkedinUrl` | String | |
| `bioEmbedding` | [Float!] | Vector |
| `createdAt` | DateTime! | Auto |
| `updatedAt` | DateTime! | Auto |

### Product (`product.ts`)

**Node label:** `Product`

| Field | Type | Notes |
|---|---|---|
| `id` | ID! | Auto-generated |
| `name` | String! | |
| `description` | String | |
| `price` | Float | |
| `currency` | String | |
| `productType` | String | |
| `url` | String | |
| `ingredients` | [String] | |
| `descriptionEmbedding` | [Float!] | Vector |
| `createdAt` | DateTime! | Auto |
| `updatedAt` | DateTime! | Auto |

### Compound / CompoundForm (`compound.ts`)

**Node label:** `Compound` / `CompoundForm`

Used for supplement ingredients, bioactive molecules.

### Study (`study.ts`)

**Node label:** `Study`

Used for clinical trials, research papers (light support now, deeper later).

---

## Direct Cypher Write Patterns

### MERGE Organization Node

```cypher
MERGE (o:Organization { name: $name })
ON CREATE SET
  o.id           = $id,
  o.createdAt    = datetime()
SET
  o.description  = coalesce($description, o.description),
  o.website      = coalesce($website, o.website),
  o.sector       = coalesce($sector, o.sector),
  o.founded      = coalesce($founded, o.founded),
  o.headquarters = coalesce($headquarters, o.headquarters),
  o.updatedAt    = datetime()
RETURN o
```

### MERGE Person Node

```cypher
MERGE (p:Person { name: $name })
ON CREATE SET
  p.id        = $id,
  p.createdAt = datetime()
SET
  p.title      = coalesce($title, p.title),
  p.bio        = coalesce($bio, p.bio),
  p.updatedAt  = datetime()
RETURN p
```

### MERGE Product Node

```cypher
MERGE (prod:Product { name: $name })
ON CREATE SET
  prod.id        = $id,
  prod.createdAt = datetime()
SET
  prod.description  = coalesce($description, prod.description),
  prod.price        = coalesce($price, prod.price),
  prod.productType  = coalesce($productType, prod.productType),
  prod.url          = coalesce($url, prod.url),
  prod.updatedAt    = datetime()
RETURN prod
```

### MERGE Relationships

```cypher
-- Organization EMPLOYS Person
MATCH (o:Organization { name: $orgName })
MATCH (p:Person { name: $personName })
MERGE (o)-[r:EMPLOYS]->(p)
ON CREATE SET r.createdAt = datetime()
SET r.roleTitle = $roleTitle, r.isCurrent = $isCurrent, r.updatedAt = datetime()
RETURN r

-- Organization OFFERS_PRODUCT Product
MATCH (o:Organization { name: $orgName })
MATCH (prod:Product { name: $productName })
MERGE (o)-[r:OFFERS_PRODUCT]->(prod)
ON CREATE SET r.createdAt = datetime()
SET r.updatedAt = datetime()
RETURN r

-- Organization FOUNDED_BY Person
MATCH (o:Organization { name: $orgName })
MATCH (p:Person { name: $personName })
MERGE (o)-[r:FOUNDED_BY]->(p)
ON CREATE SET r.createdAt = datetime()
SET r.year = $year, r.updatedAt = datetime()
RETURN r
```

---

## GraphQL Mutations (via biotech-kg API)

> The Neo4jGraphQL library auto-generates Create/Update/Delete mutations from the type definitions.

### Create Organization

```graphql
mutation CreateOrganization($input: [OrganizationCreateInput!]!) {
  createOrganizations(input: $input) {
    organizations {
      id
      name
      description
    }
  }
}
```

Variables:
```json
{
  "input": [{
    "name": "Qualia Life Sciences",
    "description": "Neuroscience-focused supplement company",
    "website": "https://www.qualialife.com",
    "sector": "Supplement / Longevity",
    "headquarters": "San Diego, CA"
  }]
}
```

### Create Person and Relate to Organization

```graphql
mutation CreatePersonAndRelate($orgId: ID!, $person: PersonCreateInput!) {
  updateOrganizations(
    where: { id: $orgId }
    update: {
      employs: {
        create: [{
          node: $person
          edge: { roleTitle: "CEO", isCurrent: true }
        }]
      }
    }
  ) {
    organizations {
      id
      name
      employs { id name title }
    }
  }
}
```

### Query Organizations with Products

```graphql
query OrgWithProducts($name: String!) {
  organizations(where: { name: $name }) {
    id
    name
    description
    headquarters
    offersProduct {
      id
      name
      price
      productType
    }
    employs {
      id
      name
      title
    }
    foundedBy {
      id
      name
    }
  }
}
```

### Vector Search (Organizations by description)

```graphql
query SearchOrganizations($text: String!, $limit: Int) {
  searchOrganizationByDescription(phrase: $text, limit: $limit) {
    id
    name
    description
    score
  }
}
```

### Full-text Search

```graphql
query SearchOrganizationsByName($text: String!) {
  searchOrganizationsByName(phrase: $text) {
    id
    name
    description
    score
  }
}
```

---

## KG Ingestion Pipeline (Planned — see PLAN_kg_ingestion.md)

The full KG ingestion flow (Schema Selection → Entity Extraction → searchText → Embedding → Neo4j Write) is planned and documented in `.cursor/PLAN_kg_ingestion.md`.

**Planned module:** `src/research/langchain_agent/kg/`

```
kg/
  __init__.py
  schema_selector.py       # Select relevant schema chunks from schema_index.json
  extraction_models.py     # KGExtractionResult, ExtractedOrganization, ExtractedPerson, etc.
  extractor.py             # LLM extraction agent with response_format=KGExtractionResult
  searchtext.py            # Field-concat + LLM-generated searchText
  embedder.py              # Batch embed with OpenAI text-embedding-3-small
  neo4j_writer.py          # MERGE helpers per entity type
  run_kg_ingestion.py      # Orchestrator — called from run_mission.py after each stage
  ingest_report.py         # Standalone CLI — process existing reports
  setup_indexes.py         # One-time vector index creation

kg/schema/
  schema_index.json        # Chunk index (chunk_id, description, keywords, node_labels, rel_types)
  Organizations.md         # Organization + Person schema chunk
  Products.md              # Product + CompoundForm schema chunk
  PanelDefinitions_LabTests.md
```

**Hook in `run_mission.py`:**
```python
# After run_single_mission_slice, if stage.ingest_to_kg:
ingestion_result = await run_kg_ingestion(
    report_text=out["final_report_text"],
    stage=stage,
    neo4j_client=neo4j_client,
    llm=extraction_llm,
    embedder=embedder,
    schema_index=schema_index,
)
out["kg_ingestion"] = ingestion_result
```

---

## Neo4j Vector Index Setup (run once)

```cypher
-- Organization
CREATE VECTOR INDEX organization_embedding_idx IF NOT EXISTS
FOR (n:Organization) ON (n.descriptionEmbedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };

-- Person
CREATE VECTOR INDEX person_embedding_idx IF NOT EXISTS
FOR (n:Person) ON (n.bioEmbedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };

-- Product
CREATE VECTOR INDEX product_embedding_idx IF NOT EXISTS
FOR (n:Product) ON (n.descriptionEmbedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };
```

Embedding model: `text-embedding-3-small` (1536 dims) — matches `biotech-kg/src/server.ts` Neo4jGraphQL vector config.

---

## Entity Types Priority (Current Focus)

Per the mission goals, the current priority entities are:

1. **Organization** — companies, brands, institutions
2. **Person** — executives, founders, scientists, advisors
3. **Product** — supplements, devices, diagnostics, services
4. **Technology** — platforms, methodologies (map to `TechnologyPlatform` node)
5. **Story** — news events, announcements (map to episodic memory for now, later a proper node)
6. **Case Study / Protocol** — lightweight now, deeper later

Scientific papers and clinical trials require a deeper layer (Study node) — defer detailed extraction for now.
