# Plan: Knowledge Graph Ingestion Pipeline

\*\* NOTE you do not need to support searchText on Relationships for right now

**Goal:** After each research mission stage produces a final report, extract structured
entities and relationships from that report, produce rich `searchText` for each node and
edge, embed them, and write precise nodes + relationships into Neo4j according to the
biotech schema.

**Constraint:** The full schema (Organization, Person, Product, CompoundForm, LabTest,
PanelDefinition, RegulatoryPathway, TechnologyPlatform, ManufacturingProcess, Listing, …)
is far too large to paste into a single LLM prompt. Schema selection must be progressive
and on-demand.

---

## 1. High-Level Architecture

```
ResearchMission
  └─ run_mission.py  (stage loop)
       └─ run_slice.py  (per-stage: research → memory)
            └─ NEW: run_kg_ingestion.py  (per-stage post-step)
                  ├─ Step 1: Schema Selection
                  │    └─ schema_selector tool → returns subset of schema chunks
                  ├─ Step 2: Entity Extraction
                  │    └─ extraction LLM call → KGExtractionResult (Pydantic)
                  ├─ Step 3: searchText Generation
                  │    └─ LLM call (rich) OR field-concat (fast) per node/edge
                  ├─ Step 4: Embedding
                  │    └─ batch embed all searchTexts
                  └─ Step 5: Neo4j Write
                       └─ MERGE nodes, MERGE relationships, store embeddings
```

The ingestion step runs **after** `run_single_mission_slice` returns, inside
`run_mission.py`. It receives the `final_report_text` and `stage_type` from the slice
output.

A new `stage_type` value — `"knowledge_graph_ingestion"` — is **not** needed.
Instead, KG ingestion is an opt-in post-step triggered by a flag on `MissionStage`
(e.g. `ingest_to_kg: bool = True`) or run as a standalone pipeline step over already
completed reports.

---

## 2. Schema Selection — Progressive Disclosure

### 2a. Schema Chunk Index

Schema chunks live as plain Markdown files in `src/test_runs/kg/schema/`.
Add a **JSON index** at:

```
src/test_runs/kg/schema/schema_index.json
```

Each entry in the index describes one chunk:

```json
[
  {
    "chunk_id": "organization",
    "label": "Organization",
    "description": "Company, nonprofit, manufacturer, clinic, or any organizational entity. Use when the report mentions a company, brand, or institution.",
    "file": "kg/schema/Organizations.md",
    "keywords": [
      "company",
      "organization",
      "brand",
      "manufacturer",
      "founded",
      "hq",
      "employees",
      "revenue"
    ],
    "node_labels": [
      "Organization",
      "PhysicalLocation",
      "Listing",
      "ManufacturingProcess",
      "TechnologyPlatform"
    ],
    "relationship_types": [
      "HAS_LOCATION",
      "OWNS_OR_CONTROLS",
      "LISTS",
      "OFFERS_PRODUCT",
      "SUPPLIES_COMPOUND_FORM",
      "MANUFACTURES",
      "EMPLOYS",
      "FOUNDED_BY",
      "HAS_BOARD_MEMBER",
      "HAS_SCIENTIFIC_ADVISOR",
      "HAS_EXECUTIVE_ROLE"
    ]
  },
  {
    "chunk_id": "person",
    "label": "Person",
    "description": "Named individual — executive, founder, scientist, advisor. Use when the report names specific people with roles or credentials.",
    "file": "kg/schema/Organizations.md",
    "keywords": [
      "founder",
      "CEO",
      "scientist",
      "advisor",
      "team",
      "leadership",
      "person",
      "doctor",
      "PhD",
      "MD"
    ],
    "node_labels": ["Person"],
    "relationship_types": [
      "EMPLOYS",
      "FOUNDED_BY",
      "HAS_BOARD_MEMBER",
      "HAS_SCIENTIFIC_ADVISOR",
      "HAS_EXECUTIVE_ROLE"
    ]
  },
  {
    "chunk_id": "product",
    "label": "Product",
    "description": "Supplement, device, diagnostic, or service product. Use when the report lists products with names, ingredients, prices, or specs.",
    "file": "kg/schema/Products.md",
    "keywords": [
      "supplement",
      "product",
      "capsule",
      "dose",
      "ingredient",
      "price",
      "SKU",
      "catalog",
      "formulation"
    ],
    "node_labels": [
      "Product",
      "CompoundForm",
      "ProductCategory",
      "RegulatoryStatus",
      "RegulatoryPathway"
    ],
    "relationship_types": [
      "OFFERS_PRODUCT",
      "CONTAINS_COMPOUND_FORM",
      "CONTAINS_COMPOUND",
      "IN_CATEGORY",
      "FOLLOWS_PATHWAY",
      "HAS_REGULATORY_STATUS"
    ]
  },
  {
    "chunk_id": "labtest",
    "label": "LabTest / PanelDefinition",
    "description": "Diagnostic lab test, biomarker panel, or specimen. Use when the report covers diagnostic products or test panels.",
    "file": "kg/schema/PanelDefinitions_LabTests.md",
    "keywords": [
      "lab test",
      "biomarker",
      "panel",
      "diagnostic",
      "LOINC",
      "CPT",
      "specimen",
      "blood draw"
    ],
    "node_labels": ["LabTest", "PanelDefinition", "Biomarker", "Specimen"],
    "relationship_types": ["DELIVERS_LAB_TEST", "IMPLEMENTS_PANEL"]
  }
]
```

### 2b. Schema Selection Tool / Skill

Schema selection runs as a `create_agent` agent with a small `SchemaSelectionResult`
response format. This makes it easy to add tools to the selector later (e.g., the
semantic schema search skill from §2c) without changing the call site.

```python
# src/test_runs/kg/schema_selector.py

from pydantic import BaseModel
from langchain.agents import create_agent

class SchemaSelectionResult(BaseModel):
    chunk_ids: list[str]
    reasoning: str          # brief explanation — useful for debugging

SELECTOR_SYSTEM_PROMPT = """
You are selecting which biotech schema chunks are needed to extract structured entities
from a research report. Choose only chunks where the report clearly contains
extractable data of that type. Return at most {top_k} chunk_ids.
"""

def build_schema_selector_agent(llm, tools: list = None, top_k: int = 3):
    return create_agent(
        llm,
        tools=tools or [],
        system_prompt=SELECTOR_SYSTEM_PROMPT.format(top_k=top_k),
        response_format=SchemaSelectionResult,
    )


async def select_schema_chunks(
    report_text: str,
    stage_type: str,
    targets: list[str],
    index: list[dict],
    selector_agent,             # built with build_schema_selector_agent()
) -> list[dict]:
    """
    Returns the relevant chunk dicts from the index.
    Uses a fast/cheap model — the prompt only contains chunk metadata,
    not the full schema text.
    """
    index_summary = "\n".join(
        f"- {c['chunk_id']}: {c['description']}  keywords={c['keywords']}"
        for c in index
    )
    user_message = f"""
Stage: {stage_type}
Targets: {targets}

Report (first 800 chars):
{report_text[:800]}

Available schema chunks:
{index_summary}

Select the chunk_ids needed to fully extract this report.
"""
    result = await selector_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message.strip()}]}
    )
    selection: SchemaSelectionResult = result["structured_response"]
    selected_ids = set(selection.chunk_ids)
    return [c for c in index if c["chunk_id"] in selected_ids]
```

**Output:** list of chunk dicts → load their `.md` files → concatenate schema text →
feed into the extraction agent.

### 2c. Future: Semantic Schema Search (Skill)

Once the schema is large enough, replace keyword matching with:

- Pre-embed each schema chunk description + keywords at startup
- Vector-search the schema index using the report's embedding
- Return top-k chunks by cosine similarity

This is the "schema searching skill" — it lives in
`src/test_runs/kg/schema_search_skill.py` and is exposed as a LangChain tool
the extraction agent can call with a query like:
`"I need to extract supplement products with compound ingredients and prices"`.

---

## 3. Entity Extraction

### 3a. Extraction Output Model

A single Pydantic model captures everything the LLM extracts:

```python
# src/test_runs/kg/extraction_models.py

class ExtractedOrganization(BaseModel):
    name: str
    aliases: list[str] = []
    orgType: str = "COMPANY"           # OrgType enum
    businessModel: str = "B2C"         # BusinessModel enum
    description: str = ""
    websiteUrl: str = ""
    legalName: str = ""
    primaryIndustryTags: list[str] = []
    regionsServed: list[str] = []
    headquartersCity: str = ""         # → PhysicalLocation node
    headquartersCountry: str = ""
    searchFields: list[str] = ["name", "aliases", "description", "businessModel", "primaryIndustryTags"]

class ExtractedPerson(BaseModel):
    canonicalName: str
    givenName: str = ""
    familyName: str = ""
    honorific: str = ""
    degrees: list[str] = []
    bio: str = ""
    primaryDomain: str = ""
    specialties: list[str] = []
    expertiseTags: list[str] = []
    linkedinUrl: str = ""
    searchFields: list[str] = ["canonicalName", "bio", "primaryDomain", "specialties", "expertiseTags"]

class ExtractedOrgPersonRelationship(BaseModel):
    org_name: str                        # matched to Organization by name
    person_name: str                     # matched to Person by canonicalName
    relationship_type: str               # EMPLOYS | FOUNDED_BY | HAS_BOARD_MEMBER | HAS_SCIENTIFIC_ADVISOR | HAS_EXECUTIVE_ROLE
    roleTitle: str = ""
    department: str = ""
    seniority: str = ""
    isCurrent: bool = True
    searchFields: list[str] = ["roleTitle", "department", "seniority"]

class ExtractedProduct(BaseModel):
    name: str
    brandName: str = ""
    productDomain: str = "SUPPLEMENT"
    productType: str = ""
    description: str = ""
    intendedUse: str = ""
    priceAmount: float | None = None
    currency: str = "USD"
    synonyms: list[str] = []
    searchFields: list[str] = ["name", "synonyms", "brandName", "intendedUse", "description"]

class ExtractedCompoundIngredient(BaseModel):
    product_name: str
    compoundName: str                    # → CompoundForm.canonicalName
    formType: str = ""
    dose: float | None = None
    doseUnit: str = ""
    role: str = "active"
    bioavailabilityNotes: str = ""
    searchFields: list[str] = ["compoundName", "formType", "bioavailabilityNotes"]

class KGExtractionResult(BaseModel):
    source_report: str                   # report path or task_slug
    organizations: list[ExtractedOrganization] = []
    persons: list[ExtractedPerson] = []
    org_person_relationships: list[ExtractedOrgPersonRelationship] = []
    products: list[ExtractedProduct] = []
    compound_ingredients: list[ExtractedCompoundIngredient] = []
    # Add more entity types as schema grows
```

### 3b. Extraction Agent

The extraction step uses `langchain.create_agent` with `response_format=KGExtractionResult`
rather than `llm.with_structured_output`. This gives three important advantages:

- The agent runs in a **tool-calling loop**, so it can call the schema search skill
  (or any other tool) before producing its final structured response.
- It's easy to extend: add tools to the agent without changing the call site.
- The final response is still a typed `KGExtractionResult` — the `response_format`
  parameter forces the last message to conform to the Pydantic model.

```python
# src/test_runs/kg/extractor.py

from langchain.agents import create_agent

EXTRACTION_SYSTEM_PROMPT = """
You are a biotech knowledge graph extraction agent.

Your job:
1. Read the research report provided in the user message.
2. If you need to look up additional schema details, use the schema_search tool.
3. Extract all entities and relationships that are explicitly stated or strongly implied.
4. Leave fields empty or omit list entries when you are not confident.
5. Use exact enum values from the schema (OrgType, BusinessModel, ProductDomain, etc.).
6. When finished, produce a final KGExtractionResult with everything you found.
"""

def build_extraction_agent(llm, tools: list = None):
    """
    Build a reusable extraction agent.
    Pass tools=[schema_search_tool, ...] to enable progressive schema lookup.
    Pass tools=[] for a simple single-pass extraction run.
    """
    return create_agent(
        llm,
        tools=tools or [],
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        response_format=KGExtractionResult,
    )


async def extract_kg_entities(
    report_text: str,
    selected_schema_text: str,
    agent,                             # built with build_extraction_agent()
    source_report: str = "",
) -> KGExtractionResult:
    user_message = f"""
Schema (selected subset for this report):
{selected_schema_text}

Research report:
{report_text}

Extract all nodes and relationships. Set source_report = "{source_report}".
"""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message.strip()}]}
    )
    # create_agent with response_format returns the structured model
    # in result["structured_response"]
    return result["structured_response"]
```

**Usage in the orchestrator:**

```python
extraction_agent = build_extraction_agent(llm, tools=[schema_search_tool])
extraction: KGExtractionResult = await extract_kg_entities(
    report_text=report_text,
    selected_schema_text=selected_schema_text,
    agent=extraction_agent,
    source_report=stage.slice_input.task_slug,
)
```

**Model choice:** Use a mid-tier model (gpt-4o or equivalent) — extraction quality
matters here. Build the agent once and reuse it across all stages in a mission run.

---

## 4. searchText Generation

Each node and relationship needs a `searchText` before embedding.
Two modes, selectable per node type:

### 4a. Field-Concat (fast, deterministic)

Combine the values of `searchFields` with space separators:

```python
def build_searchtext_from_fields(
    entity_dict: dict,
    search_fields: list[str],
) -> str:
    parts = []
    for field in search_fields:
        val = entity_dict.get(field)
        if isinstance(val, list):
            parts.append(" ".join(str(v) for v in val if v))
        elif val:
            parts.append(str(val))
    return " | ".join(parts)

# Example for Organization:
# "Elysium Health | Basis, Signal, Matter | longevity aging NAD+ supplement company | B2C | supplement longevity aging"
```

### 4b. LLM-Generated searchText (richer, semantic)

For nodes where a narrative searchText significantly improves retrieval quality
(Organization, Person, Product), use `langchain.create_agent` with a small
`SearchTextResult` response format. This keeps the pattern consistent and allows
tools to be added later (e.g., fetching additional context from memory):

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class SearchTextResult(BaseModel):
    searchText: str

SEARCHTEXT_SYSTEM_PROMPT = """
Write a 2–4 sentence dense semantic description of the given entity for use in
vector search. Include the most searchable facts: name, type, domain, key
attributes, relationships. No markdown. No preamble. Just the description.
"""

def build_searchtext_agent(llm, tools: list = None):
    return create_agent(
        llm,
        tools=tools or [],
        system_prompt=SEARCHTEXT_SYSTEM_PROMPT,
        response_format=SearchTextResult,
    )


async def generate_searchtext_llm(
    entity_type: str,
    entity_dict: dict,
    context: str,               # mission objective or report summary
    searchtext_agent,           # built with build_searchtext_agent()
) -> str:
    user_message = f"""
Entity type: {entity_type}
Context: {context}

Entity data:
{json.dumps(entity_dict, indent=2)}
"""
    result = await searchtext_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message.strip()}]}
    )
    return result["structured_response"].searchText
```

**Strategy:** Use LLM searchText for `Organization`, `Person`, `Product` nodes.
Use field-concat for relationship edges and less-critical nodes (CompoundForm, Listing).
Make it configurable per entity type via a `SEARCHTEXT_STRATEGY` dict.

---

## 5. Embedding

Use OpenAI `text-embedding-3-small` (1536 dims) or `text-embedding-3-large` (3072 dims).

```python
# src/test_runs/kg/embedder.py

from langchain_openai import OpenAIEmbeddings

async def embed_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    embedder = OpenAIEmbeddings(model=model)
    return await embedder.aembed_documents(texts)
```

Batch all `searchText` values together in one API call before writing to Neo4j.

### Vector Index Creation (one-time setup)

Run this once per node label that needs vector search:

```cypher
-- Organization
CREATE VECTOR INDEX organization_search_idx IF NOT EXISTS
FOR (n:Organization) ON (n.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };

-- Person
CREATE VECTOR INDEX person_search_idx IF NOT EXISTS
FOR (n:Person) ON (n.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };

-- Product
CREATE VECTOR INDEX product_search_idx IF NOT EXISTS
FOR (n:Product) ON (n.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };

-- CompoundForm
CREATE VECTOR INDEX compound_form_search_idx IF NOT EXISTS
FOR (n:CompoundForm) ON (n.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };
```

Add this to a `src/test_runs/kg/setup_indexes.py` script that runs once on first use.

---

## 6. Neo4j Write Patterns

### 6a. Node MERGE Pattern

All nodes use `MERGE` on their UUID `*Id` field (generated client-side with `uuid4()`
before write). The existing `Neo4jAuraClient` in `neo4j_aura.py` handles the
async driver — no changes needed there.

```python
# src/test_runs/kg/neo4j_writer.py

async def upsert_organization(client: Neo4jAuraClient, data: dict) -> None:
    await client.execute_write("""
        MERGE (o:Organization { organizationId: $organizationId })
        ON CREATE SET o.createdAt = datetime()
        SET o.name             = coalesce($name, o.name),
            o.aliases          = $aliases,
            o.orgType          = coalesce($orgType, o.orgType),
            o.businessModel    = coalesce($businessModel, o.businessModel),
            o.description      = coalesce($description, o.description),
            o.websiteUrl       = coalesce($websiteUrl, o.websiteUrl),
            o.primaryIndustryTags = $primaryIndustryTags,
            o.regionsServed    = $regionsServed,
            o.searchText       = $searchText,
            o.searchFields     = $searchFields,
            o.embedding        = $embedding,
            o.embeddingModel   = $embeddingModel,
            o.embeddingDimensions = $embeddingDimensions,
            o.validAt          = datetime()
        RETURN o
    """, parameters=data)
```

### 6b. Relationship MERGE Pattern

```python
async def upsert_org_person_rel(client, org_id: str, person_id: str, rel_type: str, props: dict) -> None:
    cypher = f"""
        MATCH (o:Organization {{ organizationId: $org_id }})
        MATCH (p:Person {{ personId: $person_id }})
        MERGE (o)-[r:{rel_type}]->(p)
        ON CREATE SET r.createdAt = datetime()
        SET r.roleTitle    = coalesce($roleTitle, r.roleTitle),
            r.isCurrent    = coalesce($isCurrent, r.isCurrent),
            r.searchText   = $searchText,
            r.searchFields = $searchFields,
            r.embedding    = $embedding,
            r.validAt      = datetime()
        RETURN r
    """
    await client.execute_write(cypher, parameters={"org_id": org_id, "person_id": person_id, **props})
```

### 6c. Entity Resolution (Name → UUID)

The extraction LLM returns names, not UUIDs. Before writing relationships, resolve
names to UUIDs via a local in-memory map built during the node-write phase:

```python
name_to_org_id: dict[str, str] = {}     # org.name → organizationId
name_to_person_id: dict[str, str] = {}  # person.canonicalName → personId
```

If no match is found, skip the relationship write and log it.
Future: implement fuzzy matching or a disambiguation LLM call.

---

## 7. Integration into the Mission Workflow

### 7a. New post-step in `run_mission.py`

Add an optional `ingest_to_kg: bool = True` field to `MissionStage`.
After `run_single_mission_slice` returns, call the KG ingestion pipeline:

```python
# In run_mission.py, inside the stage loop, after collecting out:

if stage.ingest_to_kg and neo4j_client:
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

### 7b. `run_kg_ingestion` orchestrator

```python
# src/test_runs/kg/run_kg_ingestion.py

async def run_kg_ingestion(
    report_text: str,
    stage: MissionStage,
    neo4j_client: Neo4jAuraClient,
    llm,
    embedder,
    schema_index: list[dict],
) -> dict:

    # 1. Select schema chunks
    selected_chunks = select_schema_chunks(
        report_text=report_text,
        stage_type=stage.slice_input.stage_type,
        targets=stage.slice_input.targets,
        index=schema_index,
        llm=llm,
    )
    selected_schema_text = load_schema_chunks(selected_chunks)

    # 2. Extract entities
    extraction: KGExtractionResult = await extract_kg_entities(
        report_text=report_text,
        selected_schema_text=selected_schema_text,
        llm=llm,
    )

    # 3. Generate searchTexts (LLM or field-concat per entity type)
    # 4. Batch embed all searchTexts
    # 5. Write nodes to Neo4j (MERGE)
    # 6. Write relationships to Neo4j (MERGE)

    return {"extraction": extraction, "chunks_used": [c["chunk_id"] for c in selected_chunks]}
```

### 7c. Standalone Mode (for existing reports)

Also expose as a standalone runner that processes completed reports without
re-running the research:

```python
# src/test_runs/kg/ingest_report.py
# Usage: uv run python -m src.test_runs.kg.ingest_report \
#           --report reports/elysium-products-and-specs.md \
#           --targets "Elysium Health" \
#           --stage_type targeted_extraction
```

This is immediately usable with the already-completed Elysium and Qualia reports.

---

## 8. Recommended File Structure

```
src/test_runs/kg/
  __init__.py
  schema_selector.py          # chunk selection logic (LLM + keyword)
  schema_search_skill.py      # future: semantic schema search tool
  extraction_models.py        # KGExtractionResult and sub-models (Pydantic)
  extractor.py                # extraction LLM call
  searchtext.py               # field-concat and LLM-generated searchText
  embedder.py                 # batch embedding wrapper
  neo4j_writer.py             # MERGE helpers per entity type
  run_kg_ingestion.py         # orchestrator (called from run_mission.py)
  ingest_report.py            # standalone CLI runner
  setup_indexes.py            # one-time vector index creation
```

Schema index and chunk files:

```
src/test_runs/kg/schema/schema_index.json
src/test_runs/kg/schema/Organizations.md
src/test_runs/kg/schema/Products.md
src/test_runs/kg/schema/PanelDefinitions_LabTests.md
```

---

## 9. Implementation Order (Suggested)

| Step | File                       | What it does                                        |
| ---- | -------------------------- | --------------------------------------------------- |
| 1    | `schema_index.json`        | Author the chunk index (fast)                       |
| 2    | `extraction_models.py`     | Define Pydantic extraction output models            |
| 3    | `setup_indexes.py`         | Create vector indexes in Neo4j (run once)           |
| 4    | `extractor.py`             | Extraction LLM call with structured output          |
| 5    | `neo4j_writer.py`          | MERGE helpers for Organization, Person, Product     |
| 6    | `searchtext.py`            | Field-concat implementation (start here)            |
| 7    | `embedder.py`              | Batch embedding                                     |
| 8    | `run_kg_ingestion.py`      | Wire 1–7 together                                   |
| 9    | `ingest_report.py`         | Standalone CLI — test on Elysium report immediately |
| 10   | `schema_selector.py`       | Schema selection (LLM call with index)              |
| 11   | `searchtext.py` (LLM mode) | Add LLM-generated searchText for key nodes          |
| 12   | `run_mission.py`           | Hook ingestion into the mission loop                |

**Start with step 9 first** using the existing Elysium products report and hardcoded
schema chunks to validate the extraction and write pipeline before generalizing.
The Elysium products report is already in `reports/elysium-products-and-specs.md`
and is a clean test case: Organization + Products + CompoundIngredients.

---

## 10. Key Design Decisions and Trade-offs

### Schema selection: LLM vs. keyword

- **Keyword matching** is fast, cheap, and deterministic. Build it first.
- **LLM classification** is more robust for edge cases. Add later as fallback.
- **Semantic search** over schema embeddings is the most scalable long-term solution
  but requires upfront embedding of schema chunks.

### searchText: LLM vs. field-concat

- **Field-concat** is always available, deterministic, zero-cost, and good enough for
  most nodes. Use as default.
- **LLM searchText** produces richer, more queryable text for the most important
  node types (Organization, Person, Product). It costs one extra LLM call per node
  but dramatically improves recall in semantic search. Use for these three types.
- Store `searchFields` on the node regardless so the concat can be reconstructed
  without re-running the LLM.

### Agent pattern — `langchain.create_agent` + `response_format`

All three LLM-powered steps (schema selection, entity extraction, searchText generation)
use `langchain.create_agent` (`from langchain.agents import create_agent`) with a
`response_format=PydanticModel`. This means:

- Each step runs in a tool-calling loop and can be extended with tools without changing
  the call site.
- The final output is always a typed Pydantic model via `result["structured_response"]`.
- Build each agent once at startup and reuse across all stages in a mission run.
- The system prompt is passed as `system_prompt=` (string or `SystemMessage`).
- `response_format` accepts a Pydantic class directly — it defaults to `ProviderStrategy`
  (native structured output) and falls back to `ToolStrategy` (tool-call based) when
  the provider doesn't support native structured output. Use `ToolStrategy(Model)` or
  `ProviderStrategy(Model)` explicitly if you need to force one strategy.
- **Never** use `llm.with_structured_output` in this pipeline — it bypasses the
  agentic loop and cannot be extended with tools.

### Extraction quality

- The extraction system prompt emphasizes: "extract only what is explicitly stated,
  leave null when uncertain." Hallucinated edges in a knowledge graph are worse than
  missing edges.
- For high-stakes fields (pricing, dosages), include a `confidence: float` field on
  the extraction model so low-confidence values can be flagged for human review.

### Entity resolution

- Start with exact name matching (normalized to lowercase, stripped).
- For production: add a disambiguation step — query Neo4j for fuzzy matches before
  creating a new node.

### Relationship embeddings

- Not all relationships need embeddings immediately. Prioritize node embeddings first.
  Add relationship embeddings for the most semantically important edge types
  (EMPLOYS, FOUNDED_BY, CONTAINS_COMPOUND_FORM) in a second pass.

---

## 11. Open Questions to Resolve Before Building

1. **Which LLM for extraction?** gpt-4o recommended for accuracy. Consider o3-mini for cost
   on high-volume runs.
2. **Embedding model dimensions?** 1536 (text-embedding-3-small) is a good default.
   Decide before creating vector indexes — changing dimensions later requires dropping
   and recreating the index.
3. **Confidence thresholds?** Define minimum confidence for writing a node vs. just logging it.
4. **Re-ingestion behavior?** MERGE is idempotent for nodes. For relationships: do you
   always update properties, or only on first create? Current pattern uses SET unconditionally
   (overwrite on re-run) which is fine for iterative research.
5. **Ingestion from intermediate files vs. final report only?** Starting with final report only
   is simpler and correct per your stated assumption ("final report combines all intermediates
   optimally"). If intermediate files reveal structured data not in the final report, revisit.
