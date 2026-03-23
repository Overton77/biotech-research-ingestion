Good day Mr. Opus 4.6,

I need your help re-architecting my Neo4j knowledge graph ingestion pipeline to support bitemporal validity and a model where state is separated from structure.

This will increase schema and ingestion complexity, but it will make the knowledge graph much more expressive, especially for biotech entities, evolving scientific claims, and longitudinal evidence.

This work also implies changes to the LLM research workflow:

- research configuration must carry temporal assumptions and temporal scope
- prompts must always include the current date
- extraction and report generation must surface temporal information explicitly
- ingestion must update existing state/relationships carefully rather than just append naive facts

We are applying this to my current codebase, specifically around:

@biotech-research-ingestion/src/research/langchain_agent/kg/

Please inspect the existing ontology subset, schema selection flow, report generation flow, extraction flow, and Neo4j ingestion pipeline before proposing implementation changes.

==================================================
PRIMARY GOAL
==================================================

Re-architect the current KG ingestion workflow so that it supports:

1. stable canonical identity nodes
2. immutable state snapshot nodes
3. time-bounded relationships between identity and state
4. time-bounded structural relationships between canonical entities
5. bitemporal semantics where possible:
   - validFrom / validTo
   - recordedFrom / recordedTo
6. preparation for richer scientific claim modeling using assertion nodes

The target pattern is:

STATE SEPARATED FROM STRUCTURE

- Durable identity / structure is separated from mutable descriptive state.
- Identity nodes are stable.
- State nodes are immutable snapshots.
- Identity -> State is attached through time-bounded HAS_STATE relationships.
- Structural facts between canonical entities are also time-bounded.
- Complex scientific claims should use reified assertion nodes.

==================================================
TARGET MODEL
==================================================

Core identity/state pattern:

- Identity node:
  (:Product {productId})
- State node:
  (:ProductState { ... immutable state snapshot payload ... })
- Attachment:
  (:Product)-[:HAS_STATE {
  validFrom,
  validTo,
  recordedFrom,
  recordedTo
  }]->(:ProductState)

Structural facts are also time-bounded, for example:

- (:Person)-[:AFFILIATED_WITH {validFrom, validTo, recordedFrom, recordedTo}]->(:Organization)
- (:Organization)-[:DEVELOPS {validFrom, validTo, recordedFrom, recordedTo}]->(:Product)
- (:Product)-[:HAS_MECHANISM {validFrom, validTo, recordedFrom, recordedTo}]->(:Mechanism)

For more complex scientific claims, prefer reified assertions:

- (:Product)-[:HAS_ASSERTION]->(:MechanisticAssertion)
- (:MechanisticAssertion)-[:ABOUT_MECHANISM]->(:Mechanism)
- (:MechanisticAssertion)-[:SUPPORTED_BY]->(:Evidence)
- (:MechanisticAssertion)-[:HAS_STATE {
  validFrom,
  validTo,
  recordedFrom,
  recordedTo
  }]->(:MechanisticAssertionState)

Case study / evidence pattern:

- (:CaseStudy)-[:HAS_STATE]->(:CaseStudyState)
- (:CaseStudy)-[:HAS_ASSERTION]->(:OutcomeAssertion)
- (:OutcomeAssertion)-[:ON_PRODUCT]->(:Product)
- (:OutcomeAssertion)-[:ON_ENDPOINT]->(:Endpoint)
- (:OutcomeAssertion)-[:IN_POPULATION]->(:Population)
- (:OutcomeAssertion)-[:HAS_STATE]->(:OutcomeAssertionState)

==================================================
TEMPORAL SEMANTICS
==================================================

We want both:

1. valid time

- when the fact is true in the domain

2. recorded time

- when the system knew / stored / believed the fact

For now, assume a research report being ingested is current unless the report explicitly indicates another temporal frame.

However, design the system so that later we can support:

- historical reports
- backfilled historical facts
- future-dated knowledge
- as-of valid time and recorded time queries

Important practical assumption:

- A coordinator agent may already have reasoned about some temporal scope and entity change history before this ingestion run.
- But this ingestion module must still be robust if that resolution is incomplete.
- It must defensively inspect current graph state before creating new state or changing relationships.

==================================================
CURRENT WORKFLOW CONTEXT
==================================================

Right now I have a working subset of:

- ontology / schema selection
- report -> schema selection
- LLM extraction of biotech entities
- Neo4j ingestion

This currently assumes a simpler non-temporal or insufficiently temporal model.

That will now need to change substantially.

I want you to understand the current code path first, then design and implement the migration toward the temporal state-separated model.

==================================================
IMPORTANT MODELING PRINCIPLES
==================================================

Use these principles consistently:

1. Identity is stable
   Canonical entities such as Organization, Person, Product should remain durable anchors.

2. State is immutable
   Descriptive state changes should create new state snapshots, not mutate old ones.

3. Structure changes independently from state
   Changing a Product name or summary should not require rewiring the entire structural graph.

4. Historical truth and system belief are distinct
   Whenever feasible, preserve both valid-time and recorded-time semantics.

5. Avoid silent mutation of history
   Prefer closing previous validity windows and adding new state/relationships rather than overwriting prior history.

6. Prefer assertion nodes for complex scientific claims
   Biomechanistic claims and study findings are not just simple relationships when confidence, provenance, interpretation, or curation can change over time.

==================================================
PRACTICAL INGESTION REQUIREMENTS
==================================================

The ingestion pipeline should be updated to support the following behaviors:

1. Before ingesting a new state for an entity:

- look up the current canonical identity node
- look up the current active state, if one exists
- compare whether the incoming state is materially different
- if different:
  - close the previous HAS_STATE validity interval
  - create a new immutable state node
  - create a new HAS_STATE relationship with current valid/recorded bounds

2. Before creating or updating a structural relationship:

- check whether an active matching relationship already exists
- if it exists and is unchanged, do not duplicate it
- if it exists but must be updated:
  - close the old relationship interval
  - create the new relationship interval
- if it does not exist:
  - create it with the appropriate temporal bounds

3. For current-state assumptions:

- treat the report as current unless explicit temporal evidence says otherwise
- use ingestion time as recordedFrom
- validFrom may initially default to report/publication/research-run time depending on available evidence and design choice
- validTo / recordedTo for active facts should be open-ended

4. We will need temporal metadata upstream in research

- research configuration should include temporal scope / temporal assumptions
- extraction prompts should ask for time-relevant evidence
- final report should explicitly cite temporal qualifiers when available
- prompts should always include the current date

==================================================
RESEARCH AGENT / REPORTING CHANGES
==================================================

Please also modify the research workflow so that temporal information becomes first-class.

I want you to inspect how the research agent is configured and how prompts are constructed, then propose changes so that:

1. the current date is always included in prompts
2. research configuration can include temporal scope, such as:
   - current state
   - as of a date
   - within date range
   - unknown / infer from sources
3. extracted entity facts can include temporal qualifiers
4. final report output can cite temporal context explicitly
5. ingestion can consume those temporal fields cleanly

For this phase, default assumption can remain:

- report is about current state
- but the structure should support future extension to historical reasoning

==================================================
INVARIANTS TO ENFORCE
==================================================

Design around these invariants:

1. One active current state per identity per state type
2. No overlapping valid windows for mutually exclusive states
3. No overlapping recorded windows for the same exact active belief version
4. Old state nodes are never mutated
5. Updates close old intervals and create new ones
6. Structural relationships should also be managed temporally
7. Prefer idempotent ingestion where possible
8. Avoid duplicate active states and duplicate active relationships

==================================================
WHAT I WANT YOU TO DO
==================================================

Please do this in stages.

Stage 1 — Inspect current implementation

- Identify the current ontology subset in code
- Identify where schema selection happens
- Identify where extraction output is shaped
- Identify where Neo4j node creation / merging happens
- Identify assumptions in the current ingestion code that conflict with temporal state separation

Stage 2 — Design the target architecture

- Propose the canonical entity/state model for current entity classes
- Specify which labels get state nodes
- Specify which relationships become temporal
- Specify where assertion-node modeling is needed now vs later
- Define temporal property conventions and default values
- Define identity keys and state hashing / comparison strategy if appropriate

Stage 3 — Propose implementation changes

- Show which files/modules should change
- Show what interfaces / data contracts must change
- Show how extraction output schema must change
- Show how report schema must change
- Show how ingestion logic must change
- Show how Neo4j merge/upsert logic must change

Stage 4 — Implement carefully

- Make changes incrementally
- Preserve working behavior where possible
- Avoid breaking the existing pipeline unnecessarily
- Introduce migration-safe patterns

Stage 5 — Summarize

- Explain what was changed
- Explain remaining gaps
- Explain assumptions made
- Explain what should be done next

==================================================
IMPORTANT IMPLEMENTATION DETAILS
==================================================

Please be explicit about the following:

1. Identity keys
   How should canonical entities be keyed?
   Examples:

- Organization.organizationId
- Person.personId
- Product.productId

2. State comparison
   Before creating a new state snapshot, how should the system decide whether the incoming state is actually new?
   Potentially:

- compare normalized payload
- compare selected state fields
- use payload hash

3. Open-ended intervals
   Choose and standardize representation for active intervals.

4. Bitemporal defaults
   Define what values should be used when valid time is not fully known but recorded time is known.

5. Relationship update semantics
   Be precise about when to:

- retain
- close
- supersede
- recreate

6. Provenance
   Ensure temporal state and relationships can carry provenance or link to provenance/evidence where appropriate.

==================================================
IMPORTANT DOMAIN GUIDANCE
==================================================

Model these distinctions carefully:

1. Canonical entities

- Organization
- Person
- Product
- Mechanism
- CaseStudy

2. Descriptive state
   Examples:

- Product display name
- Product regulatory status
- Organization website
- Person role title
- CaseStudy summary

3. Structural relationships
   Examples:

- Person AFFILIATED_WITH Organization
- Organization DEVELOPS Product
- Product HAS_MECHANISM Mechanism

4. Assertion-style scientific facts
   Examples:

- Product modulates a pathway with some confidence
- Product has a mechanism claim supported by evidence
- Study outcome interpretation changes over time
- Confidence / curation / evidence strength changes without changing canonical identity

These assertion-style facts should likely move toward reified assertion nodes rather than naive direct edges.

==================================================
EXPECTED OUTPUT
==================================================

I do not want a vague conceptual answer.

I want you to:

1. inspect the current code first
2. produce a concrete implementation plan mapped to the current codebase
3. identify likely modules/files to modify
4. propose any schema/data-contract changes
5. implement or partially implement the changes where reasonable
6. explain tradeoffs and unresolved design choices

Return your work in this structure:

1. Current system assessment
2. Proposed temporal architecture
3. Required code changes by module/file
4. Data contract / schema changes
5. Neo4j ingestion changes
6. Research prompt / workflow changes
7. Migration strategy
8. Risks / open questions / recommended next steps

==================================================
GUARDRAILS
==================================================

- Do not assume the current code already matches this architecture.
- Inspect before changing.
- Be concrete and implementation-oriented.
- Prefer incremental migration over a full rewrite unless the code strongly demands otherwise.
- Preserve compatibility where possible.
- Call out anywhere the existing ontology or extraction format is insufficient for temporal modeling.
- If you encounter ambiguity, make the most reasonable implementation-oriented assumption and state it explicitly.
