# Roadmap — What to Build Next

> **Scope:** `biotech-research-ingestion/src/research/langchain_agent/`  
> This document tracks everything to build, ordered by dependency and priority.  
> Cross-reference `ARCHITECTURE.md` for current state, `KG_API.md` for storage targets, `ECOSYSTEM_SKILLS.md` for how to implement.

---

## Mission Types to Add

### 1. Stage-Based Research (DAG execution)

**What:** Missions defined as a DAG of stages. Stages with no dependencies run in parallel. Dependent stages receive prior outputs.

**Current state:** The topological sort exists in `workflow/run_mission.py`, but execution is strictly sequential — each stage runs after the previous regardless of parallelism opportunities.

**What to build:**

```python
# workflow/run_mission.py — parallel execution of independent stages

import asyncio
from collections import defaultdict

async def run_mission_parallel(mission: ResearchMission, ...) -> List[Dict]:
    """
    True DAG execution: run all stages with no unmet dependencies in parallel,
    then unlock the next wave, repeat until done.
    """
    completed: dict[str, str] = {}     # task_slug -> final_report_text
    pending = list(range(len(mission.stages)))

    while pending:
        # Find all stages whose dependencies are already in `completed`
        ready = [
            i for i in pending
            if all(dep in completed for dep in mission.stages[i].dependencies)
        ]
        if not ready:
            raise RuntimeError("Cycle or unresolvable dependency detected")

        # Run ready stages in parallel
        wave_tasks = [
            run_single_mission_slice(
                run_input=_inject_deps(mission.stages[i].slice_input, completed),
                prompt_spec=mission.stages[i].prompt_spec,
                ...
            )
            for i in ready
        ]
        wave_outputs = await asyncio.gather(*wave_tasks)

        for i, out in zip(ready, wave_outputs):
            slug = mission.stages[i].slice_input.task_slug
            completed[slug] = out.get("final_report_text") or ""
            pending.remove(i)

    return list of outputs
```

**Files to touch:** `workflow/run_mission.py`, `models/mission.py`

---

### 2. Iterative Research (Fresh-Context Cycles)

**What:** A mission type that loops — after completing all stages, it evaluates whether the research is sufficient, then optionally starts a new cycle with fresh context, seeded by the prior cycle's memories and outputs.

**Use cases:**
- Deep-dives that need progressive refinement
- Missions where early discoveries reveal new targets
- Long-running research that should continue accumulating knowledge over days/weeks

**Model to add:**

```python
class IterativeResearchConfig(BaseModel):
    max_cycles: int = 3
    completion_criteria: str = ""          # Passed to evaluator agent
    seed_from_prior_outputs: bool = True   # Inject prior reports as context

class IterativeResearchMission(BaseModel):
    mission_id: str
    mission_name: str
    base_domain: str
    stage_template: MissionStage           # Template cloned for each cycle
    iterative_config: IterativeResearchConfig
    cycles_completed: int = 0
```

**Workflow:**

```python
async def run_iterative_mission(mission: IterativeResearchMission, ...):
    prior_reports: list[str] = []
    for cycle_num in range(1, mission.iterative_config.max_cycles + 1):
        # Build fresh MissionSliceInput for this cycle
        cycle_input = mission.stage_template.slice_input.model_copy(deep=True)
        cycle_input.task_id = f"{cycle_input.task_id}-cycle-{cycle_num}"
        cycle_input.task_slug = f"{cycle_input.task_slug}-cycle-{cycle_num}"
        if mission.iterative_config.seed_from_prior_outputs and prior_reports:
            cycle_input.guidance_notes += [
                f"Prior cycle {i+1} summary:\n{rpt[:800]}"
                for i, rpt in enumerate(prior_reports[-2:])
            ]

        out = await run_single_mission_slice(cycle_input, ...)
        prior_reports.append(out["final_report_text"])

        # Evaluate — simple heuristic or LLM evaluator agent
        if await _is_research_complete(out, mission.iterative_config.completion_criteria):
            break

    return prior_reports
```

**Files to create:** `models/iterative_mission.py`, `workflow/run_iterative_mission.py`

---

## Temporal Activities (Temporal.io)

**What:** Wrap each stage and iterative cycle in a Temporal Activity so the workflow is durable, retryable, and observable.

**Architecture:**

```
Temporal Worker
  └─ ResearchWorkflow (Workflow)
       ├─ execute_stage_activity(stage) → Temporal Activity
       │    └─ run_single_mission_slice(...)
       ├─ execute_parallel_stages(wave: List[stage]) → gather Temporal Activities
       └─ evaluate_and_continue_activity(outputs) → decides next iteration

# All I/O (LLM calls, tool calls, Neo4j writes) happens inside Activities
# The Workflow itself is deterministic (no I/O)
```

**Key design decisions:**
- One Activity per stage — granular retries and visibility
- Use `asyncio.gather` inside the Workflow for parallel stages in the same wave
- Store Activity results in Temporal Workflow history (automatically durable)
- Heartbeat long-running Activities (research stages can take minutes)

**Files to create:** `temporal/workflows.py`, `temporal/activities.py`, `temporal/worker.py`

**Dependencies to add:** `temporalio`

---

## Storage Targets

### Research Runs → MongoDB

Store each completed `run_single_mission_slice` output as a run document.

```python
# storage/research_run_store.py

from src.research.langchain_agent.storage.async_mongo_client import mongo_client

RUNS_COLLECTION = "research_runs"

async def save_research_run(out: dict, mission_id: str, task_slug: str) -> str:
    db = mongo_client["biotech_research"]
    doc = {
        "mission_id": mission_id,
        "task_slug": task_slug,
        "final_report_text": out["final_report_text"],
        "structured_memory_report": out["structured_memory_report"].model_dump(),
        "agent_state": out["agent_state"],
        "created_at": datetime.utcnow(),
    }
    result = await db[RUNS_COLLECTION].insert_one(doc)
    return str(result.inserted_id)
```

**Collection:** `biotech_research.research_runs`  
**Index:** `{ mission_id: 1, task_slug: 1, created_at: -1 }`

### File Artifacts → S3 (mirrored, not replaced)

The local `agent_outputs/` filesystem is **always kept** — agents continue to write to it during a run as their primary scratchpad. S3 is an additional mirror: after each stage completes, outputs and intermediate files are uploaded to S3 for durability, cross-environment sharing, and downstream pipeline access.

```python
# storage/s3_artifact_store.py — mirrors completed artifacts to S3

import aiobotocore

async def upload_artifact(content: str, key: str, bucket: str = ARTIFACTS_BUCKET):
    session = aiobotocore.get_session()
    async with session.create_client("s3") as s3:
        await s3.put_object(Bucket=bucket, Key=key, Body=content.encode())

async def mirror_stage_outputs(
    task_slug: str,
    mission_id: str,
    written_file_paths: list[str],
    root: Path = ROOT_FILESYSTEM,
):
    """After a slice completes, upload all written files to S3."""
    for rel_path in written_file_paths:
        content = await read_file_text(rel_path, root=root)
        if content:
            s3_key = f"research/{mission_id}/{task_slug}/{rel_path}"
            await upload_artifact(content, key=s3_key)
```

**S3 key pattern:** `research/{mission_id}/{task_slug}/{filename}`  
**Bucket env var:** `S3_ARTIFACTS_BUCKET`  
**Local path:** `agent_outputs/{runs|reports|scratch}/` — unchanged, always present

Call `mirror_stage_outputs()` at the end of `run_single_mission_slice()` using the `written_file_paths` already tracked in agent state. The `FilesystemMiddleware` backend stays as `FilesystemBackend` — no swap needed.

---

## Mission Creation Agent (Pre-Research Step)

**What:** Before a mission runs, an LLM agent builds the `ResearchMission` from a natural language query, optional starter sources, and optional starter content.

**Input:**
```python
class MissionCreationRequest(BaseModel):
    query: str                            # "Research Qualia Life Sciences' product line"
    entity_type: str = "organization"     # organization, person, product, technology, story
    starter_sources: List[str] = []       # URLs to use as initial context
    starter_content: str = ""             # Pasted text / document content
    depth: Literal["surface", "deep", "exhaustive"] = "deep"
```

**Output:** `ResearchMission` (with stages already populated)

**Agent pattern:**
```python
mission_agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    system_prompt=MISSION_CREATION_PROMPT,
    response_format=ResearchMission,
)
result = await mission_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
mission: ResearchMission = result["structured_response"]
```

**File to create:** `agents/mission_creator.py`

---

## Additional Tools and Data Sources

### Medical / Case Study / Protocol Data

| Source | Tool to build | Notes |
|---|---|---|
| PubMed | `tools/pubmed_tool.py` — wire existing `funcs/pubmed.py` | Abstracts, trials, NRF2 etc. |
| ClinicalTrials.gov | `tools/clinicaltrials_tool.py` | REST API, search by condition/intervention |
| OpenFDA | `tools/openfda_tool.py` | Drug/device adverse events, recall data |
| Semantic Scholar | `tools/semantic_scholar_tool.py` | Paper search with citations |

### Document Conversion / Extraction

| Capability | Tool to build | Notes |
|---|---|---|
| PDF → Markdown | `tools/pdf_extract_tool.py` | Use `pymupdf` or `pdfplumber` |
| HTML → clean Markdown | `tools/html_to_md_tool.py` | `markdownify` or `trafilatura` |
| DOCX → Markdown | `tools/docx_tool.py` | `python-docx` |
| Multi-format (PDF, DOCX, PPTX, HTML, images) | `tools/docling_tool.py` | **Docling** — unified document conversion library from IBM; handles all formats in one tool, produces clean Markdown or structured JSON; consider as the primary converter with the per-format libraries above as fallbacks or supplements |

### Tavily Fallbacks

When Tavily fails or rate-limits:

| Fallback | Library | Coverage |
|---|---|---|
| SerpAPI | `google-search-results` | General web search |
| DuckDuckGo | `duckduckgo-search` | Free, no API key |
| Bing Search | `azure-cognitiveservices-search-websearch` | Paid, reliable |

Create a `tools/search_fallback_tool.py` that tries Tavily first, falls back in order.

---

## LangSmith Evaluation

**Skill to read first:** `.agents/skills/langsmith-evaluator/SKILL.md` and `.agents/skills/langsmith-dataset/SKILL.md`

### What to evaluate

| What | How | Metric |
|---|---|---|
| Final report quality | LLM-as-judge on report content | Coverage score, factual accuracy, section completeness |
| KG extraction accuracy | Compare extracted entities vs. ground truth | Precision / recall on org names, person names, products |
| Memory recall relevance | Score recalled memories against the stage objective | Semantic similarity, hit rate |
| Tool call efficiency | Count tool calls vs. findings density | Calls per validated fact |
| Mission completion | Did the agent answer the research objective? | Pass/fail + rubric score |

---

### Dataset Strategy

Build LangSmith datasets from existing completed runs (Qualia, Elysium, Dave Asprey):

```python
# eval/build_datasets.py

from langsmith import Client

client = Client()

# Create a dataset from completed reports
dataset = client.create_dataset(
    dataset_name="biotech-research-reports-v1",
    description="Final research reports for known biotech entities",
)

# Add examples: input = mission objective + targets, output = expected report sections present
client.create_examples(
    inputs=[
        {"objective": "Research Elysium Health company fundamentals", "targets": ["Elysium Health"]},
        {"objective": "Catalog Qualia Life Sciences products", "targets": ["Qualia Life Sciences"]},
    ],
    outputs=[
        {"required_sections": ["Executive Summary", "Founding", "Products", "Leadership"]},
        {"required_sections": ["Product Catalog", "Ingredients", "Prices", "Product Count > 5"]},
    ],
    dataset_id=dataset.id,
)
```

---

### Evaluators

**Report Quality Evaluator (LLM-as-judge):**

```python
# eval/evaluators.py

from langsmith.evaluation import LangChainStringEvaluator
from langchain.agents import create_agent

REPORT_QUALITY_PROMPT = """
You are evaluating a biotech research report.

Objective: {input}
Report: {output}

Score 1-5 on each criterion:
1. Coverage — does the report cover the main entity thoroughly?
2. Accuracy — are claims sourced and plausible?
3. Structure — are all required sections present?
4. Conciseness — is the report free of filler?

Return JSON: {"coverage": N, "accuracy": N, "structure": N, "conciseness": N, "overall": N, "reasoning": "..."}
"""

report_quality_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "coverage": "Report covers the main entity thoroughly",
            "accuracy": "Claims are sourced and accurate",
            "structure": "All required sections are present",
        }
    },
)
```

**KG Extraction Evaluator:**

```python
from langsmith.schemas import Run, Example

def kg_extraction_evaluator(run: Run, example: Example) -> dict:
    """Check that extracted entity names match expected entities."""
    extracted = run.outputs.get("organizations", [])
    expected = example.outputs.get("expected_orgs", [])
    extracted_names = {e["name"].lower() for e in extracted}
    expected_names = {n.lower() for n in expected}
    hits = extracted_names & expected_names
    precision = len(hits) / len(extracted_names) if extracted_names else 0
    recall = len(hits) / len(expected_names) if expected_names else 0
    return {
        "key": "kg_extraction_f1",
        "score": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
    }
```

---

### Running Evaluations

```python
# eval/run_eval.py

from langsmith.evaluation import evaluate

results = evaluate(
    target=run_research_slice_for_eval,   # wraps run_single_mission_slice
    data="biotech-research-reports-v1",
    evaluators=[report_quality_evaluator, kg_extraction_evaluator],
    experiment_prefix="research-agent",
    metadata={"model": "gpt-5-mini", "mission": "elysium-health"},
)
```

All `create_agent()` runs emit traces to LangSmith automatically when `LANGSMITH_TRACING=true` is set. Evaluations attach to those traces so you can drill into individual tool calls, prompt injections, and memory recalls in the LangSmith UI.

---

### Env Vars

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=biotech-research-ingestion
```

---

### Files to Create

```
eval/
  __init__.py
  build_datasets.py      # create/update LangSmith datasets from completed runs
  evaluators.py          # report quality, KG extraction, memory recall evaluators
  run_eval.py            # CLI: run evaluation suite against a dataset
  rubrics.py             # scoring rubrics and criteria definitions
```

---

## Neo4j / KG Ingestion

See `KG_API.md` for the full schema and `PLAN_kg_ingestion.md` for the detailed implementation plan.

**Implementation order (start here):**

1. `kg/extraction_models.py` — `KGExtractionResult`, `ExtractedOrganization`, `ExtractedPerson`, `ExtractedProduct`
2. `kg/neo4j_writer.py` — MERGE helpers: `upsert_organization`, `upsert_person`, `upsert_product`, `upsert_relationship`
3. `kg/extractor.py` — extraction agent with `response_format=KGExtractionResult`
4. `kg/ingest_report.py` — standalone CLI, test on existing Elysium/Qualia reports immediately
5. `kg/schema_selector.py` — schema chunk selection
6. `kg/searchtext.py` + `kg/embedder.py` — searchText generation + batch embeddings
7. `kg/run_kg_ingestion.py` — orchestrator (hook into `run_mission.py`)
8. `kg/setup_indexes.py` — vector index creation script (run once)

**Add `ingest_to_kg: bool = True` to `MissionStage`** so ingestion is opt-in per stage.

---

## Entity Types — Current Focus

| Entity | Node in Neo4j | Status |
|---|---|---|
| Organization | `Organization` | Schema defined in biotech-kg; MERGE patterns ready |
| Person | `Person` | Schema defined; relationship patterns ready |
| Product | `Product` | Schema defined; ingredient sub-nodes pending |
| Technology | `TechnologyPlatform` (or new `Technology`) | Schema partial |
| Story / News | Episodic memory (short-term) → `Story` node later | Create node type |
| Case Study | `Study` (light for now) | Deeper later |
| Protocol | `Protocol` (new node) | Add to biotech-kg schema later |

---

## Cursor Rules to Add

- `rules/mission_types.mdc` — how to author Stage-Based and Iterative missions
- `rules/temporal_activities.mdc` — Temporal activity wrapping patterns
- `rules/kg_ingestion.mdc` — KG extraction agent patterns
- `rules/storage.mdc` — MongoDB run storage + S3 artifact mirror patterns
- `rules/evaluation.mdc` — LangSmith dataset + evaluator patterns for research agents

---

## Summary — Build Order

| Priority | Feature | Files |
|---|---|---|
| 1 | KG ingestion — extraction models + neo4j_writer + CLI | `kg/extraction_models.py`, `kg/neo4j_writer.py`, `kg/ingest_report.py` |
| 2 | Mission creation agent | `agents/mission_creator.py` |
| 3 | Parallel stage execution (DAG) | `workflow/run_mission.py` |
| 4 | Iterative mission type | `models/iterative_mission.py`, `workflow/run_iterative_mission.py` |
| 5 | MongoDB run storage | `storage/research_run_store.py` |
| 6 | S3 artifact mirror | `storage/s3_artifact_store.py` |
| 7 | LangSmith evaluation suite | `eval/build_datasets.py`, `eval/evaluators.py`, `eval/run_eval.py` |
| 8 | Additional tools (PubMed, ClinicalTrials, Docling, PDF, fallback search) | `tools/*.py` |
| 9 | Temporal activities wrapper | `temporal/workflows.py`, `temporal/activities.py` |
| 10 | searchText + embeddings for KG nodes | `kg/searchtext.py`, `kg/embedder.py` |
| 11 | Schema selector + full KG pipeline | `kg/schema_selector.py`, `kg/run_kg_ingestion.py` |
