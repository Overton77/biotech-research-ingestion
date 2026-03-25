# Agent Interaction Design
# How the AI agent will work within this system

> **Scope:** `src/research/langchain_agent/`  
> **Last updated:** 2026-03-21 (revised — test apparatus documented, bugs recorded)  
> This document describes exactly how the AI agent operates within this codebase:
> what surfaces it uses, what scaffolding it creates for itself, and the phased plan.

---

## 1. The Six Interaction Surfaces

### Surface 1 — Code (Read / StrReplace / Write)

**Rule:** always read a file before editing. Always read the relevant skill before implementing a feature.

#### Skill Lookup (Primary Source)

The `.agents/skills/` directory contains progressive-disclosure SKILL.md files for every part of the LangChain ecosystem used in this codebase. The registry is at `.cursor/ECOSYSTEM_SKILLS.md`. **Read the SKILL.md first, then implement.**

| I am about to… | Read this skill first |
|---|---|
| Create or modify an agent, add tools, add middleware | `langchain-fundamentals/SKILL.md` |
| Add or change middleware (dynamic prompt, request/response hooks) | `langchain-middleware/SKILL.md` |
| Work with vector stores, document loaders, RAG retrieval | `langchain-rag/SKILL.md` |
| Add or update a dependency in `pyproject.toml` | `langchain-dependencies/SKILL.md` |
| Write a `StateGraph`, add nodes/edges, use `Command` or `Send` | `langgraph-fundamentals/SKILL.md` |
| Work with checkpointers, `AsyncPostgresStore`, cross-thread memory | `langgraph-persistence/SKILL.md` |
| Add an interrupt / human-approval flow | `langgraph-human-in-the-loop/SKILL.md` |
| Use `create_deep_agent()`, harness architecture | `deep-agents-core/SKILL.md` |
| Add `SubAgentMiddleware`, `TodoListMiddleware`, or `HumanInTheLoopMiddleware` | `deep-agents-orchestration/SKILL.md` |
| Add or configure `FilesystemMiddleware`, backend types | `deep-agents-memory/SKILL.md` |
| Add `@traceable`, configure LangSmith tracing, query traces | `langsmith-trace/SKILL.md` |
| Create or update a LangSmith dataset | `langsmith-dataset/SKILL.md` |
| Write an evaluator, scoring function, or automated quality check | `langsmith-evaluator/SKILL.md` |
| Decide which framework to use for a new agent | `framework-selection/SKILL.md` |
| Use any Tavily operation (search, extract, map, crawl) | `tavily-best-practices/SKILL.md` + operation-specific skill |
| Use `search_web` specifically | `tavily-search/SKILL.md` |
| Use `extract_from_urls` specifically | `tavily-extract/SKILL.md` |
| Use `map_website` specifically | `tavily-map/SKILL.md` |
| Use `crawl_website` specifically | `tavily-crawl/SKILL.md` |
| Build an end-to-end research pattern | `tavily-research/SKILL.md` |

**Fallback:** if the relevant skill does not cover the specific API or edge case, use `user-docs-langchain-search_docs_by_lang_chain` to search the live docs. Always prefer skill → docs search → guessing. Never guess.

#### Code Conventions (always apply)
- `from __future__ import annotations` at the top of every new file
- `async def` everywhere — no sync blocking calls in async contexts
- `uv` for all package management; `pydantic-settings` for configuration
- `LANGSMITH_TRACING=true` — all `create_agent()` runs emit traces automatically
- All new Pydantic models that persist to MongoDB use Beanie documents in `biotech_research` — always `AsyncMongoClient`, never Motor
- Model string default: `"gpt-5-mini"` for dev/testing; `"gpt-5"` for quality runs

### Surface 2 — Execution (Shell → `uv run`)

All commands run from: `biotech-research-ingestion/`

#### Mission runs (primary test surface)
```bash
# Named mission (hardcoded in models/mission.py)
uv run python -m src.research.langchain_agent.run_mission --mission qualia
uv run python -m src.research.langchain_agent.run_mission --mission elysium

# JSON mission file — the preferred way to test new research missions
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/thorne_research.json \
  --output-dir   src/research/langchain_agent/test_runs/run_outputs/thorne_research

# Run only one stage from a mission file (useful for debugging a single stage)
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/thorne_research.json \
  --stage thorne-company-fundamentals
```

#### KG ingestion
```bash
# Dry run first, then live
uv run python -m src.research.langchain_agent.kg.ingest_report \
  --report reports/elysium-products-and-specs.md \
  --targets "Elysium Health" \
  --dry_run

uv run python -m src.research.langchain_agent.kg.ingest_report \
  --report reports/elysium-products-and-specs.md \
  --targets "Elysium Health"

# Batch ingest all reports
uv run python -m src.research.langchain_agent.scripts.ingest_all_reports
```

#### Inspection scripts (planned — not yet created)
```bash
uv run python -m src.research.langchain_agent.scripts.check_neo4j
uv run python -m src.research.langchain_agent.scripts.run_single_stage \
  --stage qualia-company-fundamentals
```

#### LangSmith evaluation (planned — not yet created)
```bash
uv run python -m src.research.langchain_agent.eval.build_datasets
uv run python -m src.research.langchain_agent.eval.run_eval \
  --dataset biotech-research-reports-v1 \
  --experiment baseline
```

#### Tests (planned — not yet created)
```bash
uv run pytest src/research/langchain_agent/tests/ -v
```

### Surface 3 — Output Retrieval (five channels)

| Channel | How | What I retrieve |
|---|---|---|
| **Agent files** | `Read` tool on `agent_outputs/reports/*.md` and `agent_outputs/runs/*/` | Final reports, intermediate reasoning files |
| **Neo4j (MCP)** | `user-neo4j-dev-dev-read_neo4j_cypher` | Node/relationship counts, entity data, vector index status |
| **MongoDB (MCP)** | `user-MongoDB-find` / `user-MongoDB-aggregate` | Research run documents, Beanie episode records |
| **LangSmith** | Browser → `smith.langchain.com` or `eval/run_eval.py` stdout | Trace trees, evaluation scores, experiment comparisons |
| **Terminal** | Shell stdout + `.cursor/terminals/*.txt` | Runtime logs, errors, timing |

### Surface 3b — JSON Mission Test Apparatus (active)

This is the primary hands-on testing surface. Write a mission file → run it → inspect outputs.

#### Directory layout

```
src/research/langchain_agent/
├── test_runs/
│   ├── missions/                    ← JSON mission definitions go here
│   │   ├── elysium_mini.json        ← single-stage smoke test (search_web only)
│   │   ├── qualia_mini.json         ← single-stage smoke test (search_web only)
│   │   └── thorne_research.json     ← full 2-stage mission (first live run: 2026-03-21)
│   │
│   └── run_outputs/                 ← one subdirectory per mission run
│       └── thorne_research/
│           ├── mission_summary.json              ← written at end (status, all stage records)
│           ├── stage_01_thorne-company-fundamentals.json      ← per-stage JSON record
│           ├── stage_01_thorne-company-fundamentals_report.md ← final report text
│           ├── stage_02_thorne-products-research.json
│           └── stage_02_thorne-products-research_report.md
│
└── agent_outputs/                   ← agent's own sandbox (written during run)
    ├── reports/                     ← final .md reports (relative path: reports/<slug>.md)
    └── runs/                        ← intermediate checkpoint files per stage
        ├── thorne-company-fundamentals/
        │   ├── 01_discovery_notes.md
        │   └── 02_leadership_notes.md
        └── thorne-products-research/
            ├── 01_url_map.md
            ├── 02_flagship_products.md
            ├── 03_sports_products.md
            ├── 04_foundational_products.md
            └── 05_specialty_products.md
```

**Two output trees exist simultaneously:**
- `agent_outputs/` — the agent's virtual filesystem sandbox (created during the run; paths are relative to `ROOT_FILESYSTEM`)
- `test_runs/run_outputs/<name>/` — structured JSON + markdown copies written by `run_mission.py` after each stage completes (safe to inspect without knowing sandbox internals)

#### Mission JSON schema

Every field below is what `load_mission_from_file()` in `run_mission.py` expects. All fields at the top level are required; most `prompt_spec` fields have safe defaults.

```jsonc
{
  "mission_id":   "mission-<slug>-001",          // unique, kebab-case
  "mission_name": "Human-readable name",
  "base_domain":  "https://www.example.com",

  "stages": [
    {
      // ── slice_input ── fed directly into MissionSliceInput ──────────────
      "slice_input": {
        "task_id":    "unique-task-id-001",       // unique per stage per mission
        "mission_id": "mission-<slug>-001",       // must match top-level mission_id
        "task_slug":  "kebab-task-slug",          // used in file paths + dependency refs

        "user_objective": "Full natural-language objective for the agent. Be specific: what to research, what tools to use, where to save files, what the final report must contain.",

        "targets": ["Primary Entity", "domain.com"],  // entity names the agent is researching

        // tools the agent may call — must be keys in TOOLS_MAP (agent/config.py):
        //   "search_web", "extract_from_urls", "map_website"
        //   browser_control is available via SubAgentMiddleware (not listed here)
        "selected_tool_names": ["search_web", "extract_from_urls", "map_website"],

        // section headers the final report must include
        "report_required_sections": [
          "Executive Summary",
          "Key Findings",
          "Sources",
          "Open Questions and Next Steps"
        ],

        // stage_type controls how the agent frames its work:
        //   "discovery"           — open-ended first pass
        //   "entity_validation"   — confirm entity, official domains, leadership
        //   "official_site_mapping" — map the official site structure
        //   "targeted_extraction" — pull specific structured data (products, prices, specs)
        //   "report_synthesis"    — aggregate prior outputs into a final report
        "stage_type": "entity_validation",

        "max_step_budget": 12    // hard step cap enforced by agent prompt context
      },

      // ── prompt_spec ── shapes the system prompt at each LLM call ────────
      "prompt_spec": {
        "agent_identity": "One sentence describing the agent's persona/expertise.",

        "domain_scope": [          // bullet-point context injected into the system prompt
          "Key fact 1 about the entity",
          "Key fact 2"
        ],

        "workflow": [              // numbered steps the agent should follow in order
          "1. search_web: 'query here'",
          "2. Save notes to runs/<slug>/01_notes.md",
          "3. Write final report to reports/<slug>.md"
        ],

        "tool_guidance": [         // specific tool usage hints
          "search_web: use max_results=8",
          "extract_from_urls: batch 3–5 URLs per call"
        ],

        "subagent_guidance": [     // when / how to use browser_control subagent
          "Use browser_control if extract_from_urls returns empty content (JS-rendered pages)"
        ],

        "practical_limits": [      // constraints repeated in the prompt
          "Maximum 12 steps total",
          "Report length: 1000–1500 words"
        ],

        "filesystem_rules": [      // file path rules — always use relative paths
          "Use runs/<slug>/ for all intermediate files",
          "Write final report to reports/<slug>.md",
          "Relative paths only — no absolute paths"
        ],

        "intermediate_files": [    // files the agent is expected to save before the report
          "runs/<slug>/01_notes.md",
          "runs/<slug>/02_more_notes.md"
        ]
      },

      "execution_reminders": [     // bullet-points injected at end of every prompt turn
        "Key reminder 1",
        "Write final report to reports/<slug>.md before stopping"
      ],

      // task_slugs of stages this stage depends on.
      // The runner injects those stages' final report text into dependency_reports.
      // Empty list = runs first (no dependencies).
      "dependencies": []
    }
  ]
}
```

#### Dependency wiring

When `"dependencies": ["stage-a-slug"]` is set on a stage, `run_mission.py` automatically:
1. Runs `stage-a-slug` first (topological ordering)
2. Reads that stage's final report text from the agent sandbox
3. Injects it into `MissionSliceInput.dependency_reports` as `{"stage-a-slug": "<report text>"}`
4. The dependent stage's user message then starts with the prior report for context

#### How to inspect outputs after a run

```bash
# 1. High-level summary (status, per-stage slugs, report previews)
cat src/research/langchain_agent/test_runs/run_outputs/<name>/mission_summary.json

# 2. Per-stage structured record (StageRunRecord with S3 URIs, memory report)
cat src/research/langchain_agent/test_runs/run_outputs/<name>/stage_01_<slug>.json

# 3. Final report markdown for each stage
cat src/research/langchain_agent/test_runs/run_outputs/<name>/stage_01_<slug>_report.md

# 4. Agent's intermediate checkpoint files
ls src/research/langchain_agent/agent_outputs/runs/<slug>/
cat src/research/langchain_agent/agent_outputs/runs/<slug>/01_discovery_notes.md

# 5. Agent's final reports (same content as run_outputs reports)
ls src/research/langchain_agent/agent_outputs/reports/
```

#### Known bugs fixed in the test apparatus (2026-03-21)

| Bug | Root cause | Fix location | Fix |
|---|---|---|---|
| `NotImplementedError: awrap_tool_call not available` | `monitor_filesystem_tools` used `@wrap_tool_call` on a sync `def`, producing only a sync middleware class. The agent runs async (`ainvoke`). | `tools_for_test/filesystem_middleware.py` | Changed to `async def` + `await handler(request)`. `@wrap_tool_call` detects coroutines and auto-creates `awrap_tool_call`. |
| `BadRequestError 400: tool_calls not followed by tool messages` | `thread_id` in `run_slice.py` was always `task_id` (e.g. `"thorne-company-fundamentals-001"`). On re-run the checkpointer resumed the previous broken conversation that had a dangling unanswered tool call. | `workflow/run_slice.py` | Appended a UUID suffix per invocation: `run_thread_id = f"{task_id}-{uuid4().hex[:8]}"`. Each run now starts a fresh checkpoint thread. Memory store (cross-thread) is unaffected. |
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Windows terminal defaults to cp1252. Agent memory reports contain Unicode (e.g. `≈`, `‑`). The crash happened at a `print()` call after Stage 1 completed. | `run_mission.py` `__main__` block | Added `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` + same for stderr, guarded by `sys.platform == "win32"`. |

### Surface 4 — Neo4j

**In production code:** always use `Neo4jAuraClient` (in `neo4j_aura.py`). The client connects to Aura directly via env vars `NEO4J_URI`, `NEO4J_AURA_USERNAME`, `NEO4J_AURA_PASSWORD`.

**For inspection and validation (agent-side only):** use the `user-neo4j-dev-dev-read_neo4j_cypher` / `write_neo4j_cypher` MCP tools. This is the same Aura instance. Write MCP is used only when absolutely necessary (e.g., one-off cleanup of test data, schema setup).

**Rule:** never bypass the `Neo4jAuraClient` for bulk writes. MCP is for observation and emergency fixes.

### Surface 5 — MongoDB

**In production code:** `AsyncMongoClient` with Beanie ODM. Database: `biotech_research`. Collections are either existing or newly created — the database is clean.

**For inspection:** `user-MongoDB-find` / `user-MongoDB-aggregate` MCP tools. Same instance.

**Key collections to create:**
- `research_runs` — one document per completed `run_single_mission_slice` output
- `research_missions` — one document per started `ResearchMission`
- `task_runs` — Beanie document per `TaskNode` run (aligns with AGENTS.md `TaskRun` model)

### Surface 6 — Cursor Subagents (`.cursor/agents/`)

Scoped AGENTS.md-style task definitions I can launch for focused parallel work:

```
.cursor/agents/
├── kg-validator.md         ← Query Neo4j after ingestion; compare counts vs. expected
├── eval-runner.md          ← Build LangSmith dataset + run evaluation suite
├── report-reader.md        ← Read agent_outputs/reports/ and summarize findings
└── mission-tester.md       ← Run a single stage and retrieve intermediate + final outputs
```

---

## 2. Scaffolding — Exists vs. Planned

### Already exists (active)

```
src/research/langchain_agent/
│
├── run_mission.py                  ← PRIMARY CLI entry point (fully working)
│   # --mission <name>             named hardcoded mission
│   # --mission-file <path>        JSON mission file (preferred for testing)
│   # --output-dir <dir>           write per-stage JSON + report copies here
│   # --stage <slug>               run only one stage
│
├── workflow/
│   ├── run_mission.py             ← DAG topological sort + stage orchestration
│   └── run_slice.py               ← single stage: memory recall, agent run, memory ingestion,
│                                     S3 upload, StageRunRecord assembly
│
├── agent/
│   ├── config.py                  ← MissionSliceInput, ResearchPromptSpec, TOOLS_MAP,
│   │                                ROOT_FILESYSTEM, state schema, memory helpers
│   └── factory.py                 ← build_research_agent(), build_memory_report_agent(),
│                                     filesystem_middleware, browser_control subagent
│
├── models/
│   └── mission.py                 ← ResearchMission, MissionStage, named missions
│                                     (QUALIA_RESEARCH_MISSION, ELYSIUM_RESEARCH_MISSION)
│
├── storage/
│   ├── models.py                  ← Beanie documents: MissionRunDocument, StageRunRecord,
│   │                                StageArtifacts, MemoryReportRecord
│   ├── async_mongo_client.py      ← AsyncMongoClient wrapper
│   ├── langgraph_persistence.py   ← get_persistence() → (store, checkpointer)
│   └── s3_store.py                ← persist_slice_artifacts() → S3 uploads
│
├── tools_for_test/
│   ├── tavily_tools.py            ← search_web, extract_from_urls, map_website, crawl_website
│   ├── filesystem_middleware.py   ← monitor_filesystem_tools (async @wrap_tool_call)
│   ├── playwright_agent_tool.py   ← playwright_mcp_specs for browser_control subagent
│   └── formatters.py              ← _format_tavily_event_block, _format_file_state_block
│
├── memory/
│   └── langmem_manager.py         ← build_langmem_manager() → cross-thread memory store
│
└── test_runs/                      ← TESTING APPARATUS (active)
    ├── missions/                   ← JSON mission definitions
    │   ├── elysium_mini.json       (smoke test — search_web only, 4 steps)
    │   ├── qualia_mini.json        (smoke test — search_web only, 4 steps)
    │   └── thorne_research.json    (full 2-stage mission, completed 2026-03-21)
    └── run_outputs/                ← per-run output directories
        └── thorne_research/        (first completed full run: ~9.8 min, exit 0)
```

### Planned (not yet created)

```
src/research/langchain_agent/
│
├── eval/                           ← LangSmith evaluation suite
│   ├── __init__.py
│   ├── build_datasets.py           # Build datasets from completed reports in agent_outputs/reports/
│   ├── evaluators.py               # LLM-as-judge + KG extraction F1 + memory recall
│   ├── run_eval.py                 # CLI: run evaluation suite against a dataset
│   └── rubrics.py                  # Scoring criteria + threshold constants
│
├── tests/                          ← pytest unit + integration tests
│   ├── conftest.py                 # Fixtures: mission stubs, MemorySaver, neo4j dry-run
│   ├── test_models.py              # Pydantic validation: MissionSliceInput, ResearchMission
│   ├── test_kg_extraction.py       # Extraction agent: dry-run against existing reports
│   ├── test_kg_writer.py           # Neo4j writer: integration test against live Aura
│   ├── test_mission_flow.py        # run_single_mission_slice smoke test (MemorySaver)
│   └── test_ingest_pipeline.py     # Full kg/ingest_report.py --dry_run end-to-end
│
└── scripts/                        ← CLI helpers
    ├── check_neo4j.py              # Node/rel counts + spot-check latest ingested entities
    ├── run_single_stage.py         # Run one MissionStage in isolation
    ├── ingest_all_reports.py       # Batch ingest all reports in agent_outputs/reports/
    └── list_langsmith_runs.py      # Print recent LangSmith runs
```

---

## 3. Models in Use

### LLM Model Strategy

| Phase | Model | Purpose |
|---|---|---|
| **Testing / dev** | `gpt-5-mini` | Cheap, fast — for unit tests, dry-run extraction, eval runs |
| **Full missions** | `gpt-5` | Quality runs — research agent, memory report agent, mission creator |
| **Future (post-stabilization)** | `gpt-5.2-mini`, `gpt-5.2`, `gpt-5.4-mini`, `gpt-5.4` | Swap in once pipeline is robust |

Apply `gpt-5-mini` everywhere by default. Promote to `gpt-5` only for full mission runs or when report quality scores are being evaluated. This is controlled by a `MODEL_TIER` env var or per-function parameter.

---

## 4. Mission Types

Two first-class mission types live side-by-side:

### Type A — Stage-Based (DAG execution)
```
ResearchMission
  stages: List[MissionStage]   ← each has dependencies: List[task_slug]
  execution: DAG topological sort → wave-based parallel execution
```
- Independent stages run in parallel via `asyncio.gather`
- Dependent stages receive prior stage reports in `dependency_reports`
- Files: `workflow/run_mission.py` (parallel rewrite), `models/mission.py`

### Type B — Iterative (fresh-context cycles)
```
IterativeResearchMission
  stage_template: MissionStage    ← cloned per cycle
  iterative_config:
    max_cycles: int               ← hard cap (user-controlled)
    completion_criteria: str      ← passed to evaluator agent
    seed_from_prior_outputs: bool ← inject prior cycle summaries into next
```
- Each cycle runs with fresh LangGraph context
- Prior cycle reports are injected as `guidance_notes` (last 2 cycles max)
- A lightweight evaluator (LLM-as-judge or heuristic) decides early termination
- Files: `models/iterative_mission.py`, `workflow/run_iterative_mission.py`

Both types produce the same output shape (`List[Dict]` from `run_single_mission_slice`) and feed into the same KG ingestion and MongoDB storage pipeline.

---

## 5. Temporal Integration Plan

Temporal wraps the execution layer — not the research agents themselves.

```
FastAPI server (later phase)
  └─ POST /runs/{mission_id}/start
       └─ temporal_client.start_workflow(ResearchWorkflow, mission_id)
            └─ ResearchWorkflow (deterministic, no I/O)
                 ├─ execute_stage_activity(stage)      ← Activity wraps run_single_mission_slice
                 ├─ execute_parallel_wave(stages)      ← asyncio.gather of Activities
                 └─ evaluate_and_continue_activity()  ← iterative completion check

# All LLM calls, tool calls, Neo4j writes happen inside Activities (retryable, observable)
# The Workflow itself is pure Python + Temporal SDK — no I/O
```

**Integration order:**
1. Build and stabilize the research pipeline (Phases 1–4 below)
2. Wrap `run_single_mission_slice` in a Temporal Activity
3. Build `ResearchWorkflow` for stage-based missions
4. Extend to `IterativeResearchWorkflow`
5. Wire into FastAPI via `temporalio.client.Client`
6. Mount Temporal worker as a separate process launched alongside FastAPI

**Files to create (Phase 6+):** `temporal/workflows.py`, `temporal/activities.py`, `temporal/worker.py`

---

## 6. GraphQL API Integration (biotech-kg — deferred)

The GraphQL server at `http://localhost:4002/graphql` is read + write. Integration is deferred until the Cypher-based ingestion pipeline is stable.

**When ready:**
- Use `ariadne-codegen` to generate a typed Python client from the biotech-kg schema
- The generated client lives in `kg/graphql_client/` (auto-generated, not hand-edited)
- The Neo4j GraphQL API transforms schema types into tools in a structured way — this becomes a unique tool surface for research agents (agents query the KG via GraphQL tools, not raw Cypher)
- Write path: GraphQL mutations for fields/relationships not covered by the Cypher writer
- Read path: Replace some direct Cypher reads with GraphQL queries for app-layer consistency

**Deferred to:** Phase 7+ (after Temporal integration is underway)

---

## 7. My Optimization Loop

```
WRITE code change
     │
     ▼
RUN via Shell (uv run python -m ...)
     │
     ▼
RETRIEVE outputs:
  ├── Read agent_outputs/reports/*.md (final report quality)
  ├── Query Neo4j MCP (node counts, entity spot-check)
  ├── Query MongoDB MCP (run document stored correctly)
  └── Check LangSmith (Browser or eval/run_eval.py stdout)
     │
     ├── LangSmith eval score < threshold
     │     └─ identify lowest-scoring dimension → fix that code path → repeat
     │
     ├── Neo4j counts wrong / missing nodes
     │     └─ query MCP → fix extractor or writer → --dry_run → live run → verify
     │
     ├── MongoDB document malformed
     │     └─ fix Beanie model or storage code → rerun → query MCP → verify
     │
     └── pytest fails
           └─ fix → pytest again → only proceed when green
```

---

## 8. Phase Execution Plan

### Phase 1 — Component Validation (no full LLM runs, cheap)

| Task | How | Success signal |
|---|---|---|
| Verify imports are correct | `uv run python -c "from src.research.langchain_agent.models.mission import ResearchMission; print('OK')"` | Prints OK |
| KG dry-run: Elysium products | `ingest_report --dry_run` | Prints extraction counts |
| KG dry-run: all 8 reports | `scripts/ingest_all_reports.py --dry_run` | No errors, counts plausible |
| Unit tests: models | `pytest tests/test_models.py` | All green |
| Neo4j state baseline | `scripts/check_neo4j.py` | Prints current counts |

---

### Phase 2 — KG Ingestion: Live Writes

Run `ingest_report` (no `--dry_run`) on all 8 existing reports. Query Neo4j via MCP before and after. Fix any extractor/writer issues.

**Reports to ingest (in order — fundamentals before products):**
1. `elysium-company-fundamentals.md` → targets: `Elysium Health`
2. `elysium-products-and-specs.md` → targets: `Elysium Health`
3. `qualia-company-fundamentals.md` → targets: `Qualia Life Sciences`
4. `qualia-life-sciences-product-catalog.md` → targets: `Qualia Life Sciences`
5. `qualia-products-and-specs.md` → targets: `Qualia Life Sciences`
6. `dave-asprey-research.md` → targets: `Dave Asprey`
7. `dave-asprey-upgradelabs-dangercoffee.md` → targets: `Dave Asprey`, `Upgrade Labs`, `Danger Coffee`
8. `dave-asprey-upgrade-labs-danger-coffee.md` → targets: `Dave Asprey`, `Upgrade Labs`, `Danger Coffee`

**After each batch:** query Neo4j MCP → verify node/rel counts → spot-check a few entities.

---

### Phase 3 — LangSmith Evaluation Setup

```bash
uv run python -m src.research.langchain_agent.eval.build_datasets
uv run python -m src.research.langchain_agent.eval.run_eval \
  --dataset biotech-research-reports-v1 --experiment baseline
```

Open Browser → `smith.langchain.com` → inspect experiment. Record baseline scores (coverage, accuracy, structure, conciseness). These become the benchmark for all future changes.

---

### Phase 4 — JSON Mission Testing (active as of 2026-03-21)

The primary test loop is now: **write a JSON mission file → run with `--output-dir` → inspect outputs**.

**Completed run: `thorne_research.json`**
- 2 stages: `entity_validation` (company fundamentals) + `targeted_extraction` (product catalog)
- Stage 1 used: `search_web`, `extract_from_urls` — ~2.5 min
- Stage 2 used: `search_web`, `extract_from_urls`, `map_website`, + `browser_control` subagent for JS-rendered pages — ~7 min
- Both stages completed, reports saved locally + to S3, memory updates persisted to cross-thread store
- 20 products documented with prices, ingredients, certifications

**To run a new JSON mission test:**
```bash
# 1. Write the mission JSON to test_runs/missions/<name>.json (follow schema in Surface 3b)
# 2. Run:
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/<name>.json \
  --output-dir   src/research/langchain_agent/test_runs/run_outputs/<name>
# 3. Inspect:
#    test_runs/run_outputs/<name>/mission_summary.json
#    test_runs/run_outputs/<name>/stage_NN_<slug>_report.md
#    agent_outputs/runs/<slug>/  (intermediate checkpoint files)
#    agent_outputs/reports/<slug>.md
```

**To run a single stage only (debug):**
```bash
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/<name>.json \
  --stage <task_slug>
```

---

### Phase 5 — ROADMAP Implementation (following build order)

| # | Task | Files | Approval gate |
|---|---|---|---|
| 5.1 | Wire KG ingestion into `run_slice.py` post-run | `workflow/run_slice.py` | No |
| 5.2 | MongoDB run storage (Beanie `ResearchRun` doc) | `storage/research_run_store.py`, `models/research_run.py` | No |
| 5.3 | Mission creation agent | `agents/mission_creator.py` | Yes — prompt review |
| 5.4 | Parallel DAG execution (wave-based) | `workflow/run_mission.py` | Yes — concurrency model |
| 5.5 | Iterative mission type | `models/iterative_mission.py`, `workflow/run_iterative_mission.py` | No |
| 5.6 | S3 artifact mirror (post-stage) | `storage/s3_artifact_store.py` | No |
| 5.7 | Additional tools (PubMed, ClinicalTrials, Docling, fallback search) | `tools/*.py` | No |

### Phase 6 — Temporal Integration

| # | Task | Files |
|---|---|---|
| 6.1 | Activity: wrap `run_single_mission_slice` | `temporal/activities.py` |
| 6.2 | Workflow: stage-based DAG | `temporal/workflows.py` |
| 6.3 | Workflow: iterative cycles | `temporal/workflows.py` |
| 6.4 | Worker process | `temporal/worker.py` |
| 6.5 | FastAPI integration | `src/api/routes/runs.py` + lifespan |

### Phase 7 — GraphQL Integration (ariadne-codegen)

| # | Task | Files |
|---|---|---|
| 7.1 | Generate typed client from biotech-kg schema | `kg/graphql_client/` (auto-generated) |
| 7.2 | GraphQL read validation — compare with Cypher results | `kg/graphql_validator.py` |
| 7.3 | GraphQL tool wrappers for research agents | `tools/kg_query_tool.py` |
| 7.4 | GraphQL write path for missing fields | extend `kg/neo4j_writer.py` or `kg/graphql_writer.py` |

---

## 9. Cursor Rules to Add (rolling — add after each phase completes)

| Rule file | Trigger | Content |
|---|---|---|
| `rules/evaluation.mdc` | After Phase 3 | LangSmith dataset + evaluator patterns |
| `rules/kg_ingestion.mdc` | After Phase 2 | KG extraction agent patterns, dry-run discipline |
| `rules/mission_types.mdc` | After Phase 5.4–5.5 | Stage-based vs. iterative mission authoring |
| `rules/temporal_activities.mdc` | After Phase 6.1 | Activity wrapping patterns, heartbeat, retry policy |
| `rules/storage.mdc` | After Phase 5.2 | MongoDB Beanie model patterns, collection conventions |
| `rules/self_testing.mdc` | After Phase 1 | How agent validates its own outputs (MCP queries, dry-run discipline) |

---

## 10. Working Agreements

- Always use `--dry_run` on `ingest_report.py` before any live Neo4j write when testing
- Always ask for approval before changes that affect: agent memory schema, mission Pydantic models (breaking), Neo4j node/relationship schema, Temporal workflow signatures
- Present LangSmith scores **before** and **after** every optimization pass
- Surface Neo4j state (MCP query) after every live ingestion run
- `gpt-5-mini` everywhere by default; `gpt-5` only for full quality runs and benchmark evaluations
- MongoDB writes always go through Beanie documents in `biotech_research` — never raw dict inserts
- S3 is accessed via AWS CLI in the shell for one-off inspection; in code via `boto3` / `aioboto3`
