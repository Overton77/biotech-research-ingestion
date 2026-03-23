# Biotech Research Agent — Architecture Reference

> **Scope:** `biotech-research-ingestion/src/research/langchain_agent/`  
> **Last updated:** 2026-03-21  
> This document is the single source of truth for the current architecture of the LangChain-based biotech research agent suite and its immediate surroundings.

---

## Directory Map

```
src/research/langchain_agent/
│
├── .cursor/                        ← You are here (documentation for agents & humans)
│   ├── ARCHITECTURE.md             ← This file
│   ├── ECOSYSTEM_SKILLS.md         ← LangChain / LangGraph / DeepAgents / LangMem / LangSmith skill registry
│   ├── KG_API.md                   ← biotech-kg GraphQL API + Neo4j schema + write patterns
│   └── ROADMAP.md                  ← What to build next (Iterative, Stage-based, Temporal, Storage)
│
├── .cursor/rules/                  ← Cursor rules (always-apply context)
│   ├── optimizing_agent_run.mdc
│   └── seed_knowledge_graph/       ← Neo4j schema rules for Organization, Product, LabTest
│
├── agent/
│   ├── config.py                   ← TOOLS_MAP, paths, MissionSliceInput, ResearchPromptSpec,
│   │                                  BiotechResearchAgentState, memory helpers, file helpers
│   └── factory.py                  ← build_research_agent(), build_memory_report_agent(),
│                                      FilesystemMiddleware, browser_control subagent
│
├── models/
│   └── mission.py                  ← ResearchMission, MissionStage, hardcoded Qualia + Elysium missions
│
├── memory/
│   ├── langmem_manager.py          ← build_langmem_manager() → LangMem store-backed manager
│   ├── langmem_schemas.py          ← SemanticEntityFact, EpisodicResearchRun, ProceduralResearchPlaybook
│   └── langmem_namespaces.py       ← (namespace helpers)
│
├── storage/
│   ├── langgraph_persistence.py    ← get_persistence() → AsyncPostgresStore + AsyncPostgresSaver
│   └── async_mongo_client.py       ← AsyncMongoClient singleton (currently for episodes; entities → Neo4j)
│
├── workflow/
│   ├── run_mission.py              ← run_mission() — topological sort + sequential stage loop
│   └── run_slice.py                ← run_single_mission_slice() — memory recall → agent → memory ingest
│
├── tools_for_test/
│   ├── tavily_tools.py             ← search_web, extract_from_urls, map_website, crawl_website (LangChain @tools)
│   ├── formatters.py               ← _format_tavily_event_block, _format_file_state_block, _truncate_text
│   ├── playwright_agent_tool.py    ← playwright_mcp_specs (Playwright MCP tool specs for browser subagent)
│   ├── filesystem_middleware.py    ← (additional filesystem middleware helpers)
│   └── funcs/pubmed.py             ← PubMed fetch functions (early-stage)
│
├── utils/
│   └── artifacts/                  ← Local artifact storage: test_run/, pubmed_cli/, crawl_isolated/
│
├── agent_outputs/                  ← File sandbox root (runs/, reports/, scratch/)
│   ├── runs/                       ← Per-stage intermediate files (01_*.md, 02_*.md, …)
│   ├── reports/                    ← Final stage reports (*.md)
│   └── scratch/                    ← Ephemeral working space
│
└── run_mission.py                  ← Top-level CLI entrypoint (invokes workflow/run_mission.py)
```

---

## Data Flow: One Mission Run

```
run_mission.py (CLI)
  └─ main()
       ├─ get_persistence()            → (AsyncPostgresStore, AsyncPostgresSaver) from Postgres
       ├─ build_langmem_manager()      → LangMem manager backed by the store
       └─ run_mission_workflow(mission, store, checkpointer, memory_manager)
            └─ _topological_stage_order(mission)    ← sorts stages by dependencies
                 └─ for each stage (in order):
                      ├─ inject dependency_reports into stage.slice_input
                      └─ run_single_mission_slice(run_input, prompt_spec, ...)
                           ├─ [Pre-run]  load_memories_for_prompt()   ← semantic / episodic / procedural recall
                           ├─ [Run]      build_research_agent()        ← create_agent + middleware
                           │              └─ agent.ainvoke(user_message + agent_state)
                           │                   tools: search_web, extract_from_urls, map_website
                           │                   middleware: FilesystemMiddleware, dynamic_prompt, SubAgentMiddleware
                           │                   subagent: browser_control (Playwright)
                           ├─ [Post-run] build_memory_ingestion_prompt()
                           │              └─ build_memory_report_agent().ainvoke() → ResearchTaskMemoryReport
                           └─ [Persist]  memory_manager.ainvoke()     ← writes to LangMem store
```

---

## Core Models

### `MissionSliceInput` (agent/config.py)
One bounded agent run (a stage or sub-stage).

| Field | Type | Notes |
|---|---|---|
| `task_id` | str | Unique run ID |
| `mission_id` | str | Parent mission |
| `task_slug` | str | Slug used for file paths |
| `user_objective` | str | Full LLM-facing objective |
| `targets` | List[str] | Entity names / domains |
| `dependency_reports` | Dict[str, str] | Injected prior-stage markdown reports |
| `selected_tool_names` | List[str] | Subset of TOOLS_MAP |
| `stage_type` | Literal | `discovery`, `entity_validation`, `official_site_mapping`, `targeted_extraction`, `report_synthesis` |
| `max_step_budget` | int | Tool-call step cap |
| `report_required_sections` | List[str] | Required report sections |
| `guidance_notes` | List[str] | Extra instructions |

### `MissionStage` (models/mission.py)
| Field | Type | Notes |
|---|---|---|
| `slice_input` | MissionSliceInput | |
| `prompt_spec` | ResearchPromptSpec | Per-stage identity, workflow, tool guidance |
| `execution_reminders` | List[str] | Appended to dynamic prompt |
| `dependencies` | List[str] | `task_slug` values of prerequisite stages |

### `ResearchMission` (models/mission.py)
| Field | Type | Notes |
|---|---|---|
| `mission_id` | str | |
| `mission_name` | str | |
| `base_domain` | str | Primary research domain |
| `stages` | List[MissionStage] | Ordered or unordered; runner sorts topologically |

### `BiotechResearchAgentState` (agent/config.py)
Extends LangChain `AgentState`. Key additions:

| Field | Notes |
|---|---|
| `task_id`, `mission_id`, `task_slug` | Run identity |
| `targets`, `official_domains` | Research targets |
| `visited_urls`, `findings`, `open_questions` | Research state |
| `run_dir`, `report_path` | Sandbox paths |
| `step_count`, `max_step_budget` | Budget tracking |
| `semantic_memories`, `episodic_memories`, `procedural_memories` | Recalled memory blocks |
| `tavily_search_events`, `tavily_extract_events`, `tavily_map_events` | Tool provenance |
| `filesystem_events`, `read_file_paths`, `written_file_paths`, `edited_file_paths` | File provenance |

---

## Memory System (LangMem)

Three memory schemas stored in `("memories", "{mission_id}")` namespace:

| Schema | Purpose | Key Fields |
|---|---|---|
| `SemanticEntityFact` | Durable entity facts across runs | `entity_name`, `data` (aliases, domains, founders, products) |
| `EpisodicResearchRun` | Compact per-run outcome notes | `mission_id`, `run_label`, `data` (what worked / failed) |
| `ProceduralResearchPlaybook` | Reusable research tactics | `agent_type`, `data` (query patterns, heuristics) |

**Memory flow per slice:**
1. Pre-run: `manager.asearch()` x3 (semantic/procedural/episodic) → injected into dynamic prompt
2. Post-run: `build_memory_ingestion_prompt()` → `memory_report_agent` → `ResearchTaskMemoryReport`
3. Persist: `memory_manager.ainvoke()` writes memories back to `AsyncPostgresStore`

---

## Tool Registry

All tools are LangChain `@tool` functions returning `Command` (state update + ToolMessage).

| Tool | File | What it does |
|---|---|---|
| `search_web` | `tools_for_test/tavily_tools.py` | Tavily search; updates `visited_urls`, `tavily_search_events` |
| `extract_from_urls` | `tools_for_test/tavily_tools.py` | Tavily extract; updates `visited_urls`, `tavily_extract_events` |
| `map_website` | `tools_for_test/tavily_tools.py` | Tavily map (URL discovery); updates `visited_urls`, `tavily_map_events` |
| `crawl_website` | `tools_for_test/tavily_tools.py` | Tavily crawl (content + links) |
| `playwright_mcp_specs` | `tools_for_test/playwright_agent_tool.py` | Browser subagent tool (Playwright MCP) |
| PubMed tools | `tools_for_test/funcs/pubmed.py` | PubMed fetch (early-stage, not yet wired) |

**TOOLS_MAP** (in `agent/config.py`) controls which tools can be selected per stage:
```python
TOOLS_MAP = {
    "search_web": search_web,
    "extract_from_urls": extract_from_urls,
    "map_website": map_website,
    # crawl_website not yet in TOOLS_MAP; add when needed
}
```

---

## Middleware Stack (per research agent)

Built in `agent/factory.py` via `create_agent()`:

1. **`FilesystemMiddleware`** (`deepagents.middleware.filesystem`)  
   Tools: `ls`, `read_file`, `write_file`, `edit_file`  
   Backend: `FilesystemBackend(root_dir=agent_outputs/, virtual_mode=True)`

2. **`dynamic_prompt` middleware** (research prompt fragment)  
   Injects mission context, memories, tool provenance into every LLM call.

3. **`SubAgentMiddleware`** (`deepagents.middleware.subagents`)  
   Exposes `task` tool → delegates to `browser_control` subagent (Playwright).

---

## Persistence Layer

| Concern | Technology | Location |
|---|---|---|
| LangGraph checkpoint (agent state per thread) | `AsyncPostgresSaver` (Postgres) | `storage/langgraph_persistence.py` |
| LangMem store (long-term memory) | `AsyncPostgresStore` (Postgres) | `storage/langgraph_persistence.py` |
| Episode/run storage | `AsyncMongoClient` (MongoDB) | `storage/async_mongo_client.py` (placeholder) |
| File artifacts (intermediate + reports) | Local filesystem | `agent_outputs/` |

**Env vars required:** `POSTGRES_URI`, `MONGO_URI`

---

## Existing Missions

| Mission | ID | Stages | Stage Dependencies |
|---|---|---|---|
| Qualia Life Sciences | `mission-qualia-life-sciences-001` | company-fundamentals, products-and-specs, leadership-and-advisors | products depends on fundamentals |
| Elysium Health | `mission-elysium-health-001` | company-fundamentals, products-and-specs, leadership-and-advisors | leadership depends on fundamentals |

---

## Related Repositories

| Repo | Role | Key Path |
|---|---|---|
| `biotech-kg` | GraphQL API over Neo4j for all KG entities | `src/server.ts` → `http://localhost:4002/graphql` |
| `biotech-research-ingestion` | This repo — research agents + ingestion pipeline | `src/research/langchain_agent/` |

See `KG_API.md` for Neo4j schema, GraphQL mutations, and direct Cypher write patterns.
