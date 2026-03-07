# Execution Plan — Deep Biotech Research Agent System

**Status:** `[active]`  
**Version:** 1.0  
**Scope:** `biotech-research-ingestion` + `biotech-research-web`

---

## Use non-blocking async langchain interfaces across the board

## Architecture Decision Records

### ADR-001: Coordinator is `create_agent`, not `create_deep_agent`

**Decision:** Use `create_agent` (LangChain) for the Coordinator.

**Rationale:** The Coordinator is a conversational agent that generates plans and gates HITL. It does not need isolated filesystem access, subagent spawning, or the deep agent tool surface. `create_agent` keeps it lightweight, predictable, and easy to debug. The execution layer (workers) is where `create_deep_agent` provides value.

**Migration path:** If the Coordinator later needs to manage complex state or spawn workers as a graph node, it can be promoted to a full LangGraph `StateGraph` without changing the FastAPI or Socket.IO infrastructure.

---

### ADR-002: Plan approval via `interrupt()` inside a tool, not `interrupt_on`

**Decision:** The `create_research_plan` tool calls `interrupt(plan_payload)` directly.

**Rationale:** `interrupt_on` is designed for approving/rejecting specific tool calls, not for multi-field document review. Plan approval is a structured review of a complex object. Using `interrupt()` directly gives full control over the payload sent to the frontend and the shape of the resume value.

---

### ADR-003: Execution Compiler is deterministic code, not an agent

**Decision:** The plan → execution graph transformation is deterministic Python, not an LLM call.

**Rationale:** This is a type-safe, schema-driven transformation. Making it an agent introduces variability, cost, and failure modes where none are needed. The agent generates the plan; the compiler validates and structures it for execution. This boundary keeps the system maintainable and testable.

---

### ADR-004: Workers are one `create_deep_agent` per TaskNode

**Decision:** Each `TaskRun` spawns its own `create_deep_agent` instance with scoped config from `AgentConfig`.

**Rationale:** Task isolation prevents context bleeding between stages. Each task can have a different model, tool set, and filesystem scope. Maps cleanly to the DeepAgents subagent pattern.

---

### ADR-005: `MemorySaver` in dev, `PostgresSaver` in production

**Decision:** Controlled by `settings.LANGGRAPH_CHECKPOINTER = "memory" | "postgres"`.

**Rationale:** `MemorySaver` is in-process and lost across restarts. `PostgresSaver` is multi-process safe and required for any multi-worker FastAPI deployment. The toggle allows fast local development without PostgreSQL setup.

---

## Risk Register

| Risk                                               | Likelihood | Impact | Mitigation                                                                                                                |
| -------------------------------------------------- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `PostgresSaver` multi-process state collision      | Medium     | High   | One checkpointer per process using same Postgres; LangGraph handles concurrency at DB level. Test with 2 uvicorn workers. |
| Context window overflow in Coordinator             | Medium     | Medium | Summarize web search results before adding to conversation. Use `create_agent` with message trimming.                     |
| Circular dependency in user-crafted plan           | Medium     | High   | Compiler validation catches this before execution. Surfaced as a clear error to the Coordinator → user.                   |
| Worker task deadlock on asyncio failure            | Low        | High   | Structured exception handling in orchestrator. Each task in `asyncio.TaskGroup`; failure in one does not deadlock others. |
| Socket.IO interrupt-resume race condition          | Medium     | High   | Store `{thread_id → pending interrupt}` in Redis with TTL. Validate `interrupt_id` on resume. Multi-process safe.         |
| S3 credential misconfiguration in dev              | High       | Low    | Fallback to `FilesystemBackend` in dev mode via `settings.WORKER_BACKEND = "filesystem"                                   | "composite"`. |
| LangSmith trace fragmentation (subagents)          | Low        | Low    | LangGraph propagates `parent_run_id` automatically. Verify in Phase 7.                                                    |
| Plan schema drift (Coordinator output vs compiler) | Medium     | Medium | Strict Pydantic validation at plan save time. Coordinator system prompt enforces fixed output schema.                     |

---

## Phased Implementation Plan

### Phase 0 — Repository Foundation

**Goal:** Both repos boot, deps resolve, env validates, health endpoints return 200.  
**Duration:** Days 1–2

#### Backend Tasks

| #   | Task                                                                                                                | Effort |
| --- | ------------------------------------------------------------------------------------------------------------------- | ------ |
| 0.1 | Populate `pyproject.toml` — add all deps via `uv add`                                                               | S      |
| 0.2 | Create `src/` package structure: `api/`, `agents/`, `models/`, `services/`, `config/`, `compiler/`, `orchestrator/` | S      |
| 0.3 | `src/config/settings.py` — `pydantic-settings`, all env vars, fail-fast validation                                  | S      |
| 0.4 | `src/main.py` — FastAPI app with `lifespan`, `AsyncMongoClient`, `init_beanie`                                      | S      |
| 0.5 | `src/api/routes/health.py` — health check with MongoDB/Redis/Postgres/S3 checks                                     | S      |
| 0.6 | `src/api/socketio/server.py` — `python-socketio` ASGI app, Redis adapter, `/research` namespace                     | S      |
| 0.7 | Mount Socket.IO app as ASGI middleware wrapping FastAPI                                                             | S      |
| 0.8 | Stub Beanie models: `Thread`, `Message` (minimal schema)                                                            | S      |
| 0.9 | Verify: `uv run uvicorn src.main:app` starts, `/api/v1/health` returns 200                                          | S      |

#### Frontend Tasks

| #    | Task                                                               | Effort |
| ---- | ------------------------------------------------------------------ | ------ |
| 0.10 | Install Shadcn UI, `socket.io-client`, `zustand` via pnpm          | S      |
| 0.11 | `lib/socket.ts` — Socket.IO singleton, `/research` namespace       | S      |
| 0.12 | `providers/SocketProvider.tsx` — React context for socket instance | S      |
| 0.13 | Basic layout with sidebar shell                                    | S      |
| 0.14 | Verify: `pnpm dev` starts without error                            | S      |

**Milestone:** Both apps start locally without errors. Health check passes.  
**Dependencies:** MongoDB URI, PostgreSQL URL, Redis URL in `.env`.

---

### Phase 1 — Thread & Message API

**Goal:** Frontend can create threads, send messages, retrieve history.  
**Duration:** Days 3–4

#### Backend Tasks

| #   | Task                                                           | Effort |
| --- | -------------------------------------------------------------- | ------ |
| 1.1 | Full `Thread` and `Message` Beanie models with complete schema | S      |
| 1.2 | `POST /threads`, `GET /threads`, `GET /threads/{id}` routes    | S      |
| 1.3 | `GET /threads/{id}/messages` with cursor pagination            | S      |

#### Frontend Tasks

| #   | Task                                                                       | Effort |
| --- | -------------------------------------------------------------------------- | ------ |
| 1.4 | Thread list page (`/`) — list threads, create button                       | M      |
| 1.5 | New thread flow (`/threads/new` or modal)                                  | S      |
| 1.6 | Thread page shell (`/threads/[id]`) — message list, input box, empty state | M      |

**Milestone:** User creates a thread from the UI and sees it in the list.

---

### Phase 2 — Coordinator Agent + Streaming

**Goal:** User can chat with the Coordinator and see streaming token output in real time.  
**Duration:** Days 5–9

#### Backend Tasks

| #   | Task                                                                                    | Effort |
| --- | --------------------------------------------------------------------------------------- | ------ |
| 2.1 | `uv add langchain langgraph deepagents langsmith langchain-anthropic`                   | S      |
| 2.2 | `src/agents/tools/web_search.py` — Tavily tool                                          | S      |
| 2.3 | `src/agents/coordinator.py` — `create_agent` with web search, `MemorySaver`             | M      |
| 2.4 | `src/services/coordinator_service.py` — thread → agent invocation, config builder       | M      |
| 2.5 | Socket.IO `send_message` handler → `astream_events()` → emit `coordinator_token` events | M      |
| 2.6 | Persist user message and assistant response to MongoDB after stream                     | S      |
| 2.7 | LangSmith: pass `thread_id` in config, verify traces appear                             | S      |
| 2.8 | Swap `MemorySaver` for `PostgresSaver` via `settings.LANGGRAPH_CHECKPOINTER`            | M      |

#### Frontend Tasks

| #    | Task                                                               | Effort |
| ---- | ------------------------------------------------------------------ | ------ |
| 2.9  | Render streaming token output in chat — token accumulation         | M      |
| 2.10 | Tool activity indicator (show when `coordinator_tool_start` fires) | S      |
| 2.11 | Message history on load from REST `/threads/{id}/messages`         | S      |

**Milestone:** Streaming conversation with the Coordinator works end to end. LangSmith traces grouped by thread.

**Risk:** `AsyncPostgresSaver` requires `checkpointer.setup()` on first startup — must be idempotent. Test with 2 workers.

---

### Phase 3 — Research Plan Schema + HITL

**Goal:** Coordinator generates a structured plan, pauses for approval, receives approval/rejection.  
**Duration:** Days 10–15

#### Backend Tasks

| #   | Task                                                                                                  | Effort |
| --- | ----------------------------------------------------------------------------------------------------- | ------ |
| 3.1 | Full Pydantic models: `ResearchTask`, `ResearchPlan`, `AgentConfig`, `TaskInputRef`, `TaskOutputSpec` | M      |
| 3.2 | `ResearchPlan` Beanie document — register with `init_beanie`                                          | S      |
| 3.3 | `src/agents/tools/create_plan.py` — generates plan, calls `interrupt()`                               | M      |
| 3.4 | Update Coordinator agent — add `create_research_plan` tool, update system prompt                      | S      |
| 3.5 | Socket.IO: detect `__interrupt__` in agent result, store in Redis, emit `plan_ready`                  | M      |
| 3.6 | Socket.IO: handle `plan_approved` / `plan_rejected` events, call `Command(resume=...)`                | M      |
| 3.7 | `GET /plans/{id}`, `PATCH /plans/{id}` REST routes                                                    | S      |
| 3.8 | `POST /plans/{id}/approve` — REST fallback for approval                                               | S      |

#### Frontend Tasks

| #    | Task                                                                   | Effort |
| ---- | ---------------------------------------------------------------------- | ------ |
| 3.9  | Plan review panel — renders `ResearchPlan` as stage/task hierarchy     | L      |
| 3.10 | Approve / Request Revision / Reject actions with notes field           | M      |
| 3.11 | Socket.IO subscription: show panel automatically on `plan_ready` event | S      |
| 3.12 | Plan status badge in thread view                                       | S      |

**Milestone:** User submits a research objective → receives a draft plan in the UI → approves it → plan transitions to `approved` in MongoDB.

**Risk:** Redis interrupt state management — must handle WS disconnect between interrupt and resume. Implement `interrupt_id` validation and TTL-based expiry.

---

### Phase 4 — Execution Compiler

**Goal:** Approved plan compiles to a valid `ExecutionGraph`.  
**Duration:** Days 16–18

#### Backend Tasks

| #   | Task                                                                              | Effort |
| --- | --------------------------------------------------------------------------------- | ------ |
| 4.1 | `ExecutionGraph`, `ExecutionNode`, `ExecutionEdge` Pydantic models                | M      |
| 4.2 | `src/compiler/dag.py` — Kahn's topological sort with cycle detection              | M      |
| 4.3 | `src/compiler/compiler.py` — full plan → graph compilation with validation        | M      |
| 4.4 | `CyclicDependencyError`, `MissingDependencyError`, `MissingInputError` exceptions | S      |
| 4.5 | Unit tests: valid plan, cyclic plan, missing dependency, empty plan, single task  | M      |
| 4.6 | `POST /plans/{id}/compile` — debug endpoint that returns the compiled graph       | S      |

**Milestone:** Any valid `ResearchPlan` compiles correctly. Invalid plans return structured errors.

**Risk:** Cycle detection is critical safety guardrail. Must be covered by tests before execution is built.

---

### Phase 5 — Run Orchestrator + Workers

**Goal:** Approved plans execute through multiple dependent deep agent workers with live progress streaming.  
**Duration:** Days 19–28

#### Backend Tasks

| #    | Task                                                                                | Effort |
| ---- | ----------------------------------------------------------------------------------- | ------ |
| 5.1  | `ResearchRun`, `TaskRun` Beanie models — register with `init_beanie`                | M      |
| 5.2  | `POST /runs` — validate plan approved, compile graph, create `ResearchRun`          | M      |
| 5.3  | `src/orchestrator/orchestrator.py` — async driver with `asyncio.TaskGroup`          | L      |
| 5.4  | `src/agents/worker.py` — `create_deep_agent` factory from `AgentConfig`             | M      |
| 5.5  | Worker tool registry (`src/agents/tools/registry.py`) — `tool_name → tool fn`       | M      |
| 5.6  | Worker backend: `CompositeBackend` with `/inputs/`, `/outputs/`, `/memories/`       | M      |
| 5.7  | Input injection: write prior task outputs to `/inputs/` before worker start         | M      |
| 5.8  | Output collection: read `/outputs/` after worker ends, persist to S3 + MongoDB      | M      |
| 5.9  | Socket.IO: emit `task_started`, `task_token`, `task_tool_*`, `task_complete/failed` | M      |
| 5.10 | Retry logic: `max_retries` per task from `AgentConfig`, exponential backoff         | S      |
| 5.11 | `GET /runs/{id}` — full task status response                                        | S      |

#### Frontend Tasks

| #    | Task                                                                | Effort |
| ---- | ------------------------------------------------------------------- | ------ |
| 5.12 | Run monitor page (`/runs/[id]`) — task grid with live status badges | L      |
| 5.13 | Live event log panel — scrolling Socket.IO event stream             | M      |
| 5.14 | Task detail drawer/modal — token stream, tool calls, outputs        | M      |
| 5.15 | Run status header — overall progress, elapsed time                  | S      |

**Milestone:** An approved plan with 3 tasks (2 parallel + 1 dependent) executes fully. Frontend shows live progress. Task outputs stored in MongoDB.

**Risk:** Worker LangGraph thread IDs — each `TaskRun` must use `f"task-{task_run_id}"` as its `thread_id` to avoid checkpoint collision with the conversation thread.

**Risk:** `asyncio.TaskGroup` error propagation — a failed task must not cancel the entire group. Use `try/except` inside each task coroutine; group cancellation is only for catastrophic failures.

---

### Phase 6 — Artifact Persistence

**Goal:** Completed runs have browseable, downloadable artifacts.  
**Duration:** Days 29–32

#### Backend Tasks

| #   | Task                                                                              | Effort |
| --- | --------------------------------------------------------------------------------- | ------ |
| 6.1 | `Artifact` Beanie model, register with `init_beanie`                              | S      |
| 6.2 | `src/services/s3_service.py` — upload, get_presigned_url, key conventions         | M      |
| 6.3 | `s3_upload` worker tool — uploads file, returns `s3_key`                          | M      |
| 6.4 | `GET /runs/{id}/artifacts`, `GET /artifacts/{id}`, `GET /artifacts/{id}/download` | S      |

#### Frontend Tasks

| #   | Task                                                               | Effort |
| --- | ------------------------------------------------------------------ | ------ |
| 6.5 | Artifact panel in run monitor — list with type badges              | M      |
| 6.6 | Artifact viewer page (`/artifacts/[id]`) — markdown/text rendering | M      |
| 6.7 | Download button → presigned URL                                    | S      |

**Milestone:** Completed run has artifacts viewable and downloadable from the frontend.

---

### Phase 7 — Observability Hardening

**Goal:** Full run inspectable in LangSmith. Structured logs. Health checks pass.  
**Duration:** Days 33–35

#### Backend Tasks

| #   | Task                                                                          | Effort |
| --- | ----------------------------------------------------------------------------- | ------ |
| 7.1 | Verify LangSmith thread grouping for Coordinator runs — check UI              | S      |
| 7.2 | Pass `metadata={"run_id": ..., "task_id": ...}` to all worker agent configs   | S      |
| 7.3 | Structured logging throughout: `thread_id`, `run_id`, `task_id` on every line | M      |
| 7.4 | `RunEvent` collection in MongoDB — persist all Socket.IO events               | M      |
| 7.5 | Health endpoint: test all 4 dependency checks (MongoDB, Postgres, Redis, S3)  | S      |

**Milestone:** Full research run inspectable in LangSmith by thread ID. Logs structured and queryable.

---

### Phase 8 — Long-term Memory

**Goal:** Workers persist cross-run knowledge. Coordinator improves over time.  
**Duration:** Days 36–39

#### Backend Tasks

| #   | Task                                                                       | Effort |
| --- | -------------------------------------------------------------------------- | ------ |
| 8.1 | `PostgresStore` — configure, call `store.setup()` at startup               | M      |
| 8.2 | Add `/memories/` route to worker `CompositeBackend` → `StoreBackend`       | S      |
| 8.3 | Update Coordinator system prompt — save domain insights to `/memories/`    | M      |
| 8.4 | Worker agents — read task-specific memory from `/memories/` when available | M      |

**Milestone:** Workers read and write persistent memory across runs. Coordinator builds a knowledge base.

---

### Phase 9 — KG Ingestion Pipeline (Post Phase 8)

**Goal:** Completed research outputs feed into `biotech-kg`.  
**Duration:** TBD (separate planning cycle)

| #   | Task                                                                    | Effort |
| --- | ----------------------------------------------------------------------- | ------ |
| 9.1 | `docling` document parsing and chunking worker tool                     | M      |
| 9.2 | Structured extraction agent: entity/relation mapping from report chunks | L      |
| 9.3 | Neo4j GraphQL ingestion client (via `biotech-kg` API)                   | M      |
| 9.4 | Post-run pipeline trigger: enqueue on `run_complete` event              | M      |

**Prerequisite:** Phase 7 must be complete and at least one real research run validated end to end.

---

## Recommended Build Sequence (Summary)

```
Week 1 (Days 1–9):   Phase 0 → 1 → 2
                     Foundation + Thread API + Coordinator Streaming
                     ↓
                     Validates: full stack wired, streaming works,
                     LangSmith traces flow, Socket.IO event model correct

Week 2 (Days 10–18): Phase 3 → 4
                     Plan HITL + Execution Compiler
                     ↓
                     Validates: interrupt/resume works across WS sessions,
                     plan schema stable, compiler tested independently

Weeks 3–4 (19–35):  Phase 5 → 6 → 7
                     Workers + Artifacts + Observability
                     ↓
                     Validates: end-to-end execution, real artifacts,
                     full LangSmith trace lineage

Post Week 4:         Phase 8 → 9
                     Long-term memory + KG ingestion
```

**Frontend runs in parallel with the backend at every phase.** It does not need to be feature-complete — it exposes whatever the backend can do at each phase boundary.

---

## Effort Legend

| Size | Meaning                    |
| ---- | -------------------------- |
| S    | Small — 1–3 hours          |
| M    | Medium — half day to 1 day |
| L    | Large — 1–2 days           |
| XL   | Extra large — 3+ days      |
