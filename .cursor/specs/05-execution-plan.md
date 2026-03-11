# Execution Plan — Deep Biotech Research Agent System

# This execution plan from 4 onward will change after the deep research system is implemented and tested.

**Status:** `[active]`  
**Version:** 1.0  
**Scope:** `biotech-research-ingestion` + `biotech-research-web`

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
