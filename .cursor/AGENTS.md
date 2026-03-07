# biotech-research-ingestion — Agent Instructions

**Role:** Deep Biotech Research Agent Backend  
**Status:** `[active]` — Phase 0 (Foundation Setup)  
**Spec source of truth:** `.cursor/specs/`

---

## What This Repo Is

This is the backend server for the Deep Biotech Research Agent system. It owns:

- The **Coordinator / Planner Agent** — conversational LangGraph agent that accepts research objectives, performs initial web research, generates structured `ResearchPlan` documents, and gates execution behind human approval.
- The **Execution Compiler** — deterministic plan → DAG transformer that validates and topologically sorts a `ResearchPlan` into an `ExecutionGraph`.
- The **Run Orchestrator** — async Python orchestrator that schedules and drives execution of `ExecutionGraph` task nodes in dependency order.
- The **Research Workers** — one `create_deep_agent` instance per `TaskNode`, each with a scoped tool set, filesystem, and system prompt.
- The **FastAPI + Socket.IO server** — REST API and real-time event bus for the frontend.

It does **not** own:
- The user-facing frontend (`biotech-research-web`)
- The knowledge graph layer (`biotech-kg`)
- Infrastructure (`biotech-infra`)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | `uv` |
| Web framework | FastAPI (ASGI) |
| Real-time | `python-socketio` (Socket.IO, ASGI middleware) |
| Agent framework | LangChain + LangGraph + DeepAgents |
| Observability | LangSmith |
| Database ODM | Beanie (MongoDB) using `pymongo.AsyncMongoClient` — **NOT Motor** |
| Checkpointing (LangGraph) | `langgraph-checkpoint-postgres` (`PostgresSaver`) |
| Long-term memory (LangGraph) | `langgraph.store.postgres` (`PostgresStore`) |
| Blob storage | Amazon S3 via `boto3` |
| Cache / pub-sub | Redis via `redis.asyncio` |
| Document parsing | `docling` |
| Settings | `pydantic-settings` |
| Environment | `.env` file, validated at startup |

---

## Architecture Decisions

### ADR-001: Coordinator is `create_agent`, not `create_deep_agent`
The Coordinator is a conversational, plan-generating agent. It does not need isolated filesystem access or subagent spawning. `create_agent` keeps it lightweight and debuggable. Workers use `create_deep_agent`.

### ADR-002: Plan approval uses `interrupt()` inside a tool
The `create_research_plan` tool calls `interrupt(plan_payload)` directly — not `interrupt_on`. This gives full control over the review payload shape and the resume value. The frontend receives the plan, the user approves/edits, and `Command(resume={"plan": ..., "approved": True})` resumes execution.

### ADR-003: Execution Compiler is deterministic Python, not an LLM
The plan → ExecutionGraph transformation is type-safe Pydantic + Kahn's algorithm. No LLM involvement. Cycle detection happens here before any execution starts.

### ADR-004: One `create_deep_agent` per TaskNode
Each task runs in complete isolation: separate LangGraph thread ID, separate filesystem backend, separate tool scope. This prevents context bleeding between stages.

### ADR-005: `MemorySaver` in dev, `PostgresSaver` in production
Controlled by `settings.LANGGRAPH_CHECKPOINTER = "memory" | "postgres"`.

---

## Package Structure

```
biotech-research-ingestion/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── threads.py
│   │   │   ├── plans.py
│   │   │   ├── runs.py
│   │   │   ├── artifacts.py
│   │   │   └── health.py
│   │   └── socketio/
│   │       ├── server.py          # Socket.IO app, namespace /research
│   │       ├── handlers.py        # Event handlers
│   │       └── rooms.py           # Room join/leave helpers
│   ├── agents/
│   │   ├── coordinator.py         # create_agent Coordinator
│   │   ├── worker.py              # create_deep_agent factory
│   │   └── tools/
│   │       ├── web_search.py
│   │       ├── create_plan.py     # contains interrupt() call
│   │       ├── docling_parse.py
│   │       ├── s3_upload.py
│   │       └── registry.py        # tool_name → tool fn lookup
│   ├── compiler/
│   │   ├── compiler.py            # plan → ExecutionGraph
│   │   └── dag.py                 # topological sort, cycle detection
│   ├── orchestrator/
│   │   └── orchestrator.py        # async run driver
│   ├── models/
│   │   ├── thread.py
│   │   ├── message.py
│   │   ├── plan.py                # ResearchPlan, ResearchTask, AgentConfig, etc.
│   │   ├── run.py                 # ResearchRun, TaskRun, ExecutionGraph
│   │   └── artifact.py
│   ├── services/
│   │   ├── coordinator_service.py
│   │   ├── run_service.py
│   │   └── s3_service.py
│   ├── config/
│   │   └── settings.py            # pydantic-settings, validated at startup
│   └── main.py                    # FastAPI app, lifespan, middleware
├── tests/
├── pyproject.toml
├── .env
└── .python-version
```

---

## Key Conventions

### MongoDB / Beanie
- Always use `pymongo.AsyncMongoClient` — never Motor.
- All Beanie documents are in `src/models/`.
- Call `init_beanie()` inside FastAPI `lifespan` after the `AsyncMongoClient` is created.
- Document class names match MongoDB collection names via `Settings.name`.

### FastAPI
- All routes use `async def`.
- Request/response schemas use Pydantic v2 models — separate from Beanie documents.
- All routes return a `{"data": ..., "error": null}` envelope or raise `HTTPException`.
- The FastAPI app is mounted as middleware to the Socket.IO ASGI app — not the other way around.

### Socket.IO
- Namespace: `/research`
- Room pattern: `thread:{thread_id}` and `run:{run_id}`
- All events carry a `thread_id` and optionally a `run_id`.
- Redis adapter required for multi-process deployments.

### LangGraph / LangChain
- Every agent invocation passes `config = {"configurable": {"thread_id": str(thread_id)}, "metadata": {...}}`.
- `thread_id` is always the MongoDB `Thread._id` cast to string.
- TaskRun workers use a separate `langgraph_thread_id` = `f"task-{task_run_id}"` to avoid checkpoint collisions with the conversation thread.
- Never call `.invoke()` on the Coordinator inside a Socket.IO event handler. Use `.astream_events()` with `asyncio` and emit events as they arrive.

### Environment Variables (required at startup)
```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=biotech_research
POSTGRES_URL=postgresql+psycopg://...
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=...
AWS_REGION=us-east-1
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGGRAPH_CHECKPOINTER=memory   # or: postgres
```

---

## Build Commands

```bash
# Install deps
uv sync

# Run development server
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

---

## Spec Files in This Directory

| File | Contents |
|---|---|
| `specs/01-research-findings.md` | LangChain / LangGraph / DeepAgents documentation synthesis |
| `specs/02-technical-specification.md` | Full system technical specification: models, agents, APIs, storage |
| `specs/03-execution-plan.md` | Phased implementation plan with milestones, risks, ADRs |

---

## Current Phase

**Phase 0 — Repository Foundation**

Next tasks:
1. Populate `pyproject.toml` with all dependencies via `uv add`
2. Create `src/` package structure
3. `src/config/settings.py` with `pydantic-settings`
4. FastAPI app with lifespan, health endpoint
5. Beanie init with MongoDB
6. Socket.IO ASGI middleware with Redis adapter

See `specs/03-execution-plan.md` for the full ordered task list.

---

## Cross-Repo Integration

| Repo | Integration point |
|---|---|
| `biotech-research-web` | Consumes REST API and Socket.IO `/research` namespace |
| `biotech-kg` | Future: receives post-run ingestion payloads (Phase 9) |
| `biotech-meta` | Specs originate here; this repo implements them |
| `biotech-mcp` | Future: MCP tools may be exposed to research workers |
