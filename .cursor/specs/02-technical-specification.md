# Technical Specification — Deep Biotech Research Agent System

**Status:** `[active]`  
**Version:** 1.0  
**Scope:** `biotech-research-ingestion` (backend) + `biotech-research-web` (frontend)

---

## 1. Service Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    biotech-research-ingestion                       │
│                    (Python / FastAPI / uv)                          │
│                                                                     │
│  ┌──────────────────┐   ┌──────────────────────────────────────┐   │
│  │  HTTP REST API   │   │     Socket.IO Event Server           │   │
│  │  (FastAPI)       │   │     (python-socketio / ASGI)         │   │
│  │                  │   │                                      │   │
│  │  /threads        │   │  ns: /research                       │   │
│  │  /plans          │   │  events: run_event, plan_ready,      │   │
│  │  /runs           │   │          token_stream, run_complete  │   │
│  │  /artifacts      │   │                                      │   │
│  └──────────────────┘   └──────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Coordinator Agent                           │  │
│  │              (create_agent / LangGraph)                       │  │
│  │                                                              │  │
│  │   tools: web_search, create_research_plan,                   │  │
│  │           get_context, clarify_intent                        │  │
│  │   checkpointer: PostgresSaver (prod) / MemorySaver (dev)     │  │
│  │   HITL: interrupt() in create_research_plan tool             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Execution Compiler                           │  │
│  │                  (plan → DAG — deterministic Python)          │  │
│  │                                                              │  │
│  │   Input: ApprovedResearchPlan                                │  │
│  │   Output: ExecutionGraph (topologically sorted TaskNodes)    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Research Worker Runtime                          │  │
│  │              (create_deep_agent per TaskNode)                 │  │
│  │                                                              │  │
│  │   Each TaskNode spawns one deep agent with:                  │  │
│  │   - scoped tools (web_search, docling_parse, s3_upload, ...) │  │
│  │   - scoped filesystem (CompositeBackend)                     │  │
│  │   - scoped system prompt from TaskNode config                │  │
│  │   - declared inputs/outputs                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Storage Layer:                                                     │
│  MongoDB (Beanie) · PostgreSQL (LangGraph) · S3 · Redis            │
└────────────────────────────────────────────────────────────────────┘
                        │ HTTP + Socket.IO
┌────────────────────────────────────────────────────────────────────┐
│                    biotech-research-web                             │
│                    (Next.js / TypeScript / pnpm)                    │
│                                                                     │
│   Threads · Coordinator Chat · Plan Review · Run Monitor           │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Models

### 2.1 Thread

```python
class Thread(Document):
    id: PydanticObjectId
    title: str
    created_at: datetime
    updated_at: datetime
    status: Literal["active", "archived"]
    metadata: dict                  # user_id, tags, etc.

    class Settings:
        name = "threads"
```

### 2.2 Message

```python
class Message(Document):
    thread_id: PydanticObjectId
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list             # supports multipart content
    created_at: datetime
    run_id: str | None              # LangSmith run ID
    metadata: dict

    class Settings:
        name = "messages"
```

### 2.3 AgentConfig

```python
class AgentConfig(BaseModel):
    model: str                      # e.g. "anthropic:claude-sonnet-4-6"
    system_prompt: str
    tools: list[str]                # tool names from registry
    backend_type: Literal["state", "filesystem", "composite"]
    backend_root_dir: str | None    # for filesystem backend
    interrupt_on: dict | None
    max_retries: int = 6
    timeout: int = 120
```

### 2.4 TaskInputRef and TaskOutputSpec

```python
class TaskInputRef(BaseModel):
    name: str                       # variable name inside the task
    source: Literal["task_output", "user_provided", "external"]
    source_task_id: str | None      # if source == "task_output"
    output_name: str | None         # which output of the source task
    description: str

class TaskOutputSpec(BaseModel):
    name: str
    type: Literal["text", "markdown", "json", "file", "s3_ref"]
    description: str
    required: bool = True
```

### 2.5 ResearchTask

```python
class ResearchTask(BaseModel):
    id: str                         # unique within plan, used for dependency refs
    title: str
    description: str
    stage: str                      # human-readable stage grouping
    sub_stage: str | None
    agent_config: AgentConfig
    inputs: list[TaskInputRef]
    outputs: list[TaskOutputSpec]
    dependencies: list[str]         # list of ResearchTask.id values this task depends on
    estimated_duration_minutes: int | None
```

### 2.6 ResearchPlan

```python
class ResearchPlan(Document):
    id: PydanticObjectId
    thread_id: PydanticObjectId
    title: str
    objective: str
    stages: list[str]               # ordered stage names (for display)
    tasks: list[ResearchTask]
    status: Literal[
        "draft",
        "pending_approval",
        "approved",
        "rejected",
        "executing",
        "complete",
        "failed"
    ]
    created_at: datetime
    updated_at: datetime
    approved_at: datetime | None
    approver_notes: str | None
    version: int                    # incremented on each revision

    class Settings:
        name = "research_plans"
```

### 2.7 ExecutionGraph

```python
class ExecutionEdge(BaseModel):
    from_task: str
    to_task: str
    output_name: str                # which output feeds this edge
    input_name: str                 # which input receives it

class ExecutionNode(BaseModel):
    task_id: str
    depth: int                      # topological depth (0 = no dependencies)
    can_run_parallel_with: list[str]  # task IDs at same depth level

class ExecutionGraph(BaseModel):
    nodes: list[ExecutionNode]      # topologically sorted
    edges: list[ExecutionEdge]
```

### 2.8 TaskRun

```python
class TaskRun(BaseModel):
    task_id: str                    # ref to ResearchTask.id
    status: Literal["pending", "ready", "running", "complete", "failed", "skipped"]
    started_at: datetime | None
    completed_at: datetime | None
    langgraph_thread_id: str        # per-task thread ID for LangGraph checkpointing
    langsmith_run_id: str | None
    outputs: dict                   # output_name → value or s3_key
    error: str | None
    retry_count: int = 0
```

### 2.9 ResearchRun

```python
class ResearchRun(Document):
    id: PydanticObjectId
    plan_id: PydanticObjectId
    thread_id: PydanticObjectId
    status: Literal["pending", "running", "paused", "complete", "failed", "cancelled"]
    started_at: datetime | None
    completed_at: datetime | None
    execution_graph: ExecutionGraph
    task_runs: list[TaskRun]
    artifacts: list[ArtifactRef]
    langsmith_thread_id: str        # for LangSmith trace grouping

    class Settings:
        name = "research_runs"
```

### 2.10 Artifact

```python
class Artifact(Document):
    id: PydanticObjectId
    run_id: PydanticObjectId
    task_id: str
    name: str
    type: Literal["report", "document", "dataset", "log", "intermediate"]
    storage: Literal["s3", "mongodb"]
    s3_key: str | None
    content: str | None             # small artifacts stored inline
    content_type: str               # MIME type
    created_at: datetime
    metadata: dict

    class Settings:
        name = "artifacts"
```

---

## 3. Agent Specifications

### 3.1 Coordinator Agent

**Purpose:** Conversation intake, context clarification, research plan generation, HITL gating before execution.

**Implementation:** `create_agent(model, tools, checkpointer=PostgresSaver)`

**Model:** `anthropic:claude-sonnet-4-6` (configurable)

**Tools:**

| Tool | Description |
|---|---|
| `web_search(query, max_results)` | Tavily web search for initial context gathering |
| `create_research_plan(objective, context)` | Generates draft plan, calls `interrupt()`, returns approved plan |
| `get_thread_context(thread_id)` | Retrieves prior conversation for continuity |
| `clarify_intent(question)` | Asks the user a clarifying question |
| `save_approved_plan(plan)` | Persists plan to MongoDB, sets status to `approved` |

**System prompt responsibilities:**
- Accept a biotech research objective
- Perform initial web research to understand scope
- Generate a structured `ResearchPlan` with stages and tasks
- Use `interrupt()` in `create_research_plan` to gate human approval
- After approval, hand off to the execution layer

**HITL flow:**
```python
# Inside create_research_plan tool:
approved = interrupt({
    "type": "plan_review",
    "plan": plan_draft.model_dump(),
    "message": "Please review this research plan before execution begins.",
})
```

**Checkpointing:** `PostgresSaver` in production. Thread ID = MongoDB `Thread._id` cast to string.

---

### 3.2 Execution Compiler

**Purpose:** Transform `ApprovedResearchPlan` into a validated, topologically sorted `ExecutionGraph`.

**Implementation:** Deterministic Python — no LLM involved.

**Algorithm:**
1. Build adjacency list from `ResearchTask.dependencies`
2. Topological sort using Kahn's algorithm
3. Detect cycles — raise `CompilerError` with affected task IDs
4. Assign `depth` per node (parallel groups at same depth)
5. Validate all `TaskInputRef.source_task_id` reference existing task IDs
6. Validate no missing required inputs at depth 0 (where `source != "user_provided"`)
7. Return `ExecutionGraph`

**Error types:**
- `CyclicDependencyError` — plan has circular task dependencies
- `MissingDependencyError` — `source_task_id` references a non-existent task
- `MissingInputError` — depth-0 task has a `task_output` input with no provider

---

### 3.3 Run Orchestrator

**Purpose:** Drive execution of an `ExecutionGraph` — scheduling, concurrency, output handoff, event emission.

**Implementation:** Python async orchestration code. Not an agent.

**Algorithm:**
```
1. Load ExecutionGraph, initialize all TaskRuns as "pending"
2. Find all depth=0 tasks → mark "ready" → spawn workers concurrently
3. For each completed task:
   a. Collect outputs, persist to S3 + MongoDB
   b. For each downstream task: check if all dependencies complete
   c. If unblocked: mark "ready" → spawn worker
4. On task failure:
   a. If retry_count < max_retries: re-enqueue
   b. If exhausted: mark TaskRun "failed", emit run_failed, abort
5. When all tasks complete: mark run "complete", emit run_complete
```

**Concurrency:** `asyncio.TaskGroup` for parallel tasks at the same depth. Each worker runs as a separate asyncio task.

---

### 3.4 Research Worker

**Purpose:** Execute one `ResearchTask` in complete isolation.

**Implementation:** `create_deep_agent` with task-scoped config from `AgentConfig`.

**Per-task LangGraph thread ID:** `f"task-{task_run_id}"` — separate from the conversation thread ID to prevent checkpoint collisions.

**Filesystem backend:**
```python
def make_worker_backend(rt):
    return CompositeBackend(
        default=StateBackend(rt),           # /workspace/* — ephemeral scratch
        routes={
            "/inputs/": StoreBackend(rt),   # resolved inputs from prior tasks
            "/outputs/": StoreBackend(rt),  # declared outputs to be collected
            "/memories/": StoreBackend(rt), # cross-run persistent knowledge
        }
    )
```

**Input injection:** Before starting the worker, the orchestrator writes resolved inputs from prior `TaskRun.outputs` into the worker's `/inputs/` namespace.

**Output collection:** After the worker completes, the orchestrator reads files from `/outputs/` and persists them to S3 (for large blobs) and MongoDB `Artifact` collection (for metadata).

---

## 4. Socket.IO Event Model

**Server:** `python-socketio` (ASGI middleware)  
**Namespace:** `/research`  
**Room patterns:** `thread:{thread_id}`, `run:{run_id}`  
**Redis adapter:** Required for multi-process deployments

### Server → Client Events

| Event | Payload | When |
|---|---|---|
| `coordinator_token` | `{token, run_id, thread_id}` | Streaming LLM token |
| `coordinator_tool_start` | `{tool_name, args_summary, run_id, thread_id}` | Tool begins |
| `coordinator_tool_end` | `{tool_name, result_summary, run_id, thread_id}` | Tool ends |
| `plan_ready` | `{plan, thread_id, interrupt_id}` | Plan draft ready for approval |
| `plan_revision_needed` | `{plan, notes, thread_id}` | Coordinator revised plan after rejection |
| `run_started` | `{run_id, plan_id, thread_id, task_count}` | Execution started |
| `task_started` | `{run_id, task_id, task_title, stage}` | Worker task started |
| `task_token` | `{run_id, task_id, token}` | Token from worker agent |
| `task_tool_start` | `{run_id, task_id, tool_name, args_summary}` | Worker tool starts |
| `task_tool_end` | `{run_id, task_id, tool_name, result_summary}` | Worker tool ends |
| `task_complete` | `{run_id, task_id, outputs_summary}` | Task done |
| `task_failed` | `{run_id, task_id, error, retry_count}` | Task error |
| `run_complete` | `{run_id, artifacts, duration_seconds}` | All tasks done |
| `run_failed` | `{run_id, failed_task_id, error}` | Run aborted |
| `error` | `{message, code}` | Generic error |

### Client → Server Events

| Event | Payload | When |
|---|---|---|
| `join_thread` | `{thread_id}` | Client subscribes to thread events |
| `join_run` | `{run_id}` | Client subscribes to run events |
| `send_message` | `{thread_id, content}` | User message to Coordinator |
| `plan_approved` | `{thread_id, interrupt_id, plan}` | User approves plan (with optional edits) |
| `plan_rejected` | `{thread_id, interrupt_id, notes}` | User rejects with revision notes |
| `approve_tool` | `{thread_id, interrupt_id, decisions}` | User approves tool calls |

---

## 5. REST API

**Base URL:** `/api/v1`  
**Response envelope:** `{"data": ..., "error": null}` or `{"data": null, "error": {"code": ..., "message": ...}}`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check: MongoDB, PostgreSQL, Redis, S3 |
| `POST` | `/threads` | Create thread |
| `GET` | `/threads` | List threads (paginated) |
| `GET` | `/threads/{id}` | Thread detail with recent messages |
| `GET` | `/threads/{id}/messages` | Messages (paginated) |
| `GET` | `/plans/{id}` | Plan detail |
| `PATCH` | `/plans/{id}` | Update plan (pre-approval) |
| `POST` | `/plans/{id}/approve` | REST approval fallback (when WS unavailable) |
| `POST` | `/runs` | Start run from approved plan |
| `GET` | `/runs/{id}` | Run status and task statuses |
| `GET` | `/runs/{id}/artifacts` | List artifacts for a run |
| `GET` | `/artifacts/{id}` | Artifact metadata |
| `GET` | `/artifacts/{id}/download` | Presigned S3 download URL |

---

## 6. Storage Model

| Data | Store | Rationale |
|---|---|---|
| Threads, Messages | MongoDB / Beanie | Flexible schema, fast lookup by thread |
| ResearchPlan, ResearchRun, TaskRun | MongoDB / Beanie | Structured, nested docs, queryable |
| Artifact metadata | MongoDB / Beanie | Linked to runs and tasks |
| Artifact blob content | Amazon S3 | Large files, presigned download URLs |
| LangGraph checkpoint state | PostgreSQL (`PostgresSaver`) | Required by LangGraph API, multi-process safe |
| LangGraph Store (cross-thread memory) | PostgreSQL (`PostgresStore`) | `StoreBackend` for worker `/memories/` namespace |
| Socket.IO pub/sub (rooms) | Redis | Required for multi-process Socket.IO room broadcast |
| Pending interrupt state | Redis (with TTL) | `{thread_id → interrupt config}` for WS resume |

### MongoDB Client Pattern

```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str = "biotech_research"
    # ...

# src/main.py
from pymongo import AsyncMongoClient  # NOT Motor
from beanie import init_beanie

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncMongoClient(settings.MONGODB_URI)
    await init_beanie(
        database=client[settings.MONGODB_DB],
        document_models=[Thread, Message, ResearchPlan, ResearchRun, Artifact],
    )
    yield
    client.close()
```

---

## 7. Observability

### LangSmith Tracing

```python
# All agent invocations use this config shape:
config = {
    "configurable": {
        "thread_id": str(thread_id),
    },
    "metadata": {
        "run_id": str(run_id) if run_id else None,
        "task_id": task_id if task_id else None,
    }
}
```

### Structured Logging

Every log line must carry:
- `thread_id`
- `run_id` (when executing)
- `task_id` (when in a worker)
- `level`, `timestamp`, `service = "biotech-research-ingestion"`

### Health Check Response

```json
{
  "status": "ok",
  "checks": {
    "mongodb": "ok",
    "postgres": "ok",
    "redis": "ok",
    "s3": "ok"
  },
  "version": "0.1.0"
}
```

---

## 8. Security and Environment

All environment variables are validated at startup via `pydantic-settings`. The app **refuses to start** if any required variable is missing.

Required variables:
```
MONGODB_URI
MONGODB_DB
POSTGRES_URL
REDIS_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET
AWS_REGION
ANTHROPIC_API_KEY
TAVILY_API_KEY
LANGSMITH_API_KEY
LANGSMITH_TRACING
LANGGRAPH_CHECKPOINTER   # "memory" (dev) or "postgres" (prod)
```

CORS: configured to allow `biotech-research-web` origin in development and via env var in production.

---

## 9. Phase 9 — KG Ingestion (Future)

After research execution is stable, completed run outputs feed into `biotech-kg`:

1. **Document parsing** — `docling` parses reports and documents into chunks
2. **Structured extraction** — extraction agent maps chunks to entities and relations
3. **Neo4j ingestion** — entities and relations persisted via `biotech-kg` GraphQL API
4. **Trigger** — post-run pipeline enqueued on `run_complete` event

This pipeline is implemented in a later phase and does not affect Phase 1–8 architecture.
