# Execution Plan 04 — Research Run Persistence, Progress Streaming & API

**Status:** `[active]`  
**Version:** 1.0  
**Scope:** `biotech-research-ingestion` (backend)

---

## Overview

This plan specifies:

1. **Persisting Research Run outputs** to the AWS S3 bucket and (where applicable) the local filesystem, using `AsyncS3Client` and `ResearchRunsS3Store`.
2. **Custom LangChain middleware** in Compiled SubAgents and main task agents: `after_model` and `after_agent` hooks that send progress (tool calls, prompts, agent lifecycle) to the frontend via a Socket.IO path.
3. **REST API** with pagination for ResearchPlans, ResearchMissions, ResearchRuns, plus endpoints to retrieve full outputs from S3 (mission, task runs, TaskDef outputs).
4. **Progress delivery** from Temporal workers to the frontend using a Python Socket.IO client that connects to the same API server and emits to a room the server re-broadcasts to.

No timelines are specified; only specifications and detailed instructions with examples.

---

## 1. Persisting Research Run Outputs to S3 and Filesystem

### 1.1 Current State

- **`src/research/persistence/runs_s3.py`** defines `ResearchRunsS3Store` and `ResearchRunS3Paths`. It already implements:
  - `write_mission`, `write_mission_draft`, `write_research_run`, `write_task_result`, `write_final_report_*`, `upload_task_artifact_*`, `build_task_runs_index`, `list_mission_objects`.
- **`src/research/persistence/research_run_writer.py`** writes only to MongoDB (`ResearchRun` Beanie documents).
- **`src/research/runtime/mission_runner.py`** node `persist_research_run` calls only `ResearchRunWriter.upsert_run`; it does **not** call `ResearchRunsS3Store`.
- **`src/infrastructure/aws/async_s3.py`** provides `AsyncS3Client` (`put_json`, `put_text`, `get_json`, `get_text`, `list_objects`). Bucket name from `BIOTECH_RESEARCH_RUNS_BUCKET`.

S3 key layout (from `.cursor/docs/aws_research_bucket_organization.md`):

```
missions/{mission_id}/mission/...
tasks/{task_id}/attempts/{attempt_number}/run.json, outputs.json, events.json, resolved-inputs.json, artifacts/...
indexes/task-runs.json, artifacts.json
```

### 1.2 Specification: Wire S3 Into Mission Runner and Task Lifecycle

**Objective:** Every completed task run persists to both MongoDB and S3. Mission-level artifacts (draft, final report, summary, task-runs index) are written to S3 at the appropriate points.

**Tasks:**

1. **Inject `ResearchRunsS3Store` into the mission runner**  
   - In `mission_runner.py`, obtain an `ResearchRunsS3Store` instance (e.g. module-level singleton or from a small `get_research_runs_s3_store()` that uses `AsyncS3Client()`).  
   - Use the same store in both `persist_research_run` and `finalize_mission`.

2. **After each task: persist to S3 in `persist_research_run`**  
   - After `ResearchRunWriter.upsert_run`, build a `ResearchRun` document (or use the one returned by `upsert_run`).  
   - Call `s3_store.write_research_run(mission=state["mission"], research_run=run_doc, task_def=task_def)`.  
   - Optionally call `s3_store.write_task_result(mission=..., task_result=result, task_def=task_def)` if you want the full `TaskResult` (including events) in S3 as well; the existing method writes `run.json`, `outputs.json`, `events.json` under the task attempt prefix.  
   - Use the `ResearchRun` document for `write_research_run` so that `resolved_inputs_snapshot` and `outputs_snapshot` match MongoDB.  
   - **Idempotency:** Both `write_research_run` and `write_task_result` are overwrites by key; safe to call once per (mission_id, task_id, attempt_number).

3. **Task artifacts (filesystem → S3)**  
   - `task_executor.py` currently collects artifacts from `task_root(mission_id, task_id) / "outputs"` and returns `ArtifactRef` with `storage="filesystem"` and `path=str(filepath)`.  
   - After task completion, before or inside `persist_research_run`, for each artifact in `result.artifacts` that has a local filesystem path:  
     - Read the file content (or use existing helpers).  
     - Call `s3_store.upload_task_artifact_text` or `upload_task_artifact_json` (depending on content type) so that the artifact is stored under `tasks/{task_id}/attempts/{attempt_number}/artifacts/{type}/{name}`.  
     - Update the `ArtifactRef` to use `path=s3_uri` (and optionally `storage="s3"` if you extend the model) before persisting to Mongo and S3.  
   - This keeps a single source of truth: S3 holds the full artifact content; MongoDB/ResearchRun can keep the same `ArtifactRef` with the S3 path.

4. **Mission-level S3 writes**  
   - **On mission load or after mission creation:** Optionally write the mission document and draft to S3 (e.g. in `load_mission` or when the mission is first created in `mission_creator` / before workflow start) using `write_mission` and, if you have a draft in memory, `write_mission_draft`.  
   - **On mission finalization:** In `finalize_mission`:  
     - Call `s3_store.write_mission(mission)` again so that `mission.json` reflects final status.  
     - Build the list of `ResearchRun` documents for the mission (from MongoDB or from `state`), then call `s3_store.build_task_runs_index(mission=..., research_runs=runs)` so that `indexes/task-runs.json` exists.  
     - If you have a final report or summary (e.g. aggregated from task outputs), call `write_final_report_markdown` / `write_final_report_json` / `write_final_report` as needed.

5. **Local filesystem mirror (optional)**  
   - If you want a local mirror of S3 for development or debugging, add a small layer that, when an env var is set (e.g. `MIRROR_RESEARCH_OUTPUTS_TO_DISK=1`), also writes the same JSON/text to a local directory mirroring the S3 prefix layout (e.g. `./research_outputs/missions/{mission_id}/...`).  
   - This is optional and can be a separate helper called from the same places that call `ResearchRunsS3Store`.

**Example (conceptual) for `persist_research_run`:**

```python
# In persist_research_run, after writer.upsert_run:
run_doc = await writer.upsert_run(...)
task_def = state["task_defs_by_id"].get(run_doc.task_id)
s3_store = get_research_runs_s3_store()
await s3_store.write_research_run(
    mission=state["mission"],
    research_run=run_doc,
    task_def=task_def,
)
# Optionally upload filesystem artifacts to S3 and update ArtifactRef.path
for art in run_doc.artifacts:
    if art.path and not art.path.startswith("s3://"):
        # upload local file to S3, get s3_uri, then update art.path = s3_uri
        ...
```

**Example for `finalize_mission`:**

```python
# In finalize_mission:
s3_store = get_research_runs_s3_store()
await s3_store.write_mission(mission)
runs = await ResearchRun.find(ResearchRun.mission_id == mission.id).to_list()
await s3_store.build_task_runs_index(mission=mission, research_runs=runs)
```

### 1.3 Error Handling

- S3 write failures should be logged and optionally retried (e.g. 1–2 retries with backoff). If S3 is down, mission execution can still complete and MongoDB will have the authoritative record; document the behavior (e.g. "S3 write failed; run is persisted in MongoDB only") so the API can reflect that when serving outputs from S3.

---

## 2. Custom Middleware for Progress (after_model, after_agent)

### 2.1 Goal

Capture agent progress inside the Deep Agent and Compiled SubAgents so the frontend can show:

- When a (sub)agent starts and ends.
- When the model is called and when it returns (and optionally a short summary of the last message).
- Tool calls: name, arguments summary, and optionally result summary.

Data should be minimal but sufficient for a real-time dashboard (e.g. "Task X, Subagent Y: calling tool Z", "Model returned", "Agent finished").

### 2.2 LangChain Middleware Contract

- Use **node-style hooks** as in [LangChain Custom Middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom):
  - **`after_model`** — runs after each model response. Use it to emit: `model_response` (e.g. last message type and length or a truncated content summary), and optionally `message_count`.
  - **`after_agent`** — runs once when the agent completes. Use it to emit: `agent_completed` with a small summary (e.g. final message count, completion reason if available).
- Use **wrap-style** only if you need to intercept **before** a tool runs and **after** it returns: **`wrap_tool_call`** lets you emit `tool_start` (name, args summary) and `tool_end` (name, result summary). This is the recommended way to track tool calls.
- Middleware runs inside the process that invokes the agent (the Temporal activity worker). It must not block the agent; emitting should be fire-and-forget or via a non-blocking callback.

### 2.3 Progress Callback Contract

Define a **progress callback** that middleware and the task executor can call:

- **Signature:** `async def progress_callback(event_type: str, payload: dict[str, Any]) -> None`
- **Callers:** Custom middleware (inside subagents and main agent) and, if desired, the mission runner nodes (e.g. `task_started`, `task_completed`).
- **Payload** must always include at least:
  - `mission_id: str`
  - `task_id: str`
  - `timestamp: str` (ISO)
- For **subagent** events, include:
  - `subagent_name: str`
  - `agent_role: Literal["main", "subagent"]`
- Event types (examples):
  - `agent_started` — payload: `agent_role`, `subagent_name?`, `message_count?`
  - `agent_completed` — payload: `agent_role`, `subagent_name?`, `message_count?`, `summary?`
  - `model_called` / `model_response` — payload: `agent_role`, `subagent_name?`, `content_preview?`, `message_index?`
  - `tool_start` — payload: `tool_name`, `args_summary?`, `agent_role`, `subagent_name?`
  - `tool_end` — payload: `tool_name`, `result_summary?`, `agent_role`, `subagent_name?`
  - `task_started` / `task_completed` / `task_failed` — can be emitted from the mission runner or task executor, not necessarily from middleware.

The callback should be **injectable** into the agent compilation so that the same agent code can run with or without progress reporting (e.g. tests pass `None` or a no-op).

### 2.4 Where to Inject the Callback

- **RuntimeContext** (in `agent_compiler.py`) should accept an optional `progress_callback: Callable[[str, dict], Awaitable[None]] | None`.
- **compile_subagent** and **compile_main_task_agent**:
  - Build one or more custom middleware instances that receive this callback and the current `mission_id`, `task_id`, and (for subagents) `subagent_name`.
  - Register these middleware with `create_agent` / the deep agent so that:
    - **after_model:** extract the last AI message from state, build a small payload (e.g. `content_preview=content[:200]` or message type + length), call `progress_callback("model_response", payload)`.
    - **after_agent:** call `progress_callback("agent_completed", payload)`.
    - **wrap_tool_call:** before calling the handler, call `progress_callback("tool_start", {...})`; after handler returns, call `progress_callback("tool_end", {...})`; then return the handler result.
  - Middleware must be **async-safe**: if the callback is async, use `await`; if the framework runs hooks in a sync context, use `asyncio.create_task` or a thread-safe queue that a background task drains (see below).

### 2.5 Implementing the Middleware (Class-Based)

Use **class-based middleware** (subclass of `AgentMiddleware`) so you can hold `mission_id`, `task_id`, `subagent_name`, and the callback.

**Example structure:**

```python
# src/research/middleware/progress_middleware.py
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any, Callable, Awaitable

class ResearchProgressMiddleware(AgentMiddleware):
    def __init__(
        self,
        mission_id: str,
        task_id: str,
        subagent_name: str | None,
        progress_callback: Callable[[str, dict], Awaitable[None]] | None,
    ):
        super().__init__()
        self.mission_id = mission_id
        self.task_id = task_id
        self.subagent_name = subagent_name
        self.progress_callback = progress_callback

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        if not self.progress_callback:
            return None
        # Build payload from state["messages"][-1], then schedule callback
        # (e.g. asyncio.create_task(self.progress_callback("model_response", payload)))
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        if not self.progress_callback:
            return None
        # asyncio.create_task(self.progress_callback("agent_completed", {...}))
        return None

    def wrap_tool_call(self, request, handler):
        if self.progress_callback:
            # emit tool_start, then call handler, then emit tool_end
            ...
        return handler(request)
```

- If LangChain middleware hooks are **sync**, you cannot `await` the callback inside them. Then either:
  - Use a **thread-safe queue**: middleware pushes `(event_type, payload)` to the queue; a dedicated asyncio task (started by the task executor or mission runner) drains the queue and calls the async callback and/or forwards to the Socket.IO client, or
  - Use a **sync** wrapper that pushes to a queue that the worker’s event loop drains.

### 2.6 Passing the Callback From Task Executor to Compiler

- In **task_executor.execute_task**:
  - The caller (mission runner) or a higher-level layer must create an async `progress_callback` that forwards to the Socket.IO client (see Section 3).
  - Build `RuntimeContext(mission_id=..., task_id=..., store=..., checkpointer=..., progress_callback=...)` and pass it to `compile_main_task_agent`.
- In **agent_compiler.compile_subagent** and **compile_main_task_agent**:
  - For each subagent, create `ResearchProgressMiddleware(mission_id, task_id, subagent_name=config.name, progress_callback=ctx.progress_callback)` and append it to the middleware list.
  - For the main task agent, create one `ResearchProgressMiddleware` with `subagent_name=None` and add it to the deep agent’s middleware (if the deep agent accepts middleware; otherwise add it to each subagent’s `create_agent` and ensure the main agent’s invocations are wrapped similarly).

- **Important:** `create_deep_agent` may not expose a `middleware` parameter the same way `create_agent` does. Check the Deep Agents API: if the main agent is a wrapper around an inner graph, you may add middleware only to the inner agent or to each subagent. The goal is that every tool call and model turn from both main and subagents is reported.

### 2.7 Payload Size and PII

- Keep payloads small: truncate message content to a few hundred characters, and use short `args_summary` / `result_summary` (e.g. first 200 chars or a hash for very long results).
- Do not send raw PII or full prompts in the progress stream; document that the progress channel is for operational visibility, not for storing sensitive data.

---

## 3. Sending Progress to the Frontend (Socket.IO)

### 3.1 Architecture

- **Temporal worker** runs in a separate process from the FastAPI app. The worker executes the mission and the agent (with middleware) runs inside the activity.
- **Frontend** is connected to the **FastAPI** app’s Socket.IO server (namespace `/research`) and joins a room per mission, e.g. `mission:{mission_id}`.
- **Progress** must go: Worker → API server → Socket.IO room.

**Option A — Python Socket.IO client in the worker (recommended):**

- In the **Temporal worker** (or in the ingestion app when running the mission in-process), use a **Python Socket.IO client** (`socketio.AsyncClient` or equivalent) to connect to the **same** API base URL (e.g. `http://localhost:8000` or the internal URL the worker uses to reach the API).
- The worker does **not** join rooms; instead, it emits a **server-side event** that the API server understands. For example:
  - Worker connects to namespace `/research` (or a dedicated `/research-internal` namespace).
  - Worker emits an event, e.g. `research_progress`, with payload: `{ "mission_id": "...", "event_type": "...", "payload": { ... } }`.
  - The **FastAPI Socket.IO server** has a handler for `research_progress` (or the server listens on a dedicated namespace). On receipt, it re-emits to the room `mission:{mission_id}` so that all frontend clients watching that mission receive the event.
- The worker must be able to resolve the API URL (e.g. from env `API_BASE_URL` or `SOCKETIO_SERVER_URL`).

**Option B — Redis pub/sub:**

- Worker **publishes** progress to a Redis channel, e.g. `research:progress:{mission_id}`.
- The FastAPI app subscribes to `research:progress:*` (or per-mission channels) and, when it receives a message, emits to the Socket.IO room `mission:{mission_id}`.
- This requires a shared Redis instance and a background task in the FastAPI app.

This plan specifies **Option A** (Python Socket.IO client) with a single event name `research_progress` and a payload that includes `mission_id` and the same event types as in Section 2.3.

### 3.2 Worker-Side: Progress Callback to Socket.IO

- In the **Temporal activity** `run_deep_research_mission`, before calling `run_mission(mission_id)`:
  - Create an async **progress callback** that:
    1. Builds a message `{ "mission_id": mission_id, "event_type": event_type, "payload": payload, "timestamp": utc_iso }`.
    2. Sends it to the API server. Sending can be implemented by:
      - **Option A1:** A shared **Socket.IO client** (e.g. one per worker process) that connects to the API and emits `research_progress` with that message. The server’s handler then does `await self.emit("research_progress", message, room=f"mission:{mission_id}")`.
      - **Option A2:** An **HTTP POST** to an internal endpoint, e.g. `POST /api/v1/internal/research-progress`, with the same JSON body. The endpoint then uses the app’s `sio.emit("research_progress", body, room=f"mission:{body['mission_id']}")`. This avoids a long-lived Socket.IO connection from the worker.
- **Recommendation:** Use **HTTP POST** for progress from the worker to the server, so the worker does not need to maintain a Socket.IO connection. The FastAPI app exposes `POST /api/v1/internal/research-progress` (or a path you choose), protected by an internal secret or network, and that endpoint emits to the Socket.IO room.

**Example internal endpoint:**

```python
# In a new or existing router, e.g. src/api/routes/internal.py
@router.post("/internal/research-progress")
async def research_progress(req: ResearchProgressRequest) -> None:
    # Validate internal auth (e.g. header X-Internal-Secret)
    sio = get_sio()
    room = f"mission:{req.mission_id}"
    await sio.emit("research_progress", req.model_dump(), room=room)
    return None
```

- Then the **progress callback** used in the task executor (and passed into the agent compiler) should:
  - Be created in a layer that has access to `mission_id` and to an **HTTP client** (e.g. `httpx.AsyncClient`) or a small helper that POSTs to `API_BASE_URL + "/api/v1/internal/research-progress"` with the same JSON.  
  - So: **middleware → progress_callback(event_type, payload) → HTTP POST to API → API emits to Socket.IO room.**

- **Creating the callback in the mission runner:** The mission runner runs inside the Temporal activity. So when the activity starts, it can create an `httpx.AsyncClient` and a closure that POSTs to the internal endpoint. Pass this closure (or a wrapper that adds `mission_id` and `timestamp`) into the graph state or into the task executor. The task executor receives it and passes it in `RuntimeContext.progress_callback`. The **mission runner** must be able to pass the callback into `run_task`; for that, the callback can be stored in `RunnableConfig` or in a custom context that the runner builds and passes to `execute_task_def`.

**Simplest integration:** Add an optional `progress_callback` to the **config** passed to the graph invocation. When building `RuntimeContext` in `run_task`, read `progress_callback = config.get("configurable", {}).get("progress_callback")` and pass it to the context. The Temporal activity then sets `config["configurable"]["progress_callback"] = my_http_emit_callback` before invoking the runner.

### 3.3 Socket.IO Server: Room and Event

- **Room name:** `mission:{mission_id}`.
- **Event name for frontend:** `research_progress`.
- **Payload:** `{ "mission_id": str, "event_type": str, "payload": dict, "timestamp": str }`.
- **Frontend:** On connecting to a “watch run” view, the client emits `join_mission` with `{ "mission_id": "..." }`; the server adds the socket to the room `mission:{mission_id}`. When the server emits `research_progress`, all clients in that room receive it.

**New server-side handler (example):**

```python
# In ResearchNamespace (server.py):
async def on_join_mission(self, sid: str, data: dict[str, Any]) -> None:
    mission_id = data.get("mission_id")
    if mission_id:
        room = f"mission:{mission_id}"
        await self.enter_room(sid, room)
```

### 3.4 Summary of Flow

1. Frontend opens dashboard for mission `M`, connects to Socket.IO, emits `join_mission` with `mission_id=M`.
2. User starts the mission (or it is already running). Temporal activity runs.
3. Activity creates a progress callback that POSTs to `POST /api/v1/internal/research-progress` with `{ mission_id, event_type, payload, timestamp }`.
4. Activity invokes the mission runner with this callback in config. Runner passes it to task executor → RuntimeContext → agent compiler → middleware.
5. Middleware emits `tool_start`, `tool_end`, `model_response`, `agent_completed` via the callback.
6. API receives POST, emits `research_progress` to room `mission:M`.
7. Frontend receives `research_progress` and updates the UI.

---

## 4. API Endpoints for ResearchPlans, ResearchMissions, ResearchRuns and S3 Outputs

### 4.1 List Endpoints with Pagination

- **ResearchPlans**
  - `GET /api/v1/plans`  
    - Query params: `skip` (default 0), `limit` (default 20, max 100), optional `thread_id`, optional `status`.  
    - Response: `{ "items": [ plan_dict, ... ], "total": int }` (or equivalent envelope).  
    - Implement by `ResearchPlan.find(...).skip(skip).limit(limit)` and a separate `count()` for total if needed.

- **ResearchMissions**
  - `GET /api/v1/missions`  
    - Query params: `skip`, `limit`, optional `research_plan_id`, optional `thread_id`, optional `status`.  
    - Response: `{ "items": [ mission_dict, ... ], "total": int }`.

- **ResearchRuns**
  - `GET /api/v1/runs`  
    - Query params: `skip`, `limit`, optional `mission_id`.  
    - Response: `{ "items": [ run_dict, ... ], "total": int }`.  
  - Existing `GET /api/v1/missions/{mission_id}/runs` can remain for “all runs for a mission” without pagination; the new list endpoint supports cross-mission listing and pagination.

### 4.2 Single-Resource Endpoints

- Keep existing:
  - `GET /api/v1/plans/{plan_id}`
  - `GET /api/v1/missions/{mission_id}`
  - Add: `GET /api/v1/runs/{run_id}` — return a single ResearchRun by MongoDB `_id` (envelope with full document).

### 4.3 Retrieving Full Outputs from S3

- **Mission-level**
  - `GET /api/v1/missions/{mission_id}/outputs`  
    - Returns the full mission-level outputs that are stored in S3: e.g. `mission.json`, `mission-draft.json`, `final-report.md` or `.json`, `summary.json`, and the task-runs index.  
    - Implementation: use `ResearchRunsS3Store` + `ResearchRunS3Paths(mission_id)`. Call `s3.get_json` / `s3.get_text` for each key (mission_json_key, mission_draft_json_key, final_report_markdown_key, final_report_json_key, summary_json_key, task_runs_index_key). If an object does not exist, return that key as `null` or omit it. Return a single JSON envelope, e.g. `{ "mission": {...}, "mission_draft": {...}, "final_report_markdown": "...", "final_report_json": {...}, "summary": {...}, "task_runs_index": {...} }`.

- **Task run (ResearchRun) full outputs**
  - `GET /api/v1/missions/{mission_id}/runs/{task_id}/outputs`  
    - Query param: `attempt_number` (default 1).  
    - Fetches from S3: `run.json`, `resolved-inputs.json`, `outputs.json`, `events.json` for that task attempt. Return envelope: `{ "run": {...}, "resolved_inputs": {...}, "outputs": {...}, "events": {...} }`.  
    - If the run is not in S3 (e.g. legacy run), fall back to MongoDB: load `ResearchRun` by mission_id + task_id + attempt_number and return its fields (resolved_inputs_snapshot, outputs_snapshot, etc.) so the API still works.

- **Task run artifacts**
  - `GET /api/v1/missions/{mission_id}/runs/{task_id}/artifacts`  
    - Query param: `attempt_number`.  
    - Return list of artifact metadata (from MongoDB ResearchRun.artifacts or from S3 index).  
  - `GET /api/v1/missions/{mission_id}/runs/{task_id}/artifacts/{artifact_name}/content`  
    - Query params: `attempt_number`, optional `artifact_type`.  
    - Resolve the S3 key from `ResearchRunS3Paths.artifact_key(...)` or from the artifact index; if the artifact is in S3, stream or return the object (or a presigned URL). Prefer returning presigned URL for large files so the frontend can open or download.

### 4.4 Model and Response Shapes

- **ResearchPlan:** Already has Beanie model and `_plan_to_dict`. Ensure list response uses the same dict shape.
- **ResearchMission:** Already has `_mission_to_dict`. For list, return the same shape; for `GET /missions/{id}/outputs`, add the S3-sourced fields as above.
- **ResearchRun:** Add `started_at` in the response if not already (model already has it). For S3 outputs, the response is the raw S3 document shapes plus any fallback from Mongo.

### 4.5 Errors

- 404 when mission/plan/run not found.  
- 404 or 503 when S3 object is missing and no Mongo fallback (document behavior).  
- 400 for invalid `attempt_number` or missing required query params.

---

## 5. Implementation Order (Suggested)

1. **S3 persistence**  
   - Add `get_research_runs_s3_store()`.  
   - Wire `ResearchRunsS3Store` into `persist_research_run` and `finalize_mission`.  
   - Upload task artifacts from filesystem to S3 and set `ArtifactRef.path` to the S3 URI.  
   - Optionally write mission and draft to S3 on mission creation/load and write task-runs index on finalize.

2. **Progress callback and middleware**  
   - Define `progress_callback` signature and event types.  
   - Add `ResearchProgressMiddleware` (after_model, after_agent, wrap_tool_call).  
   - Add `progress_callback` to `RuntimeContext` and wire it from mission runner → task executor → agent compiler.  
   - Implement callback to POST to an internal HTTP endpoint (no Socket.IO client in worker yet if you prefer).

3. **Internal endpoint and Socket.IO room**  
   - Add `POST /api/v1/internal/research-progress` and `on_join_mission`.  
   - From the activity, create the callback that POSTs to this endpoint; pass callback via runner config.

4. **List and get APIs**  
   - Add paginated `GET /plans`, `GET /missions`, `GET /runs` and `GET /runs/{id}`.  
   - Add `GET /missions/{id}/outputs` and `GET /missions/{id}/runs/{task_id}/outputs`, plus artifact list and artifact content (or presigned URL).

5. **Optional: Python Socket.IO client**  
   - If you prefer the worker to emit via Socket.IO instead of HTTP, introduce a long-lived client in the worker that connects to the API and emits `research_progress`; the server then re-emits to the room. Same payload contract as above.

---

## 6. References

- LangChain middleware: [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom), [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in), [Overview](https://docs.langchain.com/oss/python/langchain/middleware/overview).
- Deep Agents: [Overview](https://docs.langchain.com/oss/python/deepagents/overview).
- LangGraph: [Overview](https://docs.langchain.com/oss/python/langgraph/overview).
- In-repo: `src/research/persistence/runs_s3.py`, `src/infrastructure/aws/async_s3.py`, `src/research/runtime/mission_runner.py`, `src/research/compiler/agent_compiler.py`, `src/api/socketio/handlers.py`, `src/api/socketio/server.py`.
