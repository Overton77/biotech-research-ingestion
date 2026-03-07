# Research Findings — LangChain / LangGraph / DeepAgents / LangSmith

**Status:** `[active]`  
**Produced:** Phase 0 — pre-implementation research  
**Sources:** LangChain DeepAgents docs, LangGraph docs, LangSmith docs, official examples

---

## 1. LangChain Agent Architecture

### `create_agent()` (LangChain)

The current standard entry point for building a conversational agent from a model and a tool list. Internally compiles a LangGraph `StateGraph`, giving the full LangGraph feature set without manually writing graph nodes:

- Returns a `CompiledStateGraph`
- Supports checkpointing, streaming, interrupts, and trace propagation
- Can be wrapped as a `CompiledSubAgent` inside DeepAgents
- Supports `.astream_events()` for token-level streaming

**Use for:** The Coordinator / Planner agent — conversational, plan-generating, HITL-gating.

### `create_deep_agent()` (DeepAgents)

Extends `create_agent` with a built-in tool surface designed for autonomous, multi-step tasks:

| Built-in | Description |
|---|---|
| `write_todos` | Planning and task decomposition |
| `ls` | List filesystem |
| `read_file` | Read from virtual filesystem |
| `write_file` | Write to virtual filesystem |
| `edit_file` | Edit file in place |
| `glob` | File pattern matching |
| `grep` | Search file contents |
| `task()` | Spawn a subagent |

Full `create_deep_agent` signature:
```python
create_deep_agent(
    name,               # Agent name (used in traces)
    model,              # str "provider:model" or BaseChatModel
    tools,              # List of tool functions
    system_prompt,      # str, SystemMessage, or callable
    subagents,          # List of SubAgent dicts or CompiledSubAgent
    interrupt_on,       # Dict[tool_name, True|False|{allowed_decisions:[...]}]
    checkpointer,       # Required for HITL; use PostgresSaver in production
    backend,            # Filesystem backend or callable(runtime) → backend
    store,              # LangGraph BaseStore for long-term memory
)
```

**Use for:** Research worker agents — one per TaskNode, isolated filesystem, scoped tools.

---

## 2. LangGraph Orchestration and State

### Thread Model

Every agent invocation is associated with a `thread_id` in `config["configurable"]["thread_id"]`. The checkpointer persists the full graph state at every step, keyed by `thread_id`. This enables:

- Multi-turn conversation continuity
- HITL pause and resume across HTTP/WebSocket requests
- Failure recovery (resume from last checkpoint)
- Trace grouping in LangSmith

### Checkpointers

| Checkpointer | When to use |
|---|---|
| `MemorySaver` | Development only — in-process, lost on restart |
| `PostgresSaver` | Production — durable, cross-process, multi-worker safe |

`PostgresSaver` requires calling `checkpointer.setup()` once on first startup (idempotent).

### `interrupt()` and `Command(resume=...)`

```python
# Inside a tool or graph node:
from langgraph.types import interrupt, Command

result = interrupt({"type": "plan_review", "plan": plan.model_dump()})
# Graph pauses here — caller receives {"__interrupt__": [...]}

# To resume:
agent.invoke(Command(resume={"plan": edited_plan, "approved": True}), config=config)
```

Key rules:
- Must use the **same `thread_id` config** when resuming
- A `checkpointer` is **required** for interrupt to work
- Can interrupt inside subagents — handling is identical

### Streaming

`.astream_events(input, config, version="v2")` is the async streaming API.

| Event type | When | Frontend use |
|---|---|---|
| `on_chat_model_stream` | Each LLM token | Token streaming |
| `on_tool_start` | Before tool call | "Starting web search…" |
| `on_tool_end` | After tool call | "Search complete" |
| `on_chain_start` | Subagent invoked | "Launching sub-agent" |
| `on_chain_end` | Subagent finished | "Sub-agent complete" |

---

## 3. DeepAgents — Backends

### StateBackend (default, ephemeral)
- Files stored in LangGraph agent state for the current thread
- Lost when the thread ends
- Good for: scratch pad, intermediate notes within a task

### FilesystemBackend (local disk)
```python
from deepagents.backends import FilesystemBackend
agent = create_deep_agent(backend=FilesystemBackend(root_dir="/absolute/path/"))
```
- Reads/writes real files under `root_dir`
- `virtual_mode=True` sandboxes and normalizes paths
- Good for: local dev, CI sandboxes

### StoreBackend (cross-thread persistent)
```python
from deepagents.backends import StoreBackend
agent = create_deep_agent(backend=lambda rt: StoreBackend(rt))
```
- Backed by LangGraph `BaseStore` (PostgresStore, InMemoryStore, etc.)
- Files persist across threads and agent restarts
- Good for: long-term memory, cross-run knowledge

### CompositeBackend (path-prefix routing)
```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

def make_backend(rt):
    return CompositeBackend(
        default=StateBackend(rt),           # /workspace/* — ephemeral
        routes={
            "/inputs/": StoreBackend(rt),   # task inputs from prior tasks
            "/outputs/": StoreBackend(rt),  # declared outputs
            "/memories/": StoreBackend(rt), # long-term memory
        }
    )
```
- Routes file operations to different backends by path prefix
- The production pattern for hybrid ephemeral + persistent storage
- `CompositeBackend` strips the prefix before storing (e.g. `/memories/file.txt` is stored as `/file.txt` in the StoreBackend)

---

## 4. DeepAgents — Subagents

Two patterns for defining subagents:

### Dictionary-based SubAgent
```python
research_subagent = {
    "name": "literature-reviewer",
    "description": "Searches and summarizes scientific literature on a topic. Use for deep paper review.",
    "system_prompt": "You are a thorough scientific literature reviewer...",
    "tools": [pubmed_search, parse_pdf],
    "model": "anthropic:claude-sonnet-4-6",
    "interrupt_on": {"delete_file": True},
}
```

### CompiledSubAgent
```python
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent

inner = create_agent(model=model, tools=tools, name="specialist")
wrapped = CompiledSubAgent(
    name="specialist",
    description="Specialist with custom graph logic",
    runnable=inner,
)
```

Best practices:
- Write clear, specific descriptions — the main agent uses them to decide which subagent to call
- Minimize tool sets per subagent — only give them what they need
- Return concise results from subagents — instruct them to summarize, not dump raw data
- Keep system prompts detailed with explicit output formats

---

## 5. Human-in-the-Loop (HITL) — Two Patterns

### Pattern A: `interrupt_on` (tool approval)
```python
agent = create_deep_agent(
    tools=[delete_file, send_email],
    interrupt_on={
        "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "send_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=checkpointer
)

# Handle interrupt:
result = agent.invoke(input, config=config)
if result.get("__interrupt__"):
    interrupts = result["__interrupt__"][0].value
    # ...display to user...
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
```

Allowed decision types:
- `"approve"` — execute tool with original arguments
- `"edit"` — modify arguments before execution
- `"reject"` — skip this tool call entirely

### Pattern B: Direct `interrupt()` in tool body (used for plan approval)
```python
from langgraph.types import interrupt

@tool
def create_research_plan(objective: str, context: str) -> dict:
    """Generate and get approval for a research plan."""
    plan = _generate_plan(objective, context)
    
    # Pause and send plan to human for review
    approved = interrupt({
        "type": "plan_review",
        "plan": plan.model_dump(),
        "message": "Please review and approve this research plan.",
    })
    
    if approved.get("approved"):
        return {"status": "approved", "plan": approved["plan"]}
    else:
        return {"status": "rejected", "notes": approved.get("notes")}
```

**This is the pattern used for plan approval in this system.**

### WebSocket Integration Pattern
```
Client connects → joins thread:{thread_id} room
User sends message → Socket.IO send_message event
Server: agent.astream_events() starts
Server: streams coordinator_token events to client
Server: coordinator calls create_research_plan tool
Server: agent.astream_events() yields __interrupt__ signal
Server: emits plan_ready event to thread:{thread_id} room
         payload: {type:"plan_ready", plan:{...}, interrupt_id:"..."}
Client: renders plan review UI
User: approves/edits/rejects
Client: emits plan_approved event
         payload: {thread_id, interrupt_id, plan:{...}}
Server: calls agent.invoke(Command(resume={...}), config=config)
Server: resumes streaming events to client
```

**Critical:** The server must store `{thread_id → pending_interrupt_config}` in Redis between the interrupt and the resume. Multiple FastAPI workers need shared state.

---

## 6. Long-term Memory

### CompositeBackend with StoreBackend at `/memories/`
```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(os.environ["DATABASE_URL"])
store.setup()  # Run once at startup

agent = create_deep_agent(
    store=store,
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={"/memories/": StoreBackend(rt)}
    ),
    system_prompt="""...
    Save important findings to /memories/research/<topic>/notes.txt
    These persist across all runs and conversations.
    """
)
```

Files stored at `/memories/` paths survive thread/run boundaries. All other paths are ephemeral.

### Store Implementations

| Store | Use |
|---|---|
| `InMemoryStore` | Development/testing only — lost on restart |
| `PostgresStore` | Production — durable, queryable |

### FileData Schema (StoreBackend)
```python
{
    "content": ["line 1", "line 2"],   # list of strings, one per line
    "created_at": "2024-01-15T10:30:00Z",
    "modified_at": "2024-01-15T11:45:00Z"
}
```

---

## 7. LangSmith Observability

### Setup
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls__..."
```

### Thread Grouping
```python
config = {
    "configurable": {
        "thread_id": str(thread_id),  # groups all turns as one thread
    },
    "metadata": {
        "run_id": str(run_id),
        "task_id": task_id,
        "user_id": str(user_id),
    }
}
```

- `thread_id` in `configurable` groups all runs within a conversation into one thread in LangSmith
- LangGraph propagates `parent_run_id` to all child nodes and subagents automatically
- Metadata fields are searchable in LangSmith

### Trace Lineage
```
Conversation thread (thread_id = MongoDB thread._id)
└── Coordinator run (turn 1)
│   └── web_search tool call
│   └── create_research_plan tool call
│       └── interrupt / resume
└── Coordinator run (turn 2)
    └── save_approved_plan tool call

Research run (separate LangGraph thread per task)
└── TaskRun: literature-review
│   └── pubmed_search tool
│   └── parse_pdf tool
│   └── write_file tool
└── TaskRun: data-analysis (depends on literature-review)
    └── read_file tool
    └── statistical_analysis tool
```

---

## 8. Connection Resilience

LangChain models automatically retry failed API calls:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="anthropic:claude-sonnet-4-6",
    max_retries=10,     # default: 6 — increase for long research tasks
    timeout=120,        # seconds — increase for complex completions
)
```

For long-running research tasks on unreliable networks: `max_retries=10`, `timeout=180`, and use `PostgresSaver` so progress is preserved across retries.

---

## 9. Key API Patterns (Reference)

### Invoke with config
```python
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": user_message}]},
    config={"configurable": {"thread_id": thread_id}, "metadata": {...}}
)
```

### Async streaming
```python
async for event in agent.astream_events(
    {"messages": [{"role": "user", "content": user_message}]},
    config=config,
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        await sio.emit("coordinator_token", {"token": token}, room=f"thread:{thread_id}")
    elif kind == "on_tool_start":
        await sio.emit("coordinator_tool_start", {"tool": event["name"]}, room=f"thread:{thread_id}")
```

### Detect and handle interrupt
```python
result = await agent.ainvoke(input, config=config)

if result.get("__interrupt__"):
    interrupt_val = result["__interrupt__"][0].value
    interrupt_id = str(uuid.uuid4())
    
    # Store in Redis for multi-process resume
    await redis.setex(f"interrupt:{thread_id}", 3600, json.dumps({
        "interrupt_id": interrupt_id,
        "config": config,
        "value": interrupt_val,
    }))
    
    # Notify frontend
    await sio.emit("plan_ready", {
        "plan": interrupt_val["plan"],
        "interrupt_id": interrupt_id,
        "thread_id": str(thread_id),
    }, room=f"thread:{thread_id}")
```

### Resume from interrupt
```python
from langgraph.types import Command

stored = json.loads(await redis.get(f"interrupt:{thread_id}"))
config = stored["config"]

result = await agent.ainvoke(
    Command(resume={"plan": approved_plan, "approved": True}),
    config=config,
)
await redis.delete(f"interrupt:{thread_id}")
```
