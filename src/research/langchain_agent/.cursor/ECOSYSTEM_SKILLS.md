# LangChain Ecosystem Skills Reference

> **Location:** `biotech-research-ingestion/.agents/skills/`  
> This document is a compact reference of all available LangChain-ecosystem skills. Read the full SKILL.md for any skill before implementing the feature it covers.

---

## Skills Inventory

| Skill Directory | Name | When to invoke |
|---|---|---|
| `langchain-fundamentals/` | LangChain Fundamentals | Any time you create an agent, define tools, or add middleware |
| `langchain-middleware/` | LangChain Middleware | Custom middleware, dynamic prompt injection, request/response hooks |
| `langchain-rag/` | LangChain RAG | Retrieval-augmented generation, vector stores, document loaders |
| `langchain-dependencies/` | LangChain Dependencies | Dependency management, package versions, pyproject.toml patterns |
| `langgraph-fundamentals/` | LangGraph Fundamentals | Any time you write StateGraph, nodes, edges, Command, Send, streaming |
| `langgraph-persistence/` | LangGraph Persistence | Checkpointers, stores, memory across threads |
| `langgraph-human-in-the-loop/` | LangGraph Human-in-the-Loop | Interrupt, resume, approval workflows |
| `deep-agents-core/` | Deep Agents Core | `create_deep_agent()`, harness architecture, middleware selection |
| `deep-agents-orchestration/` | Deep Agents Orchestration | SubAgentMiddleware, TodoListMiddleware, HumanInTheLoopMiddleware |
| `deep-agents-memory/` | Deep Agents Memory | FilesystemMiddleware backends (State/Store/Filesystem/Composite) |
| `langsmith-trace/` | LangSmith Trace | Adding tracing, querying traces, @traceable decorator |
| `langsmith-dataset/` | LangSmith Dataset | Dataset creation, management, and evaluation datasets |
| `langsmith-evaluator/` | LangSmith Evaluator | Evaluators, scoring, automated quality checks |
| `framework-selection/` | Framework Selection | Choosing between LangChain, LangGraph, Deep Agents for a use case |
| `tavily-best-practices/` | Tavily Best Practices | Optimal use of all Tavily operations |
| `tavily-search/` | Tavily Search | `search_web` patterns, query construction, depth/topic selection |
| `tavily-extract/` | Tavily Extract | `extract_from_urls` patterns, chunking, format selection |
| `tavily-map/` | Tavily Map | `map_website` site discovery, depth/breadth settings |
| `tavily-crawl/` | Tavily Crawl | `crawl_website` content + links, shallow crawl patterns |
| `tavily-research/` | Tavily Research | End-to-end research patterns combining search + extract + map + crawl |
| `tavily-cli/` | Tavily CLI | CLI usage for testing Tavily calls |

---

## Key Patterns Used in This Codebase

### `create_agent()` — the only way to build agents here

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-mini",              # or any LiteLLM-compatible model string
    tools=[...],                      # LangChain @tool functions
    middleware=[...],                 # FilesystemMiddleware, dynamic_prompt, SubAgentMiddleware
    store=store,                      # AsyncPostgresStore for long-term memory
    checkpointer=checkpointer,        # AsyncPostgresSaver for thread state
    state_schema=BiotechResearchAgentState,  # custom AgentState subclass
)
result = await agent.ainvoke({"messages": [...]}, config=runtime_config)
```

> Never use `llm.with_structured_output()` in agentic pipelines — it bypasses the tool-calling loop.  
> Use `response_format=MyPydanticModel` in `create_agent()` for structured output agents.

### Structured Output Agents

For extraction, schema selection, memory reporting — build a minimal agent with `response_format`:

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class MyResult(BaseModel):
    field: str

agent = create_agent(
    model="gpt-5-mini",
    tools=[],           # add tools if the agent needs to look things up
    system_prompt="...",
    response_format=MyResult,
    store=store,
    checkpointer=checkpointer,
)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]}, config=cfg)
structured: MyResult = result["structured_response"]
```

### Dynamic Prompt Middleware

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def my_prompt_fragment(request: ModelRequest) -> str:
    state = request.state
    return f"Mission: {state.get('mission_id')}\nStep: {state.get('step_count')}"

agent = create_agent(model=..., tools=..., middleware=[my_prompt_fragment])
```

### FilesystemMiddleware (Deep Agents)

```python
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.filesystem import FilesystemBackend

backend = FilesystemBackend(root_dir="/path/to/sandbox", virtual_mode=True)
fs_middleware = FilesystemMiddleware(
    backend=backend,
    system_prompt="Use relative sandbox paths only: runs/, reports/, scratch/",
    custom_tool_descriptions={"write_file": "Create markdown or JSON files in sandbox."},
)
```

### SubAgentMiddleware (Deep Agents)

```python
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware

browser_subagent = CompiledSubAgent(
    name="browser_control",
    description="Controls a real browser via Playwright. Use for dynamic pages.",
    runnable=create_agent(model=..., tools=[playwright_mcp_specs]),
)
middleware = SubAgentMiddleware(subagents=[browser_subagent], backend=backend)
```

---

## LangMem Integration

LangMem uses `create_memory_store_manager()` from the `langmem` package.

```python
from langmem import create_memory_store_manager

manager = create_memory_store_manager(
    "openai:gpt-4.1",
    schemas=[SemanticEntityFact, EpisodicResearchRun, ProceduralResearchPlaybook],
    instructions=MEMORY_INSTRUCTIONS,
    enable_inserts=True,
    enable_deletes=False,
    namespace=("memories", "{mission_id}"),   # {mission_id} filled from config["configurable"]
    store=store,
)

# Search
items = await manager.asearch(query="Qualia Life Sciences domain", config=runtime_config)

# Write (extract + persist from conversation)
await manager.ainvoke({"messages": [user_msg, assistant_msg]}, config=runtime_config)
```

**Namespace convention:** `("memories", "{mission_id}")` — all memories are scoped per mission.

---

## LangSmith Tracing

Set these env vars — tracing is automatic for all `create_agent()` runs:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=biotech-research-ingestion
```

To add custom spans:
```python
from langsmith import traceable

@traceable(name="kg_ingestion_stage")
async def run_kg_ingestion(...):
    ...
```

---

## LangGraph Persistence (Postgres)

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

# In storage/langgraph_persistence.py
store, checkpointer = await get_persistence()
# Both are singletons — one per process, initialized lazily with double-checked locking

# Runtime config pattern
runtime_config: RunnableConfig = {
    "configurable": {
        "thread_id": run_input.task_id,      # checkpoint isolation
        "mission_id": run_input.mission_id,  # LangMem namespace
    }
}
```

---

## Tool Pattern: `Command` with State Update

All Tavily tools return `Command` to update agent state alongside the tool message:

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool(parse_docstring=False)
async def search_web(runtime: ToolRuntime, query: str, ...) -> Command:
    formatted, raw = await _tavily_search_impl(query, ...)
    # Build and persist event
    event = _build_search_event(query=query, raw_results=raw, ...)
    visited_urls = _merge_visited(runtime, event["top_urls"])
    tavily_search_events = _append_state_event(runtime, "tavily_search_events", event)

    return Command(
        update={
            "visited_urls": visited_urls,
            "tavily_search_events": tavily_search_events,
            "messages": [ToolMessage(content=formatted, tool_call_id=runtime.tool_call_id)],
        }
    )
```

---

## Framework Selection Quick Guide

| Scenario | Use |
|---|---|
| Single-purpose tool-calling agent | `create_agent()` (langchain-fundamentals) |
| Complex multi-step with planning, delegation, filesystem | `create_deep_agent()` (deep-agents-core) |
| Fine-grained graph control, loops, branching | `StateGraph` (langgraph-fundamentals) |
| Already-built `create_agent()` needs file management | Add `FilesystemMiddleware` (deep-agents-memory) |
| Agent needs to spawn specialized sub-agents | Add `SubAgentMiddleware` (deep-agents-orchestration) |
| Agent needs long-term memory across sessions | Add `MemoryMiddleware` or LangMem (deep-agents-memory) |
| Need human approval before sensitive actions | `HumanInTheLoopMiddleware` (langgraph-human-in-the-loop) |
| Observability and tracing | LangSmith env vars (langsmith-trace) |

---

## Notes on Model Strings

The codebase currently uses `gpt-5-mini` (hardcoded in `agent/factory.py`).  
Update the model string in `factory.py` to switch models globally. Use LiteLLM format:

- `"openai:gpt-4o"` — OpenAI GPT-4o
- `"openai:gpt-4.1"` — OpenAI GPT-4.1 (used in LangMem manager)
- `"anthropic:claude-sonnet-4-5"` — Anthropic Claude
