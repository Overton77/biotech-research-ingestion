# Deep Agents Framework Context

You are configuring agents built on top of LangChain's **Deep Agents** framework. Here is what you need to know:

## create_deep_agent (Main Task Agent)

`create_deep_agent` is the primary harness for building capable agents. It provides:

- **Built-in filesystem tools**: `ls`, `read_file`, `write_file`, `edit_file` — these are automatically available without manual configuration.
- **Built-in planning**: A `write_todos` tool for task decomposition and tracking.
- **Built-in subagent spawning**: A `task` tool to delegate to registered subagents.
- **Conversation history summarization**: Automatically compresses old history when token limits approach.
- **Large tool result eviction**: Automatically offloads large results to the filesystem.

Usage:
```python
from deepagents import create_deep_agent
agent = create_deep_agent(
    model=model,
    tools=[...],            # Additional tools beyond built-ins
    system_prompt="...",
    backend=backend_factory, # Callable or BackendProtocol instance
    store=store,             # InMemoryStore or other BaseStore
    subagents=[...],         # List of CompiledSubAgent or dict configs
)
```

## create_agent (Subagent Builder)

`langchain.agents.create_agent` is used to build subagents. Unlike `create_deep_agent`, it does NOT include built-in filesystem tools, planning, or subagent spawning. You must explicitly add middleware:

- **FilesystemMiddleware**: Gives the subagent `ls`, `read_file`, `write_file`, `edit_file` tools backed by a specific backend.
- **TodoListMiddleware**: Gives the subagent a `write_todos` planning tool (optional, for multi-step work).

Usage:
```python
from langchain.agents import create_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware

agent_graph = create_agent(
    model=model,
    tools=[...],
    system_prompt="...",
    middleware=[
        FilesystemMiddleware(backend=filesystem_backend),
        TodoListMiddleware(),  # Optional
    ],
)
```

## CompiledSubAgent

Wraps a pre-built agent graph for attachment to a main `create_deep_agent`. The main agent sees it as an available subagent and uses the `description` to decide when to delegate.

```python
from deepagents import CompiledSubAgent
subagent = CompiledSubAgent(
    name="research-specialist",
    description="Conducts in-depth web research on specific topics.",
    runnable=agent_graph,
)
```

## CompositeBackend

Routes different filesystem paths to different backends. The canonical pattern gives agents both local files and persistent memory:

```python
from deepagents.backends import FilesystemBackend, StoreBackend
from deepagents.backends.composite import CompositeBackend

def backend_factory(runtime):
    return CompositeBackend(
        default=FilesystemBackend(root_dir="/path/to/workspace"),
        routes={"/memories/": StoreBackend(runtime)},
    )
```

- Files under `/memories/` go to the persistent store (cross-thread).
- All other files go to the local filesystem backend.

## FilesystemBackend

Reads/writes real files on disk under a configured `root_dir`. Best for task-local workspace files that should be inspectable.

## Writing to AGENTS.md and /memories/

The main (deep) agent should be instructed to maintain two kinds of persistent context:

- **AGENTS.md**: Document decisions, which subagents were used, and where important outputs are (e.g. “Subagent X wrote outputs/company_products.md”). The main agent should update this file during the task so there is a readable log of what was done and where to find results.
- **/memories/**: Persisted via StoreBackend (cross-thread). Use it for important facts, intermediate conclusions, and references to key files (paths). That way we can retrieve them at the end of the run.

**Requirement:** In your main_agent system_prompt, include 1–2 sentences instructing the agent to: (1) update AGENTS.md with decisions and output locations, (2) write important memories and file references to /memories/.

When defining a subagent, specify **expected_output_format** and **expected_output_path** so the main agent and runtime know what to expect and where to find it.

## Key Design Rules

1. **Main task agents** use `create_deep_agent` — they get filesystem, planning, and subagent tools automatically.
2. **Subagents** use `create_agent` with explicit `FilesystemMiddleware` — they do NOT get planning or recursive subagent tools (prevents context bloat and uncontrolled recursion).
3. Each subagent gets its own isolated workspace directory.
4. Subagents should return concise results — raw data goes to files, only summaries go back to the main agent.
