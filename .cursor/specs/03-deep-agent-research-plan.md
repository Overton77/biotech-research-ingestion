# Deep Agent Research Mission — Finalized Execution Plan

**Status:** `[active — implementation ready]`  
**Version:** 1.0  
**Scope:** `biotech-research-ingestion`  
**Date:** 2026-03-09

---

## 0. Purpose & Scope

This document specifies the complete design and implementation plan for the **Deep Agent Research Mission system** — the layer that sits between the existing Coordinator (which produces a `ResearchPlan`) and actual deep research execution. It covers:

- All new domain models
- All new modules to create (`src/research/`)
- Precise changes to existing code (coordinator, plan models, routes, `__init__` files)
- LangGraph `MissionRunner` graph design with node contracts
- Deep Agent and compiled subagent construction patterns
- Filesystem backend layout
- Tool policy
- Mongo persistence strategy
- ADRs for every non-obvious decision
- Phased implementation tasks

**What the user will experience after this is built:**

1. Chat with the Coordinator → it creates a `ResearchPlan` with stages/tasks.
2. Approve the plan via HITL interrupt.
3. **A Mission Compiler LLM Agent** reads the approved plan and reasons about Deep Agent topology — producing a fully detailed `ResearchMission` with per-task agent system prompts, subagent roles and system prompts, tool profiles, input/output schemas, and acceptance criteria. The LLM is given Deep Agents framework documentation as part of its context.
4. _(Optional HITL gate — Phase 8)_: The compiled mission config is shown to the user before execution starts.
5. A LangGraph `MissionRunner` graph drives sequential task execution.
6. Each task runs as a `create_deep_agent` instance with explicitly compiled `CompiledSubAgent` workers.
7. Every subtask completion persists a `ResearchRun` snapshot to Mongo.
8. The final mission output and all artifacts are available via the REST API.

---

## 1. Architecture Overview (Updated)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   biotech-research-ingestion                         │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Coordinator Agent  (create_agent — UNCHANGED)                │  │
│  │  tools: web_search, create_research_plan                      │  │
│  │  HITL: interrupt() in create_research_plan → plan_ready WS   │  │
│  └────────────────────────┬──────────────────────────────────────┘  │
│                            │  approved ResearchPlan                  │
│                            ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Mission Compiler Agent  (create_agent, structured output)    │  │
│  │  ResearchPlan → ResearchMission  (LLM-driven)                 │  │
│  │                                                               │  │
│  │  System prompt includes:                                      │  │
│  │  - Full Deep Agents framework documentation                   │  │
│  │  - Available tool profiles + subagent role catalogue          │  │
│  │  - ResearchMission JSON schema                                │  │
│  │                                                               │  │
│  │  For each ResearchTask → produces:                            │  │
│  │  - TaskDef with detailed main agent system_prompt             │  │
│  │  - 1–3 CompiledSubAgentConfigs with distinct roles            │  │
│  │  - Input/output schema, acceptance criteria                   │  │
│  │  - Tool profile assignments                                   │  │
│  │                                                               │  │
│  │  Post-LLM: deterministic validation + topology build          │  │
│  │  (cycle detection, dependency_map, reverse_dependency_map)    │  │
│  │  Saves ResearchMission to Mongo.                              │  │
│  └────────────────────────┬──────────────────────────────────────┘  │
│                            │  ResearchMission                        │
│                            ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  MissionRunner  (LangGraph StateGraph)                        │  │
│  │                                                               │  │
│  │  load_mission → initialize_runtime_state                      │  │
│  │    → compute_ready_queue → select_next_task                   │  │
│  │      → execute_task  ←─── ONLY agentic node                  │  │
│  │        → merge_task_result → persist_research_run             │  │
│  │          → check_completion ──→ finalize_mission              │  │
│  │                          ↑                                    │  │
│  │                          └── loops via compute_ready_queue    │  │
│  └────────────────────────┬──────────────────────────────────────┘  │
│                            │                                         │
│                            ▼  per TaskDef                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Deep Agent Worker  (create_deep_agent)                       │  │
│  │  - own FilesystemBackend + /memories/ StoreBackend            │  │
│  │  - tavily tools + file-search tools                           │  │
│  │  - explicitly compiled CompiledSubAgents                      │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │  CompiledSubAgent  (create_agent + middleware)          │ │  │
│  │  │  - own FilesystemBackend (subagent workspace)           │ │  │
│  │  │  - FilesystemMiddleware (explicit, not inherited)       │ │  │
│  │  │  - tavily tools + file-search tools                     │ │  │
│  │  │  - optional TodoListMiddleware                          │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Storage: MongoDB (Beanie) · InMemoryStore (dev) · local filesystem  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Existing Code Inventory & What Changes

### 2.1 Files that are UNCHANGED

| File                                         | Status        | Note                                                                        |
| -------------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| `src/models/thread.py`                       | ✅ keep as-is | Thread document unchanged                                                   |
| `src/models/message.py`                      | ✅ keep as-is | Message document unchanged                                                  |
| `src/models/plan.py`                         | ✅ keep as-is | `ResearchPlan` is the Mission Compiler Agent input; no schema change needed |
| `src/agents/tools/tavily_search_tools.py`    | ✅ keep as-is | Used by task agents                                                         |
| `src/agents/tools/utils/tavily_functions.py` | ✅ keep as-is | Used by task agents                                                         |
| `src/agents/tools/web_search.py`             | ✅ keep as-is | Coordinator tool only                                                       |
| `src/api/routes/threads.py`                  | ✅ keep as-is | Thread CRUD unchanged                                                       |
| `src/api/routes/plans.py`                    | ⚠️ extend     | Add `POST /plans/{id}/launch` to trigger mission creation                   |
| `src/api/routes/health.py`                   | ✅ keep as-is | May add mission store to health check later                                 |
| `src/api/socketio/handlers.py`               | ⚠️ extend     | Add mission/run Socket.IO events                                            |

### 2.2 Files that CHANGE

#### `src/models/__init__.py`

- Add imports for `ResearchMission`, `TaskDef`, `ResearchRun` (new model) when those modules exist.
- Register all new Beanie documents in `init_beanie`.

#### `src/main.py`

- Add `ResearchMission` and the new `ResearchRun` to `init_beanie` document list.
- Add new router: `src/api/routes/missions.py`.

#### `src/agents/coordinator.py`

- **MINIMAL CHANGE**: update the `COORDINATOR_SYSTEM_PROMPT` to mention that after plan approval the system will automatically compile and launch a ResearchMission. No structural changes to the agent.
- The system prompt should tell the coordinator that after approval it can confirm to the user that execution is about to begin, but it does NOT launch the mission itself — that's triggered externally via `POST /plans/{id}/launch`.

#### `src/api/routes/plans.py`

- Add `POST /plans/{id}/launch` — validates plan is `approved`, triggers `ResearchMissionCreator`, returns `mission_id`.

### 2.3 Files that are DEPRECATED/REPLACED

| File                            | Disposition                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `src/models/openai_research.py` | Keep for now — OpenAI deep research path is separate. The new `ResearchRun` in `src/research/models/mission.py` is independent. |
| `src/infrastructure/temporal/`  | Keep as-is; Temporal is a different execution path. MissionRunner uses LangGraph, not Temporal.                                 |

---

## 3. New Directory Structure

```
src/
└── research/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── mission.py                  ← ALL new domain models
    ├── compiler/
    │   ├── __init__.py
    │   ├── mission_compiler_agent.py   ← LLM agent: ResearchPlan → ResearchMission draft
    │   ├── mission_creator.py          ← Orchestrates LLM call + post-LLM validation/save
    │   └── agent_compiler.py           ← compile_main_task_agent, compile_subagent (runtime)
    ├── runtime/
    │   ├── __init__.py
    │   ├── backends.py                 ← workspace paths + backend factories
    │   ├── tools.py                    ← tool profile resolution
    │   ├── task_executor.py            ← input resolution + agent invocation
    │   └── mission_runner.py           ← LangGraph StateGraph
    └── persistence/
        ├── __init__.py
        └── research_run_writer.py      ← idempotent Mongo persistence
```

New API routes:

```
src/api/routes/
└── missions.py                     ← GET /missions/{id}, GET /missions/{id}/runs
```

---

## 4. Domain Models — `src/research/models/mission.py`

All models in one file to keep imports simple in v1. Split into separate files if the file grows beyond ~400 lines.

### 4.1 InputBinding

```python
class InputBinding(BaseModel):
    """Declares how a task input is resolved from prior task outputs."""

    source_task_id: str
    # Which key in task_outputs[source_task_id] to pull from
    source_key: str
    # If False, task can start even if this binding resolves to None
    required: bool = True
    # Optional jmespath or dot-notation transform applied to the resolved value
    transform: str | None = None
```

### 4.2 MainDeepAgentConfig

```python
class MainDeepAgentConfig(BaseModel):
    """Configuration for the primary create_deep_agent for a TaskDef."""

    model_name: str = "openai:gpt-5.2"
    system_prompt: str
    # Key into ToolRegistry (resolved at runtime)
    tool_profile_name: str = "default_research"
    # Resolved at runtime to a FilesystemBackend root path
    filesystem_profile: str = "task_local"
    # Resolved at runtime to InMemoryStore (dev) or PostgresStore (prod)
    memory_profile: str = "in_memory"
    allow_general_purpose_subagent: bool = True
    max_iterations: int | None = None
    notes: dict[str, Any] = Field(default_factory=dict)
```

### 4.3 CompiledSubAgentConfig

```python
class CompiledSubAgentConfig(BaseModel):
    """Configuration for a compiled subagent worker attached to a task agent."""

    name: str
    description: str
    system_prompt: str
    model_name: str | None = None     # falls back to main agent model if None
    tool_profile_name: str = "default_research"
    filesystem_profile: str = "subagent_local"
    use_todo_middleware: bool = False
    memory_profile: str = "in_memory"
    # Appended to the task workspace path: tasks/{task_id}/subagents/{workspace_suffix}/
    workspace_suffix: str
    max_invocations: int = 1
```

### 4.4 TaskExecutionPolicy

```python
class TaskExecutionPolicy(BaseModel):
    timeout_seconds: int = 300
    max_retries: int = 1
    persist_run_after_completion: bool = True
```

### 4.5 TaskDef

```python
class TaskDef(BaseModel):
    """The schedulable runtime unit. One TaskDef = one Deep Agent invocation."""

    task_id: str
    name: str
    stage_label: str | None = None
    description: str
    depends_on: list[str] = Field(default_factory=list)
    # key = local input name; value = binding to prior task output
    input_bindings: dict[str, InputBinding] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    main_agent: MainDeepAgentConfig
    compiled_subagents: list[CompiledSubAgentConfig] = Field(default_factory=list)
    execution: TaskExecutionPolicy = Field(default_factory=TaskExecutionPolicy)
```

### 4.6 ResearchMission (Beanie Document)

```python
class ResearchMission(Document):
    """Fully compiled, executable mission. Stored in MongoDB."""

    research_plan_id: PydanticObjectId
    thread_id: PydanticObjectId
    title: str
    goal: str
    global_context: dict[str, Any] = Field(default_factory=dict)
    global_constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    task_defs: list[TaskDef] = Field(default_factory=list)
    # task_id → [dependency task_ids]
    dependency_map: dict[str, list[str]] = Field(default_factory=dict)
    # task_id → [task_ids that depend on this task]
    reverse_dependency_map: dict[str, list[str]] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "research_missions"
        indexes = [
            [("research_plan_id", 1)],
            [("thread_id", 1)],
            [("status", 1)],
            [("created_at", -1)],
        ]
```

### 4.7 ArtifactRef

```python
class ArtifactRef(BaseModel):
    """Reference to an artifact produced by a task."""

    task_id: str
    name: str
    artifact_type: str     # "report", "document", "json", "log"
    storage: Literal["filesystem", "mongo_inline"] = "filesystem"
    path: str | None = None          # local workspace path
    content_inline: str | None = None
    content_type: str = "text/plain"
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### 4.8 ResearchEvent

```python
class ResearchEvent(BaseModel):
    """An event emitted during task or mission execution."""

    event_type: str    # "task_started", "task_completed", "agent_token", etc.
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 4.9 TaskResult

```python
class TaskResult(BaseModel):
    """Normalized output of a single TaskDef execution. Never mutates mission state directly."""

    task_id: str
    status: Literal["completed", "failed"]
    # Namespaced by output key — merged into task_outputs[task_id]
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    events: list[ResearchEvent] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_number: int = 1
```

### 4.10 ResearchRun (Beanie Document)

This replaces the old `TaskRun` concept from `02-technical-specification.md` for the mission execution path. The OpenAI-specific `OpenAIResearchRun` remains unchanged.

```python
class ResearchRun(Document):
    """One task execution record. Written after every task completion."""

    mission_id: PydanticObjectId
    task_id: str
    attempt_number: int = 1
    status: Literal["completed", "failed"]
    resolved_inputs_snapshot: dict[str, Any] = Field(default_factory=dict)
    outputs_snapshot: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "research_runs"
        indexes = [
            # Unique per (mission, task, attempt) — enables idempotent upserts
            [("mission_id", 1), ("task_id", 1), ("attempt_number", 1)],
            [("mission_id", 1)],
            [("status", 1)],
        ]
```

---

## 5. Module Specifications

### 5.1 `src/research/compiler/mission_compiler_agent.py` + `mission_creator.py`

**Design:** Mission compilation is a two-step pipeline:

1. **`mission_compiler_agent.py`** — an LLM agent that reasons about the plan and produces a richly detailed `ResearchMission` draft (as a validated Pydantic object).
2. **`mission_creator.py`** — the orchestrator that calls the agent, applies post-LLM deterministic validation (cycle detection, topology build), saves to Mongo, and returns the final `ResearchMission`.

The LLM does the intellectual work (what subagents are needed, what their system prompts should be, what tools they need). Deterministic code does the structural work (verifying no cycles, building the `dependency_map`, enforcing schema constraints).

---

#### `mission_compiler_agent.py` — LLM Agent

**Purpose:** Accept an approved `ResearchPlan` and produce a fully-specified `ResearchMissionDraft` (a plain Pydantic model — not yet saved to Mongo).

**Implementation:** `create_agent` with **structured output mode** (not `create_deep_agent`). The agent makes a single LLM call and returns a validated structured object. No tool calls needed — all context is injected into the system prompt.

**Model:** `openai:gpt-5.2` (or configurable). Use structured output / response format mode for reliable schema adherence.

**System prompt sections:**

```
SECTION 1 — ROLE
You are a Deep Agent Mission Compiler. Your job is to read an approved research plan
and produce a fully executable ResearchMission — a detailed configuration of
Deep Agent workers and their specialized subagents.

SECTION 2 — DEEP AGENTS FRAMEWORK CONTEXT
[Injected at runtime from src/research/compiler/prompts/deep_agents_context.md]
Contents: explanation of create_deep_agent, create_agent, FilesystemMiddleware,
CompiledSubAgent, CompositeBackend, tool profiles, SubAgentMiddleware, TodoListMiddleware.
This gives the LLM enough understanding to make good architectural decisions.

SECTION 3 — AVAILABLE TOOL PROFILES
[Injected at runtime]
- default_research: Tavily search tools + file-search tools
- search_only: Tavily search tools only
- write_only: File-read/write utilities only

SECTION 4 — SUBAGENT ROLE CATALOGUE
[Injected at runtime from src/research/compiler/prompts/subagent_roles.md]
Named roles with descriptions, recommended tool profiles, and when to use them:
- source_finder: Discovers and validates primary sources. Use search_only profile.
- evidence_extractor: Extracts structured evidence from documents. Use default_research.
- verifier: Cross-checks claims against multiple sources. Use search_only.
- synthesizer: Integrates evidence into coherent summaries. Use write_only.
- writer: Produces final structured reports. Use write_only.
- domain_specialist: Deep-dives into a specific biotech subdomain. Use default_research.

SECTION 5 — OUTPUT CONTRACT
You MUST return a JSON object conforming to the ResearchMissionDraft schema below.
For each task in the plan, you MUST produce:
- A detailed main agent system_prompt (2–4 paragraphs) explaining its role, approach,
  how to use its filesystem, what to write to /outputs/, when to delegate to subagents.
- 1–3 CompiledSubAgentConfigs with distinct roles, specific system_prompts,
  and clear delegation criteria the main agent can use to decide when to call them.
- A list of acceptance_criteria strings the output will be judged against.
- input_bindings derived from the plan's task dependencies.
- output_schema describing expected output keys and their types.

Do not invent new task IDs. Use exactly the task IDs from the plan.
Do not add dependencies that are not in the plan.
```

**`ResearchMissionDraft` schema (Pydantic, not a Beanie Document):**

```python
class ResearchMissionDraft(BaseModel):
    """
    Structured output from the Mission Compiler Agent.
    Validated by Pydantic before post-processing. Not saved to Mongo directly.
    """
    title: str
    goal: str
    global_context: dict[str, Any]
    global_constraints: list[str]
    success_criteria: list[str]
    task_defs: list[TaskDef]

    model_config = ConfigDict(extra="forbid")   # reject hallucinated fields
```

**Public interface:**

```python
async def compile_mission_draft(
    plan: ResearchPlan,
    model_name: str = "openai:gpt-5.2",
) -> ResearchMissionDraft:
    """
    Call the Mission Compiler LLM Agent with the approved plan.
    Returns a validated ResearchMissionDraft.
    Raises MissionDraftValidationError if the LLM output fails Pydantic validation.
    """
```

**Implementation sketch:**

```python
async def compile_mission_draft(plan, model_name="openai:gpt-5.2"):
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    system_prompt = _build_compiler_system_prompt()  # loads prompt files + injects schemas
    user_message = _plan_to_compiler_prompt(plan)    # serializes plan as readable JSON + context

    model = init_chat_model(model_name).with_structured_output(ResearchMissionDraft)



    # Single-shot structured output call — no tool loop needed
    draft: ResearchMissionDraft = await model.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return draft
```

**Note on structured output vs. `create_agent`:** Since this is a single-turn structured output call (no tool calls, no loops), using `model.with_structured_output(ResearchMissionDraft)` directly is cleaner than wrapping it in `create_agent`. Use `create_agent` only if we later add tools (e.g., `web_search` to research what tools are appropriate for the domain).

---

#### `mission_creator.py` — Orchestrator

**Purpose:** Orchestrate the LLM call, apply deterministic post-processing, save to Mongo.

**Public interface:**

```python
async def create_mission_from_plan(
    plan: ResearchPlan,
    model_name: str = "openai:gpt-5.2",
) -> ResearchMission:
    """
    Full pipeline: approve-check → LLM compile → validate → topology build → save.
    Raises MissionCompilationError (or subclasses) on any failure.
    """
```

**Algorithm:**

```
1. Guard: plan.status == "approved" → else raise UnapprovedPlanError

2. LLM step:
   draft = await compile_mission_draft(plan, model_name)
   # draft is a validated ResearchMissionDraft

3. Structural validation (deterministic):
   a. All task_ids in draft.task_defs match plan.tasks ids (no hallucinated ids)
   b. All depends_on in every TaskDef reference only valid task_ids
   c. Cycle detection on the dependency graph (Kahn's algorithm)
      → raise CyclicDependencyError if cycles detected
   d. All required InputBindings reference existing (source_task_id, source_key) pairs
      → raise MissingInputError if not resolvable

4. Build topology:
   dependency_map = {td.task_id: td.depends_on for td in draft.task_defs}
   reverse_dependency_map = _invert_dependency_map(dependency_map)

5. Construct ResearchMission document:
   mission = ResearchMission(
       research_plan_id=plan.id,
       thread_id=plan.thread_id,
       title=draft.title,
       goal=draft.goal,
       global_context=draft.global_context,
       global_constraints=draft.global_constraints,
       success_criteria=draft.success_criteria,
       task_defs=draft.task_defs,
       dependency_map=dependency_map,
       reverse_dependency_map=reverse_dependency_map,
       status="pending",
   )

6. Save to Mongo:
   await mission.insert()

7. Return mission
```

**Error types:**

```python
class MissionCompilationError(Exception):
    """Base class for all mission compilation errors."""

class UnapprovedPlanError(MissionCompilationError):
    """Plan is not in approved status."""

class MissionDraftValidationError(MissionCompilationError):
    """LLM output failed Pydantic validation or schema constraint."""

class HallucinatedTaskIdError(MissionCompilationError):
    """LLM produced a task_id not present in the original plan."""

class CyclicDependencyError(MissionCompilationError):
    """LLM introduced or preserved a cyclic dependency in TaskDefs."""

class MissingInputError(MissionCompilationError):
    """A required InputBinding references a non-existent task/output key."""
```

**Retry policy:** If `MissionDraftValidationError` is raised, retry the LLM call up to 2 times with the validation error appended to the user message as feedback. If still failing after 3 attempts, raise and surface to the caller.

---

#### `src/research/compiler/prompts/` — Prompt files

```
src/research/compiler/prompts/
├── deep_agents_context.md      ← Framework documentation injected into system prompt
├── subagent_roles.md           ← Subagent role catalogue
└── mission_schema.md           ← Human-readable ResearchMissionDraft schema description
```

`deep_agents_context.md` content summary (written at implementation time):

- What `create_deep_agent` is and what harness features it provides
- What `create_agent` + `FilesystemMiddleware` does for subagents
- What `CompiledSubAgent(name, description, runnable=graph)` means
- How `CompositeBackend` with `/memories/` works
- What `TodoListMiddleware` adds
- The difference between the main agent's context and subagent context isolation
- When to assign `use_todo_middleware: true` (complex multi-step subtasks) vs. `false`

`subagent_roles.md` content summary:

- Named roles with purpose, recommended tool_profile_name, example system_prompt skeleton, and `use_todo_middleware` recommendation

These are **static files loaded at startup**, not generated at runtime. They encode the institutional knowledge about the framework that the LLM needs to make good decisions.

---

### 5.2 `src/research/runtime/backends.py`

**Purpose:** Path builders and backend factory functions. Pure functions, no side effects.

**Path layout:**

```
/tmp/research_missions/{mission_id}/
└── tasks/
    └── {task_id}/
        ├── workspace/
        ├── outputs/
        ├── scratch/
        └── subagents/
            └── {subagent_name}/
                ├── workspace/
                ├── outputs/
                └── scratch/
```

**Public interface:**

```python
def mission_root(mission_id: str) -> Path:
    return Path("/tmp/research_missions") / mission_id

def task_root(mission_id: str, task_id: str) -> Path:
    return mission_root(mission_id) / "tasks" / task_id

def subagent_root(mission_id: str, task_id: str, subagent_name: str) -> Path:
    return task_root(mission_id, task_id) / "subagents" / subagent_name

async def ensure_task_workspace(mission_id: str, task_id: str) -> Path:
    """Create workspace directories for a task. Returns task root path."""
    root = task_root(mission_id, task_id)
    for subdir in ("workspace", "outputs", "scratch"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root

async def ensure_subagent_workspace(
    mission_id: str, task_id: str, subagent_name: str
) -> Path:
    """Create workspace directories for a subagent. Returns subagent root."""
    root = subagent_root(mission_id, task_id, subagent_name)
    for subdir in ("workspace", "outputs", "scratch"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root

def build_task_backend(
    mission_id: str,
    task_id: str,
    store: InMemoryStore,
) -> Callable:
    """
    Return a CompositeBackend factory (callable that takes runtime).
    Routes /memories/ → StoreBackend(runtime), everything else → FilesystemBackend.
    """
    root = task_root(mission_id, task_id)

    def backend_factory(runtime):
        from deepagents.backends import FilesystemBackend, StoreBackend
        from deepagents.backends.composite import CompositeBackend
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(root)),
            routes={"/memories/": StoreBackend(runtime)},
        )

    return backend_factory

def build_subagent_backend(
    mission_id: str,
    task_id: str,
    subagent_name: str,
) -> FilesystemBackend:
    """
    Return a plain FilesystemBackend for a subagent.
    Subagents get their own isolated workspace, not a composite backend.
    """
    from deepagents.backends import FilesystemBackend
    root = subagent_root(mission_id, task_id, subagent_name)
    return FilesystemBackend(root_dir=str(root))
```

**Key design note:** The backend factory for the main task agent is a **callable** (not an instance), because `create_deep_agent(backend=...)` accepts either a backend instance or a callable that takes the runtime. For subagents, a plain `FilesystemBackend` instance is sufficient since `create_agent` with `FilesystemMiddleware` takes a backend instance directly.

---

### 5.3 `src/research/runtime/tools.py`

**Purpose:** Resolve tool profile names to lists of tool functions. This decouples `TaskDef` configs from concrete tool imports.

**Tool profiles (v1):**

```python
TOOL_PROFILES: dict[str, Callable[[], list]] = {
    "default_research": _build_default_research_tools,
    "search_only": _build_search_only_tools,
    "write_only": _build_write_only_tools,
}
```

**`_build_default_research_tools()`** returns:

- All tools from `src/agents/tools/tavily_search_tools.py` (the exported `@tool` functions)
- File search utility tools (to be defined — wrappers around `grep`/`glob` on the agent's workspace)

**`_build_search_only_tools()`** returns:

- Tavily search tools only (no file-write utilities)

**`_build_write_only_tools()`** returns:

- Filesystem utilities (read/write/format) but no web search

**Public interface:**

```python
def resolve_tool_profile(profile_name: str) -> list:
    """Resolve a profile name to a list of LangChain tool instances."""
    if profile_name not in TOOL_PROFILES:
        raise ValueError(f"Unknown tool profile: {profile_name!r}")
    return TOOL_PROFILES[profile_name]()
```

**Important:** Tools are instantiated fresh per call — do not cache stateful tool instances across task executions.

---

### 5.4 `src/research/compiler/agent_compiler.py`

**Purpose:** Build `create_deep_agent` and `CompiledSubAgent` instances from `TaskDef` configs and a runtime context.

**RuntimeContext:**

```python
@dataclass
class RuntimeContext:
    mission_id: str
    task_id: str
    store: InMemoryStore
```

**Public interface:**

```python
async def compile_main_task_agent(
    task_def: TaskDef,
    ctx: RuntimeContext,
) -> Any:
    """
    Build and return a create_deep_agent instance for the given TaskDef.
    Creates workspace directories. Returns a compiled, invocable agent.
    """

async def compile_subagent(
    config: CompiledSubAgentConfig,
    ctx: RuntimeContext,
) -> CompiledSubAgent:
    """
    Build a CompiledSubAgent wrapping a create_agent with explicit
    FilesystemMiddleware + tools. Never inherits from main agent.
    """
```

**`compile_main_task_agent` implementation:**

```python
async def compile_main_task_agent(task_def, ctx):
    from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model

    cfg = task_def.main_agent

    # 1. Ensure workspace exists
    await ensure_task_workspace(ctx.mission_id, ctx.task_id)

    # 2. Resolve tools
    tools = resolve_tool_profile(cfg.tool_profile_name)

    # 3. Build backend factory
    backend = build_task_backend(ctx.mission_id, ctx.task_id, ctx.store)

    # 4. Compile declared subagents
    subagents = [
        await compile_subagent(sub_cfg, ctx)
        for sub_cfg in task_def.compiled_subagents
    ]

    # 5. Build model
    model = init_chat_model(cfg.model_name)

    # 6. Assemble agent
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.system_prompt,
        backend=backend,
        store=ctx.store,
        subagents=subagents,
    )

    return agent
```

**`compile_subagent` implementation:**

```python
async def compile_subagent(config, ctx):
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents import CompiledSubAgent

    # 1. Ensure workspace
    await ensure_subagent_workspace(ctx.mission_id, ctx.task_id, config.name)

    # 2. Resolve tools (subagents get their own tool instances)
    tools = resolve_tool_profile(config.tool_profile_name)

    # 3. Build filesystem backend
    backend = build_subagent_backend(ctx.mission_id, ctx.task_id, config.name)

    # 4. Build middleware — FilesystemMiddleware is EXPLICIT, not inherited
    middleware = [FilesystemMiddleware(backend=backend)]
    if config.use_todo_middleware:
        from deepagents.middleware.todo import TodoListMiddleware
        middleware.append(TodoListMiddleware())

    # 5. Build model (fall back to main agent model if not specified)
    model_name = config.model_name or "openai:gpt-5.2"
    model = init_chat_model(model_name)

    # 6. Build the agent graph
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=config.system_prompt,
        middleware=middleware,
    )

    # 7. Wrap as CompiledSubAgent for attachment to main agent
    return CompiledSubAgent(
        name=config.name,
        description=config.description,
        runnable=agent_graph,
    )
```

**Critical rule:** `compile_subagent` must **never** use `create_deep_agent`. Subagents are built with `create_agent` and `FilesystemMiddleware`. This is per-spec: Deep Agents' middleware docs explicitly show `FilesystemMiddleware` being added via `create_agent(middleware=[FilesystemMiddleware(...)])`, and the `create_deep_agent` harness includes it by default only for the main agent.

---

### 5.5 `src/research/runtime/task_executor.py`

**Purpose:** Resolve task inputs, invoke the main Deep Agent, and normalize its result into a `TaskResult`.

**Public interface:**

```python
async def execute_task(
    task_def: TaskDef,
    task_outputs: dict[str, dict[str, Any]],
    ctx: RuntimeContext,
) -> TaskResult:
    """
    Full task execution pipeline:
    1. Resolve inputs from task_outputs using task_def.input_bindings
    2. Compile main task agent
    3. Build invocation message from resolved inputs
    4. Invoke agent (async)
    5. Normalize into TaskResult
    Returns TaskResult regardless of success/failure (errors are captured in result).
    """
```

**Input resolution logic:**

```python
def _resolve_inputs(
    bindings: dict[str, InputBinding],
    task_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved = {}
    for local_name, binding in bindings.items():
        source_data = task_outputs.get(binding.source_task_id, {})
        value = source_data.get(binding.source_key)
        if value is None and binding.required:
            raise InputResolutionError(
                f"Required input '{local_name}' could not be resolved from "
                f"task '{binding.source_task_id}' key '{binding.source_key}'"
            )
        if binding.transform and value is not None:
            value = _apply_transform(value, binding.transform)
        resolved[local_name] = value
    return resolved
```

**Agent invocation:**

- The resolved inputs are formatted into a user message as a structured JSON block (or Markdown table for readability).
- The agent is invoked with `await agent.ainvoke({"messages": [{"role": "user", "content": invocation_message}]}, config={"configurable": {"thread_id": f"task-{ctx.task_id}"}})`.
- The last assistant message in the result is used as the primary output.
- Artifacts are collected from the task workspace's `/outputs/` directory after completion.

**Result normalization:**

- On success: `TaskResult(task_id=..., status="completed", outputs={"response": last_msg_content, "files": artifact_list})`
- On exception: `TaskResult(task_id=..., status="failed", error_message=str(e))`
- Both paths capture `started_at` and `completed_at`.

---

### 5.6 `src/research/runtime/mission_runner.py`

**Purpose:** LangGraph `StateGraph` that drives sequential mission execution.

#### MissionRunnerState

```python
from typing import Annotated, Any
from operator import add
from typing_extensions import TypedDict

class MissionRunnerState(TypedDict):
    # Static mission data (set once during initialization)
    mission_id: str
    mission: ResearchMission
    task_defs_by_id: dict[str, TaskDef]
    dependency_map: dict[str, list[str]]
    reverse_dependency_map: dict[str, list[str]]

    # Mutable execution state
    task_statuses: dict[str, str]  # task_id → "pending"|"running"|"completed"|"failed"
    task_outputs: dict[str, dict[str, Any]]  # task_id → {output_key → value}

    # Append-only via reducers
    task_results: Annotated[list[TaskResult], add]
    artifacts: Annotated[list[ArtifactRef], add]
    events: Annotated[list[ResearchEvent], add]
    completed_task_ids: Annotated[list[str], add]
    failed_task_ids: Annotated[list[str], add]

    # Overwrite semantics (last write wins)
    ready_queue: list[str]
    current_task_id: str | None
    mission_status: str  # "pending"|"running"|"completed"|"failed"
    final_outputs: dict[str, Any]
```

**Why `Annotated[list, add]` for the accumulator fields:** LangGraph's state merge strategy uses the annotated reducer. `add` (i.e., list concatenation) ensures that appending to `task_results` from each node call accumulates correctly through checkpointed state. Without this, a partial state return from a node would overwrite the entire list.

**Why `dict` fields use overwrite semantics for `task_statuses` and `task_outputs`:** These are keyed dictionaries, not append-only lists. Merging is done explicitly in `merge_task_result` by returning a full updated copy of the dict.

#### Graph Nodes

**`load_mission(state) -> dict`**

```python
async def load_mission(state: MissionRunnerState) -> dict:
    mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))
    if not mission:
        raise MissionNotFoundError(state["mission_id"])
    return {"mission": mission}
```

**`initialize_runtime_state(state) -> dict`**

```python
async def initialize_runtime_state(state: MissionRunnerState) -> dict:
    mission = state["mission"]
    task_defs_by_id = {td.task_id: td for td in mission.task_defs}
    task_statuses = {td.task_id: "pending" for td in mission.task_defs}
    return {
        "task_defs_by_id": task_defs_by_id,
        "dependency_map": mission.dependency_map,
        "reverse_dependency_map": mission.reverse_dependency_map,
        "task_statuses": task_statuses,
        "task_outputs": {},
        "task_results": [],
        "artifacts": [],
        "events": [],
        "completed_task_ids": [],
        "failed_task_ids": [],
        "ready_queue": [],
        "current_task_id": None,
        "mission_status": "running",
        "final_outputs": {},
    }
```

**`compute_ready_queue(state) -> dict`**

```python
async def compute_ready_queue(state: MissionRunnerState) -> dict:
    ready = []
    for task_id, status in state["task_statuses"].items():
        if status != "pending":
            continue
        deps = state["dependency_map"].get(task_id, [])
        if not all(state["task_statuses"].get(d) == "completed" for d in deps):
            continue
        # Validate required input bindings resolve
        task_def = state["task_defs_by_id"][task_id]
        if not _all_required_inputs_resolvable(task_def, state["task_outputs"]):
            continue
        ready.append(task_id)
    # Stable order: preserve declaration order from mission.task_defs
    ordered = [td.task_id for td in state["mission"].task_defs if td.task_id in ready]
    return {"ready_queue": ordered}
```

**`select_next_task(state) -> dict`**

```python
async def select_next_task(state: MissionRunnerState) -> dict:
    queue = state["ready_queue"]
    if not queue:
        return {"current_task_id": None}
    return {"current_task_id": queue[0]}
```

**`execute_task(state, config) -> dict`** ← Only agentic node

```python
async def execute_task(state: MissionRunnerState, config: RunnableConfig) -> dict:
    task_id = state["current_task_id"]
    if not task_id:
        return {}

    task_def = state["task_defs_by_id"][task_id]
    store = _get_store()  # module-level InMemoryStore singleton
    ctx = RuntimeContext(
        mission_id=state["mission_id"],
        task_id=task_id,
        store=store,
    )

    result = await execute_task_def(task_def, state["task_outputs"], ctx)

    return {
        "task_results": [result],
        "events": result.events,
    }
```

**`merge_task_result(state) -> dict`**

```python
async def merge_task_result(state: MissionRunnerState) -> dict:
    # The last result in task_results is the one we just completed
    if not state["task_results"]:
        return {}
    result = state["task_results"][-1]
    task_id = result.task_id

    new_statuses = dict(state["task_statuses"])
    new_outputs = dict(state["task_outputs"])

    if result.status == "completed":
        new_statuses[task_id] = "completed"
        new_outputs[task_id] = result.outputs
        return {
            "task_statuses": new_statuses,
            "task_outputs": new_outputs,
            "artifacts": result.artifacts,
            "completed_task_ids": [task_id],
        }
    else:
        new_statuses[task_id] = "failed"
        return {
            "task_statuses": new_statuses,
            "failed_task_ids": [task_id],
        }
```

**`persist_research_run(state) -> dict`**

```python
async def persist_research_run(state: MissionRunnerState) -> dict:
    if not state["task_results"]:
        return {}
    result = state["task_results"][-1]
    writer = ResearchRunWriter()
    await writer.upsert_run(
        mission_id=state["mission_id"],
        task_result=result,
        resolved_inputs=state["task_outputs"].get(result.task_id, {}),
    )
    return {}
```

**`check_completion(state) -> str`** ← Routing function (not a node)

```python
def check_completion(state: MissionRunnerState) -> str:
    total = len(state["task_defs_by_id"])
    done = len(state["completed_task_ids"]) + len(state["failed_task_ids"])
    if done >= total:
        return "finalize_mission"
    if not state["ready_queue"] and state["current_task_id"] is None:
        # No progress possible — stalled
        return "finalize_mission"
    return "compute_ready_queue"
```

**`finalize_mission(state) -> dict`**

```python
async def finalize_mission(state: MissionRunnerState) -> dict:
    failed = state["failed_task_ids"]
    status = "failed" if failed else "completed"
    mission = state["mission"]
    mission.status = status
    mission.updated_at = datetime.utcnow()
    await mission.save()
    final_outputs = {
        tid: state["task_outputs"].get(tid, {})
        for tid in state["completed_task_ids"]
    }
    return {"mission_status": status, "final_outputs": final_outputs}
```

#### Graph Assembly

```python
def build_mission_runner() -> CompiledStateGraph:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    builder = StateGraph(MissionRunnerState)

    # Add nodes
    builder.add_node("load_mission", load_mission)
    builder.add_node("initialize_runtime_state", initialize_runtime_state)
    builder.add_node("compute_ready_queue", compute_ready_queue)
    builder.add_node("select_next_task", select_next_task)
    builder.add_node("execute_task", execute_task)
    builder.add_node("merge_task_result", merge_task_result)
    builder.add_node("persist_research_run", persist_research_run)
    builder.add_node("finalize_mission", finalize_mission)

    # Linear flow
    builder.set_entry_point("load_mission")
    builder.add_edge("load_mission", "initialize_runtime_state")
    builder.add_edge("initialize_runtime_state", "compute_ready_queue")
    builder.add_edge("compute_ready_queue", "select_next_task")
    builder.add_edge("select_next_task", "execute_task")
    builder.add_edge("execute_task", "merge_task_result")
    builder.add_edge("merge_task_result", "persist_research_run")

    # Conditional routing after persistence
    builder.add_conditional_edges(
        "persist_research_run",
        check_completion,
        {
            "compute_ready_queue": "compute_ready_queue",
            "finalize_mission": "finalize_mission",
        },
    )
    builder.add_edge("finalize_mission", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton
_runner: CompiledStateGraph | None = None

def get_mission_runner() -> CompiledStateGraph:
    global _runner
    if _runner is None:
        _runner = build_mission_runner()
    return _runner
```

**Invocation pattern:**

```python
async def run_mission(mission_id: str) -> dict:
    runner = get_mission_runner()
    config = {"configurable": {"thread_id": mission_id}}
    result = await runner.ainvoke(
        {"mission_id": mission_id},
        config=config,
    )
    return result
```

---

### 5.7 `src/research/persistence/research_run_writer.py`

**Purpose:** Idempotent Mongo persistence of task execution results.

```python
class ResearchRunWriter:
    async def upsert_run(
        self,
        mission_id: str,
        task_result: TaskResult,
        resolved_inputs: dict[str, Any],
    ) -> ResearchRun:
        """
        Insert or update a ResearchRun document for (mission_id, task_id, attempt_number).
        Uses Beanie's upsert with a unique filter to ensure idempotency.
        LangGraph may replay this node on resume — the upsert must be safe to call twice.
        """
        doc = ResearchRun(
            mission_id=PydanticObjectId(mission_id),
            task_id=task_result.task_id,
            attempt_number=task_result.attempt_number,
            status=task_result.status,
            resolved_inputs_snapshot=resolved_inputs,
            outputs_snapshot=task_result.outputs,
            artifacts=task_result.artifacts,
            error_message=task_result.error_message,
            started_at=task_result.started_at,
            completed_at=task_result.completed_at,
        )
        # Upsert by (mission_id, task_id, attempt_number) — unique index enforces this
        existing = await ResearchRun.find_one({
            "mission_id": PydanticObjectId(mission_id),
            "task_id": task_result.task_id,
            "attempt_number": task_result.attempt_number,
        })
        if existing:
            existing.status = doc.status
            existing.outputs_snapshot = doc.outputs_snapshot
            existing.artifacts = doc.artifacts
            existing.error_message = doc.error_message
            existing.completed_at = doc.completed_at
            await existing.save()
            return existing
        await doc.insert()
        return doc
```

**Why idempotent:** LangGraph's `MemorySaver` checkpointer replays nodes when resuming from a checkpoint. If the process crashed between `execute_task` and `persist_research_run`, LangGraph may re-enter `persist_research_run` with the same state. The upsert must produce the same result regardless of how many times it's called.

---

## 6. New API Routes

### `src/api/routes/plans.py` — Add `POST /plans/{id}/launch`

```python
@router.post("/{plan_id}/launch", status_code=status.HTTP_202_ACCEPTED)
async def launch_plan(plan_id: str) -> dict:
    """
    Trigger ResearchMissionCreator for an approved plan.
    Returns mission_id immediately. Mission execution runs in the background.
    """
    plan = await _get_plan_or_404(plan_id)
    if plan.status != "approved":
        raise HTTPException(status_code=400, detail="Plan must be approved before launch")

    mission = await create_mission_from_plan(plan)

    # Fire-and-forget: launch in background task
    # asyncio.create_task is safe here because FastAPI's event loop outlives the request
    asyncio.create_task(_run_mission_bg(str(mission.id)))

    plan.status = "executing"
    plan.updated_at = datetime.utcnow()
    await plan.save()

    return envelope({"mission_id": str(mission.id), "status": "accepted"})


async def _run_mission_bg(mission_id: str) -> None:
    try:
        await run_mission(mission_id)
    except Exception:
        logger.exception("Mission %s failed", mission_id)
```

### `src/api/routes/missions.py` — New file

```python
@router.get("/{mission_id}")
async def get_mission(mission_id: str) -> dict:
    """Return current mission state."""

@router.get("/{mission_id}/runs")
async def get_mission_runs(mission_id: str) -> dict:
    """Return all ResearchRun documents for a mission."""
```

---

## 7. Architecture Decision Records

### ADR-006: Mission Compilation is LLM-Driven, Not Deterministic

**Decision:** The `ResearchPlan → ResearchMission` transformation uses an LLM agent (Mission Compiler Agent) to produce agent configurations, subagent configs, system prompts, and tool profiles. Only structural validation (cycle detection, schema enforcement) is deterministic.

**Rationale:** A `ResearchPlan` contains task titles, descriptions, and dependency IDs. It does not contain — and should not contain — deep agent engineering decisions such as subagent roles, system prompt content, tool profile assignments, acceptance criteria, and delegation strategies. These require reasoning about:

- The semantics of the research task (what kind of specialized workers would help?)
- The Deep Agents framework (what does `FilesystemMiddleware` buy a subagent? When is `TodoListMiddleware` useful?)
- The flow of information through the mission (what does task B need from task A?)

Encoding this as rule-based Python would require a brittle heuristic engine that would need constant tuning. An LLM with the framework documentation in context produces dramatically better configs and can adapt to unusual research topologies.

**Why not just enhance the Coordinator to generate missions directly:** The Coordinator focuses on research planning — understanding the user's objective, gathering context, and breaking work into logical stages. Asking it to simultaneously reason about Deep Agent architecture introduces cognitive overload and conflates two very different concerns. The Mission Compiler Agent is a specialist that runs after human approval of the research plan, with a system prompt specifically designed for agent engineering.

**Why not Option 2 (enrich `ResearchPlan` schema):** Making `ResearchPlan` carry full subagent configs and agent system prompts would make the human approval step (HITL) review a wall of technical configuration rather than a readable research plan. The plan the user sees and approves should remain human-readable. The agent config is an implementation detail of execution.

**Retry with feedback:** If the LLM output fails Pydantic validation or structural checks, the error is fed back to the LLM for up to 2 retries before propagating as a hard error.

---

### ADR-007: Sequential Execution First (v1)

**Decision:** The MissionRunner runs one task at a time. `select_next_task` always picks the first ready task.

**Rationale:** Sequential execution is correct, debuggable, and testable. LangGraph's `StateGraph` already supports concurrent super-steps via `Send` API, but that requires carefully designed reducers and adds complexity. The current reducer design (overwrite dict for `task_statuses`/`task_outputs`, `add` for lists) is safe for sequential execution. Parallelism can be added in v2 by introducing a `parallel_execute_tasks` super-step node that uses `Send`.

---

### ADR-008: Subagents Built with `create_agent`, Not `create_deep_agent`

**Decision:** Every compiled subagent uses `langchain.agents.create_agent` with explicit `FilesystemMiddleware`.

**Rationale:** Deep Agents docs: "For custom subagents built with langchain.create_agent, explicitly add FilesystemMiddleware." The `FilesystemMiddleware` docs explicitly show this pattern: `create_agent(model=..., middleware=[FilesystemMiddleware(backend=...)])`. Using `create_deep_agent` for subagents would add recursive planning, todo tools, and subagent-spawning to each subagent, which bloats context and introduces uncontrolled recursion.

**Source:** Deep Agents middleware docs, compiled subagent example in the spec above.

---

### ADR-009: CompiledSubAgent via `runnable=agent_graph`

**Decision:** Wrap `create_agent` output in `CompiledSubAgent(name=..., description=..., runnable=...)`.

**Rationale:** This is the `CompiledSubAgent` pattern from the Deep Agents docs. It allows pre-built graphs to be used as subagents without further wrapping. The description string is what the main agent uses to decide when to delegate.

**Source:**

```python
# From Deep Agents docs:
custom_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="Specialized agent for complex data analysis tasks",
    runnable=custom_graph
)
```

---

### ADR-010: CompositeBackend with `/memories/` Route

**Decision:** Main task agents get a `CompositeBackend` routing `/memories/` to `StoreBackend(runtime)` and everything else to `FilesystemBackend(root_dir=task_root)`.

**Rationale:** This is the canonical pattern from the Deep Agents backends docs. It gives the agent both task-local durable files (inspectable on disk) and a cross-thread memory namespace. In v1 the `InMemoryStore` is ephemeral but the routing is in place for production migration to `PostgresStore`.

**Source:** Deep Agents backends doc, `CompositeBackend` section.

---

### ADR-011: Subagents Get Plain `FilesystemBackend`, Not Composite

**Decision:** Compiled subagents get `FilesystemBackend(root_dir=subagent_root)` with no `/memories/` route.

**Rationale:** Subagents are short-lived workers with bounded context. They do not need cross-thread memory — their job is to complete a focused subtask and return concise results. Adding `StoreBackend` routing to subagents adds complexity with no benefit in v1.

---

### ADR-012: ResearchRun is Separate from OpenAIResearchRun

**Decision:** The new `ResearchRun` document in `src/research/models/mission.py` is independent of `OpenAIResearchRun` in `src/models/openai_research.py`.

**Rationale:** These are two different execution paths (LangGraph Deep Agent mission vs. OpenAI native deep research). Keeping them separate avoids schema coupling and allows each path to evolve independently.

---

### ADR-013: `task_outputs` Keyed by `task_id`, Never Flattened

**Decision:** All task outputs live in `state["task_outputs"][task_id][output_key]`. No global flat output dict.

**Rationale:** Preserves provenance. When multiple tasks produce a field named `"summary"`, they don't collide. Input bindings explicitly reference `source_task_id` + `source_key`, making data flow auditable. This also makes the state schema deterministic regardless of task execution order.

---

### ADR-014: `persist_research_run` is a Separate Node

**Decision:** Mongo writes happen in `persist_research_run`, not inside `execute_task` or `merge_task_result`.

**Rationale:** LangGraph's durable execution model replays nodes on resume. If a side effect (Mongo write) is inside `execute_task`, it may be re-executed incorrectly on replay. Isolating it in a dedicated node with idempotent upsert logic makes it safe to replay. This is the explicit recommendation in LangGraph's durable execution docs.

---

### ADR-015: `model_name = "openai:gpt-5.2"` as Default

**Decision:** Use `init_chat_model("openai:gpt-5.2")` as the default for main task agents.

**Rationale:** Per spec: "gpt-5.2 is the most capable general-purpose model in that family and replaces gpt-5.1 for broad, agentic tasks." `init_chat_model` accepts `provider:model` format and is the canonical LangChain model factory.

---

### ADR-016: `asyncio.create_task` for Background Mission Execution

**Decision:** `POST /plans/{id}/launch` fires the mission runner as a background `asyncio.Task`.

**Rationale:** FastAPI runs on a single asyncio event loop. `asyncio.create_task` schedules the mission coroutine concurrently without blocking the HTTP response. This is appropriate for v1 single-process deployments. For multi-process production deployments, this should be replaced with a Celery task or a background worker queue (Temporal or ARQ). The v1 approach is acceptable because the spec explicitly says "do not introduce multiprocessing yet."

---

## 8. Changes to Existing Files Summary

### `src/models/__init__.py`

```python
# Add to existing imports
from src.research.models.mission import ResearchMission, ResearchRun

__all__ = ["Thread", "Message", "ResearchMission", "ResearchRun"]
```

### `src/main.py` — `init_beanie` registration

```python
await init_beanie(
    database=client[settings.MONGODB_DB],
    document_models=[
        Thread,
        Message,
        ResearchPlan,
        ResearchMission,   # NEW
        ResearchRun,       # NEW
        # OpenAI path — unchanged
        OpenAIResearchPlan,
        OpenAIResearchRun,
    ],
)
```

### `src/main.py` — Include new router

```python
from src.api.routes.missions import router as missions_router
app.include_router(missions_router, prefix="/api/v1")
```

### `src/agents/coordinator.py`

- Update `COORDINATOR_SYSTEM_PROMPT` to add a sentence at the end:
  > "After a plan is approved, inform the user that the system will automatically compile and execute the research mission. You do not trigger execution yourself."
- No structural changes to the agent.

### `src/api/routes/plans.py`

- Add `POST /plans/{id}/launch` endpoint (see Section 6).
- Add import for `create_mission_from_plan` and `run_mission`.

---

## 9. Tool Profile Registry — v1 Tool Mapping

| Profile Name       | Tavily Tools                                   | File Search | Notes                                           |
| ------------------ | ---------------------------------------------- | ----------- | ----------------------------------------------- |
| `default_research` | All Tavily tools from `tavily_search_tools.py` | Yes         | Default for main task agents and most subagents |
| `search_only`      | All Tavily tools                               | No          | For source-finder subagents                     |
| `write_only`       | None                                           | Yes         | For writer subagents                            |

**File search tools (v1 stub):** Thin wrappers that call Python's `pathlib.glob` and `re.search` against the agent's workspace root. These are NOT the Deep Agent built-in filesystem tools (those come from `FilesystemMiddleware`). File search tools allow the agent to search content within its workspace using LangChain `@tool` decorated functions.

---

## 10. Subagent Role Catalogue (LLM Guidance)

These are the named roles provided to the Mission Compiler Agent as a reference catalogue in `subagent_roles.md`. The LLM selects and configures them intelligently per task — it is not forced to use them, but they represent validated patterns.

| Role Name            | Purpose                                                                                       | Recommended Tool Profile | `use_todo_middleware` |
| -------------------- | --------------------------------------------------------------------------------------------- | ------------------------ | --------------------- |
| `source_finder`      | Discovers and validates primary sources via targeted web search                               | `search_only`            | True                  |
| `evidence_extractor` | Extracts structured evidence claims from gathered documents                                   | `default_research`       | True                  |
| `verifier`           | Cross-checks key claims against independent sources                                           | `search_only`            | False                 |
| `synthesizer`        | Integrates evidence into coherent structured summaries                                        | `write_only`             | False                 |
| `writer`             | Produces the final structured report or document                                              | `write_only`             | False                 |
| `domain_specialist`  | Deep-dives into a specific biotech subdomain (e.g., CRISPR mechanisms, clinical trial design) | `default_research`       | True                  |

The LLM is instructed to:

- Assign 1–3 subagents per TaskDef based on task complexity and stage
- Write distinct, specific system prompts per subagent (not generic "you are a researcher" prompts)
- Include in each main agent system prompt explicit instructions on **when** to delegate to each subagent — this is what makes the `description` field on `CompiledSubAgentConfig` effective
- Prefer `source_finder` for the first task in a dependency chain, `writer`/`synthesizer` for terminal tasks
- Only assign `use_todo_middleware: true` for subagents that will perform multi-step, iterative work

---

## 11. Testing Plan

### 11.1 Unit Tests

**`tests/research/test_mission_compiler_agent.py`**

- `test_compile_mission_draft_returns_valid_draft` — mock LLM returns valid JSON → Pydantic validates successfully
- `test_compile_mission_draft_retries_on_validation_error` — mock LLM returns invalid JSON on first call, valid on second → verifies retry with feedback appended
- `test_compile_mission_draft_raises_after_max_retries` — mock LLM returns invalid JSON 3× → raises `MissionDraftValidationError`
- `test_draft_contains_subagents_for_each_task`
- `test_draft_system_prompts_are_non_empty`

**`tests/research/test_mission_creator.py`**

- `test_creates_mission_from_valid_plan` — mocked `compile_mission_draft` returns a valid draft → mission saved to Mongo
- `test_raises_on_unapproved_plan`
- `test_raises_on_hallucinated_task_id` — LLM draft has a task_id not in the original plan → `HallucinatedTaskIdError`
- `test_raises_on_cyclic_dependencies` — LLM draft introduces a cycle → `CyclicDependencyError`
- `test_dependency_map_is_correct_for_two_task_chain`
- `test_input_bindings_preserved_from_draft`

**`tests/research/test_agent_compiler.py`**

- `test_compile_subagent_has_filesystem_middleware` — asserts `FilesystemMiddleware` in middleware list
- `test_compile_subagent_does_not_use_create_deep_agent` — the returned object must be a `CompiledSubAgent`
- `test_compile_main_agent_attaches_subagents` — main agent has compiled subagents in its config
- `test_subagent_gets_own_workspace` — workspace directories are created and not shared with main

**`tests/research/test_task_executor.py`**

- `test_resolves_inputs_from_prior_outputs`
- `test_raises_on_missing_required_input`
- `test_returns_failed_result_on_agent_exception`

**`tests/research/test_mission_runner.py`**

- `test_two_task_sequential_dependency_chain` — task B only runs after task A completes
- `test_merge_task_result_updates_statuses_and_outputs`
- `test_check_completion_routes_to_finalize_when_all_done`
- `test_check_completion_routes_to_loop_when_tasks_remain`
- `test_failed_task_captured_in_failed_ids`

**`tests/research/test_research_run_writer.py`**

- `test_upsert_inserts_on_first_call`
- `test_upsert_does_not_duplicate_on_second_call`
- `test_upsert_updates_status_on_second_call`

### 11.2 Integration Test Fixture

A minimal 2-task mission fixture for integration testing:

```python
# tests/research/fixtures/two_task_mission.py

def make_two_task_plan() -> ResearchPlan:
    """Returns an approved plan with two tasks: A → B (B depends on A)."""
    return ResearchPlan(
        thread_id=PydanticObjectId(),
        title="Test Plan",
        objective="Test objective",
        stages=["Stage 1", "Stage 2"],
        tasks=[
            ResearchTask(
                id="task_a",
                title="Task A",
                description="First task",
                stage="Stage 1",
                agent_config=AgentConfig(system_prompt="You are task A."),
                outputs=[TaskOutputSpec(name="result", type="text")],
            ),
            ResearchTask(
                id="task_b",
                title="Task B",
                description="Second task",
                stage="Stage 2",
                agent_config=AgentConfig(system_prompt="You are task B."),
                dependencies=["task_a"],
                inputs=[TaskInputRef(
                    name="prior_result",
                    source="task_output",
                    source_task_id="task_a",
                    output_name="result",
                )],
            ),
        ],
        status="approved",
    )
```

---

## 12. Implementation Phases (Updated)

This section replaces Phase 4 and Phase 5 from `04-execution-plan.md` for the mission execution layer. Phases 0–3 remain unchanged. Phases 6–9 remain unchanged.

### Phase 4 — Mission Models + Mission Compiler Agent

**Goal:** `ResearchPlan` → `ResearchMission` via LLM. Models saved to Mongo.

| #    | Task                                                                                                                     | Effort |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | ------ |
| 4.1  | `src/research/models/mission.py` — all domain models (`ResearchMission`, `TaskDef`, `ResearchMissionDraft`, etc.)        | M      |
| 4.2  | Register `ResearchMission`, `ResearchRun` in `init_beanie`                                                               | S      |
| 4.3  | `src/research/compiler/prompts/deep_agents_context.md` — framework documentation for LLM                                 | M      |
| 4.4  | `src/research/compiler/prompts/subagent_roles.md` — role catalogue for LLM                                               | S      |
| 4.5  | `src/research/compiler/prompts/mission_schema.md` — human-readable schema for LLM                                        | S      |
| 4.6  | `src/research/compiler/mission_compiler_agent.py` — `compile_mission_draft()` with structured output                     | M      |
| 4.7  | `src/research/compiler/mission_creator.py` — `create_mission_from_plan()`: LLM call + structural validation + Mongo save | M      |
| 4.8  | Cycle detection + hallucination guards (Kahn's algorithm, task_id validation)                                            | M      |
| 4.9  | Retry-with-feedback logic for LLM validation failures                                                                    | S      |
| 4.10 | Unit tests: LLM draft validation, cycle detection, task_id guard, unapproved plan guard                                  | M      |
| 4.11 | `POST /plans/{id}/launch` route (compile + save mission, no execution yet)                                               | S      |

**Milestone:** An approved plan triggers the Mission Compiler Agent → a `ResearchMission` with per-task agent configs and subagent configs is saved to Mongo. Invalid LLM outputs are retried with feedback. Structural errors return HTTP 400.

---

### Phase 5 — Backend + Tool Infrastructure

**Goal:** Workspace paths work, backends instantiate, tools resolve correctly.

| #   | Task                                                                            | Effort |
| --- | ------------------------------------------------------------------------------- | ------ |
| 5.1 | `src/research/runtime/backends.py` — path builders + backend factories          | M      |
| 5.2 | `src/research/runtime/tools.py` — tool profile registry                         | M      |
| 5.3 | Wire Tavily tools from `tavily_search_tools.py` into `default_research` profile | S      |
| 5.4 | File search tool stubs (v1: simple `glob`/`grep` wrappers)                      | S      |
| 5.5 | Unit tests for backends: paths correct, `CompositeBackend` routes correctly     | M      |

**Milestone:** `resolve_tool_profile("default_research")` returns tools. Backend factories produce valid instances. Workspace directories created correctly.

---

### Phase 6 — Agent Compiler + Task Executor

**Goal:** `TaskDef` compiles to a runnable Deep Agent with CompiledSubAgents.

| #   | Task                                                                                      | Effort |
| --- | ----------------------------------------------------------------------------------------- | ------ |
| 6.1 | `src/research/compiler/agent_compiler.py` — `compile_main_task_agent`                     | M      |
| 6.2 | `compile_subagent` — `create_agent` + `FilesystemMiddleware` + `CompiledSubAgent`         | M      |
| 6.3 | `src/research/runtime/task_executor.py` — `execute_task_def`                              | M      |
| 6.4 | Input resolution logic + `InputResolutionError`                                           | S      |
| 6.5 | Unit tests: subagent has FilesystemMiddleware, main agent has subagents, input resolution | M      |

**Milestone:** A `TaskDef` compiles to a `create_deep_agent` instance with CompiledSubAgents attached. Invoking it returns a `TaskResult`.

---

### Phase 7 — MissionRunner LangGraph

**Goal:** Full mission executes sequentially through the LangGraph graph.

| #   | Task                                                                  | Effort |
| --- | --------------------------------------------------------------------- | ------ |
| 7.1 | `src/research/runtime/mission_runner.py` — state schema + all nodes   | L      |
| 7.2 | Graph assembly + `build_mission_runner()`                             | M      |
| 7.3 | `src/research/persistence/research_run_writer.py` — idempotent upsert | M      |
| 7.4 | `POST /plans/{id}/launch` — background task invocation                | S      |
| 7.5 | `src/api/routes/missions.py` — GET mission, GET runs                  | S      |
| 7.6 | Integration test: two-task dependency chain executes in order         | M      |
| 7.7 | Integration test: task result merges correctly into state             | M      |

**Milestone:** An approved plan launches, the mission runner executes tasks in dependency order, ResearchRun documents appear in Mongo after each task, final_outputs assembled on completion.

---

### Phase 8 — Socket.IO Mission Events (was: Artifact Persistence)

**Goal:** Frontend receives live mission/task events over Socket.IO.

| #   | Task                                                                                               | Effort |
| --- | -------------------------------------------------------------------------------------------------- | ------ |
| 8.1 | Add `mission_started`, `task_started`, `task_completed`, `task_failed`, `mission_completed` events | M      |
| 8.2 | Emit events from `execute_task` and `finalize_mission` nodes                                       | M      |
| 8.3 | Frontend: mission monitor page consuming events                                                    | L      |

---

## 13. Async Policy

Per the user's requirement: **always use async methods and the event loop for I/O operations.**

- All Beanie operations: `await doc.save()`, `await Doc.find_one(...)`, `await Doc.insert()` ✅
- All filesystem operations: `await asyncio.to_thread(path.mkdir, ...)` or `aiofiles` for file reads ✅
- All agent invocations: `await agent.ainvoke(...)`, `await runner.ainvoke(...)` ✅
- No `asyncio.run()` inside async contexts ✅
- `asyncio.create_task` for background work, not `threading.Thread` ✅

---

## 14. Dependencies to Add **Already Added**

```bash
uv add deepagents langgraph langchain langchain-openai langchain-core
# deepagents already pulled in by langgraph if using LangChain ecosystem
# Verify deepagents is pinned to a version with CompiledSubAgent + FilesystemMiddleware
```

Ensure `pyproject.toml` includes:

```toml
[tool.uv.dependencies]
deepagents = ">=0.1.0"
langgraph = ">=0.2.0"
langchain = ">=0.3.0"
langchain-openai = ">=0.2.0"
```

---

## 15. Open Questions (Resolve Before Implementation)

| #   | Question                                                                                                      | Recommendation                                                                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | Does the current `deepagents` package version expose `CompiledSubAgent`?                                      | Run `python -c "from deepagents import CompiledSubAgent; print('ok')"` to verify                                                                                             |
| Q2  | Does `FilesystemMiddleware` accept `backend=<FilesystemBackend instance>` directly?                           | Check deep agents middleware source or docs — backend param may need to be a factory                                                                                         |
| Q3  | Does `create_agent` from `langchain.agents` accept `middleware=` in the currently installed version?          | Verify import: `from langchain.agents import create_agent`                                                                                                                   |
| Q4  | Is `init_chat_model("openai:gpt-5.2")` the correct model string?                                              | Verify with OpenAI API or fall back to `"openai:gpt-4o"` if gpt-5.2 is not yet available                                                                                     |
| Q5  | Should `run_mission` stream events back via Socket.IO in v1?                                                  | No — keep v1 simple (poll via GET /missions/{id}/runs). Add Socket.IO streaming in Phase 8.                                                                                  |
| Q6  | Does `model.with_structured_output(ResearchMissionDraft)` work reliably for the large nested schema?          | Test with a sample plan before committing to this approach. Fallback: use JSON mode + manual `ResearchMissionDraft.model_validate(json.loads(response.content))`.            |
| Q7  | Should the Mission Compiler Agent output be shown to the user for review (second HITL gate) before execution? | Recommended for v1.1, not v1.0. The mission config is highly technical. Add a "preview compiled mission" step in Phase 8 if stakeholders want visibility into agent configs. |
| Q8  | What is the token budget for the Mission Compiler system prompt (framework docs + schema)?                    | Estimate: ~3000 tokens for `deep_agents_context.md` + `subagent_roles.md` + schema. Well within gpt-5.2 context window. Compress if needed by removing examples.             |

---

_End of finalized execution plan._
