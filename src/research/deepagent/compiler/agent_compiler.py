"""Agent Compiler — builds create_deep_agent and CompiledSubAgent from TaskDef configs.

compile_main_task_agent: TaskDef → create_deep_agent instance with subagents
compile_subagent: CompiledSubAgentConfig → CompiledSubAgent wrapping create_agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import BaseChatModel, init_chat_model

from src.research.models.mission import (
    CompiledSubAgentConfig,
    TaskDef,
    TaskExecutionStructuredOutput,
)
from src.research.middleware.progress_middleware import (
    ProgressCallback,
    ResearchProgressMiddleware,
)
from src.research.middleware.source_tracking import (
    SourceTrackingMiddleware,
    write_source_index,
)
from src.research.middleware.quality_validation import QualityValidationMiddleware
from src.research.middleware.subagent_streaming import SubagentStreamingMiddleware
from src.research.middleware.artifact_collection import ArtifactCollectionMiddleware
from src.research.middleware.token_tracking import track_tokens
from src.research.runtime.backends import (
    build_subagent_backend,
    build_task_backend,
    ensure_subagent_workspace,
    ensure_task_workspace,
)
from src.research.runtime.tools import resolve_tool_profile

logger = logging.getLogger(__name__)

_TIER_MAP = {
    "fast": "openai:gpt-5-mini",
    "standard": "openai:gpt-5",
    "powerful": "openai:gpt-5",
}


@dataclass
class RuntimeContext:
    """Runtime context passed to agent compilation functions."""

    mission_id: str
    task_id: str
    store: InMemoryStore | AsyncPostgresStore 
    checkpointer: Optional[InMemorySaver | AsyncPostgresSaver]
    progress_callback: ProgressCallback | None = field(default=None)


async def compile_subagent(
    config: CompiledSubAgentConfig,
    ctx: RuntimeContext,
) -> CompiledSubAgent:
    """
    Build a CompiledSubAgent wrapping a create_agent with explicit
    FilesystemMiddleware + tools. Never inherits from main agent.
    """
    # 1. Ensure workspace
    await ensure_subagent_workspace(ctx.mission_id, ctx.task_id, config.name)

    # 2. Resolve tools (subagents get their own tool instances)
    tools = resolve_tool_profile(config.tool_profile_name)

    # 3. Build filesystem backend
    backend = build_subagent_backend(ctx.mission_id, ctx.task_id, config.name)

    # 4. Build middleware stack — lighter than main agent
    middleware: list[AgentMiddleware[Any]] = [
        # Error recovery (§8)
        ModelRetryMiddleware(max_retries=2, on_failure="continue"),
        ToolRetryMiddleware(max_retries=2, on_failure="continue"),
        # Cost control (§9)
        ModelCallLimitMiddleware(run_limit=30, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="tavily_search", run_limit=15),
        # Source tracking (§3) — subagents also build provenance
        SourceTrackingMiddleware(),
        write_source_index,
        # Filesystem — EXPLICIT, not inherited
        FilesystemMiddleware(backend=backend),
    ]
    if config.use_todo_middleware:
        middleware.append(TodoListMiddleware())

    if ctx.progress_callback:
        middleware.append(ResearchProgressMiddleware(
            mission_id=ctx.mission_id,
            task_id=ctx.task_id,
            subagent_name=config.name,
            progress_callback=ctx.progress_callback,
        ))

    # 5. Build model (fall back to default if not specified)
    model_name = config.model_name or "openai:gpt-5-mini"
    model = init_chat_model(model_name)

    # 6. Build the agent graph
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=config.system_prompt,
        middleware=middleware, 
        checkpointer=ctx.checkpointer,
        store=ctx.store,
    )

    logger.info(
        "Compiled subagent '%s' for task '%s' (model=%s, tools=%s, todo=%s, skills=%s)",
        config.name, ctx.task_id, model_name, config.tool_profile_name,
        config.use_todo_middleware, config.skills,
    )

    # 7. Wrap as CompiledSubAgent
    return CompiledSubAgent(
        name=config.name,
        description=config.description,
        runnable=agent_graph,
    )


async def compile_main_task_agent(
    task_def: TaskDef,
    ctx: RuntimeContext,
) -> CompiledStateGraph:
    """
    Build and return a create_deep_agent instance for the given TaskDef.
    Creates workspace directories. Returns a compiled, invocable agent.
    """
    cfg = task_def.main_agent

    # 1. Ensure workspace exists
    await ensure_task_workspace(ctx.mission_id, ctx.task_id)

    # 2. Resolve tools
    tools = resolve_tool_profile(cfg.tool_profile_name)

    # 3. Build backend factory — StoreBackend inside the composite reads runtime.store
    #    at invocation time; no need to pass ctx.store at build time.
    backend = build_task_backend(ctx.mission_id, ctx.task_id)

    # 4. Compile declared subagents
    subagents: list[CompiledSubAgent | SubAgent] = []
    for sub_cfg in task_def.compiled_subagents:
        compiled = await compile_subagent(sub_cfg, ctx)
        subagents.append(compiled)

    # 5. Build model — use model_tier if model_name is the default
    model_name = cfg.model_name
    if model_name == "openai:gpt-5" and cfg.model_tier != "standard":
        model_name = _TIER_MAP.get(cfg.model_tier, model_name)
    model: BaseChatModel = init_chat_model(model_name)

    # 6. Build main agent middleware stack (§3, §8, §9)
    fallback_models = task_def.execution.fallback_models
    main_middleware: list[AgentMiddleware[Any] | None] = [
        # Error recovery — §8
        ModelRetryMiddleware(max_retries=2, on_failure="continue"),
        ModelFallbackMiddleware(*fallback_models) if fallback_models else None,
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=(ConnectionError, TimeoutError),
            on_failure="continue",
        ),
        # Cost control — §9
        ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="tavily_search", run_limit=25),
        SummarizationMiddleware(
            model="openai:gpt-5-mini",
            trigger=("tokens", 80_000),
            keep=("messages", 20),
        ),
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=100_000,
                keep=5,
                exclude_tools=["write_file", "edit_file"],
            )],
        ),
        # Source tracking — §3
        SourceTrackingMiddleware(),
        write_source_index,
        # Quality validation — §7
        QualityValidationMiddleware(validation_model="openai:gpt-5-mini"),
        # Streaming — §10
        SubagentStreamingMiddleware(mission_id=ctx.mission_id, task_id=ctx.task_id),
        # Artifact collection — §11
        ArtifactCollectionMiddleware(mission_id=ctx.mission_id, task_id=ctx.task_id),
        # Token tracking — §9d
        track_tokens,
    ]

    # Progress tracking for the deep agent
    if ctx.progress_callback:
        main_middleware.append(ResearchProgressMiddleware(
            mission_id=ctx.mission_id,
            task_id=ctx.task_id,
            subagent_name=None,
            progress_callback=ctx.progress_callback,
        ))

    # Filter out None entries (e.g. no fallback models configured)
    middleware_clean: list[AgentMiddleware] = [m for m in main_middleware if m is not None]

    # 7. Assemble agent — pass both store and checkpointer so the deep agent's
    #    internal graph is fully persisted with the deep-agents Postgres backend.
    #    response_format instructs the agent to return TaskExecutionStructuredOutput
    #    (subagent_final_output_locations, final_synthesis_reports) as result["structured_response"].
    agent: CompiledStateGraph = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.system_prompt,
        memory=["./AGENTS.md"],
        skills=cfg.skills if cfg.skills else None,
        backend=backend,
        store=ctx.store,
        checkpointer=ctx.checkpointer,
        subagents=subagents,
        middleware=middleware_clean if middleware_clean else [],
        # response_format=TaskExecutionStructuredOutput, 
        # TODO: Bring back response format later on after research is succesful 
        debug=True,
    )

    logger.info(
        "Compiled main task agent for '%s' (model=%s, tier=%s, tools=%s, subagents=%d, skills=%s)",
        task_def.task_id, model_name, cfg.model_tier, cfg.tool_profile_name,
        len(subagents), cfg.skills,
    )

    return agent
