"""Agent Compiler — builds create_deep_agent and CompiledSubAgent from TaskDef configs.

compile_main_task_agent: TaskDef → create_deep_agent instance with subagents
compile_subagent: CompiledSubAgentConfig → CompiledSubAgent wrapping create_agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model

from src.research.models.mission import CompiledSubAgentConfig, TaskDef
from src.research.middleware.progress_middleware import (
    ProgressCallback,
    ResearchProgressMiddleware,
)
from src.research.runtime.backends import (
    build_subagent_backend,
    build_task_backend,
    ensure_subagent_workspace,
    ensure_task_workspace,
)
from src.research.runtime.tools import resolve_tool_profile

logger = logging.getLogger(__name__)


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

    # 4. Build middleware — FilesystemMiddleware is EXPLICIT, not inherited
    middleware: list[Any] = [FilesystemMiddleware(backend=backend)]
    if config.use_todo_middleware:
        middleware.append(TodoListMiddleware())

    # Progress middleware for real-time dashboard
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
        "Compiled subagent '%s' for task '%s' (model=%s, tools=%s, todo=%s)",
        config.name, ctx.task_id, model_name, config.tool_profile_name,
        config.use_todo_middleware,
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
) -> Any:
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
    subagents: list[CompiledSubAgent] = []
    for sub_cfg in task_def.compiled_subagents:
        compiled = await compile_subagent(sub_cfg, ctx)
        subagents.append(compiled)

    # 5. Build model
    model = init_chat_model(cfg.model_name)

    # 6. Build main agent middleware (progress tracking for the deep agent itself)
    main_middleware: list[Any] = []
    if ctx.progress_callback:
        main_middleware.append(ResearchProgressMiddleware(
            mission_id=ctx.mission_id,
            task_id=ctx.task_id,
            subagent_name=None,
            progress_callback=ctx.progress_callback,
        ))

    # 7. Assemble agent — pass both store and checkpointer so the deep agent's
    #    internal graph is fully persisted with the deep-agents Postgres backend.
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.system_prompt,
        backend=backend,
        store=ctx.store,
        checkpointer=ctx.checkpointer,
        subagents=subagents,
        middleware=main_middleware if main_middleware else None,
    )

    logger.info(
        "Compiled main task agent for '%s' (model=%s, tools=%s, subagents=%d)",
        task_def.task_id, cfg.model_name, cfg.tool_profile_name, len(subagents),
    )

    return agent
