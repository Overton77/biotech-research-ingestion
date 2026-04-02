from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Sequence

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.research.langchain_agent.agent.filesystem_support import (
    build_shared_filesystem_middleware,
    filesystem_backend,
)
from src.research.langchain_agent.agent.subagent_types import (
    ALL_SUBAGENT_NAMES,
    BROWSER_CONTROL_SUBAGENT,
    CLINICALTRIALS_RESEARCH_SUBAGENT,
    DOCLING_DOCUMENT_SUBAGENT,
    EDGAR_RESEARCH_SUBAGENT,
    SUBAGENT_DESCRIPTIONS,
    TAVILY_RESEARCH_SUBAGENT,
    VERCEL_AGENT_BROWSER_SUBAGENT,
    dedupe_subagent_names,
)
from src.research.langchain_agent.agent.vercel_agent_browser import (
    build_vercel_agent_browser_subagent,
)
from src.research.langchain_agent.tools.medical.clinical_trials import (
    clinical_trials_tools,
)
from src.research.langchain_agent.tools.browser.playwright import (
    browser_interaction_task,
)
from src.research.langchain_agent.tools.middleware.filesystem import (
    monitor_filesystem_tools,
)
from src.research.langchain_agent.tools.search.tavily import (
    crawl_website,
    extract_from_urls,
    map_website,
    search_web,
)
from src.research.langchain_agent.tools.document.docling import docling_document_tools
from src.research.langchain_agent.unstructured.edgar_subagent import (
    EDGAR_SPECIALTY_PROMPT,
)
from src.research.langchain_agent.tools.financials.edgar_tools import (
    EDGAR_RESEARCH_TOOLS,
) 
from src.research.langchain_agent.agent.subagent_system_prompts import (
    BROWSER_CONTROL_SPECIALTY_PROMPT,
    CLINICALTRIALS_SPECIALTY_PROMPT,
    DOCLING_SPECIALTY_PROMPT,
    EDGAR_SPECIALTY_PROMPT,
    TAVILY_SPECIALTY_PROMPT, 
    SUBAGENT_HANDOFF_CONTRACT,
)


gpt_5_4_mini = "gpt-5.4-mini"




def _build_subagent_prompt(*, subagent_name: str, specialty_prompt: str) -> str:
    return (
        f"{specialty_prompt}\n\n"
        f"{SUBAGENT_HANDOFF_CONTRACT.replace('<subagent_name>', subagent_name)}"
    ).strip()


def _build_compiled_subagent(
    *,
    name: str,
    description: str,
    system_prompt: str,
    tools: Sequence[BaseTool],
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    runnable_graph = create_agent(
        model=gpt_5_4_mini,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=[
            ModelRetryMiddleware(max_retries=2, on_failure="continue"),
            ToolRetryMiddleware(max_retries=2, on_failure="continue"),
            ModelCallLimitMiddleware(run_limit=14, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=10, exit_behavior="continue"),
            monitor_filesystem_tools,
            build_shared_filesystem_middleware(backend=backend),
        ],
        store=store,
        checkpointer=checkpointer,
        name=name,
    ).with_config({"recursion_limit": 60})
    return CompiledSubAgent(
        name=name,
        description=description,
        runnable=runnable_graph, 
        
    )


async def _build_browser_control_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    return _build_compiled_subagent(
        name=BROWSER_CONTROL_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[BROWSER_CONTROL_SUBAGENT],
        system_prompt=_build_subagent_prompt(
            subagent_name=BROWSER_CONTROL_SUBAGENT,
            specialty_prompt=BROWSER_CONTROL_SPECIALTY_PROMPT,
        ),
        tools=[browser_interaction_task],
        backend=backend,
        store=store,
        checkpointer=checkpointer,
    )


async def _build_vercel_agent_browser_control_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    del backend
    return build_vercel_agent_browser_subagent(
        store=store,
        checkpointer=checkpointer,
    )


async def _build_clinicaltrials_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    return _build_compiled_subagent(
        name=CLINICALTRIALS_RESEARCH_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[CLINICALTRIALS_RESEARCH_SUBAGENT],
        system_prompt=_build_subagent_prompt(
            subagent_name=CLINICALTRIALS_RESEARCH_SUBAGENT,
            specialty_prompt=CLINICALTRIALS_SPECIALTY_PROMPT,
        ),
        tools=clinical_trials_tools,
        backend=backend,
        store=store,
        checkpointer=checkpointer,
    )


async def _build_tavily_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    return _build_compiled_subagent(
        name=TAVILY_RESEARCH_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[TAVILY_RESEARCH_SUBAGENT],
        system_prompt=_build_subagent_prompt(
            subagent_name=TAVILY_RESEARCH_SUBAGENT,
            specialty_prompt=TAVILY_SPECIALTY_PROMPT,
        ),
        tools=[search_web, extract_from_urls, map_website, crawl_website],
        backend=backend,
        store=store,
        checkpointer=checkpointer,
    )


async def _build_docling_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    return _build_compiled_subagent(
        name=DOCLING_DOCUMENT_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[DOCLING_DOCUMENT_SUBAGENT],
        system_prompt=_build_subagent_prompt(
            subagent_name=DOCLING_DOCUMENT_SUBAGENT,
            specialty_prompt=DOCLING_SPECIALTY_PROMPT,
        ),
        tools=docling_document_tools,
        backend=backend,
        store=store,
        checkpointer=checkpointer,
    )


async def _build_edgar_subagent(
    *,
    backend: FilesystemBackend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledSubAgent:
    return _build_compiled_subagent(
        name=EDGAR_RESEARCH_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[EDGAR_RESEARCH_SUBAGENT],
        system_prompt=_build_subagent_prompt(
            subagent_name=EDGAR_RESEARCH_SUBAGENT,
            specialty_prompt=EDGAR_SPECIALTY_PROMPT,
        ),
        tools=EDGAR_RESEARCH_TOOLS,
        backend=backend,
        store=store,
        checkpointer=checkpointer,
    )


_SUBAGENT_BUILDERS: dict[
    str,
    Callable[..., Awaitable[CompiledSubAgent]],
] = {
    BROWSER_CONTROL_SUBAGENT: _build_browser_control_subagent,
    VERCEL_AGENT_BROWSER_SUBAGENT: _build_vercel_agent_browser_control_subagent,
    CLINICALTRIALS_RESEARCH_SUBAGENT: _build_clinicaltrials_subagent,
    TAVILY_RESEARCH_SUBAGENT: _build_tavily_subagent,
    DOCLING_DOCUMENT_SUBAGENT: _build_docling_subagent,
    EDGAR_RESEARCH_SUBAGENT: _build_edgar_subagent,
}


async def build_compiled_subagents(
    selected_subagent_names: Sequence[str],
    *,
    backend: FilesystemBackend = filesystem_backend,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
) -> list[CompiledSubAgent]:
    names = dedupe_subagent_names(selected_subagent_names)
    unknown = [name for name in names if name not in ALL_SUBAGENT_NAMES]
    if unknown:
        raise ValueError(f"Unknown subagent names: {unknown}")

    tasks = [
        _SUBAGENT_BUILDERS[name](
            backend=backend,
            store=store,
            checkpointer=checkpointer,
        )
        for name in names
    ]
    return list(await asyncio.gather(*tasks))
