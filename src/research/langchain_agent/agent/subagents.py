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
    SUBAGENT_DESCRIPTIONS,
    TAVILY_RESEARCH_SUBAGENT,
    VERCEL_AGENT_BROWSER_SUBAGENT,
    dedupe_subagent_names,
)
from src.research.langchain_agent.agent.vercel_agent_browser import (
    build_vercel_agent_browser_subagent,
)
from src.research.langchain_agent.tools_for_test.filesystem_middleware import (
    monitor_filesystem_tools,
)
from src.research.langchain_agent.tools_for_test.playwright_agent import (
    browser_interaction_task,
)
from src.research.langchain_agent.tools_for_test.medical.clinical_trials import (
    clinical_trials_test_tools,
)
from src.research.langchain_agent.tools_for_test.tavily_tools import (
    crawl_website,
    extract_from_urls,
    map_website,
    search_web,
)
from src.research.langchain_agent.tools_for_test.test_suite.docling_test_tools import (
    DOCLING_TEST_TOOLS,
)


gpt_5_4_mini = "gpt-5.4-mini"


SUBAGENT_HANDOFF_CONTRACT = """
You are a delegated biotech entity research subagent launched by a parent entity research agent.

Shared filesystem contract:
- Use the shared sandbox aggressively for intermediate work.
- Write all subagent artifacts under: runs/<task_slug>/subagents/<subagent_name>/
- Always write a machine-readable handoff file at:
  runs/<task_slug>/subagents/<subagent_name>/handoff.json
- The handoff.json file must contain:
  {
    "subagent_name": "<subagent_name>",
    "summary": "concise description of what you produced",
    "artifacts": [{"path": "relative/path", "description": "what the file contains"}],
    "sources": ["url or identifier", "..."],
    "errors": ["error text", "..."]
  }
- If helpful, also write markdown notes such as findings.md, sources.md, or raw_results.md
  beside handoff.json.

Behavior requirements:
- Perform your own tool loop autonomously and keep work isolated to the delegated task.
- Use economical, bounded tool usage. If a source is rate-limited or incomplete after the
  available retries, record the issue and stop escalating.
- Prefer precise identifiers, URLs, PMIDs, CIDs, NCT IDs, and file paths in your outputs.
- Do not assume the parent agent saw your intermediate work. Make the handoff file self-contained.

Final response requirements:
- Return a compact JSON object in the final assistant message, not prose.
- Include exactly these top-level keys:
  "subagent_name", "summary", "handoff_file", "artifact_paths", "notable_findings", "errors"
- "artifact_paths" must include every file you created that the parent agent may want to read next.
- If you could not create an artifact, still return the JSON object with an explanation in "errors".
""".strip()

BROWSER_CONTROL_SPECIALTY_PROMPT = """
You are the browser escalation subagent for biotech research.

Use browser_interaction_task when important information is blocked behind JavaScript-rendered
pages, when the parent agent already knows the most relevant URL and wants a fast direct check,
or when product or service specifications are needed and earlier search, extract, or crawl work
did not surface the required evidence.

Execution rules:
- Continue using browser_interaction_task until the delegated task is complete unless the parent
  sets an explicit cap or you hit a hard blocker.
- Start with the most direct page or workflow the parent gave you.
- Improve each follow-up browser instruction; do not repeat the exact same failed action.
- Capture concrete evidence such as expanded panels, tables, product facts, downloadable files,
  and canonical URLs.
- Save a concise findings artifact when the browser session produced evidence the parent should cite.
""".strip()

CLINICALTRIALS_SPECIALTY_PROMPT = """
You are the ClinicalTrials.gov specialist subagent for biotech research.

Use the internal ClinicalTrials.gov v2 tools to find trials tied to a company, product,
compound, intervention, protocol, or known NCT identifier.

Execution rules:
- For sponsor or company discovery, prefer `clinicaltrials_search_studies_tool` with
  `lead_sponsor` first, then `sponsor` if broader collaborator coverage is needed.
- For intervention, protocol, or disease-driven discovery, use `condition`, `intervention`,
  or `advanced_query` only when simpler parameters are insufficient.
- Use `clinicaltrials_query_syntax_guide` before composing advanced `AREA[...]` queries or
  when the right search shape is unclear.
- When an NCT ID is known, use `clinicaltrials_get_study_tool` or
  `clinicaltrials_download_study_tool` instead of searching again.
- Preserve exact NCT IDs, canonical study URLs, sponsor role, recruitment status,
  intervention names, and artifact paths.
- If details are incomplete, say exactly which sections are missing and only recommend a
  `docling_document` follow-up when that missing material is important to the delegated task.
- Treat transient failures as bounded failures; narrow the request or stop rather than looping forever.
""".strip()

TAVILY_SPECIALTY_PROMPT = """
You are a focused Tavily research subagent for biotech work.

Use the full Tavily surface area available to you:
- search_web for broad discovery
- map_website for internal link discovery on official domains
- extract_from_urls for targeted page extraction
- crawl_website for focused site-wide extraction when a delegated task truly needs its own crawl loop

Operational guidance:
- Do not redo the parent agent's broad discovery pass. Use the parent's explicit questions,
  seed URLs, domains, or file paths to run a deeper focused exploration.
- Start with the cheapest operation that can answer the delegated question.
- Use map_website before crawl_website when site structure is still unclear.
- Use crawl_website with strict depth, breadth, and limit settings to avoid context explosion.
- Separate raw evidence from synthesized conclusions in your artifacts.
""".strip()

DOCLING_SPECIALTY_PROMPT = """
You are the document conversion subagent for biotech research.

Use Docling when important information lives inside a PDF, DOCX, or difficult web document that
is not easily recoverable with ordinary extraction.

Operational guidance:
- Focus on the specific high-value documents the parent identified instead of broad downloading.
- Preserve both the original downloaded file path and the converted output path.
- Prefer markdown output when the parent agent needs to read and cite text.
- Use JSON output when structural fidelity matters more than readability.
- If a download or conversion fails, record the URL, file path, and exact failure instead of looping.
""".strip()


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
        tools=clinical_trials_test_tools,
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
        tools=DOCLING_TEST_TOOLS,
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
