"""Agent factory: prompt middleware, shared filesystem, and named subagent wiring."""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from deepagents.middleware.subagents import SubAgentMiddleware

from src.research.langchain_agent.agent.filesystem_support import (
    build_shared_filesystem_middleware,
    filesystem_backend,
)
from src.research.langchain_agent.agent.subagents import build_compiled_subagents
from src.research.langchain_agent.agent.config import (
    BiotechResearchAgentState,
    NextStepsArtifact,
    ResearchPromptSpec,
    ResearchTaskMemoryReport,
    TOOLS_MAP,
)
from src.research.langchain_agent.agent.formatting_helpers import _format_tavily_event_block
from src.research.langchain_agent.tools.middleware.filesystem import monitor_filesystem_tools

from src.research.langchain_agent.agent.constants import GPT_5_4_MINI
from src.research.langchain_agent.agent.memory_report_agent import build_memory_report_agent
from src.research.langchain_agent.agent.next_steps_agent import build_next_steps_agent

# -----------------------------------------------------------------------------
# Dynamic prompt middleware (parameterized by prompt_spec + execution_reminders)
# -----------------------------------------------------------------------------


def _create_research_prompt_middleware(
    prompt_spec: ResearchPromptSpec,
    execution_reminders: Sequence[str],
):
    """Build a dynamic_prompt middleware using the given prompt_spec and reminders."""

    @dynamic_prompt
    def research_prompt_fragment(request: ModelRequest) -> str:
        state = request.state
        task_slug = state.get("task_slug", "unknown-task")
        mission_id = state.get("mission_id", "unknown-mission")
        targets = state.get("targets", []) or []
        selected_tool_names = state.get("selected_tool_names", []) or []
        selected_subagent_names = state.get("selected_subagent_names", []) or []
        stage_type = state.get("stage_type", "discovery")
        search_stage = state.get("search_stage", "initialized")
        official_domains = state.get("official_domains", []) or []
        open_questions = state.get("open_questions", []) or []
        report_path = state.get("report_path", f"reports/{task_slug}.md")
        run_dir = state.get("run_dir", f"runs/{task_slug}")
        step_count = state.get("step_count", 0)
        max_step_budget = state.get("max_step_budget", 12)
        final_report_ready = state.get("final_report_ready", False)
        report_required_sections = state.get("report_required_sections", []) or []

        current_date = state.get("current_date", "")
        research_date = state.get("research_date", "")
        temporal_scope_mode = state.get("temporal_scope_mode", "current")
        temporal_scope_description = state.get("temporal_scope_description", "")

        base_prompt = prompt_spec.render_base_prompt(current_date=current_date or None)

        semantic_memories = state.get("semantic_memories", "") or "Semantic:\n- none"
        episodic_memories = state.get("episodic_memories", "") or "Episodic:\n- none"
        procedural_memories = state.get("procedural_memories", "") or "Procedural:\n- none"

        lines = [base_prompt, "", "Live run context:"]
        lines.extend(
            [
                f"- mission_id: {mission_id}",
                f"- task_slug: {task_slug}",
                f"- stage_type: {stage_type}",
                f"- search_stage: {search_stage}",
                f"- progress: step {step_count} of {max_step_budget}",
                f"- targets: {', '.join(targets) if targets else '(none specified)'}",
                f"- enabled_tools: {', '.join(selected_tool_names) if selected_tool_names else '(none)'}",
                f"- enabled_subagents: {', '.join(selected_subagent_names) if selected_subagent_names else '(none)'}",
                f"- run_dir: {run_dir}",
                f"- report_path: {report_path}",
                f"- final_report_ready: {final_report_ready}",
            ]
        )

        if current_date or research_date or temporal_scope_mode != "current":
            lines.append("")
            lines.append("Temporal context:")
            if current_date:
                lines.append(f"- current_date: {current_date}")
            if research_date:
                lines.append(f"- research_date: {research_date}")
            lines.append(f"- temporal_scope: {temporal_scope_mode}")
            if temporal_scope_description:
                lines.append(f"- temporal_note: {temporal_scope_description}")

        if official_domains:
            lines.append(f"- official_domains: {', '.join(official_domains)}")
        else:
            lines.append("- official_domains: (not confirmed yet)")

        if selected_subagent_names:
            lines.extend(
                [
                    "",
                    "Delegation rules:",
                    "- You remain the main research agent. Delegate only when a specialist loop is materially better than continuing yourself.",
                    "- Give each subagent a precise objective, concrete success criteria, and expected artifact paths.",
                    "- After every subagent run, read the handoff artifact, validate any returned file paths, then continue the stage yourself.",
                    "- Do not stop after delegation unless the final report is already complete and sourced.",
                ]
            )
            if "browser_control" in selected_subagent_names:
                lines.append(
                    "- Use `browser_control` for the existing Playwright-based browser loop when you want targeted interactive validation inside the current browser tool stack."
                )
            if "vercel_agent_browser" in selected_subagent_names:
                lines.append(
                    "- Use `vercel_agent_browser` for shell-driven Vercel agent-browser automation when JS-rendered pages, direct product pages, or interaction-heavy flows need a dedicated Deep Agent with skills and execute access."
                )
            if "docling_document" in selected_subagent_names:
                lines.append(
                    "- Use `docling_document` for important PDFs, DOCX files, or difficult documents that need conversion before you can inspect them."
                )
            if "tavily_research" in selected_subagent_names:
                lines.append(
                    "- Use `tavily_research` for a deeper focused retrieval loop with explicit seed URLs, domains, or follow-up questions."
                )
            if "clinicaltrials_research" in selected_subagent_names:
                lines.append(
                    "- Use `clinicaltrials_research` for sponsor, company, intervention, protocol, or NCT-driven trial discovery and verification."
                )

        if report_required_sections:
            lines.append("")
            lines.append("REQUIRED REPORT SECTIONS (use these EXACT headings as ## level-2 markdown headers):")
            for section in report_required_sections:
                lines.append(f"  ## {section}")
            lines.append("")
            lines.append("Your final report MUST contain ALL of these sections with these exact heading names.")
            lines.append("The ## Sources section MUST list every URL you consulted as markdown links.")
            lines.append("Missing sections will cause the report to FAIL evaluation.")

        lines.extend(["", "Open questions:"])
        lines.extend([f"- {q}" for q in open_questions] if open_questions else ["- (none)"])

        lines.extend(
            [
                "",
                "<Memories>",
                f"<Procedural>\n{procedural_memories}\n</Procedural>",
                f"<Episodic>\n{episodic_memories}\n</Episodic>",
                f"<Semantic>\n{semantic_memories}\n</Semantic>",
                "</Memories>",
                "",
                "Execution reminders:",
                *[f"- {r}" for r in execution_reminders],
            ]
        )

        tavily_search_events = state.get("tavily_search_events", []) or []
        tavily_map_events = state.get("tavily_map_events", []) or []
        tavily_extract_events = state.get("tavily_extract_events", []) or []
        tavily_crawl_events = state.get("tavily_crawl_events", []) or []
        visited_urls = state.get("visited_urls", []) or []

        if tavily_search_events or tavily_map_events or tavily_extract_events or tavily_crawl_events:
            lines.extend(
                [
                    "",
                    "Recent research provenance:",
                    _format_tavily_event_block("Search", tavily_search_events, max_events=2),
                    _format_tavily_event_block("Map", tavily_map_events, max_events=2),
                    _format_tavily_event_block("Extract", tavily_extract_events, max_events=2),
                    _format_tavily_event_block("Crawl", tavily_crawl_events, max_events=2),
                    f"Visited URL count: {len(visited_urls)}",
                ]
            )

        if visited_urls:
            lines.append("")
            lines.append("URLs visited so far (include relevant ones in your ## Sources section):")
            for url in visited_urls[-20:]:
                lines.append(f"  - {url}")

        return "\n".join(lines).strip()

    return research_prompt_fragment


# -----------------------------------------------------------------------------
# Agent construction
# -----------------------------------------------------------------------------


async def build_research_agent(
    *,
    prompt_spec: ResearchPromptSpec,
    execution_reminders: Sequence[str],
    selected_tool_names: List[str],
    selected_subagent_names: List[str],
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any], 
    model_name: str = GPT_5_4_MINI,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a research agent with the given prompt spec and reminders."""
    selected_tools: List[BaseTool] = [TOOLS_MAP[n] for n in selected_tool_names]
    selected_subagents = await build_compiled_subagents(
        selected_subagent_names,
        backend=filesystem_backend,
        store=store,
        checkpointer=checkpointer,
    )
    prompt_middleware = _create_research_prompt_middleware(
        prompt_spec=prompt_spec,
        execution_reminders=execution_reminders,
    )

    middleware: list[AgentMiddleware] = [
        monitor_filesystem_tools,  # intercepts write_file/read_file/edit_file → updates written/read/edited_file_paths in state
        build_shared_filesystem_middleware(backend=filesystem_backend),
        prompt_middleware,
    ]
    if selected_subagents:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="task",
                run_limit=max(2, len(selected_subagents) + 1),
                exit_behavior="continue",
            )
        )
        middleware.append(
            SubAgentMiddleware(
                subagents=selected_subagents,
                backend=filesystem_backend, 
               
            )
        )

    return create_agent(
        model=model_name,
        tools=selected_tools,
        middleware=middleware,
        store=store,
        checkpointer=checkpointer,
        state_schema=BiotechResearchAgentState,
    )

