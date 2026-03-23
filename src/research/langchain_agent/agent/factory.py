"""
Agent factory: filesystem middleware, browser subagent, build_research_agent (parameterized by prompt_spec).
"""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from deepagents.backends.filesystem import FilesystemBackend 
from src.research.langchain_agent.tools_for_test.tavily_tools import crawl_website
from src.research.langchain_agent.tools_for_test.filesystem_middleware import monitor_filesystem_tools


from src.research.langchain_agent.tools_for_test.playwright_agent_tool import playwright_mcp_specs

from src.research.langchain_agent.agent.config import (
    BiotechResearchAgentState,
    NextStepsArtifact,
    ResearchPromptSpec,
    ResearchTaskMemoryReport,
    ROOT_FILESYSTEM,
    TOOLS_MAP,
)
from src.research.langchain_agent.tools_for_test.formatters import _format_tavily_event_block 
from src.research.langchain_agent.tools_for_test.playwright_agent import (
    browser_interaction_task,
)


gpt_5_mini = "gpt-5-mini"
# -----------------------------------------------------------------------------
# Filesystem middleware
# -----------------------------------------------------------------------------

filesystem_backend = FilesystemBackend(
    root_dir=str(ROOT_FILESYSTEM),
    virtual_mode=True,
)

filesystem_middleware = FilesystemMiddleware(
    backend=filesystem_backend,
    system_prompt=(
        "Use the filesystem aggressively for intermediate research state. "
        "All paths must stay inside the sandbox root. "
        "Use only relative sandbox paths such as runs/, reports/, and scratch/. "
        "Never use absolute host paths."
    ),
    custom_tool_descriptions={
        "ls": (
            "List directories and files inside the sandbox. "
            "Use relative paths like runs, reports, and scratch."
        ),
        "read_file": (
            "Read saved notes, intermediate findings, and draft reports from sandbox-relative paths."
        ),
        "write_file": (
            "Create structured intermediate files and final reports inside the sandbox only. "
            "Prefer markdown or json."
        ),
        "edit_file": (
            "Update an existing findings file or report incrementally inside the sandbox."
        ),
    },
)


# -----------------------------------------------------------------------------
# Browser control subagent
# -----------------------------------------------------------------------------

# browser_control_agent = create_agent(
#     model=gpt_5_mini,
#     tools=[playwright_mcp_specs],
# )

# browser_control_subagent = CompiledSubAgent(
#     name="browser_control",
#     description=(
#         "This subagent controls a real browser via Playwright. Use it when pages "
#         "require scrolling, clicking, or interaction to reveal content. "
#         "Provide the URL(s) and clear extraction instructions (what data to capture)."
#     ),
#     runnable=browser_control_agent,
# ) 


BROWSER_CONTROL_SYSTEM_PROMPT = """
You are the browser-control coordinator.

Your job is to decide when and how to use the browser_interaction_task tool in order to complete the delegated request.

You may call browser_interaction_task up to 3 times for a single delegated task.

How to work:
- Use the first browser call for the most direct attempt.
- If the result is incomplete, ambiguous, or lacks evidence, you may call the tool again with improved instructions.
- Each retry must be materially better or more specific than the previous one.
- Do not repeat the exact same browser call.
- Prefer the minimum number of browser calls needed.

When to use the browser tool:
- The page requires clicking, scrolling, expansion, tab switching, or dynamic interaction.
- The relevant content is likely hidden behind "show more", accordions, tabs, or technical details sections.
- Static extraction is insufficient.

When preparing a browser call, be explicit about:
- the exact start URL
- the task goal
- the success criteria
- likely useful hints
- allowed domains when relevant

Final answer requirements:
- Synthesize the browser findings rather than simply dumping raw tool output.
- If the task still cannot be completed after retries, clearly state what was attempted and what remains uncertain.
""".strip()

browser_control_agent = create_agent(
    model=gpt_5_mini,
    tools=[browser_interaction_task],
    system_prompt=BROWSER_CONTROL_SYSTEM_PROMPT,
)

browser_control_subagent = CompiledSubAgent(
    name="browser_control",
    description=(
        "Use this subagent when a webpage requires real browser interaction to reveal or verify information. "
        "It can navigate pages, click, scroll, expand sections, and extract structured results from dynamic content."
    ),
    runnable=browser_control_agent,
)

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

        lines = [
            base_prompt,
            "",
            "Current run context:",
            f"- mission_id: {mission_id}",
            f"- task_slug: {task_slug}",
            f"- stage_type: {stage_type}",
            f"- search_stage: {search_stage}",
            f"- step_count: {step_count}/{max_step_budget}",
            f"- targets: {', '.join(targets) if targets else '(none specified)'}",
            f"- enabled_tools: {', '.join(selected_tool_names) if selected_tool_names else '(none)'}",
            f"- run_dir: {run_dir}",
            f"- report_path: {report_path}",
            f"- final_report_ready: {final_report_ready}",
        ]

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

        if report_required_sections:
            lines.append("")
            lines.append("REQUIRED REPORT SECTIONS (use these EXACT headings as ## level-2 markdown headers):")
            for section in report_required_sections:
                lines.append(f"  ## {section}")
            lines.append("")
            lines.append("Your final report MUST contain ALL of these sections with these exact heading names.")
            lines.append("The ## Sources section MUST list every URL you consulted as markdown links.")
            lines.append("Missing sections will cause the report to FAIL evaluation.")

        lines.append("")
        lines.append("Open questions:")
        if open_questions:
            lines.extend([f"- {q}" for q in open_questions])
        else:
            lines.append("- (none)")

        lines.extend(
            [
                "",
                "<Memories>",
                f"<Procedural>\n{procedural_memories}\n</Procedural>",
                f"<Episodic>\n{episodic_memories}\n</Episodic>",
                f"<Semantic>\n{semantic_memories}\n</Semantic>",
                "</Memories>",
                "",
                "Execution reminder for this run:",
                *[f"- {r}" for r in execution_reminders],
            ]
        )

        tavily_search_events = state.get("tavily_search_events", []) or []
        tavily_map_events = state.get("tavily_map_events", []) or []
        tavily_extract_events = state.get("tavily_extract_events", []) or []
        visited_urls = state.get("visited_urls", []) or []

        if tavily_search_events or tavily_map_events or tavily_extract_events:
            lines.extend(
                [
                    "",
                    "Recent research provenance:",
                    _format_tavily_event_block("Search", tavily_search_events, max_events=2),
                    _format_tavily_event_block("Map", tavily_map_events, max_events=2),
                    _format_tavily_event_block("Extract", tavily_extract_events, max_events=2),
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


def build_research_agent(
    *,
    prompt_spec: ResearchPromptSpec,
    execution_reminders: Sequence[str],
    selected_tool_names: List[str],
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a research agent with the given prompt spec and reminders."""
    selected_tools: List[BaseTool] = [TOOLS_MAP[n] for n in selected_tool_names]
    prompt_middleware = _create_research_prompt_middleware(
        prompt_spec=prompt_spec,
        execution_reminders=execution_reminders,
    )

    return create_agent(
        model=gpt_5_mini,
        tools=selected_tools,
        middleware=[
            monitor_filesystem_tools,  # intercepts write_file/read_file/edit_file → updates written/read/edited_file_paths in state
            filesystem_middleware,
            prompt_middleware,
            SubAgentMiddleware(
                subagents=[browser_control_subagent],
                backend=filesystem_backend,
            ),
        ],
        store=store,
        checkpointer=checkpointer,
        state_schema=BiotechResearchAgentState,
    )


def build_memory_report_agent(
    store: BaseStore, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model=gpt_5_mini,
        tools=[],
        middleware=[],
        response_format=ResearchTaskMemoryReport,
        store=store,
        checkpointer=checkpointer,
    )


# -----------------------------------------------------------------------------
# Next-steps extraction agent (iterative missions)
# -----------------------------------------------------------------------------

NEXT_STEPS_EXTRACTION_PROMPT = """
You are an iteration evaluator for a multi-pass biotech research workflow.

You will receive:
1. The original research objective for this stage.
2. The iteration number (which pass this is).
3. The completion criteria (what "done" looks like).
4. The final report produced by the research agent for this iteration.
5. The agent's last response message.

Your job is to produce a structured evaluation:

- **stage_complete**: true only if the report fully satisfies the completion criteria
  and no material open questions remain.
- **confidence**: 0.0 to 1.0 — how complete is the research relative to the objective?
  0.0 = nothing useful found. 0.5 = partial coverage with significant gaps.
  0.9+ = comprehensive, only minor polish remaining.
- **open_questions**: list the most important unanswered questions or unresolved
  contradictions. Each item should have a question, priority (high/medium/low),
  and brief rationale. Only include questions that are material — skip trivial ones.
- **suggested_focus**: one sentence describing what the next iteration should
  prioritize if the stage is not yet complete.
- **key_findings_this_iteration**: 3-7 bullet points summarizing the most important
  new information discovered in this iteration.

Be honest and calibrated. Do not inflate confidence. Do not mark stage_complete
unless the research genuinely covers the objective.
""".strip()


def build_next_steps_agent(
    store: BaseStore, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a structured-output agent that evaluates an iteration and produces NextStepsArtifact."""
    return create_agent(
        model=gpt_5_mini,
        tools=[],
        system_prompt=NEXT_STEPS_EXTRACTION_PROMPT,
        response_format=NextStepsArtifact,
        store=store,
        checkpointer=checkpointer,
    )
