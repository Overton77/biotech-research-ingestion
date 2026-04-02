"""
Agent configuration: tools, paths, input models, state schema, memory helpers, file helpers.
Single source of truth for one research agent run (no Qualia-specific defaults in the base spec).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

import aiofiles
from pydantic import BaseModel, Field, field_validator

from langchain.agents import AgentState
from langchain.tools import BaseTool

from src.research.langchain_agent.agent.subagent_types import (
    ALL_SUBAGENT_NAMES,
    DEFAULT_STAGE_SUBAGENT_NAMES,
)
from src.research.langchain_agent.kg.extraction_models import TemporalScope
from src.research.langchain_agent.tools.formatters import (
    _format_file_state_block,
    _format_tavily_event_block,
    _truncate_text,
)
from src.research.langchain_agent.tools.search.tavily import (
    crawl_website,
    extract_from_urls,
    map_website,
    search_web,
)

# -----------------------------------------------------------------------------
# Tool registry
# -----------------------------------------------------------------------------

TOOLS_MAP: Dict[str, BaseTool] = {
    "search_web": search_web,
    "extract_from_urls": extract_from_urls,
    "map_website": map_website,
    "crawl_website": crawl_website,
}

# -----------------------------------------------------------------------------
# Paths (root is test_runs/agent_outputs)
# -----------------------------------------------------------------------------

ROOT_FILESYSTEM = (Path(__file__).resolve().parent.parent / "agent_outputs").resolve()
RUNS_DIR = ROOT_FILESYSTEM / "runs"
REPORTS_DIR = ROOT_FILESYSTEM / "reports"
SCRATCH_DIR = ROOT_FILESYSTEM / "scratch"


async def ensure_dirs() -> None:
    for p in (ROOT_FILESYSTEM, RUNS_DIR, REPORTS_DIR, SCRATCH_DIR):
        await asyncio.to_thread(p.mkdir, parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Input models
# -----------------------------------------------------------------------------


class MissionSliceInput(BaseModel):
    """
    One bounded agent run representing a major stage or sub-stage
    within a larger mission.
    """

    task_id: str
    mission_id: str
    task_slug: str
    user_objective: str

    targets: List[str] = Field(default_factory=list)
    dependency_reports: Dict[str, str] = Field(
        default_factory=dict,
        description="task_slug -> final report markdown from stages this one depends on (set by runner)",
    )

    selected_tool_names: List[str] = Field(
        default_factory=lambda: ["search_web", "extract_from_urls", "map_website"]
    )
    selected_subagent_names: List[str] = Field(
        default_factory=lambda: list(DEFAULT_STAGE_SUBAGENT_NAMES)
    )

    report_required_sections: List[str] = Field(
        default_factory=lambda: [
            "Executive Summary",
            "Key Findings",
            "Sources",
            "Open Questions and Next Steps",
        ]
    )

    guidance_notes: List[str] = Field(default_factory=list)

    stage_type: Literal[
        "discovery",
        "entity_validation",
        "official_site_mapping",
        "targeted_extraction",
        "report_synthesis",
    ] = "discovery"

    max_step_budget: int = 12

    # --- Temporal configuration ---
    temporal_scope: TemporalScope = Field(
        default_factory=TemporalScope,
        description="Temporal scope for this research stage. Defaults to 'current'.",
    )
    research_date: Optional[str] = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD) of when the research is considered current. "
            "Defaults to today if not set. Used as validFrom for ingested facts."
        ),
    )

    @field_validator("selected_tool_names")
    @classmethod
    def validate_tool_names(cls, value: List[str]) -> List[str]:
        unknown = [name for name in value if name not in TOOLS_MAP]
        if unknown:
            raise ValueError(f"Unknown tool names: {unknown}")
        return value

    @field_validator("selected_subagent_names")
    @classmethod
    def validate_subagent_names(cls, value: List[str]) -> List[str]:
        unknown = [name for name in value if name not in ALL_SUBAGENT_NAMES]
        if unknown:
            raise ValueError(f"Unknown subagent names: {unknown}")
        return value

    @property
    def effective_research_date(self) -> str:
        """Return the research_date or today's date as ISO string."""
        return self.research_date or date.today().isoformat()

    @property
    def effective_current_date(self) -> str:
        """Return today's date as ISO string (always real wall-clock date)."""
        return date.today().isoformat()


class RunPathsInput(BaseModel):
    run_dir: str
    report_path: str
    scratch_dir: str


class ResearchTaskMemoryReport(BaseModel):
    """Structured artifact used to feed a clean summary of the run into LangMem."""

    mission_id: str
    summary: str
    file_paths: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Iterative stage: next-steps artifact
# -----------------------------------------------------------------------------


class NextStepItem(BaseModel):
    """One open question or action item identified at the end of an iteration."""

    question: str
    priority: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


class NextStepsArtifact(BaseModel):
    """Structured next-steps produced by the evaluator after each iteration.

    The iterative runner uses this to decide whether to continue and what
    the next iteration should focus on.
    """

    stage_complete: bool = False
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed completeness (0.0 = nothing done, 1.0 = fully complete)",
    )
    open_questions: List[NextStepItem] = Field(default_factory=list)
    suggested_focus: str = ""
    key_findings_this_iteration: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Prompt data structure (generic default)
# -----------------------------------------------------------------------------


@dataclass
class ResearchPromptSpec:
    agent_identity: str = "You are a biotech entity research agent."
    domain_scope: Sequence[str] = field(
        default_factory=lambda: [
            "companies",
            "brands",
            "products",
            "founders",
            "investors",
            "operating subsidiaries",
        ]
    )
    workflow: Sequence[str] = field(
        default_factory=lambda: [
            "Start with the cheapest path that can answer the stage, then narrow quickly toward official evidence.",
            "Use search_web to frame the problem, map_website to locate high-value official pages, and extract_from_urls only on the pages that matter.",
            "Escalate to subagents only when a specialist loop will materially outperform continuing in the main context.",
            "Save meaningful intermediate findings before changing search direction.",
            "Before finalizing, read your own artifacts back and synthesize from evidence rather than memory.",
        ]
    )
    tool_guidance: Sequence[str] = field(
        default_factory=lambda: [
            "Use search_web broadly first, then narrow with precise entity, domain, sponsor, compound, or product terms.",
            "Use include_domains when official-source confirmation is needed and exclude_domains when noisy aggregators dominate.",
            "Keep max_results, max_depth, max_breadth, and crawl limits economical.",
            "Do not pull large batches of low-value pages when a smaller targeted extraction can answer the question.",
        ]
    )
    subagent_guidance: Sequence[str] = field(
        default_factory=lambda: [
            "Use the task tool for specialized work that benefits from its own tool loop and isolated context window.",
            "You are still the main research agent. Delegate, inspect the result, then continue orchestrating the stage yourself.",
            "Delegate with precise instructions, success criteria, seed URLs or identifiers, and expected output files.",
            "Ask subagents to write handoff artifacts under runs/<task_slug>/subagents/<subagent_name>/ and return the file paths they created.",
            "When a subagent returns file paths, validate them before relying on the contents in the final report.",
        ]
    )
    practical_limits: Sequence[str] = field(
        default_factory=lambda: [
            "Do not map very large sites unless necessary.",
            "Do not extract too many URLs at once.",
            "Avoid redundant searches.",
            "Record uncertainty explicitly when sources conflict.",
        ]
    )
    filesystem_rules: Sequence[str] = field(
        default_factory=lambda: [
            "Treat the agent filesystem root as a sandbox.",
            "Only write relative sandbox paths like runs/<task_slug>/... and reports/<task_slug>.md.",
            "Do not use absolute host paths.",
            "Use the filesystem as the primary scratchpad for intermediate work.",
        ]
    )
    intermediate_files: Sequence[str] = field(
        default_factory=lambda: [
            "runs/<task_slug>/01_search_plan.md",
            "runs/<task_slug>/02_broad_search_summary.md",
            "runs/<task_slug>/03_official_sites_and_targets.md",
            "runs/<task_slug>/04_extracted_facts.md",
            "runs/<task_slug>/05_open_questions.md",
            "runs/<task_slug>/06_draft_report.md",
        ]
    )

    def render_base_prompt(self, *, current_date: str | None = None) -> str:
        effective_date = current_date or date.today().isoformat()
        lines: List[str] = [
            self.agent_identity,
            "",
            f"Today's date: {effective_date}",
            "",
            "Your job is to complete one bounded biotech research stage or sub-stage.",
            "Produce a sourced markdown report and keep the work bounded to this stage.",
            "Always be explicit about temporal context: when citing facts, note the date or",
            "time frame they apply to (for example 'as of March 2026', 'since 2023', or 'formerly').",
            "",
            "Domain scope:",
        ]
        lines.extend([f"- {item}" for item in self.domain_scope])
        lines.extend(
            [
                "",
                "Common workflow:",
                *[f"- {item}" for item in self.workflow],
                "",
                "Tool usage guidance:",
                *[f"- {item}" for item in self.tool_guidance],
                "",
                "Subagent (task tool) guidance:",
                *[f"- {item}" for item in self.subagent_guidance],
                "",
                "Practical limits:",
                *[f"- {item}" for item in self.practical_limits],
                "",
                "Filesystem rules:",
                *[f"- {item}" for item in self.filesystem_rules],
                "",
                "Minimum intermediate files to write:",
                *[f"- {item}" for item in self.intermediate_files],
                "",
                "Behavior rules:",
                "- Use the filesystem as your primary scratchpad and checkpoint surface.",
                "- Save intermediate outputs before continuing to the next search step.",
                "- Read your saved files before writing the final report.",
                "- Stay concise in your internal notes and final synthesis; prefer precise facts over padded prose.",
                "- Mention the final report path when finished.",
                "",
                "CRITICAL — Final report formatting rules:",
                "- The final report MUST be valid markdown written to the report_path.",
                "- Each required section MUST appear as a level-2 heading (## Section Name) exactly as specified.",
                "- The ## Sources section MUST list every URL you used, formatted as a markdown list:",
                "  - [Page Title](https://url) — brief description of what was found",
                "- If a required section has no data, still include the heading with a note: '(No data found for this section.)'",
                "- The report MUST begin with a level-1 heading (# Report Title) followed by ## Executive Summary.",
                "- Do NOT skip any required section. The evaluation system checks for exact heading matches.",
            ]
        )
        return "\n".join(lines).strip()


PROMPT_SPEC = ResearchPromptSpec()
BASE_SYSTEM_PROMPT = PROMPT_SPEC.render_base_prompt()


# -----------------------------------------------------------------------------
# Agent state
# -----------------------------------------------------------------------------


def _merge_unique_str_list(
    current: List[str] | None,
    incoming: List[str] | None,
    *,
    max_items: int = 2000,
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for value in (current or []) + (incoming or []):
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged[-max_items:]


def _merge_event_list(
    current: List[Dict[str, Any]] | None,
    incoming: List[Dict[str, Any]] | None,
    *,
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in (current or []) + (incoming or []):
        if not value:
            continue
        signature = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(value)
    return merged[-max_items:]


class BiotechResearchAgentState(AgentState):
    task_id: str
    mission_id: str
    task_slug: str
    user_objective: str

    targets: List[str]
    selected_tool_names: List[str]
    selected_subagent_names: List[str]
    report_required_sections: List[str]

    stage_type: str
    search_stage: str

    official_domains: List[str]
    visited_urls: Annotated[List[str], _merge_unique_str_list]
    findings: List[Dict[str, Any]]
    open_questions: List[str]

    run_dir: str
    report_path: str

    step_count: int
    max_step_budget: int
    final_report_ready: bool

    semantic_memories: str
    episodic_memories: str
    procedural_memories: str

    tavily_search_events: Annotated[List[Dict[str, Any]], _merge_event_list]
    tavily_extract_events: Annotated[List[Dict[str, Any]], _merge_event_list]
    tavily_map_events: Annotated[List[Dict[str, Any]], _merge_event_list]
    tavily_crawl_events: Annotated[List[Dict[str, Any]], _merge_event_list]

    filesystem_events: Annotated[List[Dict[str, Any]], _merge_event_list]
    read_file_paths: Annotated[List[str], _merge_unique_str_list]
    written_file_paths: Annotated[List[str], _merge_unique_str_list]
    edited_file_paths: Annotated[List[str], _merge_unique_str_list]

    # Temporal context
    current_date: str
    research_date: str
    temporal_scope_mode: str
    temporal_scope_description: str


def build_run_paths(task_slug: str) -> RunPathsInput:
    return RunPathsInput(
        run_dir=f"runs/{task_slug}",
        report_path=f"reports/{task_slug}.md",
        scratch_dir="scratch",
    )


def input_to_agent_state(run_input: MissionSliceInput) -> Dict[str, Any]:
    paths = build_run_paths(run_input.task_slug)

    return {
        "task_id": run_input.task_id,
        "mission_id": run_input.mission_id,
        "task_slug": run_input.task_slug,
        "user_objective": run_input.user_objective,
        "targets": run_input.targets,
        "selected_tool_names": run_input.selected_tool_names,
        "selected_subagent_names": run_input.selected_subagent_names,
        "report_required_sections": run_input.report_required_sections,
        "stage_type": run_input.stage_type,
        "search_stage": "initialized",
        "official_domains": [],
        "visited_urls": [],
        "findings": [],
        "open_questions": [
            "What are the official domains?",
            "What are the most important validated claims for this research slice?",
            "What still needs confirmation before the final report?",
        ],
        "run_dir": paths.run_dir,
        "report_path": paths.report_path,
        "step_count": 0,
        "max_step_budget": run_input.max_step_budget,
        "final_report_ready": False,
        "semantic_memories": "",
        "episodic_memories": "",
        "procedural_memories": "",
        "tavily_search_events": [],
        "tavily_extract_events": [],
        "tavily_map_events": [],
        "tavily_crawl_events": [],
        "filesystem_events": [],
        "read_file_paths": [],
        "written_file_paths": [],
        "edited_file_paths": [],
        # Temporal context
        "current_date": run_input.effective_current_date,
        "research_date": run_input.effective_research_date,
        "temporal_scope_mode": run_input.temporal_scope.mode,
        "temporal_scope_description": run_input.temporal_scope.description,
    }


# -----------------------------------------------------------------------------
# Memory helpers
# -----------------------------------------------------------------------------


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _extract_item_value(item: Any) -> Any:
    if hasattr(item, "value"):
        return item.value
    return item


def _format_memory_items(title: str, items: List[Any], max_items: int = 5) -> str:
    if not items:
        return f"{title}:\n- none"

    lines = [f"{title}:"]
    for idx, item in enumerate(items[:max_items], start=1):
        value = _extract_item_value(item)
        if isinstance(value, dict):
            kind = value.get("kind", "memory")
            data = value.get("data", {})
            evidence = value.get("evidence", [])
            sources = value.get("sources", [])
            lines.append(f"- [{idx}] kind={kind}")
            if data:
                lines.append(f"  data={_safe_json_dumps(data)}")
            if evidence:
                lines.append(f"  evidence={_safe_json_dumps(evidence[:3])}")
            if sources:
                lines.append(f"  sources={_safe_json_dumps(sources[:3])}")
        else:
            lines.append(f"- [{idx}] {value}")
    return "\n".join(lines)


async def load_memories_for_prompt(
    *,
    manager: Any,
    run_input: MissionSliceInput,
    config: Dict[str, Any],
) -> Dict[str, str]:
    entity_query = " ; ".join(run_input.targets) if run_input.targets else run_input.user_objective
    procedural_query = (
        f"procedural tactics biotech research workflow for mission {run_input.mission_id} "
        f"and stage {run_input.stage_type}"
    )
    episodic_query = (
        f"episodic prior research outcomes for mission {run_input.mission_id} "
        f"about {' ; '.join(run_input.targets) if run_input.targets else 'current targets'}"
    )

    semantic_items = await manager.asearch(query=entity_query, config=config)
    procedural_items = await manager.asearch(query=procedural_query, config=config)
    episodic_items = await manager.asearch(query=episodic_query, config=config)

    return {
        "semantic_memories": _format_memory_items("Semantic", semantic_items, max_items=5),
        "procedural_memories": _format_memory_items("Procedural", procedural_items, max_items=5),
        "episodic_memories": _format_memory_items("Episodic", episodic_items, max_items=5),
    }


def build_memory_ingestion_prompt(
    *,
    run_input: MissionSliceInput,
    final_agent_response: str,
    final_report_text: str,
    final_report_path: str,
    visited_urls: List[str],
    tavily_search_events: List[Dict[str, Any]],
    tavily_extract_events: List[Dict[str, Any]],
    tavily_map_events: List[Dict[str, Any]],
    tavily_crawl_events: List[Dict[str, Any]],
    filesystem_events: List[Dict[str, Any]],
    read_file_paths: List[str],
    written_file_paths: List[str],
    edited_file_paths: List[str],
) -> str:
    search_block = _format_tavily_event_block(
        "Tavily search provenance",
        tavily_search_events,
        max_events=4,
    )
    extract_block = _format_tavily_event_block(
        "Tavily extract provenance",
        tavily_extract_events,
        max_events=4,
    )
    map_block = _format_tavily_event_block(
        "Tavily map provenance",
        tavily_map_events,
        max_events=4,
    )
    crawl_block = _format_tavily_event_block(
        "Tavily crawl provenance",
        tavily_crawl_events,
        max_events=4,
    )
    file_block = _format_file_state_block(
        written_file_paths=written_file_paths,
        edited_file_paths=edited_file_paths,
        read_file_paths=read_file_paths,
        filesystem_events=filesystem_events,
    )

    visited_urls_block = _safe_json_dumps(visited_urls[-30:]) if visited_urls else "[]"

    return f"""
You are preparing a clean memory-ingestion summary for a biotech research run.

Mission ID:
{run_input.mission_id}

Task slug:
{run_input.task_slug}

Stage type:
{run_input.stage_type}

Targets:
{", ".join(run_input.targets)}

Original objective:
{run_input.user_objective}

Final agent response:
{final_agent_response}

Final report path:
{final_report_path}

Final report text:
{_truncate_text(final_report_text, max_chars=12000)}

Visited URLs:
{visited_urls_block}

{search_block}

{map_block}

{crawl_block}

{extract_block}

{file_block}

Use all of the material above to produce a concise but information-dense summary that preserves:
- durable semantic entity facts
- episodic notes about what this run accomplished
- procedural tactics worth reusing

Prioritize:
- official domains
- high-yield sources/pages
- validated relationships and claims
- reusable search/map/extract tactics
- useful output file paths for future stages

Do not repeat raw logs verbatim.
Do not include irrelevant low-signal URLs.
Also include the file paths that should be remembered as useful outputs for this run.
""".strip()


# -----------------------------------------------------------------------------
# Async file helpers
# -----------------------------------------------------------------------------


async def list_agent_files(root: Path = ROOT_FILESYSTEM) -> list[str]:
    def _scan() -> list[str]:
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
        )

    return await asyncio.to_thread(_scan)


async def print_agent_files(root: Path = ROOT_FILESYSTEM) -> None:
    files = await list_agent_files(root)
    print("\n=== AGENT FILES ===")
    if not files:
        print("(no files)")
        return
    for f in files:
        print(f"- {f}")


async def read_file_text(path_relative: str, root: Path = ROOT_FILESYSTEM) -> str:
    path = root / path_relative
    if not await asyncio.to_thread(path.exists):
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def dump_file(path_relative: str, root: Path = ROOT_FILESYSTEM) -> None:
    text = await read_file_text(path_relative, root=root)
    print(f"\n=== {path_relative} ===")
    if not text:
        print("(missing)")
        return
    print(text)
