"""
Agent configuration: graph state schema, path builders, and barrel re-exports.

Domain models live in ``state/``, prompts in ``prompts/``, helpers in ``utils/``.
Import from here for backward compatibility with runners and workers.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain.agents import AgentState

from src.research.langchain_agent.agent.constants import (
    GPT_5_4_MINI,
    ROOT_FILESYSTEM,
    RUNS_DIR,
    REPORTS_DIR,
    SCRATCH_DIR,
    TOOLS_MAP,
)
from src.research.langchain_agent.agent.formatting_helpers import (
    _safe_json_dumps,
    _truncate_text,
)
from src.research.langchain_agent.agent.prompts.memory_prompts import build_memory_ingestion_prompt
from src.research.langchain_agent.agent.prompts.prompt_builders import (
    BASE_SYSTEM_PROMPT,
    PROMPT_SPEC,
    ResearchPromptSpec,
)
from src.research.langchain_agent.agent.state.agent_state import (
    MissionSliceInput,
    NextStepItem,
    NextStepsArtifact,
    ResearchTaskMemoryReport,
    RunPathsInput,
)
from src.research.langchain_agent.agent.utils.file_helpers import (
    dump_file,
    ensure_dirs,
    list_agent_files,
    print_agent_files,
    read_file_text,
)
from src.research.langchain_agent.agent.utils.memories.load_memories import load_memories_for_prompt
from src.research.langchain_agent.agent.utils.reducers import _merge_event_list, _merge_unique_str_list


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
        "current_date": run_input.effective_current_date,
        "research_date": run_input.effective_research_date,
        "temporal_scope_mode": run_input.temporal_scope.mode,
        "temporal_scope_description": run_input.temporal_scope.description,
    }


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "BiotechResearchAgentState",
    "GPT_5_4_MINI",
    "MissionSliceInput",
    "NextStepItem",
    "NextStepsArtifact",
    "PROMPT_SPEC",
    "ROOT_FILESYSTEM",
    "REPORTS_DIR",
    "ResearchPromptSpec",
    "ResearchTaskMemoryReport",
    "RUNS_DIR",
    "RunPathsInput",
    "SCRATCH_DIR",
    "TOOLS_MAP",
    "build_memory_ingestion_prompt",
    "build_run_paths",
    "dump_file",
    "ensure_dirs",
    "input_to_agent_state",
    "list_agent_files",
    "load_memories_for_prompt",
    "print_agent_files",
    "read_file_text",
    "_safe_json_dumps",
    "_truncate_text",
]
