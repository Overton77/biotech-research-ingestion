"""Prompt text for post-run memory ingestion."""

from __future__ import annotations

from typing import Any, Dict, List

from src.research.langchain_agent.agent.formatting_helpers import (
    _format_file_state_block,
    _format_tavily_event_block,
    _safe_json_dumps,
    _truncate_text,
)
from src.research.langchain_agent.agent.state.agent_state import MissionSliceInput


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
