"""Static agent constants: tool registry, filesystem roots, model id."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Final

from langchain.tools import BaseTool

from src.research.langchain_agent.agent.subagent_types import DEFAULT_STAGE_SUBAGENT_NAMES
from src.research.langchain_agent.tools.search.tavily import (
    crawl_website,
    extract_from_urls,
    map_website,
    search_web,
)

TOOLS_MAP: Dict[str, BaseTool] = {
    "search_web": search_web,
    "extract_from_urls": extract_from_urls,
    "map_website": map_website,
    "crawl_website": crawl_website,
}

# -----------------------------------------------------------------------------
# Paths (root is …/langchain_agent/agent_outputs)
# -----------------------------------------------------------------------------

ROOT_FILESYSTEM = (Path(__file__).resolve().parent.parent / "agent_outputs").resolve()
RUNS_DIR = ROOT_FILESYSTEM / "runs"
REPORTS_DIR = ROOT_FILESYSTEM / "reports"
SCRATCH_DIR = ROOT_FILESYSTEM / "scratch"

GPT_5_4_MINI: Final[str] = "gpt-5.4-mini"

_WIN_LONG_PREFIX = "\\\\?\\"

__all__ = [
    "DEFAULT_STAGE_SUBAGENT_NAMES",
    "GPT_5_4_MINI",
    "REPORTS_DIR",
    "ROOT_FILESYSTEM",
    "RUNS_DIR",
    "SCRATCH_DIR",
    "TOOLS_MAP",
    "_WIN_LONG_PREFIX",
]
