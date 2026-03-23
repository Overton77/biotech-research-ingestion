"""Tool profile registry — resolves profile names to lists of tool functions.

Decouples TaskDef configs from concrete tool imports.
Tools are instantiated fresh per call to avoid stale state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Tool profile builders
# ---------------------------------------------------------------------------

def _build_tavily_tools() -> list[Any]:
    """Import and return Tavily tools (search, extract, map). Crawl excluded for now."""
    from src.agents.tools.tavily_search_tools import (
        tavily_extract,
        tavily_map,
        tavily_search,
    )
    return [tavily_search, tavily_extract, tavily_map]


def _build_file_search_tools() -> list[Any]:
    return [workspace_glob, workspace_search]


def _build_default_research_tools() -> list[Any]:
    """Tavily only. Subagents get filesystem via FilesystemMiddleware, not workspace_*."""
    return _build_tavily_tools()


def _build_search_only_tools() -> list[Any]:
    return _build_tavily_tools()


def _build_write_only_tools() -> list[Any]:
    return _build_file_search_tools()


TOOL_PROFILES: dict[str, Any] = {
    "default_research": _build_default_research_tools,
    "search_only": _build_search_only_tools,
    "write_only": _build_write_only_tools,
}


def resolve_tool_profile(profile_name: str) -> list[Any]:
    """Resolve a profile name to a list of LangChain tool instances."""
    if profile_name not in TOOL_PROFILES:
        raise ValueError(f"Unknown tool profile: {profile_name!r}")
    return TOOL_PROFILES[profile_name]()
