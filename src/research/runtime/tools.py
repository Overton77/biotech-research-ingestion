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
# File search tool stubs (v1: simple workspace search wrappers)
# These complement the built-in filesystem tools from FilesystemMiddleware.
# ---------------------------------------------------------------------------

@tool
def workspace_glob(
    pattern: str,
    root_dir: str = ".",
) -> str:
    """Search for files matching a glob pattern in the workspace.

    Args:
        pattern: Glob pattern (e.g., '**/*.md', 'outputs/*.json')
        root_dir: Root directory to search from (defaults to current directory)
    """
    root = Path(root_dir)
    if not root.exists():
        return f"Directory not found: {root_dir}"
    matches = sorted(root.glob(pattern))
    if not matches:
        return f"No files matching '{pattern}' found in {root_dir}"
    result_lines = [str(m.relative_to(root)) for m in matches[:50]]
    return f"Found {len(matches)} files:\n" + "\n".join(result_lines)


@tool
def workspace_search(
    pattern: str,
    file_pattern: str = "**/*",
    root_dir: str = ".",
) -> str:
    """Search file contents for a regex pattern within the workspace.

    Args:
        pattern: Regex pattern to search for in file contents
        file_pattern: Glob pattern to filter which files to search (e.g., '**/*.md')
        root_dir: Root directory to search from
    """
    root = Path(root_dir)
    if not root.exists():
        return f"Directory not found: {root_dir}"

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    results: list[str] = []
    for filepath in root.glob(file_pattern):
        if not filepath.is_file() or filepath.stat().st_size > 10 * 1024 * 1024:
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    rel = filepath.relative_to(root)
                    results.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(results) >= 50:
                        break
        except Exception:
            continue
        if len(results) >= 50:
            break

    if not results:
        return f"No matches for '{pattern}' in {root_dir}"
    return f"Found {len(results)} matches:\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Tool profile builders
# ---------------------------------------------------------------------------

def _build_tavily_tools() -> list[Any]:
    """Import and return all Tavily tools from the existing codebase."""
    from src.agents.tools.tavily_search_tools import (
        tavily_crawl,
        tavily_extract,
        tavily_map,
        tavily_search,
    )
    return [tavily_search, tavily_extract, tavily_map, tavily_crawl]


def _build_file_search_tools() -> list[Any]:
    return [workspace_glob, workspace_search]


def _build_default_research_tools() -> list[Any]:
    return _build_tavily_tools() + _build_file_search_tools()


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
