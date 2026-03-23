"""Isolated Tavily crawl API test (instructions + path focus).

Aligned with ``.agents/skills/tavily-crawl/SKILL.md``:
  - Agentic / LLM use: ``instructions`` + ``chunks_per_source`` (relevant chunks, not full pages).
  - Conservative: ``max_depth=1``, low ``limit``, explicit ``--limit`` equivalent.
  - Narrow scope: ``select_paths`` regex on the docs subtree.
  - Same-site: ``allow_external=False`` (cheaper than following external links).

Uses ``tavily_crawl`` / ``format_tavily_crawl_response`` from ``tavily_functions.py``.
Writes under ``test_runs/utils/artifacts/crawl_isolated/``.

Usage (repo root)::

    uv run python -m src.research.test_runs.test_tavily_crawl
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from src.agents.tools.utils.tavily_functions import (
    format_tavily_crawl_response,
    tavily_crawl,
)
from src.clients.async_tavily_client import async_tavily_client
from src.research.langchain_agent.utils import save_json_artifact, save_text_artifact

logger = logging.getLogger(__name__)


def _stdout_safe(text: str) -> str:
    """Avoid UnicodeEncodeError on Windows cp1252 consoles when printing crawl content."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc)


RUN_NAME = "crawl_isolated"

# Inexpensive but precise: official docs root, semantic focus, documentation paths only.
CRAWL_URL = "https://docs.tavily.com/"
INSTRUCTIONS = (
    "Python SDK: pip install, TavilyClient, API key, and a minimal search() example."
)
SELECT_PATHS = [r"/documentation/.*"]


async def run_isolated_crawl() -> tuple[Path, Path]:
    slug = "docs_tavily_semantic"

    raw = await tavily_crawl(
        client=async_tavily_client,
        url=CRAWL_URL,
        instructions=INSTRUCTIONS,
        chunks_per_source=2,
        max_depth=1,
        max_breadth=10,
        limit=6,
        select_paths=SELECT_PATHS,
        allow_external=False,
        extract_depth="basic",
        format="markdown",
        include_images=False,
        timeout=60.0,
        include_usage=False,
    )

    payload = {
        "url": CRAWL_URL,
        "instructions": INSTRUCTIONS,
        "args": {
            "chunks_per_source": 2,
            "max_depth": 1,
            "max_breadth": 10,
            "limit": 6,
            "select_paths": SELECT_PATHS,
            "allow_external": False,
            "extract_depth": "basic",
            "format": "markdown",
            "timeout": 60.0,
        },
        "results": raw,
    }
    json_path = await save_json_artifact(
        payload,
        RUN_NAME,
        "tavily_crawl_isolated_raw",
        suffix=slug,
    )

    formatted = format_tavily_crawl_response(
        raw,
        max_results=6,
        max_content_chars=2400,
    )
    md_path = await save_text_artifact(
        formatted,
        RUN_NAME,
        "tavily_crawl_isolated_formatted",
        suffix=slug,
        extension="md",
    )

    n = len(raw.get("results") or [])
    logger.info("Crawl finished: %s page(s), json=%s md=%s", n, json_path, md_path)
    print(f"Pages returned: {n}")
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved MD:   {md_path.resolve()}")
    if n == 0:
        print(
            "(No pages — try relaxing select_paths or instructions; "
            "API may have changed doc URL patterns.)"
        )
    else:
        preview = formatted[:1800] + ("\n\n... [truncated]" if len(formatted) > 1800 else "")
        print("\n--- Formatted preview ---\n")
        print(_stdout_safe(preview))

    return json_path, md_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run_isolated_crawl())


if __name__ == "__main__":
    main()
