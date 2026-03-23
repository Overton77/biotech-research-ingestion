"""Run live Tavily smoke checks (search -> extract -> crawl).

Usage from repo root:
    uv run python -m src.research.test_runs.tavily_tools
"""

from __future__ import annotations

import asyncio
import logging

from src.research.langchain_agent.tools_for_test.tavily_tools import run_tavily_smoke_test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run_tavily_smoke_test())


if __name__ == "__main__":
    main()
