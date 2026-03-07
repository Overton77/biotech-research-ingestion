"""Tavily web search tool for the Coordinator."""

from langchain_core.tools import tool


def get_web_search_tool(max_results: int = 5):
    """Return a Tavily search tool (or a stub if no API key)."""
    from src.config import get_settings
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        @tool
        def web_search_stub(query: str) -> str:
            """Web search (stub: TAVILY_API_KEY not set)."""
            return f"[Web search stub] No TAVILY_API_KEY. Query was: {query}"
        return web_search_stub

    import os
    os.environ.setdefault("TAVILY_API_KEY", settings.TAVILY_API_KEY)

    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=max_results)
