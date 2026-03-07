"""Tavily web search tool for the Coordinator."""

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


def get_web_search_tool(max_results: int = 5):
    """Return a Tavily search tool (or a stub if no API key)."""
    from src.config import get_settings
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        # Stub that returns a message so the agent can still run in dev
        @tool
        def web_search_stub(query: str) -> str:
            """Web search (stub: TAVILY_API_KEY not set)."""
            return f"[Web search stub] No TAVILY_API_KEY. Query was: {query}"

        return web_search_stub
    return TavilySearchResults(
        max_results=max_results,
        api_key=settings.TAVILY_API_KEY,
        search_depth="basic",
    )
