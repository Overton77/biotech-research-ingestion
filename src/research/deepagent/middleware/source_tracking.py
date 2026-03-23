"""SourceTrackingMiddleware — intercepts Tavily search tool calls to build
a source provenance chain for every task agent and subagent.

Uses wrap_tool_call to capture every search result as a SourceReference dict,
appending to state["sources_collected"] via Command(update=...).

write_source_index is a standalone @after_agent hook that deduplicates
collected sources and writes sources/index.json to the store.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState, after_agent
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAMES = frozenset({
    "tavily_search", "tavily_extract", "tavily_map", "tavily_crawl",
})


class SourceTrackingState(AgentState):
    """Extended agent state that tracks discovered sources."""
    sources_collected: NotRequired[list[dict[str, Any]]]


class SourceTrackingMiddleware(AgentMiddleware[SourceTrackingState]):
    """Intercepts search tool calls; appends SourceReference dicts to state."""
    state_schema = SourceTrackingState

    def _parse_sources(
        self, tool_name: str, content: str, query: str | None
    ) -> list[dict[str, Any]]:
        """Parse Tavily JSON results into SourceReference-shaped dicts."""
        sources: list[dict[str, Any]] = []
        try:
            data = json.loads(content) if isinstance(content, str) else content
            results_list = data.get("results", []) if isinstance(data, dict) else []
            for r in results_list:
                sources.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": (r.get("content", "") or "")[:300],
                    "accessed_at": datetime.utcnow().isoformat(),
                    "source_type": "web",
                    "tool_name": tool_name,
                    "query": query,
                })
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return sources

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        tool_name = request.tool_call.get("name", "")
        if tool_name in SEARCH_TOOL_NAMES:
            content = result.content if isinstance(result, ToolMessage) else str(result)
            query = request.tool_call.get("args", {}).get("query")
            new_sources = self._parse_sources(tool_name, content, query)
            if new_sources:
                existing = request.state.get("sources_collected", [])
                return Command(update={"sources_collected": existing + new_sources})
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        tool_name = request.tool_call.get("name", "")
        if tool_name in SEARCH_TOOL_NAMES:
            content = result.content if isinstance(result, ToolMessage) else str(result)
            query = request.tool_call.get("args", {}).get("query")
            new_sources = self._parse_sources(tool_name, content, query)
            if new_sources:
                existing = request.state.get("sources_collected", [])
                return Command(update={"sources_collected": existing + new_sources})
        return result


@after_agent(state_schema=SourceTrackingState)
async def write_source_index(
    state: SourceTrackingState, runtime: Runtime
) -> dict | None:
    """Deduplicate collected sources and write sources/index.json to the store."""
    sources = state.get("sources_collected", [])
    if not sources:
        return None

    seen_urls: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    index_data = {
        "total_sources": len(unique_sources),
        "sources": unique_sources,
    }

    if runtime.store:
        try:
            runtime.store.put(
                ("sources",),
                "index",
                index_data,
            )
        except Exception:
            logger.debug("Failed to write source index to store", exc_info=True)

    logger.info(
        "Source index: %d unique sources from %d total collected",
        len(unique_sources), len(sources),
    )
    return None
