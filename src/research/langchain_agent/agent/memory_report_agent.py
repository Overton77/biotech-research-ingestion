from __future__ import annotations
from typing import Any
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore 
from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.research.langchain_agent.agent.config import ResearchTaskMemoryReport
from src.research.langchain_agent.agent.constants import GPT_5_4_MINI


def build_memory_report_agent(
    store: BaseStore, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model=GPT_5_4_MINI,
        tools=[],
        middleware=[],
        response_format=ResearchTaskMemoryReport,
        store=store,
        checkpointer=checkpointer,
    )
