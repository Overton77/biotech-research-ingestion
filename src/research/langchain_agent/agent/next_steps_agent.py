from __future__ import annotations
from typing import Any
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.research.langchain_agent.agent.config import NextStepsArtifact
from src.research.langchain_agent.agent.constants import GPT_5_4_MINI
from src.research.langchain_agent.agent.prompts.next_steps_prompt import NEXT_STEPS_EXTRACTION_PROMPT
from langchain.agents import create_agent

def build_next_steps_agent(
    store: BaseStore, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a structured-output agent that evaluates an iteration and produces NextStepsArtifact."""
    return create_agent(
        model=GPT_5_4_MINI,
        tools=[],
        system_prompt=NEXT_STEPS_EXTRACTION_PROMPT,
        response_format=NextStepsArtifact,
        store=store,
        checkpointer=checkpointer,
    )
