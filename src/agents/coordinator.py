"""Coordinator agent — langchain.create_agent with HumanInTheLoopMiddleware."""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple, Literal, Optional, Union, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from src.agents.persistence import (
    ENV_COORDINATOR_URI,
    ENV_POSTGRES_URI,
    ENV_POSTGRES_URL,
    close_persistence,
    get_coordinator_persistence,
)
from src.agents.tools.create_plan import create_research_plan
from src.agents.tools.openai_prebuilt import openai_web_search
from src.config import get_settings 
from src.prompts.coordinator_prompt_builders import COORDINATOR_SYSTEM_PROMPT_TEXT 


# TODO: Replace raw prompts with langsmith later on after research is succesful 

load_dotenv()

logger = logging.getLogger(__name__)

_coordinator_graph: CompiledStateGraph | None = None



async def create_coordinator_graph(tools: list[Union[BaseTool, Dict[str, str]]] | None = None) -> CompiledStateGraph:
    settings = get_settings()  

    

    
    persistence_bundle: Tuple[AsyncPostgresStore, AsyncPostgresSaver] = await get_coordinator_persistence() 
    store: AsyncPostgresStore = persistence_bundle[0]
    checkpointer: AsyncPostgresSaver = persistence_bundle[1]
    

    model = ChatOpenAI(
        model="gpt-5-mini",
        api_key=settings.OPENAI_API_KEY or "not-set", 
        use_responses_api=True, 

        temperature=0,
    )

    
    if tools is None:
        tools = [create_research_plan, openai_web_search]

    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=COORDINATOR_SYSTEM_PROMPT_TEXT,
        checkpointer=checkpointer,
        store=store,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "create_research_plan": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                },
                description_prefix="Research plan pending review",
            ),
        ],
    )

    logger.info("Coordinator agent created with HumanInTheLoopMiddleware")
    return agent


async def get_coordinator_graph(tools: list[Union[BaseTool, Dict[str, str]]] | None = None) -> CompiledStateGraph:
    """Return the coordinator graph, creating it lazily on first call."""
    global _coordinator_graph
    if _coordinator_graph is None:
        _coordinator_graph = await create_coordinator_graph(tools=tools)
    return _coordinator_graph


async def reset_coordinator_graph() -> None:
    """Reset cached graph on shutdown."""
    global _coordinator_graph
    _coordinator_graph = None
    uri = (
        os.environ.get(ENV_COORDINATOR_URI)
        or os.environ.get(ENV_POSTGRES_URL)
        or os.environ.get(ENV_POSTGRES_URI)
    )
    if uri:
        await close_persistence(uri)