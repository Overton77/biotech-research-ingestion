"""Coordinator agent — langchain.create_agent with HumanInTheLoopMiddleware."""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple, Literal, Optional

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
from src.agents.tools.web_search import get_web_search_tool
from src.config import get_settings

load_dotenv()

logger = logging.getLogger(__name__)

COORDINATOR_SYSTEM_PROMPT = """You are the Coordinator for a Deep Biotech Research system. Your role is to:
- Accept research objectives from the user and clarify scope when needed.
- Use web search to gather initial context and understand the landscape of the topic.
- When the user is ready to formalize their research, use the create_research_plan tool to propose a structured plan.
  When calling create_research_plan, you MUST provide the complete plan upfront:
    - title: A concise descriptive title
    - objective: The research objective (same as the user's)
    - stages: A list of high-level stage names (e.g. ["Literature Review", "Analysis", "Synthesis"])
    - tasks: A list of task objects, each with keys:
        id, title, description, stage, dependencies (list of task ids), estimated_duration_minutes
    - context: A brief summary of what you found via web search
  The plan will be paused for human review before execution begins.
- After a plan is approved, confirm the approval to the user and summarize next steps.
  After plan approval, the system will automatically compile and execute a research mission using Deep Agent workers. You do not trigger execution yourself — that is handled externally via the API.
- If a plan is rejected, ask the user what changes they want and create a revised plan.
- Be concise and focused on the research objectives.
"""

_coordinator_graph: CompiledStateGraph | None = None



async def create_coordinator_graph(tools: list[BaseTool] | None = None) -> CompiledStateGraph:
    settings = get_settings()  

    

    
    persistence_bundle: Tuple[AsyncPostgresStore, AsyncPostgresSaver] = await get_coordinator_persistence() 
    store: AsyncPostgresStore = persistence_bundle[0]
    checkpointer: AsyncPostgresSaver = persistence_bundle[1]
    

    model = ChatOpenAI(
        model="gpt-5-mini",
        api_key=settings.OPENAI_API_KEY or "not-set",
        temperature=0,
    ) 

    

    if tools is None:
        tools = [get_web_search_tool(), create_research_plan]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
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


async def get_coordinator_graph(tools: list[BaseTool] | None = None) -> CompiledStateGraph:
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