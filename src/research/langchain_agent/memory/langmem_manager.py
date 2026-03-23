from __future__ import annotations

from langgraph.store.base import BaseStore 
from langmem import create_memory_store_manager  
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.research.langchain_agent.storage.langgraph_persistence import get_persistence
from src.research.langchain_agent.memory.langmem_schemas import (
    SemanticEntityFact,
    EpisodicResearchRun,
    ProceduralResearchPlaybook,
)


MEMORY_INSTRUCTIONS = """
You are a memory extraction engine for a biotech research agent.

Extract only useful long-term memories in these three categories:

1. semantic
- durable entity facts
- examples: official domains, aliases, founders, products/services, validated relationships

2. episodic
- compact notes about a completed research run
- examples: what worked, what failed, high-yield source types, remaining gaps

3. procedural
- reusable tactics that improve future runs
- examples: query patterns, broad-to-narrow search tactics, mapping heuristics, extraction heuristics

Do not store:
- raw scraped text
- long transcripts
- low-confidence speculation
- repetitive notes
- temporary conversational filler

Always include:
- kind
- confidence
- evidence
- sources

Prefer concise, reusable, durable memory entries.
For volatile facts, include observed_at and optionally ttl_days.
"""


async def build_langmem_manager(
    model: str = "openai:gpt-4.1",
    query_model: str | None = None,
    query_limit: int = 5, 
    store: BaseStore | None = None, 
   
):
    """
    Build a store-backed LangMem manager.

    Namespace is mission-scoped via runtime config:
        config = {"configurable": {"mission_id": "..."}}

    The manager will automatically:
    - search relevant memories from the store
    - extract/update memories
    - write changes back to the store
    """
    

    manager = create_memory_store_manager(
        model,
        schemas=[
            SemanticEntityFact,
            EpisodicResearchRun,
            ProceduralResearchPlaybook,
        ],
        instructions=MEMORY_INSTRUCTIONS,
        enable_inserts=True,
        enable_deletes=False,
        query_model=query_model,
        query_limit=query_limit,
        namespace=("memories", "{mission_id}"),
        store=store, 
       
        
    )
    return manager