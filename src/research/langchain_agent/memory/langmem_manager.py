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
from src.research.langchain_agent.memory.langmem_prompts import MEMORY_INSTRUCTIONS 
from src.research.langchain_agent.constants import GPT_5_MINI



async def build_langmem_manager(
    model: str = GPT_5_MINI,
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