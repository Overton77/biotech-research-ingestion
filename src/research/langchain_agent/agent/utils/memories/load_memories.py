from __future__ import annotations
from typing import Any, Dict
from src.research.langchain_agent.agent.constants import GPT_5_4_MINI
from src.research.langchain_agent.agent.formatting_helpers import _format_memory_items
from src.research.langchain_agent.agent.state.agent_state import MissionSliceInput 



async def load_memories_for_prompt(
    *,
    manager: Any,
    run_input: MissionSliceInput,
    config: Dict[str, Any],
) -> Dict[str, str]:
    entity_query = " ; ".join(run_input.targets) if run_input.targets else run_input.user_objective
    procedural_query = (
        f"procedural tactics biotech research workflow for mission {run_input.mission_id} "
        f"and stage {run_input.stage_type}"
    )
    episodic_query = (
        f"episodic prior research outcomes for mission {run_input.mission_id} "
        f"about {' ; '.join(run_input.targets) if run_input.targets else 'current targets'}"
    )

    semantic_items = await manager.asearch(query=entity_query, config=config)
    procedural_items = await manager.asearch(query=procedural_query, config=config)
    episodic_items = await manager.asearch(query=episodic_query, config=config)

    return {
        "semantic_memories": _format_memory_items("Semantic", semantic_items, max_items=5),
        "procedural_memories": _format_memory_items("Procedural", procedural_items, max_items=5),
        "episodic_memories": _format_memory_items("Episodic", episodic_items, max_items=5),
    }