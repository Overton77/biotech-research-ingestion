from typing import Dict, List
from src.research.langchain_agent.models.mission import ResearchMission


def _topological_stage_order(mission: ResearchMission) -> List[int]:
    """
    Return stage indices in an order that respects dependencies.
    For each stage, every stage in its dependencies list (by task_slug) must appear before it.
    """
    slug_to_index: Dict[str, int] = {
        stage.slice_input.task_slug: i for i, stage in enumerate(mission.stages)
    }
    ordered: List[int] = []

    while len(ordered) < len(mission.stages):
        added = False
        for i, stage in enumerate(mission.stages):
            if i in ordered:
                continue
            deps_met = True
            for dep_slug in stage.dependencies:
                if dep_slug in slug_to_index:
                    if slug_to_index[dep_slug] not in ordered:
                        deps_met = False
                        break
            if deps_met:
                ordered.append(i)
                added = True
                break
        if not added:
            # Cycle or missing dependency; include remaining in current order
            for i in range(len(mission.stages)):
                if i not in ordered:
                    ordered.append(i)
            break

    return ordered
