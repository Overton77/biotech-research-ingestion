from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Namespace = Tuple[str, ...]


@dataclass(frozen=True)
class NS:
    @staticmethod
    def semantic_org(org_id: str) -> Namespace:
        return ("memories", "semantic", "org", org_id)

    @staticmethod
    def semantic_person(person_id: str) -> Namespace:
        return ("memories", "semantic", "person", person_id)

    @staticmethod
    def semantic_unscoped() -> Namespace:
        return ("memories", "semantic", "unscoped")

    @staticmethod
    def episodic_mission(mission_id: str) -> Namespace:
        return ("memories", "episodic", "mission", mission_id)

    @staticmethod
    def episodic_unscoped() -> Namespace:
        return ("memories", "episodic", "unscoped")

    @staticmethod
    def procedural_agent(agent_type: str) -> Namespace:
        return ("memories", "procedural", "agent", agent_type)

    @staticmethod
    def procedural_general() -> Namespace:
        return ("memories", "procedural", "general")


def choose_semantic_namespace(
    *,
    org_id: Optional[str],
    person_id: Optional[str],
) -> Namespace:
    if org_id:
        return NS.semantic_org(org_id)
    if person_id:
        return NS.semantic_person(person_id)
    return NS.semantic_unscoped()