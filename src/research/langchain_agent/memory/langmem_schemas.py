from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


MemoryKind = Literal["semantic", "episodic", "procedural"]


class MemoryBase(BaseModel):
    kind: MemoryKind
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    observed_at: Optional[str] = None
    ttl_days: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class SemanticEntityFact(MemoryBase):
    """
    Durable entity facts useful across future runs.
    """
    kind: Literal["semantic"] = "semantic"
    entity_name: str
    org_id: Optional[str] = None
    person_id: Optional[str] = None
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Canonical-ish entity facts such as aliases, official domains, founders, "
            "products/services, validated relationships, or other durable attributes."
        ),
    )


class EpisodicResearchRun(MemoryBase):
    """
    Compact note about what happened in one research run.
    """
    kind: Literal["episodic"] = "episodic"
    mission_id: Optional[str] = None
    run_label: Optional[str] = None
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Compact run outcome: what worked, what failed, best sources, remaining gaps, "
            "and notable decisions from the run."
        ),
    )


class ProceduralResearchPlaybook(MemoryBase):
    """
    Reusable tactic for future research runs.
    """
    kind: Literal["procedural"] = "procedural"
    agent_type: str = "biotech_research_slice_agent"
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Reusable tactics such as query patterns, broad-to-narrow search strategy, "
            "mapping heuristics, extraction heuristics, and source validation workflows."
        ),
    )