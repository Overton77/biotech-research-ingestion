"""
Unit tests: Pydantic model validation for MissionSliceInput, ResearchMission,
and the mission JSON loader. No LLM calls, no network, no external services.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.langchain_agent.agent.config import MissionSliceInput
from src.research.langchain_agent.models.mission import (
    ELYSIUM_RESEARCH_MISSION,
    QUALIA_RESEARCH_MISSION,
    MissionStage,
    ResearchMission,
)
from src.research.langchain_agent.run_mission import load_mission_from_file

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "missions"


# ---------------------------------------------------------------------------
# MissionSliceInput
# ---------------------------------------------------------------------------


def test_mission_slice_input_defaults():
    s = MissionSliceInput(
        task_id="t1",
        mission_id="m1",
        task_slug="my-stage",
        user_objective="do something",
    )
    assert s.max_step_budget == 12
    assert s.stage_type == "discovery"
    assert s.dependency_reports == {}
    assert "search_web" in s.selected_tool_names


def test_mission_slice_input_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool names"):
        MissionSliceInput(
            task_id="t1",
            mission_id="m1",
            task_slug="s",
            user_objective="x",
            selected_tool_names=["nonexistent_tool"],
        )


# ---------------------------------------------------------------------------
# ResearchMission structure
# ---------------------------------------------------------------------------


def test_qualia_mission_has_3_stages():
    assert len(QUALIA_RESEARCH_MISSION.stages) == 3


def test_elysium_mission_has_3_stages():
    assert len(ELYSIUM_RESEARCH_MISSION.stages) == 3


def test_elysium_leadership_depends_on_fundamentals():
    leadership = next(
        s for s in ELYSIUM_RESEARCH_MISSION.stages
        if s.slice_input.task_slug == "elysium-leadership-and-advisors"
    )
    assert "elysium-company-fundamentals" in leadership.dependencies


def test_mission_stage_slugs_unique():
    for mission in (QUALIA_RESEARCH_MISSION, ELYSIUM_RESEARCH_MISSION):
        slugs = [s.slice_input.task_slug for s in mission.stages]
        assert len(slugs) == len(set(slugs)), f"Duplicate slugs in {mission.mission_name}"


# ---------------------------------------------------------------------------
# Mission JSON loader
# ---------------------------------------------------------------------------


def test_load_elysium_mini_from_file():
    path = MISSIONS_DIR / "elysium_mini.json"
    mission = load_mission_from_file(path)
    assert isinstance(mission, ResearchMission)
    assert mission.mission_id == "mission-elysium-mini-001"
    assert len(mission.stages) == 1
    stage = mission.stages[0]
    assert stage.slice_input.task_slug == "elysium-fundamentals-mini"
    assert stage.slice_input.max_step_budget == 4
    assert stage.slice_input.selected_tool_names == ["search_web"]
    assert stage.dependencies == []


def test_load_qualia_mini_from_file():
    path = MISSIONS_DIR / "qualia_mini.json"
    mission = load_mission_from_file(path)
    assert mission.mission_id == "mission-qualia-mini-001"
    assert mission.stages[0].slice_input.stage_type == "entity_validation"


def test_prompt_spec_loaded_correctly():
    path = MISSIONS_DIR / "elysium_mini.json"
    mission = load_mission_from_file(path)
    spec = mission.stages[0].prompt_spec
    # agent_identity describes the mini test persona
    assert "biotech" in spec.agent_identity.lower() or "research agent" in spec.agent_identity.lower()
    # domain_scope references Elysium
    assert any("Elysium" in item for item in spec.domain_scope)
    assert len(spec.workflow) >= 1
    assert len(spec.intermediate_files) >= 1


def test_load_mission_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_mission_from_file(Path("nonexistent_mission.json"))
