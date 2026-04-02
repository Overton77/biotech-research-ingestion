"""
Unit tests: Pydantic model validation for MissionSliceInput, ResearchMission,
and the mission JSON loader. No LLM calls, no network, no external services.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.langchain_agent.agent.config import MissionSliceInput
from src.research.langchain_agent.models.mission import (
    MissionStage,
    ResearchMission,
)
from src.research.langchain_agent.cli.mission_loader import load_mission_from_file
from src.research.langchain_agent.models.plan import (
    ResearchPlanOutput,
    ResearchPlanTask,
)

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
    assert "vercel_agent_browser" in s.selected_subagent_names


def test_mission_slice_input_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool names"):
        MissionSliceInput(
            task_id="t1",
            mission_id="m1",
            task_slug="s",
            user_objective="x",
            selected_tool_names=["nonexistent_tool"],
        )


def test_mission_slice_input_rejects_unknown_subagent():
    with pytest.raises(ValueError, match="Unknown subagent names"):
        MissionSliceInput(
            task_id="t1",
            mission_id="m1",
            task_slug="s",
            user_objective="x",
            selected_subagent_names=["nonexistent_subagent"],
        )


@pytest.mark.parametrize("removed_name", ["pubmed_research", "pubchem_research"])
def test_mission_slice_input_rejects_removed_subagents(removed_name: str):
    with pytest.raises(ValueError, match="Unknown subagent names"):
        MissionSliceInput(
            task_id="t1",
            mission_id="m1",
            task_slug="s",
            user_objective="x",
            selected_subagent_names=[removed_name],
        )


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


def test_load_elysium_subagents_from_file():
    path = MISSIONS_DIR / "elysium_subagents.json"
    mission = load_mission_from_file(path)
    assert mission.mission_id == "mission-elysium-subagents-001"
    assert len(mission.stages) == 3
    assert mission.stages[0].slice_input.selected_subagent_names == [
        "clinicaltrials_research",
        "docling_document",
    ]
    assert mission.stages[2].dependencies == ["elysium-products-compounds"]


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


def test_loaded_mission_stage_slugs_unique():
    for filename in ("elysium_mini.json", "elysium_subagents.json", "qualia_mini.json"):
        path = MISSIONS_DIR / filename
        mission = load_mission_from_file(path)
        slugs = [s.slice_input.task_slug for s in mission.stages]
        assert len(slugs) == len(set(slugs)), f"Duplicate slugs in {mission.mission_name}"


# ---------------------------------------------------------------------------
# LangChain ResearchPlanOutput / ResearchPlanTask
# ---------------------------------------------------------------------------


def _sample_task(**overrides: object) -> dict:
    base = {
        "id": "task-1",
        "title": "Scan landscape",
        "description": "Review public sources",
        "stage": "Discovery",
        "dependencies": [],
        "estimated_duration_minutes": 30,
        "selected_tool_names": ["search_web", "extract_from_urls", "map_website"],
        "selected_subagent_names": ["vercel_agent_browser"],
    }
    base.update(overrides)
    return base


def test_research_plan_output_accepts_valid_tasks():
    out = ResearchPlanOutput(
        title="T",
        objective="O",
        stages=["Discovery"],
        tasks=[
            ResearchPlanTask(
                id="task-1",
                title="A",
                description="D",
                stage="Discovery",
                selected_tool_names=["search_web"],
                selected_subagent_names=["tavily_research"],
            )
        ],
    )
    assert out.tasks[0].stage_type is None
    assert "search_web" in out.tasks[0].selected_tool_names


def test_research_plan_task_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool names"):
        ResearchPlanTask(
            id="x",
            title="t",
            description="d",
            stage="S",
            selected_tool_names=["not_a_real_tool"],
            selected_subagent_names=["vercel_agent_browser"],
        )


def test_research_plan_task_rejects_unknown_subagent():
    with pytest.raises(ValueError, match="Unknown subagent names"):
        ResearchPlanTask(
            id="x",
            title="t",
            description="d",
            stage="S",
            selected_tool_names=["search_web"],
            selected_subagent_names=["phantom_subagent"],
        )


def test_research_plan_output_stage_must_match_stages_list():
    with pytest.raises(ValueError, match="not in stages"):
        ResearchPlanOutput(
            title="T",
            objective="O",
            stages=["A"],
            tasks=[
                ResearchPlanTask(
                    id="task-1",
                    title="t",
                    description="d",
                    stage="WrongStage",
                    selected_tool_names=["search_web"],
                    selected_subagent_names=["vercel_agent_browser"],
                )
            ],
        )


def test_research_plan_task_applies_defaults_for_empty_tool_lists():
    raw = _sample_task(selected_tool_names=[], selected_subagent_names=[])
    task = ResearchPlanTask.model_validate(raw)
    assert task.selected_tool_names
    assert task.selected_subagent_names


def test_coordinator_prompt_lists_tools():
    from src.prompts.coordinator_prompt_builders import RESEARCH_PLAN_SCHEMA_DESCRIPTION

    assert "search_web" in RESEARCH_PLAN_SCHEMA_DESCRIPTION
    assert "vercel_agent_browser" in RESEARCH_PLAN_SCHEMA_DESCRIPTION
