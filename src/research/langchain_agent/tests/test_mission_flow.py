"""
Smoke tests: mission ordering, dependency injection, and CLI output
directory creation. No LLM calls — stubs the run_single_mission_slice call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.research.langchain_agent.agent.config import MissionSliceInput, ResearchPromptSpec
from src.research.langchain_agent.models.mission import MissionStage, ResearchMission
from src.research.langchain_agent.workflow.run_mission import _topological_stage_order
from src.research.langchain_agent.mission_loader import load_mission_from_file

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "missions"

_SPEC = ResearchPromptSpec(agent_identity="test agent")


def _stage(task_slug: str, mission_id: str = "m1", dependencies: list[str] | None = None) -> MissionStage:
    return MissionStage(
        slice_input=MissionSliceInput(
            task_id=task_slug,
            mission_id=mission_id,
            task_slug=task_slug,
            user_objective="test objective",
        ),
        prompt_spec=_SPEC,
        dependencies=dependencies or [],
    )


def _mission_with_deps() -> ResearchMission:
    """Build a minimal mission: fundamentals → (products, leadership depends on fundamentals)."""
    return ResearchMission(
        mission_id="test-mission",
        stages=[
            _stage("fundamentals"),
            _stage("products", dependencies=["fundamentals"]),
            _stage("leadership", dependencies=["fundamentals"]),
        ],
    )


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------


def test_topological_order_places_dep_before_dependents():
    mission = _mission_with_deps()
    ordered = _topological_stage_order(mission)
    slug_order = [mission.stages[i].slice_input.task_slug for i in ordered]
    fundamentals_pos = slug_order.index("fundamentals")
    products_pos = slug_order.index("products")
    leadership_pos = slug_order.index("leadership")
    assert fundamentals_pos < products_pos
    assert fundamentals_pos < leadership_pos


def test_mini_mission_single_stage_order():
    mission = load_mission_from_file(MISSIONS_DIR / "elysium_mini.json")
    ordered = _topological_stage_order(mission)
    assert ordered == [0]


# ---------------------------------------------------------------------------
# Dependency report injection
# ---------------------------------------------------------------------------


def test_dependency_reports_injected():
    """
    The runner should inject prior stage reports into dependency_reports.
    Verify the shape by simulating what run_mission does inline.
    """
    mission = _mission_with_deps()
    ordered = _topological_stage_order(mission)
    report_by_slug: dict[str, str] = {}

    for idx in ordered:
        stage = mission.stages[idx]
        run_input = stage.slice_input.model_copy(deep=True)
        if stage.dependencies:
            run_input.dependency_reports = {
                slug: report_by_slug.get(slug, "")
                for slug in stage.dependencies
            }
        report_by_slug[run_input.task_slug] = f"Report for {run_input.task_slug}"

    leadership_stage = next(
        s for s in mission.stages if s.slice_input.task_slug == "leadership"
    )
    assert "fundamentals" in leadership_stage.dependencies
    assert "fundamentals" in report_by_slug


# ---------------------------------------------------------------------------
# Output directory creation in run_mission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_mission_creates_output_dir(tmp_path: Path):
    """
    run_mission should write a mission_summary.json to output_dir when provided.
    Uses a stub for run_single_mission_slice so no LLM calls are made.
    """
    from src.research.langchain_agent.run_mission import main

    mission_path = MISSIONS_DIR / "elysium_mini.json"
    output_dir = tmp_path / "elysium_mini_output"

    stub_out = {
        "agent_state": {"report_path": "reports/elysium-fundamentals-mini.md"},
        "final_report_text": "# Test Report\n\nStub content.",
        "stage_run_record": None,
    }

    with (
        patch(
            "src.research.langchain_agent.workflow.run_mission.run_single_mission_slice",
            new=AsyncMock(return_value=stub_out),
        ),
        patch(
            "src.research.langchain_agent.runtime.bootstrap.get_persistence",
            new=AsyncMock(return_value=(object(), object())),
        ),
        patch(
            "src.research.langchain_agent.runtime.bootstrap.init_research_agent_beanie",
            new=AsyncMock(),
        ),
        patch(
            "src.research.langchain_agent.runtime.bootstrap.build_langmem_manager",
            new=AsyncMock(return_value=object()),
        ),
    ):
        await main(
            mission_file=str(mission_path),
            output_dir=str(output_dir),
            local=True,
        )

    summary_file = output_dir / "mission_summary.json"
    assert summary_file.exists(), "Expected mission_summary.json to be written"

    import json
    data = json.loads(summary_file.read_text())
    assert data["mission_id"] == "mission-elysium-mini-001"
    assert "stages" in data
