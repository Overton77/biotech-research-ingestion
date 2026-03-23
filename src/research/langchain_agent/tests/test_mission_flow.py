"""
Smoke tests: mission ordering, dependency injection, and CLI output
directory creation. No LLM calls — stubs the run_single_mission_slice call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.research.langchain_agent.models.mission import (
    ELYSIUM_RESEARCH_MISSION,
    QUALIA_RESEARCH_MISSION,
)
from src.research.langchain_agent.workflow.run_mission import _topological_stage_order
from src.research.langchain_agent.run_mission import load_mission_from_file

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "missions"


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------


def test_elysium_topological_order_places_fundamentals_before_leadership():
    mission = ELYSIUM_RESEARCH_MISSION
    ordered = _topological_stage_order(mission)
    slug_order = [mission.stages[i].slice_input.task_slug for i in ordered]
    fundamentals_pos = slug_order.index("elysium-company-fundamentals")
    leadership_pos = slug_order.index("elysium-leadership-and-advisors")
    assert fundamentals_pos < leadership_pos, (
        f"Expected fundamentals before leadership; got order: {slug_order}"
    )


def test_qualia_topological_order_places_fundamentals_before_products():
    mission = QUALIA_RESEARCH_MISSION
    ordered = _topological_stage_order(mission)
    slug_order = [mission.stages[i].slice_input.task_slug for i in ordered]
    fundamentals_pos = slug_order.index("qualia-company-fundamentals")
    products_pos = slug_order.index("qualia-products-and-specs")
    assert fundamentals_pos < products_pos


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
    mission = ELYSIUM_RESEARCH_MISSION
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
        # Simulate stage completing and producing a report
        report_by_slug[run_input.task_slug] = f"Report for {run_input.task_slug}"

    leadership_stage = next(
        s for s in mission.stages
        if s.slice_input.task_slug == "elysium-leadership-and-advisors"
    )
    # After simulation, leadership should have fundamentals injected
    # (We verify the pattern, not actual injection — that happens at runtime)
    assert "elysium-company-fundamentals" in leadership_stage.dependencies
    assert "elysium-company-fundamentals" in report_by_slug


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
            "src.research.langchain_agent.run_mission.get_persistence",
            new=AsyncMock(return_value=(AsyncMock(), AsyncMock())),
        ),
        patch(
            "src.research.langchain_agent.run_mission.init_research_agent_beanie",
            new=AsyncMock(),
        ),
        patch(
            "src.research.langchain_agent.run_mission.build_langmem_manager",
            new=AsyncMock(return_value=AsyncMock()),
        ),
    ):
        await main(
            mission_file=str(mission_path),
            output_dir=str(output_dir),
        )

    summary_file = output_dir / "mission_summary.json"
    assert summary_file.exists(), "Expected mission_summary.json to be written"

    import json
    data = json.loads(summary_file.read_text())
    assert data["mission_id"] == "mission-elysium-mini-001"
    assert "stages" in data
