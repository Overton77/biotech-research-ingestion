"""Load ``ResearchMission`` definitions from JSON (human- or compiler-written)."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.langchain_agent.agent.config import MissionSliceInput, ResearchPromptSpec
from src.research.langchain_agent.models.mission import (
    IterativeStageConfig,
    MissionStage,
    ResearchMission,
)
from src.research.langchain_agent.unstructured.models import UnstructuredIngestionConfig


def load_mission_from_file(path: Path | str) -> ResearchMission:
    """
    Load a ResearchMission from a JSON file.

    The JSON schema mirrors the Pydantic ResearchMission model.
    ``prompt_spec`` is reconstructed as a ``ResearchPromptSpec`` dataclass.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mission file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    stages: list[MissionStage] = []
    for stage_data in raw.get("stages", []):
        spec_data: dict = stage_data.get("prompt_spec", {})
        prompt_spec = ResearchPromptSpec(
            agent_identity=spec_data.get("agent_identity", "You are a biotech research agent."),
            domain_scope=spec_data.get("domain_scope", []),
            workflow=spec_data.get("workflow", []),
            tool_guidance=spec_data.get("tool_guidance", []),
            subagent_guidance=spec_data.get("subagent_guidance", []),
            practical_limits=spec_data.get("practical_limits", []),
            filesystem_rules=spec_data.get("filesystem_rules", []),
            intermediate_files=spec_data.get("intermediate_files", []),
        )

        iter_cfg_data = stage_data.get("iterative_config")
        iterative_config = IterativeStageConfig(**iter_cfg_data) if iter_cfg_data else None

        stages.append(
            MissionStage(
                slice_input=MissionSliceInput(**stage_data["slice_input"]),
                prompt_spec=prompt_spec,
                execution_reminders=stage_data.get("execution_reminders", []),
                dependencies=stage_data.get("dependencies", []),
                iterative_config=iterative_config,
            )
        )

    ui_raw = raw.get("unstructured_ingestion")
    unstructured_ingestion = (
        UnstructuredIngestionConfig(**ui_raw) if ui_raw else UnstructuredIngestionConfig()
    )

    return ResearchMission(
        mission_id=raw["mission_id"],
        mission_name=raw.get("mission_name", raw["mission_id"]),
        base_domain=raw.get("base_domain", ""),
        stages=stages,
        run_kg=raw.get("run_kg", False),
        unstructured_ingestion=unstructured_ingestion,
    )
