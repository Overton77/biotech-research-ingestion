"""
Entrypoint: run a full ResearchMission with dependency ordering and report injection.

Usage examples
--------------
# Named mission (hardcoded definitions in models/mission.py)
uv run python -m src.research.langchain_agent.run_mission --mission qualia
uv run python -m src.research.langchain_agent.run_mission --mission elysium

# Mission from a JSON file (written by agent or human)
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/elysium_mini.json

# Mission from file + write outputs to a directory
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/elysium_mini.json \
  --output-dir   src/research/langchain_agent/test_runs/run_outputs/elysium_mini

# Run only one stage (by task_slug)
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/elysium_mini.json \
  --stage elysium-fundamentals-mini

From code
---------
  from src.research.langchain_agent.run_mission import main
  await main(mission=my_research_mission, output_dir="/path/to/out")

Output directory layout (when --output-dir is given)
------------------------------------------------------
  <output-dir>/
    mission_summary.json      ← written at the end (or on failure)
    stage_<slug>.json         ← written immediately after each stage
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.research.langchain_agent.agent.config import (
    ROOT_FILESYSTEM,
    ResearchPromptSpec,
    list_agent_files,
    print_agent_files,
    dump_file,
)
from src.research.langchain_agent.models.mission import (
    QUALIA_RESEARCH_MISSION,
    ELYSIUM_RESEARCH_MISSION,
    IterativeStageConfig,
    MissionStage,
    ResearchMission,
)
from src.research.langchain_agent.observability.tracing import mission_tracing_context
from src.research.langchain_agent.workflow.run_mission import run_mission as run_mission_workflow

logger = logging.getLogger(__name__)

# Named missions for --mission flag
NAMED_MISSIONS: dict[str, ResearchMission] = {
    "qualia": QUALIA_RESEARCH_MISSION,
    "elysium": ELYSIUM_RESEARCH_MISSION,
}


# ---------------------------------------------------------------------------
# Mission JSON loader
# ---------------------------------------------------------------------------


def load_mission_from_file(path: Path | str) -> ResearchMission:
    """
    Load a ResearchMission from a JSON file.

    The JSON schema mirrors the Pydantic ResearchMission model.
    prompt_spec is reconstructed as a ResearchPromptSpec dataclass.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mission file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    stages: list[MissionStage] = []
    for stage_data in raw.get("stages", []):
        # Reconstruct ResearchPromptSpec from plain dict
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

        from src.research.langchain_agent.agent.config import MissionSliceInput

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

    return ResearchMission(
        mission_id=raw["mission_id"],
        mission_name=raw.get("mission_name", raw["mission_id"]),
        base_domain=raw.get("base_domain", ""),
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _serialize_stage_output(out: dict[str, Any]) -> dict[str, Any]:
    """Convert a stage output dict to a JSON-safe dict."""
    record = out.get("stage_run_record")
    record_dict: dict | None = None
    if record is not None:
        try:
            record_dict = record.model_dump(mode="json")
        except Exception:
            record_dict = {"error": "non-serializable stage_run_record"}

    return {
        "task_slug": (
            out.get("agent_state", {}).get("report_path", "unknown")
            .replace("reports/", "").replace(".md", "")
        ),
        "report_path": out.get("agent_state", {}).get("report_path", ""),
        "final_report_text_preview": (out.get("final_report_text") or "")[:500],
        "stage_run_record": record_dict,
    }


def _write_output_file(output_dir: Path, filename: str, data: dict) -> None:
    """Write a JSON file to output_dir; log and continue on failure."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write output file %s: %s", filename, exc)


def _write_iterative_stage_summaries(
    mission: ResearchMission,
    outputs: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Write combined reports and iteration logs for iterative stages."""
    from src.research.langchain_agent.workflow.iteration_control import (
        synthesize_iteration_reports,
    )

    for stage in mission.stages:
        if stage.iterative_config is None:
            continue

        slug = stage.slice_input.task_slug
        stage_outputs = [
            o for o in outputs
            if (o.get("stage_run_record") and o["stage_run_record"].task_slug == slug)
        ]
        if not stage_outputs:
            continue

        reports = [o.get("final_report_text") or "" for o in stage_outputs]
        combined = synthesize_iteration_reports(
            reports=reports,
            task_slug=slug,
            user_objective=stage.slice_input.user_objective,
        )
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"stage_{slug}_combined_report.md").write_text(
                combined, encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write combined report for %s: %s", slug, exc)

        next_steps_history = []
        for o in stage_outputs:
            record = o.get("stage_run_record")
            if record and record.iteration is not None:
                next_steps_history.append({
                    "iteration": record.iteration,
                    "task_slug": slug,
                })

        iteration_log = {
            "task_slug": slug,
            "max_iterations": stage.iterative_config.max_iterations,
            "iterations_completed": len(stage_outputs),
            "iteration_reports": [
                {"iteration": i + 1, "report_preview": (r or "")[:300]}
                for i, r in enumerate(reports)
            ],
        }
        _write_output_file(out_dir, f"stage_{slug}_iteration_log.json", iteration_log)


# ---------------------------------------------------------------------------
# Resolve mission
# ---------------------------------------------------------------------------


def _get_mission(
    mission: Optional[ResearchMission],
    mission_name: Optional[str],
    mission_file: Optional[str],
) -> ResearchMission:
    if mission is not None:
        return mission
    if mission_file:
        return load_mission_from_file(Path(mission_file))
    if mission_name and mission_name in NAMED_MISSIONS:
        return NAMED_MISSIONS[mission_name]
    return QUALIA_RESEARCH_MISSION


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def main(
    mission: Optional[ResearchMission] = None,
    *,
    mission_name: Optional[str] = None,
    mission_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    stage_filter: Optional[str] = None,
) -> None:
    """
    Run a full ResearchMission.

    Args:
        mission:       ResearchMission object. Takes precedence over other args.
        mission_name:  Named key ('qualia', 'elysium') for CLI use.
        mission_file:  Path to a JSON mission file.
        output_dir:    Directory to write JSON summaries after each stage + final.
        stage_filter:  If set, run only the stage whose task_slug matches this value.
    """
    from src.research.langchain_agent.memory.langmem_manager import build_langmem_manager
    from src.research.langchain_agent.storage.langgraph_persistence import get_persistence
    from src.research.langchain_agent.storage.models import init_research_agent_beanie

    store, checkpointer = await get_persistence()
    await init_research_agent_beanie()
    memory_manager = await build_langmem_manager(store=store)

    resolved_mission = _get_mission(mission, mission_name, mission_file)
    out_dir = Path(output_dir) if output_dir else None

    # Filter stages if --stage is given
    if stage_filter:
        filtered_stages = [
            s for s in resolved_mission.stages
            if s.slice_input.task_slug == stage_filter
        ]
        if not filtered_stages:
            available = [s.slice_input.task_slug for s in resolved_mission.stages]
            raise ValueError(
                f"Stage '{stage_filter}' not found. Available: {available}"
            )
        resolved_mission = resolved_mission.model_copy(
            update={"stages": filtered_stages}
        )

    print(f"\n=== Running mission: {resolved_mission.mission_name} ===")
    print(f"Mission ID : {resolved_mission.mission_id}")
    print(f"Source     : {mission_file or mission_name or 'direct object'}")
    print(f"Stages     : {len(resolved_mission.stages)}")
    if out_dir:
        print(f"Output dir : {out_dir}")
    for s in resolved_mission.stages:
        deps = f" (depends on: {', '.join(s.dependencies)})" if s.dependencies else ""
        iterative = (
            f" [iterative: max={s.iterative_config.max_iterations}]"
            if s.iterative_config else ""
        )
        print(f"  - {s.slice_input.task_slug}{deps}{iterative}")
    print()

    mission_meta = {
        "mission_id": resolved_mission.mission_id,
        "mission_name": resolved_mission.mission_name,
        "base_domain": resolved_mission.base_domain,
        "source_file": mission_file,
        "stage_filter": stage_filter,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stages": [],
    }

    # Intercept run_mission to write per-stage JSON files
    # We replicate the dependency-ordered loop here so we can write intermediate files
    from src.research.langchain_agent.workflow.run_mission import (
        _topological_stage_order,
        run_mission as _wf_run_mission,
    )

    has_iterative = any(
        s.iterative_config is not None for s in resolved_mission.stages
    )
    mission_type = "iterative" if has_iterative else "stage_based"

    try:
        snapshot_dir = (out_dir / "state_snapshots") if out_dir else None

        with mission_tracing_context(
            mission_id=resolved_mission.mission_id,
            mission_name=resolved_mission.mission_name,
            mission_type=mission_type,
            extra_metadata={
                "source_file": mission_file,
                "stage_count": len(resolved_mission.stages),
            },
        ):
            outputs = await _wf_run_mission(
                resolved_mission,
                store=store,
                checkpointer=checkpointer,
                memory_manager=memory_manager,
                root_filesystem=ROOT_FILESYSTEM,
                snapshot_output_dir=snapshot_dir,
            )
    except Exception as e:
        mission_meta["status"] = "failed"
        mission_meta["error"] = str(e)
        if out_dir:
            _write_output_file(out_dir, "mission_summary.json", mission_meta)
        print(f"Mission failed: {e}")
        raise

    # Write per-stage output files
    for i, out in enumerate(outputs):
        serialized = _serialize_stage_output(out)
        slug = serialized["task_slug"]
        record = out.get("stage_run_record")
        iteration = record.iteration if record else None

        mission_meta["stages"].append(serialized)
        if out_dir:
            if iteration is not None:
                base = f"stage_{slug}_iter{iteration:02d}"
            else:
                base = f"stage_{i+1:02d}_{slug}"

            _write_output_file(out_dir, f"{base}.json", serialized)
            report_text = out.get("final_report_text") or ""
            if report_text:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / f"{base}_report.md").write_text(
                        report_text, encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning("Failed to write report file: %s", exc)

    # Write combined reports and iteration logs for iterative stages
    if out_dir:
        from src.research.langchain_agent.workflow.run_iterative_stage import IterativeStageResult
        _write_iterative_stage_summaries(resolved_mission, outputs, out_dir)

    mission_meta["status"] = "completed"
    mission_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    if out_dir:
        _write_output_file(out_dir, "mission_summary.json", mission_meta)

    # Console summary
    print("\n=== MISSION SUMMARY ===")
    for out in outputs:
        task_slug = out["agent_state"]["report_path"].replace("reports/", "").replace(".md", "")
        record = out.get("stage_run_record")
        s3_report = (
            record.artifacts.final_report.s3_uri
            if (record and record.artifacts and record.artifacts.final_report)
            else "—"
        )
        print(f"  {task_slug}: ok  -> {out['agent_state']['report_path']}  s3={s3_report}")

    print("\n=== AGENT FILES ===")
    await print_agent_files()

    for rel_path in await list_agent_files():
        if rel_path.startswith("reports/") and rel_path.endswith(".md"):
            await dump_file(rel_path)

    if out_dir:
        print(f"\n=== OUTPUTS WRITTEN TO: {out_dir} ===")
        for f in sorted(out_dir.iterdir()):
            print(f"  {f.name}")

    print("\n=== DONE ===")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    # psycopg3 (AsyncPostgresStore) requires SelectorEventLoop on Windows
    if sys.platform == "win32":
        import selectors
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
        )
        # Windows terminals default to cp1252; switch to UTF-8 so agent output
        # (which contains Unicode characters) doesn't crash the process.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run a ResearchMission. Supports named missions and JSON mission files.",
    )

    mission_group = parser.add_mutually_exclusive_group()
    mission_group.add_argument(
        "--mission",
        type=str,
        choices=list(NAMED_MISSIONS),
        help="Named mission to run (e.g. qualia, elysium).",
    )
    mission_group.add_argument(
        "--mission-file",
        type=str,
        dest="mission_file",
        metavar="PATH",
        help="Path to a JSON mission definition file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        metavar="DIR",
        help="Directory to write per-stage JSON outputs and mission_summary.json.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        dest="stage_filter",
        metavar="SLUG",
        help="Run only the stage with this task_slug.",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            mission_name=args.mission,
            mission_file=args.mission_file,
            output_dir=args.output_dir,
            stage_filter=args.stage_filter,
        )
    )
