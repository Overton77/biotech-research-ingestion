"""
Build LangSmith evaluation datasets from completed research mission outputs.

Reads mission JSON definitions and their corresponding output reports, then
creates LangSmith datasets that can be used with evaluate() / aevaluate().

Usage:
    # All missions in the default directory
    uv run python -m src.research.langchain_agent.eval.build_datasets

    # Single mission file (most common for targeted evals)
    uv run python -m src.research.langchain_agent.eval.build_datasets \
        --mission-file src/research/langchain_agent/test_runs/missions/analytical_resource_labs.json \
        --outputs-dir  src/research/langchain_agent/test_runs/run_outputs/analytical_resource_labs

    # Multiple specific mission files (comma-separated or repeated flag)
    uv run python -m src.research.langchain_agent.eval.build_datasets \
        --mission-file src/.../missions/analytical_resource_labs.json \
        --mission-file src/.../missions/elysium_mini.json

    # All missions in a directory, custom outputs dir, named dataset
    uv run python -m src.research.langchain_agent.eval.build_datasets \
        --missions-dir src/research/langchain_agent/test_runs/missions \
        --outputs-dir  src/research/langchain_agent/test_runs/run_outputs \
        --dataset-name biotech-research-reports-v1

    # Dry-run (print examples without uploading)
    uv run python -m src.research.langchain_agent.eval.build_datasets \
        --mission-file .../analytical_resource_labs.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langsmith import Client 
import os 
from dotenv import load_dotenv 

load_dotenv() 



logger = logging.getLogger(__name__)

DEFAULT_MISSIONS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "missions"
DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "test_runs" / "run_outputs"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "agent_outputs" / "reports"


def _load_mission_stages(mission_path: Path) -> List[Dict[str, Any]]:
    """Load stage definitions from a mission JSON file."""
    raw = json.loads(mission_path.read_text(encoding="utf-8"))
    return raw.get("stages", [])


def _candidate_output_dirs(outputs_dir: Path) -> list[Path]:
    """Return directories to search for stage files.

    Handles two layouts:
      • outputs_dir = parent that contains one subdir per mission run
        (e.g. test_runs/run_outputs/ which has analytical_resource_labs/ inside)
      • outputs_dir = specific mission run dir
        (e.g. test_runs/run_outputs/analytical_resource_labs/)

    In both cases we return the dir itself PLUS any immediate subdirectories
    so that a direct dir with stage files is also found.
    """
    candidates: list[Path] = [outputs_dir]
    if outputs_dir.is_dir():
        candidates.extend(
            d for d in outputs_dir.iterdir()
            if d.is_dir() and d.name != "state_snapshots"
        )
    return candidates


def _find_stage_report(
    task_slug: str,
    outputs_dir: Path,
    reports_dir: Path,
) -> str | None:
    """Find the report text for a given stage, checking multiple locations."""
    for search_dir in _candidate_output_dirs(outputs_dir):
        for f in search_dir.glob(f"*{task_slug}*_report.md"):
            text = f.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text

    report_file = reports_dir / f"{task_slug}.md"
    if report_file.exists():
        text = report_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            return text

    return None


def _find_stage_output_json(task_slug: str, outputs_dir: Path) -> Dict[str, Any] | None:
    """Find the structured JSON output for a stage."""
    for search_dir in _candidate_output_dirs(outputs_dir):
        for f in search_dir.glob(f"*{task_slug}*.json"):
            if "summary" in f.name:
                continue
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return None


def _extract_visited_urls_from_record(record: Dict[str, Any]) -> List[str]:
    """Pull visited URLs out of a stage_run_record.

    The agent embeds URLs in the memory_report.summary as plain text and in
    artifact paths. We extract all https?:// links from the summary string.
    """
    import re

    summary = record.get("memory_report", {}).get("summary", "")
    if not summary:
        return []
    url_pattern = re.compile(r"https?://[^\s\)\"'>,]+")
    return list(dict.fromkeys(url_pattern.findall(summary)))  # deduplicated, ordered


def build_stage_dataset(
    missions_dir: Path | None,
    outputs_dir: Path,
    reports_dir: Path,
    mission_files: List[Path] | None = None,
) -> List[Dict[str, Any]]:
    """Build dataset examples from completed stage outputs.

    Args:
        missions_dir:  Directory of mission JSON files. Ignored when
                       ``mission_files`` is provided.
        outputs_dir:   Directory containing run outputs. Can be the parent
                       that holds one subdir per mission run, OR a specific
                       mission run directory directly.
        reports_dir:   Fallback directory for final report .md files.
        mission_files: Explicit list of mission JSON paths. When given,
                       ``missions_dir`` is not used.

    Each example has:
      inputs: mission config for the stage (objective, tools, targets, required_sections)
      outputs: the actual report text, visited_urls, and metadata
    """
    examples: List[Dict[str, Any]] = []

    if mission_files:
        paths = sorted(mission_files)
    elif missions_dir is not None:
        paths = sorted(missions_dir.glob("*.json"))
    else:
        raise ValueError("Either missions_dir or mission_files must be provided")

    for mission_file in paths:
        stages = _load_mission_stages(mission_file)
        mission_name = mission_file.stem

        for stage_data in stages:
            slice_input = stage_data.get("slice_input", {})
            task_slug = slice_input.get("task_slug", "")

            if not task_slug:
                continue

            report_text = _find_stage_report(task_slug, outputs_dir, reports_dir)
            if not report_text:
                logger.info("No report found for %s — skipping", task_slug)
                continue

            stage_output_json = _find_stage_output_json(task_slug, outputs_dir)

            inputs = {
                "mission_name": mission_name,
                "task_slug": task_slug,
                "user_objective": slice_input.get("user_objective", ""),
                "targets": slice_input.get("targets", []),
                "selected_tool_names": slice_input.get("selected_tool_names", []),
                "report_required_sections": slice_input.get("report_required_sections", []),
                "stage_type": slice_input.get("stage_type", "discovery"),
                "max_step_budget": slice_input.get("max_step_budget", 12),
            }

            outputs: Dict[str, Any] = {
                "report": report_text,
                "final_report_text": report_text,
                "task_slug": task_slug,
                "visited_urls": [],
            }

            if stage_output_json:
                record = stage_output_json.get("stage_run_record", {})
                if record:
                    outputs["stage_type"] = record.get("stage_type", "")
                    outputs["visited_urls"] = _extract_visited_urls_from_record(record)

            examples.append({"inputs": inputs, "outputs": outputs})
            logger.info(
                "Added example: %s / %s (%d chars, %d urls)",
                mission_name,
                task_slug,
                len(report_text),
                len(outputs["visited_urls"]),
            )

    return examples


def upload_dataset(
    examples: List[Dict[str, Any]],
    dataset_name: str,
    description: str = "",
) -> None:
    """Upload examples to a LangSmith dataset. Creates or replaces the dataset."""
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        logger.info("Deleting existing dataset: %s", dataset_name)
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description or f"Research mission evaluation dataset ({len(examples)} examples)",
    )

    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )

    logger.info(
        "Uploaded %d examples to dataset '%s' (id=%s)",
        len(examples), dataset_name, dataset.id,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build LangSmith evaluation datasets from mission outputs")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--mission-file",
        action="append",
        dest="mission_files",
        metavar="PATH",
        help=(
            "Path to a single mission JSON file. Can be repeated for multiple files. "
            "Mutually exclusive with --missions-dir."
        ),
    )
    source_group.add_argument(
        "--missions-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of mission JSON files (default when --mission-file is not given).",
    )

    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help=(
            "Directory containing run outputs. Can be the parent dir (holds one subdir "
            "per mission run) OR a specific mission run directory."
        ),
    )
    parser.add_argument("--reports-dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--dataset-name", type=str, default="biotech-research-reports-v1")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print examples without uploading")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    reports_dir = Path(args.reports_dir)

    if args.mission_files:
        mission_paths = [Path(p) for p in args.mission_files]
        missing = [p for p in mission_paths if not p.exists()]
        if missing:
            for p in missing:
                logger.error("Mission file not found: %s", p)
            return
        examples = build_stage_dataset(
            missions_dir=None,
            outputs_dir=outputs_dir,
            reports_dir=reports_dir,
            mission_files=mission_paths,
        )
    else:
        missions_dir = Path(args.missions_dir) if args.missions_dir else DEFAULT_MISSIONS_DIR
        if not missions_dir.exists():
            logger.error("Missions directory not found: %s", missions_dir)
            return
        examples = build_stage_dataset(
            missions_dir=missions_dir,
            outputs_dir=outputs_dir,
            reports_dir=reports_dir,
        )

    if not examples:
        logger.warning("No examples found — check that missions have completed outputs")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Examples: {len(examples)}")
    print(f"{'='*60}\n")

    for i, ex in enumerate(examples, 1):
        inp = ex["inputs"]
        out = ex["outputs"]
        print(f"  [{i}] {inp['mission_name']} / {inp['task_slug']}")
        print(f"      objective: {inp['user_objective'][:80]}...")
        print(f"      report: {len(out.get('report', ''))} chars")
        print()

    if args.dry_run:
        print("  (dry run — not uploading)")
        return

    upload_dataset(examples, args.dataset_name, args.description)
    print(f"\n  Uploaded to LangSmith: {args.dataset_name}")


if __name__ == "__main__":
    main()
