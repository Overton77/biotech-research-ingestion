"""
Run evaluation experiments against a LangSmith dataset.

Executes a research mission stage for each dataset example and scores the output
using the configured evaluators. Results are stored as a LangSmith experiment.

Usage:
    # Run evaluation against a pre-built LangSmith dataset
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --dataset biotech-research-reports-v1 \
        --experiment baseline-gpt4omini

    # Run with specific evaluators only
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --dataset biotech-research-reports-v1 \
        --experiment baseline-gpt4omini \
        --evaluators report_structure source_quality

    # Evaluate existing outputs (no re-execution — uses outputs from the dataset)
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --dataset biotech-research-reports-v1 \
        --experiment static-eval \
        --static

    # --- Targeted eval from a specific mission file + its run outputs ---
    # Builds a temporary dataset inline, runs eval, no separate build step needed.

    # Single mission file
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --from-mission-file src/research/langchain_agent/test_runs/missions/analytical_resource_labs.json \
        --from-run-outputs-dir src/research/langchain_agent/test_runs/run_outputs/analytical_resource_labs \
        --experiment arl-eval-v1

    # Multiple mission files (repeat flag)
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --from-mission-file .../missions/analytical_resource_labs.json \
        --from-mission-file .../missions/elysium_mini.json \
        --from-run-outputs-dir src/research/langchain_agent/test_runs/run_outputs \
        --experiment multi-eval-v1

    # Re-execute live against a specific mission
    uv run python -m src.research.langchain_agent.eval.run_eval \
        --from-mission-file .../missions/analytical_resource_labs.json \
        --from-run-outputs-dir .../run_outputs/analytical_resource_labs \
        --experiment arl-live-v1 \
        --no-static
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
from dotenv import load_dotenv
import os


from langsmith import aevaluate

from src.research.langchain_agent.eval.evaluators import (
    iteration_convergence,
    report_accuracy,
    report_completeness,
    report_structure,
    source_quality,
    tool_efficiency,
)
from src.research.langchain_agent.observability.tracing import eval_tracing_context

logger = logging.getLogger(__name__)  

load_dotenv() 

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Registry of available evaluators
EVALUATOR_REGISTRY: Dict[str, Callable] = {
    "report_completeness": report_completeness,
    "report_accuracy": report_accuracy,
    "report_structure": report_structure,
    "source_quality": source_quality,
    "tool_efficiency": tool_efficiency,
    "iteration_convergence": iteration_convergence,
}

DEFAULT_EVALUATORS = [
    "report_completeness",
    "report_structure",
    "source_quality",
]


async def static_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through target for evaluating existing outputs without re-execution.

    LangSmith calls: evaluator(inputs, outputs=target(inputs), reference_outputs)
    Here `outputs` will be the same dict as `inputs` (no "report" key).
    Evaluators detect this and fall back to `reference_outputs` (the stored
    dataset outputs that do contain the report text). See _extract_report().
    """
    return inputs


async def live_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a research mission stage and return outputs for evaluation.

    This is the full re-execution path: it runs the research agent on the
    given inputs and returns the actual outputs for scoring.
    """
    from src.research.langchain_agent.agent.config import (
        MissionSliceInput,
        ResearchPromptSpec,
    )
    from src.research.langchain_agent.memory.langmem_manager import build_langmem_manager
    from src.research.langchain_agent.storage.langgraph_persistence import get_persistence
    from src.research.langchain_agent.workflow.run_slice import run_single_mission_slice

    store, checkpointer = await get_persistence()
    memory_manager = await build_langmem_manager(store=store)

    slice_input = MissionSliceInput(
        task_id=f"eval-{inputs['task_slug']}",
        mission_id=f"eval-{inputs.get('mission_name', 'unknown')}",
        task_slug=inputs["task_slug"],
        user_objective=inputs["user_objective"],
        targets=inputs.get("targets", []),
        selected_tool_names=inputs.get("selected_tool_names", ["search_web", "extract_from_urls"]),
        report_required_sections=inputs.get("report_required_sections", []),
        stage_type=inputs.get("stage_type", "discovery"),
        max_step_budget=inputs.get("max_step_budget", 12),
    )

    prompt_spec = ResearchPromptSpec(
        agent_identity="You are a biotech research agent running an evaluation task.",
        domain_scope=[],
        workflow=[],
    )

    result = await run_single_mission_slice(
        run_input=slice_input,
        prompt_spec=prompt_spec,
        store=store,
        checkpointer=checkpointer,
        memory_manager=memory_manager,
    )

    return {
        "report": result.get("final_report_text", ""),
        "final_report_text": result.get("final_report_text", ""),
        "visited_urls": result.get("agent_state", {}).get("visited_urls", []),
        "messages": result.get("result", {}).get("messages", []),
    }


def _build_and_upload_inline_dataset(
    mission_files: List[Path],
    run_outputs_dir: Path,
    dataset_name: str,
    description: str = "",
) -> None:
    """Build a LangSmith dataset from local mission files + run outputs and upload it.

    This is the inline equivalent of running build_datasets.py separately.
    The dataset is created/replaced in LangSmith so that run_evaluation() can
    reference it by name.
    """
    from src.research.langchain_agent.eval.build_datasets import (
        DEFAULT_REPORTS_DIR,
        build_stage_dataset,
        upload_dataset,
    )

    examples = build_stage_dataset(
        missions_dir=None,
        outputs_dir=run_outputs_dir,
        reports_dir=DEFAULT_REPORTS_DIR,
        mission_files=mission_files,
    )

    if not examples:
        raise RuntimeError(
            f"No dataset examples found. Check that mission files have matching "
            f"run outputs in: {run_outputs_dir}"
        )

    logger.info("Built %d inline dataset examples", len(examples))
    upload_dataset(examples, dataset_name, description)


async def run_evaluation(
    dataset_name: str,
    experiment_prefix: str,
    evaluator_names: List[str],
    static: bool = False,
    max_concurrency: int = 2,
    description: str = "",
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Run an evaluation experiment against a LangSmith dataset."""
    evaluators = [EVALUATOR_REGISTRY[name] for name in evaluator_names]
    target = static_target if static else live_target

    experiment_metadata = {
        "mode": "static" if static else "live",
        "evaluators": evaluator_names,
    }
    if metadata:
        experiment_metadata.update(metadata)

    with eval_tracing_context(
        dataset_name=dataset_name,
        experiment_name=experiment_prefix,
        extra_metadata=experiment_metadata,
    ):
        results = await aevaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            description=description or f"{'Static' if static else 'Live'} evaluation: {', '.join(evaluator_names)}",
            max_concurrency=max_concurrency,
            metadata=experiment_metadata,
        )

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_prefix}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Mode: {'static (existing outputs)' if static else 'live (re-execution)'}")
    print(f"  Evaluators: {', '.join(evaluator_names)}")
    print(f"{'='*60}")
    print(f"\n  View results at: https://smith.langchain.com")
    print(f"  Look for experiment prefix: {experiment_prefix}")


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run evaluation experiments on research mission outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Dataset source ──────────────────────────────────────────────────────
    dataset_group = parser.add_argument_group(
        "Dataset source",
        "Use --dataset for a pre-built LangSmith dataset, OR use --from-mission-file "
        "+ --from-run-outputs-dir to build a dataset inline from local files.",
    )
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of a pre-built LangSmith dataset.",
    )
    dataset_group.add_argument(
        "--from-mission-file",
        action="append",
        dest="from_mission_files",
        metavar="PATH",
        help=(
            "Path to a mission JSON file. Can be repeated. "
            "When provided, a dataset is built inline from these files + "
            "--from-run-outputs-dir."
        ),
    )
    dataset_group.add_argument(
        "--from-run-outputs-dir",
        type=str,
        dest="from_run_outputs_dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory containing run outputs for the given mission file(s). "
            "Can point directly at a specific mission run dir or at the parent "
            "that holds multiple run subdirs."
        ),
    )

    # ── Experiment options ──────────────────────────────────────────────────
    parser.add_argument("--experiment", type=str, required=True, help="Experiment prefix name")
    parser.add_argument(
        "--evaluators",
        nargs="+",
        choices=list(EVALUATOR_REGISTRY.keys()),
        default=DEFAULT_EVALUATORS,
        help="Evaluators to run (default: %(default)s)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        default=True,
        help="Evaluate existing outputs without re-running the agent (default: on).",
    )
    parser.add_argument(
        "--no-static",
        dest="static",
        action="store_false",
        help="Re-execute the agent for each dataset example (live mode).",
    )
    parser.add_argument("--max-concurrency", type=int, default=2, help="Parallel evaluation threads")
    parser.add_argument("--description", type=str, default="", help="Experiment description")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model used (metadata only)")
    args = parser.parse_args()

    # ── Resolve dataset name ────────────────────────────────────────────────
    if args.from_mission_files:
        mission_paths = [Path(p) for p in args.from_mission_files]
        missing = [p for p in mission_paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"ERROR: Mission file not found: {p}", file=sys.stderr)
            sys.exit(1)

        run_outputs_dir = Path(args.from_run_outputs_dir) if args.from_run_outputs_dir else None
        if run_outputs_dir is None:
            print(
                "ERROR: --from-run-outputs-dir is required when using --from-mission-file",
                file=sys.stderr,
            )
            sys.exit(1)

        # Use experiment prefix as the (temporary) dataset name so each run is isolated
        dataset_name = args.dataset or f"eval-inline-{args.experiment}"
        print(f"\nBuilding inline dataset '{dataset_name}' from {len(mission_paths)} mission file(s)...")
        _build_and_upload_inline_dataset(
            mission_files=mission_paths,
            run_outputs_dir=run_outputs_dir,
            dataset_name=dataset_name,
            description=args.description or f"Inline dataset for experiment: {args.experiment}",
        )
    elif args.dataset:
        dataset_name = args.dataset
    else:
        print(
            "ERROR: Provide either --dataset <name> or --from-mission-file + --from-run-outputs-dir",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_evaluation(
            dataset_name=dataset_name,
            experiment_prefix=args.experiment,
            evaluator_names=args.evaluators,
            static=args.static,
            max_concurrency=args.max_concurrency,
            description=args.description,
            metadata={"model": args.model},
        )
    )


if __name__ == "__main__":
    main()
