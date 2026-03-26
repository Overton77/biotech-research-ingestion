"""Standalone deep-agent test harness.

Bypasses mission planning entirely. Defines a concrete TaskDef and drives the
full deep-agent stack (subagents + middleware) with InMemory checkpointer/store.

Outputs are written to the filesystem under:
    .deepagents/research_missions/<MISSION_ID>/tasks/<TASK_ID>/

Usage:
    uv run python -m src.research.tests.test_deep_agent

LangGraph Studio / LangSmith tracing:
    Add to langgraph.json:
        {
          "graphs": {
            "deep_agent_test": "src.research.tests.test_deep_agent:make_graph"
          }
        }
    The module-level CHECKPOINTER / STORE are shared across Studio invocations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from deepagents import create_deep_agent

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore 
from dotenv import load_dotenv 

from src.research.deepagent.compiler.agent_compiler import RuntimeContext, compile_main_task_agent
from src.research.deepagent.models.mission import (
    CompiledSubAgentConfig,
    MainDeepAgentConfig,
    TaskDef,
    TaskExecutionPolicy,
)
import os
from src.research.deepagent.runtime.backends import task_root
from src.research.deepagent.runtime.task_executor import _build_invocation_message

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "biotech_research"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stable test identifiers — workspace is reused across runs
# ---------------------------------------------------------------------------

TEST_MISSION_ID = "test-deep-agent-001"
TEST_TASK_ID = "task-crispr-landscape-001"

# ---------------------------------------------------------------------------
# TaskDef — concrete biotech research scenario defined at module level
# so LangGraph Studio can inspect state schema and graph topology.
# ---------------------------------------------------------------------------

TEST_TASK_DEF = TaskDef(
    task_id=TEST_TASK_ID,
    name="CRISPR Therapeutic Landscape Research",
    description=(
        "Conduct a comprehensive research on the current CRISPR gene editing therapeutic landscape. "
        "Cover: (1) major CRISPR platforms currently in clinical trials (Cas9, Cas12, base editing, "
        "prime editing), (2) leading biotech companies and their pipeline stage, "
        "(3) key disease indications being targeted, (4) recent regulatory approvals or milestones "
        "from 2023–2026, and (5) major technical challenges and current mitigation approaches."
    ),
    acceptance_criteria=[
        "At least 5 clinical-stage CRISPR therapies identified with company name and indication",
        "Recent regulatory milestones from 2023–2026 documented with approximate dates",
        "Technical challenges section covers delivery, off-target effects, and immunogenicity",
        "A final synthesis report written to /outputs/crispr_landscape_report.md",
    ],
    main_agent=MainDeepAgentConfig(
        model_name="openai:gpt-5",
        model_tier="standard",
        system_prompt=(
            "You are a senior biotech research analyst specializing in gene editing therapeutics. "
            "Produce a thorough, well-cited research report on the current CRISPR therapeutic landscape. "
            "Use your search tools actively to gather current information. "
            "Delegate specialized subtasks to your subagents: use 'literature_scout' for academic "
            "and clinical trial data, and 'company_analyst' for company pipeline details. "
            "Synthesize everything into a final report written to /outputs/crispr_landscape_report.md. "
            "Be precise, cite sources inline with URLs, and provide actionable insights."
        ),
        tool_profile_name="default_research",
        skills=[],
    ),
    compiled_subagents=[
        CompiledSubAgentConfig(
            name="literature_scout",
            description=(
                "Specialist in searching academic databases, preprints, clinicaltrials.gov, "
                "and biotech news for CRISPR clinical trial data, recent publications, "
                "and regulatory filings."
            ),
            system_prompt=(
                "You are a biomedical literature scout. Your job is to search for CRISPR gene editing "
                "clinical trials, recent approvals, and key publications from 2022–2026. "
                "Search clinicaltrials.gov summaries, PubMed abstracts via search, FDA/EMA press releases, "
                "and major biotech news sources. "
                "Compile your findings into a structured markdown document at /outputs/literature_summary.md "
                "with the following sections: "
                "1) Clinical Trials Table (name, company, indication, phase, status), "
                "2) Regulatory Milestones Timeline, "
                "3) Key Recent Publications. "
                "Cite every claim with a URL."
            ),
            tool_profile_name="default_research",
            workspace_suffix="lit",
            use_todo_middleware=True,
            expected_output_format="Markdown: clinical trials table + regulatory timeline + publications",
            expected_output_path="outputs/literature_summary.md",
        ),
        CompiledSubAgentConfig(
            name="company_analyst",
            description=(
                "Specialist in analyzing biotech company pipelines, key programs, partnerships, "
                "and strategic positioning in the CRISPR therapeutics space."
            ),
            system_prompt=(
                "You are a biotech industry analyst. Research the leading companies developing "
                "CRISPR therapeutics including: Intellia Therapeutics, CRISPR Therapeutics, "
                "Beam Therapeutics, Editas Medicine, Prime Medicine, Graphite Bio, and Verve Therapeutics. "
                "For each company cover: lead programs, pipeline stage, key indications, major partnerships, "
                "and any commercial approvals or recent clinical readouts. "
                "Write a structured company comparison to /outputs/company_analysis.md with: "
                "1) Company Overview Table, "
                "2) Pipeline Details per company, "
                "3) Notable Partnerships & Financials summary. "
                "Cite sources with URLs."
            ),
            tool_profile_name="default_research",
            workspace_suffix="co",
            use_todo_middleware=True,
            expected_output_format="Markdown: company comparison table + pipeline details per company",
            expected_output_path="outputs/company_analysis.md",
        ),
    ],
    execution=TaskExecutionPolicy(
        timeout_seconds=900,
        max_retries=1,
        persist_run_after_completion=False,
        fallback_models=[],
    ),
)

# ---------------------------------------------------------------------------
# Persistent InMemory store + checkpointer — module-level so LangGraph Studio
# reuses the same state across multiple invocations in the same process.
# ---------------------------------------------------------------------------

CHECKPOINTER: InMemorySaver = InMemorySaver()
STORE: InMemoryStore = InMemoryStore()

# ---------------------------------------------------------------------------
# Graph factory — used by LangGraph Studio and by run_test() below.
# Keeping this async with no required args satisfies the simplest factory
# signature documented at https://docs.langchain.com/langsmith/graph-rebuild
# ---------------------------------------------------------------------------


async def make_graph():
    """Compile and return the deep-agent StateGraph for the test TaskDef.

    LangGraph Studio / LangSmith: reference as
        "src.research.tests.test_deep_agent:make_graph"
    in langgraph.json graphs section.
    """
    ctx = RuntimeContext(
        mission_id=TEST_MISSION_ID,
        task_id=TEST_TASK_ID,
        store=STORE,
        checkpointer=CHECKPOINTER,
        progress_callback=None,  # No ResearchProgressMiddleware in tests
    )
    return await compile_main_task_agent(TEST_TASK_DEF, ctx)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _extract_last_ai_message(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _write_run_summary(
    task_dir: Path,
    result: dict,
    started_at: datetime,
    elapsed: float,
) -> Path:
    messages = result.get("messages", [])
    last_content = _extract_last_ai_message(messages)

    summary = {
        "task_id": TEST_TASK_ID,
        "task_name": TEST_TASK_DEF.name,
        "mission_id": TEST_MISSION_ID,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "message_count": len(messages),
        "last_message_preview": last_content[:2000],
        "workspace_root": str(task_dir),
        "sources_collected": result.get("sources_collected", []),
        "total_tokens_used": result.get("total_tokens_used", 0),
        "quality_assessment": result.get("quality_assessment"),
    }

    summary_path = task_dir / "outputs" / "test_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


async def run_test() -> None:
    """Compile and invoke the deep agent, then write all outputs to the filesystem."""

    logger.info("=" * 60)
    logger.info("Deep Agent Test: %s", TEST_TASK_DEF.name)
    logger.info("Mission ID : %s", TEST_MISSION_ID)
    logger.info("Task ID    : %s", TEST_TASK_ID)
    logger.info("Subagents  : %s", [s.name for s in TEST_TASK_DEF.compiled_subagents])
    logger.info("=" * 60)

    # 1. Compile the deep agent (creates workspace dirs, builds subagents + middleware)
    logger.info("Compiling deep agent...")
    agent = await make_graph()
    logger.info("Agent compiled.")

    # 2. Build the invocation message exactly as task_executor does
    message = _build_invocation_message(
        TEST_TASK_DEF,
        resolved_inputs={},
        global_context={
            "research_domain": "gene editing therapeutics",
            "focus_period": "2022–2026",
            "output_style": "structured markdown with inline citations",
        },
    )
    logger.info("Invocation message ready (%d chars).", len(message))

    # 3. Invoke the agent
    started_at = datetime.utcnow()
    logger.info("Invoking deep agent (this may take several minutes)...")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config={
            "configurable": {
                "thread_id": f"test-{TEST_TASK_ID}",
            }
        },
    )

    elapsed = (datetime.utcnow() - started_at).total_seconds()
    logger.info("Agent finished in %.1fs.", elapsed)

    # 4. Write run summary + collect artifacts
    task_dir = task_root(TEST_MISSION_ID, TEST_TASK_ID)
    summary_path = _write_run_summary(task_dir, result, started_at, elapsed)
    logger.info("Run summary: %s", summary_path)

    outputs_dir = task_dir / "outputs"
    artifacts = sorted(outputs_dir.iterdir()) if outputs_dir.exists() else []
    artifact_names = [f.name for f in artifacts if f.is_file()]
    logger.info("Artifacts  : %s", artifact_names)

    # 5. Print final agent message
    last_content = _extract_last_ai_message(result.get("messages", []))
    logger.info("\n%s\nAGENT FINAL MESSAGE\n%s\n%s", "=" * 60, "=" * 60, last_content[:3000])

    if len(last_content) > 3000:
        logger.info("[... truncated — full content in run summary JSON ...]")

    logger.info("=" * 60)
    logger.info("Workspace root : %s", task_dir)
    logger.info(
        "Token usage    : %s",
        result.get("total_tokens_used", "not tracked"),
    )
    sources = result.get("sources_collected", [])
    logger.info("Sources found  : %d", len(sources))
    logger.info("=" * 60)


agent = create_deep_agent( 
    model="openai:gpt-5", 
    tools=[], 
    system_prompt="You are a helpful assistant"
)

if __name__ == "__main__":
    asyncio.run(run_test())
