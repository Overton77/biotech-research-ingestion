"""Mission Compiler Agent — LLM-driven ResearchPlan → ResearchMissionDraft.

Uses structured output (model.with_structured_output) for a single-shot call
that produces a fully validated ResearchMissionDraft Pydantic object.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from src.agents.persistence import get_deep_agents_persistence
from src.research.models.mission import ResearchMissionDraft

if TYPE_CHECKING:
    from src.models.plan import ResearchPlan

load_dotenv()
logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class MissionDraftValidationError(Exception):
    """LLM output failed Pydantic validation or schema constraint."""


def _load_prompt_file(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _build_compiler_system_prompt() -> str:
    deep_agents_ctx = _load_prompt_file("deep_agents_context.md")
    subagent_roles = _load_prompt_file("subagent_roles.md")
    mission_schema = _load_prompt_file("mission_schema.md")

    return f"""You are a Deep Agent Mission Compiler. Your job is to read an approved research plan
and produce a fully executable ResearchMission — a detailed configuration of
Deep Agent workers and their specialized subagents for biotech research.

You MUST reason carefully about:
- What kind of specialized subagent workers would help each task
- How information flows between tasks via input_bindings
- What tools each agent and subagent needs
- What detailed system prompts will produce the best research output
- Which skills to assign to agents and subagents

--- DEEP AGENTS FRAMEWORK DOCUMENTATION ---
{deep_agents_ctx}

--- AVAILABLE TOOL PROFILES ---
- default_research: Tavily search tools (search, extract, map). Subagents also get FilesystemMiddleware (ls, read_file, write_file, edit_file). Default for main task agents and most subagents.
- search_only: Tavily search tools only. For source-finder subagents that only need web access.
- write_only: Filesystem utilities only (no web search). For writer/synthesizer subagents that work with gathered data.

--- AVAILABLE SKILLS ---
One skill is available for assignment to task agents and subagents:

- /skills/source-citation/ — Source citation and provenance tracking.
  Assign this to ALL main task agents (via main_agent.skills) and to any
  subagent that conducts web searches (via compiled_subagents[].skills).

Skills are passed as a list of directory paths, e.g.:
  "skills": ["/skills/source-citation/"]

--- SUBAGENT DESIGN GUIDANCE ---
{subagent_roles}

--- OUTPUT CONTRACT ---
{mission_schema}

CRITICAL RULES:
- Do NOT invent new task IDs. Use EXACTLY the task IDs from the input plan.
- Do NOT add dependencies that are not in the original plan.
- Design 1-3 subagents per task with distinct roles; use the examples in the doc only as inspiration — role types and names are not fixed.
- Each main_agent system_prompt MUST be 2-4 paragraphs with specific instructions.
- Each main_agent system_prompt MUST include guidance on updating AGENTS.md and writing to /memories/ (important memories, intermediate and final output paths).
- Each subagent system_prompt MUST be specific to its role and task context.
- workspace_suffix values MUST use snake_case and be unique within a task.
- For tasks with dependencies, create input_bindings mapping to source_task_id + source_key.
- source_key should typically be "response" for text outputs from prior tasks.
- Use model_name "openai:gpt-5" for main task agents and "openai:gpt-5-mini" for subagents.
- Assign skills=["/skills/source-citation/"] to every main_agent and to every
  subagent whose tool_profile_name is "default_research" or "search_only".
- For input_bindings, set max_tokens when a prior task produces large text outputs
  to avoid overwhelming downstream agents. Use transform="summarize" when you want
  an LLM to condense the input rather than simply truncating."""


def _plan_to_compiler_prompt(plan: ResearchPlan) -> str:
    tasks_data = []
    for t in plan.tasks:
        task_info = {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "stage": t.stage,
            "dependencies": t.dependencies,
            "inputs": [inp.model_dump() for inp in t.inputs] if t.inputs else [],
            "outputs": [out.model_dump() for out in t.outputs] if t.outputs else [],
        }
        tasks_data.append(task_info)

    plan_data: dict = {
        "title": plan.title,
        "objective": plan.objective,
        "stages": plan.stages,
        "tasks": tasks_data,
    }
    if plan.starter_sources:
        plan_data["starter_sources"] = [s.model_dump() for s in plan.starter_sources]

    starter_blurb = ""
    if plan.starter_sources:
        starter_blurb = (
            "\n- Starter sources (use these when designing tasks and subagent instructions): "
            + ", ".join(f"{s.url} ({s.description or 'no description'})" for s in plan.starter_sources)
        )

    return f"""Here is the approved Research Plan. Produce a ResearchMissionDraft for it.

RESEARCH PLAN:
```json
{json.dumps(plan_data, indent=2)}
```

Produce the ResearchMissionDraft JSON now. Remember:
- Use EXACTLY these task IDs: {[t.id for t in plan.tasks]}
- The plan objective is: {plan.objective}
- There are {len(plan.stages)} stages: {plan.stages}
- For each task, design 1-3 specialized subagents with specific roles and detailed system prompts.{starter_blurb}"""


async def compile_mission_draft(
    plan: ResearchPlan,
    model_name: str = "openai:gpt-5",
    max_retries: int = 2,
) -> ResearchMissionDraft:
    """
    Call the Mission Compiler LLM Agent with the approved plan.
    Returns a validated ResearchMissionDraft.
    Retries up to max_retries times with validation error feedback.
    """ 

    config: RunnableConfig = {"configurable": {"thread_id": plan.thread_id}} 
    system_prompt = _build_compiler_system_prompt()
    user_message = _plan_to_compiler_prompt(plan) 

    store, checkpointer = await get_deep_agents_persistence()

    model = init_chat_model(model_name)

    mission_agent = create_agent(
        model,
        system_prompt=system_prompt,
        response_format=ResearchMissionDraft, 
        checkpointer=checkpointer,
        store=store,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0 and last_error is not None:
                feedback = (
                    f"\n\nYour previous attempt (attempt {attempt}) failed validation:\n"
                    f"{last_error}\n\n"
                    "Please fix the issues and try again. Ensure strict schema compliance."
                )
                current_message = user_message + feedback
            else:
                current_message = user_message

            result = await mission_agent.ainvoke({
                "messages": [
                    {"role": "user", "content": current_message},
                ],
            }, config=config)

            mission_draft: ResearchMissionDraft = result["structured_response"]

            logger.info(
                "Mission draft compiled successfully on attempt %d with %d task_defs",
                attempt + 1,
                len(mission_draft.task_defs),
            )
            return mission_draft

        except Exception as e:
            last_error = e
            logger.warning(
                "Mission draft compilation attempt %d failed: %s",
                attempt + 1,
                str(e)[:200],
            )
            if attempt >= max_retries:
                raise MissionDraftValidationError(
                    f"Failed to compile mission draft after {max_retries + 1} attempts: {e}"
                ) from e

    raise MissionDraftValidationError("Exhausted all retry attempts")
