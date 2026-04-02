"""
searchText generation for KG nodes.

Two modes:
  - field_concat: deterministic, zero-cost.  Joins searchFields values.
  - llm:          richer semantic text for the most important node types
                  (Organization, Person, Product).

SEARCHTEXT_STRATEGY maps entity type → mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel

from src.research.langchain_agent.kg.prompts.search_text_prompts import _SEARCHTEXT_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

SearchTextMode = Literal["llm", "field_concat"]

SEARCHTEXT_STRATEGY: dict[str, SearchTextMode] = {
    "Organization": "llm",
    "Person": "llm",
    "Product": "llm",
    "CompoundForm": "field_concat",
    "Listing": "field_concat",
}


def get_searchtext_mode(entity_type: str) -> SearchTextMode:
    return SEARCHTEXT_STRATEGY.get(entity_type, "field_concat")


# ---------------------------------------------------------------------------
# Field-concat (fast, deterministic)
# ---------------------------------------------------------------------------


def build_searchtext_from_fields(
    entity_dict: dict,
    search_fields: list[str],
) -> str:
    """
    Concatenate the values of the given fields with ' | ' as separator.

    Lists are joined with a space; scalars are cast to str.
    Empty/None values are skipped.

    Example output:
        "Elysium Health | Basis, Signal, Matter | longevity aging NAD+ | B2C"
    """
    parts: list[str] = []
    for field in search_fields:
        val = entity_dict.get(field)
        if isinstance(val, list):
            joined = " ".join(str(v) for v in val if v)
            if joined:
                parts.append(joined)
        elif val:
            parts.append(str(val))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# LLM-generated searchText (richer, semantic)
# ---------------------------------------------------------------------------


class SearchTextResult(BaseModel):
    searchText: str





def build_searchtext_agent(llm, tools: list | None = None):
    """
    Build a reusable searchText generation agent.
    Build once; reuse across all nodes in a session.
    """
    return create_agent(
        model=llm,
        tools=tools or [],
        system_prompt=_SEARCHTEXT_SYSTEM_PROMPT,
        response_format=SearchTextResult,
    )


async def generate_searchtext_llm(
    entity_type: str,
    entity_dict: dict,
    context: str,
    searchtext_agent,
) -> str:
    """
    Generate a rich searchText via LLM for a single entity.

    Args:
        entity_type:       e.g. "Organization", "Product", "Person".
        entity_dict:       The entity's field dict (as returned from extraction).
        context:           Mission objective or a brief summary for grounding.
        searchtext_agent:  Built with build_searchtext_agent().

    Returns:
        Plain-text searchText string.
    """
    user_message = (
        f"Entity type: {entity_type}\n"
        f"Context: {context}\n\n"
        f"Entity data:\n{json.dumps(entity_dict, indent=2)}"
    )
    result = await searchtext_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    return result["structured_response"].searchText


# ---------------------------------------------------------------------------
# Batch searchText generation
# ---------------------------------------------------------------------------


async def generate_all_searchtexts(
    entities: list[tuple[str, dict]],
    context: str,
    searchtext_agent,
) -> list[str]:
    """
    Generate searchTexts for a list of (entity_type, entity_dict) pairs.

    Uses the strategy map: LLM for Organization/Person/Product,
    field_concat for everything else.  LLM calls are run concurrently.

    Returns a list of searchText strings in the same order as *entities*.
    """
    tasks: list[asyncio.Task | None] = []
    for entity_type, entity_dict in entities:
        mode = get_searchtext_mode(entity_type)
        if mode == "llm":
            tasks.append(
                asyncio.create_task(
                    generate_searchtext_llm(
                        entity_type=entity_type,
                        entity_dict=entity_dict,
                        context=context,
                        searchtext_agent=searchtext_agent,
                    )
                )
            )
        else:
            tasks.append(None)

    results: list[str] = []
    for (entity_type, entity_dict), task in zip(entities, tasks):
        if task is not None:
            results.append(await task)
        else:
            search_fields = entity_dict.get("searchFields", list(entity_dict.keys()))
            results.append(
                build_searchtext_from_fields(entity_dict, search_fields)
            )
    return results
