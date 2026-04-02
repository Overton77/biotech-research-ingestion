"""
Schema chunk selection — progressive disclosure.

Loads schema_index.json once and uses a lightweight LLM agent to select
which schema chunks (Organizations, Products, LabTests, …) are relevant
for the given research report.  Only the chunk *metadata* (descriptions +
keywords) is sent to the model — not the full schema text.

After selection, load_schema_chunks() reads schema_registry.json and
builds a compact JSON extraction contract for the selected chunks.
No .md files are loaded at extraction time.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain.agents import create_agent
from pydantic import BaseModel

from src.research.langchain_agent.kg.schema_loader import (
    build_extraction_contract,
    contract_to_prompt_string,
    load_schema_registry,
)
from src.research.langchain_agent.kg.prompts.schema_selector_prompts import _SELECTOR_SYSTEM_PROMPT

# Paths -----------------------------------------------------------------------

_KG_DIR = Path(__file__).parent
_INDEX_PATH = _KG_DIR / "schema" / "schema_index.json"


# Index helpers ---------------------------------------------------------------


def load_schema_index(index_path: Path | None = None) -> list[dict]:
    """Load and return the schema chunk index from disk (synchronous)."""
    path = index_path or _INDEX_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_schema_chunks(
    selected_chunks: list[dict],
    registry: dict | None = None,
    schema_root: Path | None = None,  # kept for API compatibility; unused
) -> str:
    """
    Build a compact JSON extraction contract for the selected chunks.

    Reads schema_registry.json (not .md files) and returns a JSON string
    ready for injection into the extraction LLM prompt.

    Args:
        selected_chunks: Subset of schema_index.json entries chosen by the
                         schema-selector agent.
        registry:        Pre-loaded registry dict (optional; loads from disk
                         if not supplied).
        schema_root:     Ignored — kept for backward-compatibility with any
                         callers that pass it.

    Returns:
        JSON string describing the typed extraction contract.
    """
    reg = registry or load_schema_registry()
    contract = build_extraction_contract(selected_chunks, reg)
    return contract_to_prompt_string(contract)


# Agent -----------------------------------------------------------------------


class SchemaSelectionResult(BaseModel):
    chunk_ids: list[str]
    reasoning: str





def build_schema_selector_agent(llm, top_k: int = 4):
    """
    Build a reusable schema selector agent.
    Call once at startup; reuse across all ingestion runs.
    """
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=_SELECTOR_SYSTEM_PROMPT.format(top_k=top_k),
        response_format=SchemaSelectionResult,
    )


async def select_schema_chunks(
    report_text: str,
    stage_type: str,
    targets: list[str],
    index: list[dict],
    selector_agent,
) -> list[dict]:
    """
    Invoke the selector agent and return only the relevant chunk dicts.

    Args:
        report_text:    Full text of the final research report.
        stage_type:     Stage type string (e.g. "targeted_extraction").
        targets:        List of entity/domain targets (e.g. ["Elysium Health"]).
        index:          Full schema chunk index (from load_schema_index()).
        selector_agent: Built with build_schema_selector_agent().

    Returns:
        Subset of index dicts whose chunk_id was selected by the agent.
    """
    index_summary = "\n".join(
        f"- {c['chunk_id']}: {c['description']}  keywords={c['keywords']}"
        for c in index
    )
    user_message = (
        f"Stage: {stage_type}\n"
        f"Targets: {', '.join(targets)}\n\n"
        f"Report (first 800 chars):\n{report_text[:800]}\n\n"
        f"Available schema chunks:\n{index_summary}\n\n"
        f"Select the chunk_ids needed to fully extract this report."
    )

    result = await selector_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    selection: SchemaSelectionResult = result["structured_response"]
    selected_ids = set(selection.chunk_ids)
    chosen = [c for c in index if c["chunk_id"] in selected_ids]
    print(
        f"[schema_selector] Selected {len(chosen)} chunk(s): "
        f"{[c['chunk_id'] for c in chosen]}"
    )
    # Encode to ASCII with replacement to avoid Windows cp1252 console errors
    # on Unicode characters returned by the LLM (e.g. non-breaking hyphens).
    safe_reasoning = selection.reasoning.encode("ascii", errors="replace").decode("ascii")
    print(f"[schema_selector] Reasoning: {safe_reasoning}")
    return chosen
