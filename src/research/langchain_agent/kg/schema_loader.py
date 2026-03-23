"""
Schema loader — builds compact JSON extraction contracts from schema_registry.json.

Replaces the .md file loading that was done by load_schema_chunks() in
schema_selector.py.  The LLM now receives a structured, typed property map
instead of free-form markdown prose.

Typical call chain:
    registry = load_schema_registry()
    contract  = build_extraction_contract(selected_chunks, registry)
    prompt_str = contract_to_prompt_string(contract)
"""

from __future__ import annotations

import json
from pathlib import Path

_KG_DIR = Path(__file__).parent
_REGISTRY_PATH = _KG_DIR / "schema" / "schema_registry.json"

# Fields stripped from every property spec before sending to the LLM —
# these are internal registry annotations, not schema semantics.
_INTERNAL_KEYS = {"system", "extract", "extractionNote", "note"}


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------


def load_schema_registry(registry_path: Path | None = None) -> dict:
    """Load and return the schema registry from disk (synchronous)."""
    path = registry_path or _REGISTRY_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Contract builder
# ---------------------------------------------------------------------------


def _clean_prop(spec: dict) -> dict:
    """Return a copy of a property spec with internal-only keys removed."""
    return {k: v for k, v in spec.items() if k not in _INTERNAL_KEYS}


def build_extraction_contract(
    selected_chunks: list[dict],
    registry: dict,
) -> dict:
    """
    Slice schema_registry.json to only the nodes and relationships that appear
    in the selected chunks.  System-only and non-extractable properties are
    stripped so the LLM receives only what it needs to populate.

    The contract groups extractable relationships (e.g. the five
    Organization→Person types) under their ``extractionClass`` name so the
    LLM can map directly to the Pydantic response_format without ambiguity.

    Args:
        selected_chunks: Subset of schema_index.json entries chosen by the
                         schema-selector agent.
        registry:        Full dict loaded by load_schema_registry().

    Returns:
        {
          "nodes": {
              "<Label>": {
                  "mergeKey": "...",
                  "searchFields": [...],
                  "properties": { "<prop>": {"type": "...", ...}, ... }
              }, ...
          },
          "relationships": {
              "<extractionClass>": {
                  "covers": ["REL_TYPE", ...],   # populated from coveredTypes
                  "from": "Label",
                  "to": "Label",
                  "properties": { ... }
              }, ...
          }
        }
    """
    wanted_nodes: set[str] = set()
    wanted_rels: set[str] = set()

    for chunk in selected_chunks:
        wanted_nodes.update(chunk.get("node_labels", []))
        wanted_rels.update(chunk.get("relationship_types", []))

    # --- Nodes ---------------------------------------------------------------
    nodes_out: dict = {}
    for label, node in registry.get("nodes", {}).items():
        if label not in wanted_nodes:
            continue

        props: dict = {}
        for pname, pspec in node.get("properties", {}).items():
            if pspec.get("system"):
                continue
            if not pspec.get("extract", True):
                continue
            props[pname] = _clean_prop(pspec)

        nodes_out[label] = {
            "mergeKey": node["mergeKey"],
            "searchFields": node.get("searchFields", []),
            "properties": props,
        }

    # --- Relationships -------------------------------------------------------
    # Build a map: coveredType → registry entry for quick lookup.
    # Some registry entries use "_OrgPersonRelationship" (a virtual grouping key)
    # and list their individual rel types in "coveredTypes".
    covered_type_to_entry: dict[str, dict] = {}
    for rel_key, rel in registry.get("relationships", {}).items():
        for ct in rel.get("coveredTypes", []):
            covered_type_to_entry[ct] = rel
        # Also map the key itself for direct matches
        covered_type_to_entry.setdefault(rel_key, rel)

    rels_out: dict = {}
    for rel_type in sorted(wanted_rels):
        rel = covered_type_to_entry.get(rel_type)
        if rel is None:
            continue
        if not rel.get("extractable", False):
            continue

        class_name = rel.get("extractionClass", rel_type)
        if class_name in rels_out:
            # Already emitted via another coveredType in the same group
            continue

        props: dict = {}
        for pname, pspec in rel.get("properties", {}).items():
            if pspec.get("system"):
                continue
            props[pname] = _clean_prop(pspec)

        entry: dict = {
            "from": rel["from"],
            "to": rel["to"],
            "properties": props,
        }
        if rel.get("coveredTypes"):
            entry["covers"] = rel["coveredTypes"]
        if rel.get("note"):
            entry["note"] = rel["note"]

        rels_out[class_name] = entry

    return {"nodes": nodes_out, "relationships": rels_out}


# ---------------------------------------------------------------------------
# Prompt serialisation
# ---------------------------------------------------------------------------


def contract_to_prompt_string(contract: dict) -> str:
    """
    Serialise the extraction contract to a compact JSON string suitable for
    injection into the extraction LLM prompt.
    """
    return json.dumps(contract, indent=2, ensure_ascii=False)
