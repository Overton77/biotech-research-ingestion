"""
Code generator: reads schema_registry.json and writes extraction_models.py.

Run this script after ANY change to schema_registry.json to keep the Pydantic
extraction models in sync with the registry.

Usage:
    python -m src.research.langchain_agent.kg.codegen_extraction_models

The generated file carries a header comment warning that it is auto-generated.
Do not edit extraction_models.py by hand — edit the registry and re-run.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

_KG_DIR = Path(__file__).parent
_REGISTRY_PATH = _KG_DIR / "schema" / "schema_registry.json"
_OUTPUT_PATH = _KG_DIR / "extraction_models.py"

# ---------------------------------------------------------------------------
# Type mapping: registry type → Python annotation string
# ---------------------------------------------------------------------------

_SCALAR_MAP: dict[str, str] = {
    "string":   "str",
    "float":    "float",
    "int":      "int",
    "bool":     "bool",
    "datetime": "str",
}


def _py_type(spec: dict) -> str:
    t = spec["type"]
    if t == "array":
        item_t = _SCALAR_MAP.get(spec.get("items", "string"), "str")
        return f"list[{item_t}]"
    return _SCALAR_MAP.get(t, "str")


def _py_type_annotated(spec: dict) -> str:
    base = _py_type(spec)
    if not spec.get("required") and spec["type"] in ("float", "int", "bool"):
        return f"{base} | None"
    return base


def _default_expr(spec: dict) -> str | None:
    if spec.get("required"):
        return None

    explicit = spec.get("default")
    t = spec["type"]

    if explicit is not None:
        if isinstance(explicit, bool):
            return str(explicit)
        if isinstance(explicit, str):
            return f'"{explicit}"'
        return str(explicit)

    if t == "array":
        return "[]"
    if t in ("float", "int", "bool"):
        return "None"
    return '""'


# ---------------------------------------------------------------------------
# Model generators
# ---------------------------------------------------------------------------


def _gen_node_model(label: str, node: dict) -> list[str]:
    """Generate lines for an Extracted<Label> Pydantic model."""
    lines = [f"class Extracted{label}(BaseModel):"]
    search_fields = node.get("searchFields", [])
    has_body = False

    for pname, pspec in node.get("properties", {}).items():
        if pspec.get("system"):
            continue
        if not pspec.get("extract", True):
            continue

        py_t = _py_type_annotated(pspec)
        default = _default_expr(pspec)

        if default is None:
            lines.append(f"    {pname}: {py_t}")
        else:
            lines.append(f"    {pname}: {py_t} = {default}")
        has_body = True

    # Temporal qualifier
    lines.append('    temporal: Optional[TemporalQualifier] = Field(')
    lines.append('        default=None,')
    lines.append('        description="Temporal evidence for this entity, if available in the report.",')
    lines.append('    )')

    sf_repr = repr(search_fields)
    lines.append(f"    searchFields: list[str] = Field(")
    lines.append(f"        default={sf_repr},")
    lines.append(f'        description="Fields used for searchText generation — do not alter.",')
    lines.append(f"    )")

    if not has_body:
        lines.insert(1, "    pass")

    return lines


def _gen_rel_model(rel: dict) -> list[str]:
    """Generate lines for an extraction relationship Pydantic model."""
    class_name = rel["extractionClass"]
    lines = [f"class {class_name}(BaseModel):"]
    has_body = False

    for pname, pspec in rel.get("properties", {}).items():
        if pspec.get("system"):
            continue

        py_t = _py_type_annotated(pspec)
        default = _default_expr(pspec)

        if default is None:
            lines.append(f"    {pname}: {py_t}")
        else:
            lines.append(f"    {pname}: {py_t} = {default}")
        has_body = True

    # Temporal qualifier
    lines.append('    temporal: Optional[TemporalQualifier] = Field(')
    lines.append('        default=None,')
    lines.append('        description="Temporal evidence for this relationship, if available.",')
    lines.append('    )')

    search_fields = rel.get("searchFields", [])
    if search_fields:
        sf_repr = repr(search_fields)
        lines.append(f"    searchFields: list[str] = Field(")
        lines.append(f"        default={sf_repr},")
        lines.append(f'        description="Fields used for searchText generation — do not alter.",')
        lines.append(f"    )")

    if not has_body:
        lines.append("    pass")

    return lines


# ---------------------------------------------------------------------------
# KGExtractionResult template
# ---------------------------------------------------------------------------


def _kg_result_fields(registry: dict) -> list[tuple[str, str]]:
    """
    Return (field_name, class_name) pairs for KGExtractionResult, ordered by:
    1. Extractable nodes (alphabetical by label)
    2. Extractable relationships (alphabetical by extractionClass)
    """
    fields: list[tuple[str, str]] = []

    for label, node in sorted(registry.get("nodes", {}).items()):
        if not node.get("extractable"):
            continue
        # e.g. Organization → organizations, Study → studies
        base = label[0].lower() + label[1:]
        if base.endswith("y") and not base.endswith("ey"):
            field_name = base[:-1] + "ies"
        else:
            field_name = base + "s"
        fields.append((field_name, f"Extracted{label}"))

    seen: set[str] = set()
    for rel in sorted(registry.get("relationships", {}).values(), key=lambda r: r.get("extractionClass", "")):
        if not rel.get("extractable"):
            continue
        class_name = rel.get("extractionClass", "")
        if not class_name or class_name in seen:
            continue
        seen.add(class_name)

        raw = class_name.replace("Extracted", "")
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
        field_name = snake + "s"
        fields.append((field_name, class_name))

    return fields


# ---------------------------------------------------------------------------
# File renderer
# ---------------------------------------------------------------------------

_HEADER = '''\
# AUTO-GENERATED — do not edit directly.
# Source of truth: schema/schema_registry.json
# Regenerate:  python -m src.research.langchain_agent.kg.codegen_extraction_models
"""
Pydantic models for KG entity extraction output.

All entity types the LLM may extract from a research report.
searchFields on each model drives deterministic searchText generation;
the LLM should not alter this list.

Temporal qualifiers: each entity and relationship carries optional
temporal_qualifier and temporal_context fields.  The LLM fills these
when the report provides explicit temporal evidence (e.g. "as of 2024",
"since March 2023", "formerly known as").

Generated from: schema/schema_registry.json
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Temporal context (passed INTO extraction, not extracted BY LLM)
# ---------------------------------------------------------------------------


class TemporalScope(BaseModel):
    """
    Temporal scope carried by the research configuration.

    Tells the extraction system what time frame the research covers.
    The LLM extraction prompt includes this so temporal reasoning is grounded.
    """

    mode: Literal["current", "as_of_date", "date_range", "unknown"] = "current"
    as_of_date: Optional[str] = Field(
        default=None,
        description="ISO date string (YYYY-MM-DD) when mode='as_of_date'.",
    )
    range_start: Optional[str] = Field(
        default=None,
        description="ISO date string for range start when mode='date_range'.",
    )
    range_end: Optional[str] = Field(
        default=None,
        description="ISO date string for range end when mode='date_range'.",
    )
    description: str = Field(
        default="Current state as of research date.",
        description="Human-readable description of the temporal scope.",
    )


class IngestionTemporalContext(BaseModel):
    """
    System-level temporal context passed through the ingestion pipeline.
    Not produced by the LLM — set by the orchestrator or coordinator.
    """

    research_date: Optional[datetime] = Field(
        default=None,
        description="When the research was conducted. Used as validFrom default.",
    )
    ingestion_time: Optional[datetime] = Field(
        default=None,
        description="When the ingestion run started. Used as recordedFrom.",
    )
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    source_report: str = ""


# ---------------------------------------------------------------------------
# Temporal qualifier (extracted BY the LLM when evidence exists)
# ---------------------------------------------------------------------------


class TemporalQualifier(BaseModel):
    """Optional temporal evidence extracted from the report text."""

    valid_from: Optional[str] = Field(
        default=None,
        description="ISO date or descriptive string (e.g. '2023-01', 'founded 2014'). When the fact became true.",
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="ISO date or descriptive string. When the fact ceased being true (null = still active).",
    )
    temporal_note: str = Field(
        default="",
        description="Free-text temporal context from the report, e.g. 'as of Q2 2025', 'since founding'.",
    )


# ---------------------------------------------------------------------------
# Extracted entities
# ---------------------------------------------------------------------------

'''


def render(registry: dict) -> str:
    sections: list[str] = [_HEADER]

    # ---- Extractable node models -------------------------------------------
    for label, node in sorted(registry.get("nodes", {}).items()):
        if not node.get("extractable"):
            continue
        lines = _gen_node_model(label, node)
        sections.append("\n".join(lines))
        sections.append("\n")

    # ---- Extractable relationship models -----------------------------------
    seen_classes: set[str] = set()
    for rel in registry.get("relationships", {}).values():
        if not rel.get("extractable"):
            continue
        class_name = rel.get("extractionClass", "")
        if not class_name or class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        lines = _gen_rel_model(rel)
        sections.append("\n".join(lines))
        sections.append("\n")

    # ---- KGExtractionResult ------------------------------------------------
    result_fields = _kg_result_fields(registry)
    result_lines = ['class KGExtractionResult(BaseModel):']
    result_lines.append('    """Top-level extraction output returned by the extraction agent."""')
    result_lines.append("")
    result_lines.append('    source_report: str = ""')
    for field_name, class_name in result_fields:
        result_lines.append(f"    {field_name}: list[{class_name}] = []")
    sections.append("\n".join(result_lines))
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    output = render(registry)

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(output)

    print(f"[codegen] Wrote {_OUTPUT_PATH}")

    node_count = sum(1 for n in registry["nodes"].values() if n.get("extractable"))
    rel_count = len({
        r["extractionClass"]
        for r in registry["relationships"].values()
        if r.get("extractable") and r.get("extractionClass")
    })
    print(f"[codegen] {node_count} node model(s), {rel_count} relationship model(s)")


if __name__ == "__main__":
    main()
