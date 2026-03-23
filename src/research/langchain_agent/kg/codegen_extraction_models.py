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
    "datetime": "str",   # stored / sent as ISO-8601 string
}


def _py_type(spec: dict) -> str:
    t = spec["type"]
    if t == "array":
        item_t = _SCALAR_MAP.get(spec.get("items", "string"), "str")
        return f"list[{item_t}]"
    return _SCALAR_MAP.get(t, "str")


def _py_type_annotated(spec: dict) -> str:
    """Return the full Python type, adding '| None' for optional numerics."""
    base = _py_type(spec)
    if not spec.get("required") and spec["type"] in ("float", "int"):
        return f"{base} | None"
    return base


def _default_expr(spec: dict) -> str | None:
    """
    Return the Python default expression string, or None if the field is
    required (no default).
    """
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
    if t in ("float", "int"):
        return "None"
    if t == "bool":
        return "False"
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

    # searchFields directive — always last
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

    # Emit searchFields when the relationship defines them (e.g. denormalized
    # models like ExtractedCompoundIngredient that also create Neo4j nodes —
    # the searchtext engine reads searchFields from the entity dict).
    search_fields = rel.get("searchFields", [])
    if search_fields:
        sf_repr = repr(search_fields)
        lines.append(f"    searchFields: list[str] = Field(")
        lines.append(f"        default={sf_repr},")
        lines.append(f'        description="Fields used for searchText generation — do not alter.",')
        lines.append(f"    )")
        has_body = True

    if not has_body:
        lines.append("    pass")

    return lines


# ---------------------------------------------------------------------------
# KGExtractionResult template
# ---------------------------------------------------------------------------
# The top-level result is a fixed aggregation of all extractable entity lists.
# Adjust the field list here if new extractable types are added to the registry.

def _kg_result_fields(registry: dict) -> list[tuple[str, str]]:
    """
    Return (field_name, class_name) pairs for KGExtractionResult, ordered by:
    1. Extractable nodes (alphabetical by label)
    2. Extractable relationships (alphabetical by extractionClass)
    """
    fields: list[tuple[str, str]] = []

    # Nodes → list[Extracted<Label>]
    for label, node in sorted(registry.get("nodes", {}).items()):
        if not node.get("extractable"):
            continue
        field_name = label[0].lower() + label[1:] + "s"  # e.g. Organization → organizations
        fields.append((field_name, f"Extracted{label}"))

    # Relationships → list[<extractionClass>]
    seen: set[str] = set()
    for rel in sorted(registry.get("relationships", {}).values(), key=lambda r: r.get("extractionClass", "")):
        if not rel.get("extractable"):
            continue
        class_name = rel.get("extractionClass", "")
        if not class_name or class_name in seen:
            continue
        seen.add(class_name)

        # Derive a reasonable field name from the class name
        # ExtractedOrgPersonRelationship → org_person_relationships
        # ExtractedCompoundIngredient    → compound_ingredients
        raw = class_name.replace("Extracted", "")
        # CamelCase → snake_case
        import re
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
        # Pluralise
        if snake.endswith("ship"):
            field_name = snake + "s"
        elif snake.endswith("ient"):
            field_name = snake + "s"
        else:
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

Generated from: schema/schema_registry.json
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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

    # Report what was generated
    node_count = sum(1 for n in registry["nodes"].values() if n.get("extractable"))
    rel_count = len({
        r["extractionClass"]
        for r in registry["relationships"].values()
        if r.get("extractable") and r.get("extractionClass")
    })
    print(f"[codegen] {node_count} node model(s), {rel_count} relationship model(s)")


if __name__ == "__main__":
    main()
