"""
Generate schema_registry.json and schema_index.json from:
  1. biotech-kg schema-map.json  (raw GraphQL-derived node/relationship data)
  2. node_annotations.json       (ingestion-specific node metadata)
  3. relationship_annotations.json (ingestion-specific relationship metadata)
  4. chunk_definitions.json      (schema-selector chunk catalog)

Usage:
    python -m src.research.langchain_agent.kg.schema.generate_registry
    python -m src.research.langchain_agent.kg.schema.generate_registry --schema-map /path/to/schema-map.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_DIR = Path(__file__).parent

_DEFAULT_SCHEMA_MAP = (
    _SCHEMA_DIR.parents[5]  # Proyectos/Biotech
    / "biotech-kg"
    / "src"
    / "schema"
    / "reference"
    / "schema-map.json"
)

_NODE_ANNOTATIONS_PATH = _SCHEMA_DIR / "node_annotations.json"
_REL_ANNOTATIONS_PATH = _SCHEMA_DIR / "relationship_annotations.json"
_CHUNK_DEFS_PATH = _SCHEMA_DIR / "chunk_definitions.json"
_REGISTRY_OUT = _SCHEMA_DIR / "schema_registry.json"
_INDEX_OUT = _SCHEMA_DIR / "schema_index.json"

# GraphQL type signature -> registry type
_TYPE_MAP: dict[str, tuple[str, dict]] = {
    "ID!":          ("string", {"format": "uuid", "system": True}),
    "String!":      ("string", {"required": True}),
    "String":       ("string", {}),
    "Int":          ("int", {}),
    "Int!":         ("int", {"required": True}),
    "Float":        ("float", {}),
    "Float!":       ("float", {"required": True}),
    "Boolean":      ("bool", {}),
    "Boolean!":     ("bool", {"required": True}),
    "DateTime":     ("datetime", {}),
    "DateTime!":    ("datetime", {}),
    "[String!]":    ("array", {"items": "string"}),
    "[String]":     ("array", {"items": "string"}),
    "[Float!]":     ("array", {"items": "float"}),
    "[Float]":      ("array", {"items": "float"}),
    "[Int!]":       ("array", {"items": "int"}),
    "[Int]":        ("array", {"items": "int"}),
}

# Enum types from the SDL get mapped to string
_ENUM_TYPES = {"StudyPhase", "EvidenceStrength", "AssociationPolarity"}


def _map_gql_type(gql_sig: str) -> dict:
    """Convert a GraphQL type signature to a registry property spec."""
    if gql_sig in _TYPE_MAP:
        base_type, extras = _TYPE_MAP[gql_sig]
        return {"type": base_type, **extras}
    clean = gql_sig.rstrip("!")
    if clean in _ENUM_TYPES:
        return {"type": "string", "enum": clean}
    return {"type": "string"}


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate(schema_map_path: Path | None = None) -> None:
    schema_map_path = schema_map_path or _DEFAULT_SCHEMA_MAP
    if not schema_map_path.exists():
        print(f"[generate_registry] ERROR: schema-map.json not found at {schema_map_path}")
        sys.exit(1)

    schema_map = _load_json(schema_map_path)
    node_annotations = _load_json(_NODE_ANNOTATIONS_PATH)
    rel_annotations = _load_json(_REL_ANNOTATIONS_PATH)
    chunk_defs = _load_json(_CHUNK_DEFS_PATH)

    defaults = node_annotations.get("defaults", {})
    system_props = set(defaults.get("systemProperties", []))
    default_merge_key = defaults.get("mergeKey", "id")

    sm_nodes: dict = schema_map.get("nodes", {})
    ann_nodes: dict = node_annotations.get("nodes", {})
    ann_rels: dict = rel_annotations.get("relationships", {})

    # -----------------------------------------------------------------
    # Build registry nodes
    # -----------------------------------------------------------------
    registry_nodes: dict = {}

    for label, sm_node in sorted(sm_nodes.items()):
        ann = ann_nodes.get(label, {})
        extractable = ann.get("extractable", False)
        domain = sm_node.get("domain", "unknown")
        implements = sm_node.get("implements", [])

        node_entry: dict = {
            "domain": domain,
            "implements": implements,
            "extractable": extractable,
            "mergeKey": ann.get("mergeKey", default_merge_key),
            "nameProperty": ann.get("nameProperty", defaults.get("nameProperty", "name")),
            "searchFields": ann.get("searchFields", []),
            "fulltext": sm_node.get("fulltext", []),
            "vector": sm_node.get("vector", []),
        }

        if ann.get("stateLabel"):
            node_entry["stateLabel"] = ann["stateLabel"]
        if ann.get("identityProperties"):
            node_entry["identityProperties"] = ann["identityProperties"]
        if ann.get("extractionNote"):
            node_entry["extractionNote"] = ann["extractionNote"]

        domain_overrides = ann.get("domainProperties", {})

        props: dict = {}
        for prop_name, gql_sig in sm_node.get("properties", {}).items():
            spec = _map_gql_type(gql_sig)

            if prop_name in system_props:
                spec["system"] = True

            if prop_name in domain_overrides:
                override = domain_overrides[prop_name]
                if "extract" in override:
                    spec["extract"] = override["extract"]
                for k, v in override.items():
                    if k != "extract":
                        spec[k] = v

            props[prop_name] = spec

        node_entry["properties"] = props
        registry_nodes[label] = node_entry

    # -----------------------------------------------------------------
    # Build registry relationships
    # -----------------------------------------------------------------
    registry_rels: dict = {}

    for rel_key, ann_rel in sorted(ann_rels.items()):
        entry: dict = {
            "extractable": ann_rel.get("extractable", False),
            "from": ann_rel["from"],
            "to": ann_rel["to"],
        }
        if ann_rel.get("extractionClass"):
            entry["extractionClass"] = ann_rel["extractionClass"]
        if ann_rel.get("coveredTypes"):
            entry["coveredTypes"] = ann_rel["coveredTypes"]
        if ann_rel.get("note"):
            entry["note"] = ann_rel["note"]
        if ann_rel.get("searchFields"):
            entry["searchFields"] = ann_rel["searchFields"]
        if ann_rel.get("properties"):
            entry["properties"] = ann_rel["properties"]

        registry_rels[rel_key] = entry

    # -----------------------------------------------------------------
    # Build state nodes from stateLabel annotations
    # -----------------------------------------------------------------
    state_nodes: dict = {}

    for label, node_entry in registry_nodes.items():
        state_label = node_entry.get("stateLabel")
        if not state_label:
            continue

        sm_state = sm_nodes.get(state_label, {})
        state_entry: dict = {
            "mergeKey": "id",
            "parentLabel": label,
        }

        state_props: dict = {}
        for prop_name, gql_sig in sm_state.get("properties", {}).items():
            spec = _map_gql_type(gql_sig)
            if prop_name in system_props:
                spec["system"] = True
            state_props[prop_name] = spec

        # Add standard state-node system properties
        for sp in ["stateHash", "sourceReport"]:
            if sp not in state_props:
                state_props[sp] = {"type": "string", "system": True}

        state_entry["properties"] = state_props
        state_nodes[state_label] = state_entry

    # -----------------------------------------------------------------
    # Build relationship property types from schema-map
    # -----------------------------------------------------------------
    rel_prop_types: dict = {}
    for rpt_name, rpt_fields in schema_map.get("relationshipPropertyTypes", {}).items():
        mapped: dict = {}
        for field_name, gql_sig in rpt_fields.items():
            mapped[field_name] = _map_gql_type(gql_sig)
        rel_prop_types[rpt_name] = mapped

    # -----------------------------------------------------------------
    # Assemble registry
    # -----------------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    registry = {
        "$schema": "biotech-kg-schema-registry",
        "version": "3.0.0",
        "generatedAt": now,
        "sourceSchemaMap": str(schema_map_path.name),
        "sourceSchemaMapVersion": schema_map.get("version", "unknown"),
        "notes": "Auto-generated from schema-map.json + sidecar annotations. All nodes use id as mergeKey and name as the universal name property.",
        "nodes": registry_nodes,
        "relationships": registry_rels,
        "stateNodes": state_nodes,
        "relationshipPropertyTypes": rel_prop_types,
    }

    with open(_REGISTRY_OUT, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    extractable_count = sum(1 for n in registry_nodes.values() if n.get("extractable"))
    extractable_rels = sum(1 for r in registry_rels.values() if r.get("extractable"))
    print(
        f"[generate_registry] Wrote {_REGISTRY_OUT.name}: "
        f"{len(registry_nodes)} nodes ({extractable_count} extractable), "
        f"{len(registry_rels)} relationships ({extractable_rels} extractable), "
        f"{len(state_nodes)} state nodes"
    )

    # -----------------------------------------------------------------
    # Build schema_index.json from chunk_definitions
    # -----------------------------------------------------------------
    index = list(chunk_defs)

    with open(_INDEX_OUT, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"[generate_registry] Wrote {_INDEX_OUT.name}: "
        f"{len(index)} chunks"
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate schema_registry.json and schema_index.json")
    parser.add_argument(
        "--schema-map",
        type=Path,
        default=None,
        help="Path to schema-map.json (default: biotech-kg/src/schema/reference/schema-map.json)",
    )
    args = parser.parse_args()
    generate(schema_map_path=args.schema_map)


if __name__ == "__main__":
    main()
