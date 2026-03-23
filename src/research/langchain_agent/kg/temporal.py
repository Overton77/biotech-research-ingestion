"""
Bitemporal utilities for the KG ingestion pipeline.

Provides:
  - State payload hashing (SHA-256 of normalized property dicts)
  - Bitemporal interval defaults (validFrom/validTo, recordedFrom/recordedTo)
  - Constants for open-ended interval representation
  - Identity vs state property partitioning per entity type

Design decisions:
  - Open-ended intervals use None (mapped to Neo4j null).
  - validFrom defaults to the research_date (when the fact is true).
  - recordedFrom defaults to ingestion time (when the system learned it).
  - State hashing ignores searchText/embedding/system fields — only domain
    properties participate in the hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPEN_ENDED = None  # Neo4j null — represents an active/current interval

# Properties that are NEVER included in state hash comparisons.
_HASH_EXCLUDED_KEYS = frozenset({
    "searchText", "searchFields", "embedding",
    "embeddingModel", "embeddingDimensions",
    "stateId", "createdAt", "stateHash",
    "sourceReport", "recordedFrom", "recordedTo",
})

# Per-label definition of which properties stay on the identity node
# vs which move to the state node.  Everything not listed as identity
# goes to the state node (minus system/search/embedding fields which
# are handled on the state node automatically).
IDENTITY_PROPERTIES: dict[str, set[str]] = {
    "Organization": {"organizationId", "name", "aliases", "createdAt"},
    "Person": {"personId", "canonicalName", "createdAt"},
    "Product": {"productId", "name", "synonyms", "createdAt"},
    "CompoundForm": {"compoundFormId", "canonicalName", "createdAt"},
}

STATE_LABEL_MAP: dict[str, str] = {
    "Organization": "OrganizationState",
    "Person": "PersonState",
    "Product": "ProductState",
    "CompoundForm": "CompoundFormState",
}

# The merge key on each identity node
IDENTITY_MERGE_KEY: dict[str, str] = {
    "Organization": "organizationId",
    "Person": "personId",
    "Product": "productId",
    "CompoundForm": "compoundFormId",
}

# The merge key on each state node
STATE_MERGE_KEY = "stateId"


# ---------------------------------------------------------------------------
# Temporal defaults
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def default_bitemporal_props(
    *,
    research_date: datetime | None = None,
    ingestion_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Return default bitemporal properties for a new HAS_STATE or structural
    relationship.

    Args:
        research_date:  When the fact is considered true in the domain.
                        Falls back to ingestion_time if not provided.
        ingestion_time: When the system recorded the fact.  Defaults to now.
    """
    now = ingestion_time or now_utc()
    valid_from = research_date or now

    return {
        "validFrom": valid_from.isoformat(),
        "validTo": OPEN_ENDED,
        "recordedFrom": now.isoformat(),
        "recordedTo": OPEN_ENDED,
    }


# ---------------------------------------------------------------------------
# State hashing
# ---------------------------------------------------------------------------


def compute_state_hash(state_props: dict[str, Any]) -> str:
    """
    Compute a SHA-256 hash of the domain-relevant state properties.

    The hash is used to detect whether an incoming state snapshot is
    materially different from the current active state.

    - Keys in _HASH_EXCLUDED_KEYS are dropped.
    - Remaining keys are sorted alphabetically.
    - Values are JSON-serialized with sort_keys=True for determinism.
    - Empty strings, empty lists, and None values are normalized.
    """
    filtered: dict[str, Any] = {}
    for k, v in state_props.items():
        if k in _HASH_EXCLUDED_KEYS:
            continue
        filtered[k] = _normalize_value(v)

    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_value(v: Any) -> Any:
    """Normalize a value for consistent hashing."""
    if v is None:
        return None
    if isinstance(v, str):
        stripped = v.strip()
        return stripped if stripped else None
    if isinstance(v, list):
        normalized = [_normalize_value(item) for item in v]
        return normalized if normalized else None
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


# ---------------------------------------------------------------------------
# State property partitioning
# ---------------------------------------------------------------------------


def partition_entity_props(
    label: str,
    full_props: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a flat entity property dict into (identity_props, state_props).

    Identity properties: stay on the durable identity node.
    State properties: go onto the immutable state snapshot node.
    """
    identity_keys = IDENTITY_PROPERTIES.get(label, set())
    identity = {}
    state = {}

    for k, v in full_props.items():
        if k in identity_keys:
            identity[k] = v
        else:
            state[k] = v

    return identity, state
