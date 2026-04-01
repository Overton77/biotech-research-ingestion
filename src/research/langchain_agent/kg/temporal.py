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

All nodes now use `id` as the merge key and `name` as the universal name
property. State labels and identity properties are driven by the registry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPEN_ENDED = None

_HASH_EXCLUDED_KEYS = frozenset({
    "searchText", "searchFields", "embedding",
    "embeddingModel", "embeddingDimensions",
    "stateId", "createdAt", "stateHash",
    "sourceReport", "recordedFrom", "recordedTo",
    "searchEmbedding",
})

MERGE_KEY = "id"


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
    filtered: dict[str, Any] = {}
    for k, v in state_props.items():
        if k in _HASH_EXCLUDED_KEYS:
            continue
        filtered[k] = _normalize_value(v)

    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        stripped = v.strip()
        return stripped if stripped else None
    if isinstance(v, list):
        normalized = [_normalize_value(item) for item in v]
        return normalized if normalized else None
    if isinstance(v, float) and v != v:
        return None
    return v
