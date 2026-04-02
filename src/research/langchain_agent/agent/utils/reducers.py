from typing import List, Dict, Any
import json

def _merge_unique_str_list(
    current: List[str] | None,
    incoming: List[str] | None,
    *,
    max_items: int = 2000,
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for value in (current or []) + (incoming or []):
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged[-max_items:]


def _merge_event_list(
    current: List[Dict[str, Any]] | None,
    incoming: List[Dict[str, Any]] | None,
    *,
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in (current or []) + (incoming or []):
        if not value:
            continue
        signature = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(value)
    return merged[-max_items:]
